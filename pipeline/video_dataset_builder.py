"""Build video-first training datasets from dashcam folders.

Pipeline:
1) Extract per-frame CV features from videos in 0009/0010/1001.
2) Aggregate sliding windows (default 20 frames, stride 5) into one feature row.
3) Generate a proxy eco score target for initial XGBoost training.

Run example:
    python pipeline/video_dataset_builder.py --sample-every 5 --window-size 20 --stride 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import math

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.cv_pipeline import cv_pipeline

OUT_DIR = ROOT / "data" / "video"
FRAMES_OUT = OUT_DIR / "frames_master.csv"
WINDOWS_OUT = OUT_DIR / "windows_master.csv"

VIDEO_DIRS_DEFAULT = ["0009", "0010", "1001"]

# Keep this in sync with runtime schema in cv/cv_pipeline.py
XGB_FEATURE_SCHEMA = [
    "vehicle_count",
    "proximity_score",
    "pedestrian_flag",
    "mean_flow",
    "flow_variance",
    "braking_flag",
    "lane_change_flag",
    "road_type_id",
    "weather_id",
]


def iter_videos(video_dirs: list[str], max_videos_per_dir: int | None) -> list[Path]:
    videos: list[Path] = []
    for folder in video_dirs:
        d = ROOT / folder
        if not d.exists():
            print(f"[video_builder] WARNING: missing folder {d}")
            continue
        folder_videos = sorted(d.glob("*.mp4"))
        if max_videos_per_dir is not None:
            folder_videos = folder_videos[:max_videos_per_dir]
        videos.extend(folder_videos)
    return videos


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def extract_per_frame_features(
    video_path: Path,
    sample_every: int,
    max_frames_per_video: int | None = None,
) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[video_builder] WARNING: unable to open {video_path}")
        return pd.DataFrame()

    fps = _safe_float(cap.get(cv2.CAP_PROP_FPS), 30.0)
    if fps <= 0:
        fps = 30.0

    rows: list[dict] = []
    sampled_prev = None
    frame_idx = 0
    sampled_count = 0
    error_count = 0
    first_error = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue

        try:
            feats = cv_pipeline(frame, sampled_prev, telemetry=None)
            cv_error = ""
        except Exception as e:
            error_count += 1
            if first_error is None:
                first_error = repr(e)
            feats = {
                "vehicle_count": 0,
                "proximity_score": 0.0,
                "pedestrian_flag": 0,
                "mean_flow": 0.0,
                "flow_variance": 0.0,
                "braking_flag": 0,
                "lane_change_flag": 0,
                "road_type_id": 0,
                "weather_id": 0,
            }
            cv_error = repr(e)

        row = {
            "video_id": video_path.stem,
            "source_dir": video_path.parent.name,
            "frame_idx": frame_idx,
            "timestamp_sec": frame_idx / fps,
        }
        for c in XGB_FEATURE_SCHEMA:
            row[c] = _safe_float(feats.get(c, 0.0), 0.0)
        row["cv_error"] = cv_error
        rows.append(row)

        sampled_prev = frame
        sampled_count += 1
        frame_idx += 1

        if max_frames_per_video is not None and sampled_count >= max_frames_per_video:
            break

    cap.release()

    if sampled_count > 0 and error_count > 0:
        print(
            f"[video_builder] WARNING: {video_path.name} had {error_count}/{sampled_count} "
            f"CV extraction failures. First error: {first_error}"
        )

    return pd.DataFrame(rows)


def process_video(video_path: Path, sample_every: int, max_frames_per_video: int | None) -> pd.DataFrame:
    """Backward-compatible wrapper for per-frame feature extraction."""
    return extract_per_frame_features(video_path, sample_every, max_frames_per_video)


def _mode_or_default(series: pd.Series, default: float = 0.0) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return default
    m = s.mode()
    return float(m.iloc[0]) if not m.empty else default


def _proxy_eco_score(row: dict) -> float:
    # More sensitive proxy target so neighboring windows separate more clearly.
    # This intentionally amplifies unsafe maneuvers and motion instability.
    proximity = float(row["proximity_score"])
    mean_flow = float(row["mean_flow"])
    flow_variance = max(float(row["flow_variance"]), 0.0)
    braking = float(row["braking_flag"])
    lane = float(row["lane_change_flag"])
    pedestrian = float(row["pedestrian_flag"])

    score = 96.0
    score -= proximity * 36.0
    score -= mean_flow * 12.0
    score -= np.sqrt(flow_variance) * 42.0
    score -= flow_variance * 12.0
    score -= braking * 22.0
    score -= lane * 18.0
    score -= pedestrian * 10.0

    # Interaction penalties increase sensitivity when multiple risks co-occur.
    score -= proximity * braking * 10.0
    score -= mean_flow * lane * 6.0

    # Keep a modest floor to preserve learnable low-score spread.
    score = max(5.0, score)
    return float(max(0.0, min(100.0, score)))


def _balanced_video_split(frames_df: pd.DataFrame) -> dict[str, str]:
    """Build a source-aware split map with minimum-safe counts per split.

    Rules per source_dir group:
    - n=1: train
    - n=2: train/test
    - n>=3: at least one in each of train/val/test
    """
    rng = np.random.default_rng(42)
    split_map: dict[str, str] = {}

    for source_dir, g in frames_df.groupby("source_dir", sort=True):
        vids = sorted(g["video_id"].astype(str).unique().tolist())
        vids = [vids[i] for i in rng.permutation(len(vids))]
        n = len(vids)

        if n == 1:
            split_map[vids[0]] = "train"
            continue

        if n == 2:
            split_map[vids[0]] = "train"
            split_map[vids[1]] = "test"
            continue

        n_val = max(1, int(math.floor(n * 0.15)))
        n_test = max(1, int(math.floor(n * 0.15)))
        n_train = n - n_val - n_test

        # Guarantee at least one training video.
        if n_train < 1:
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            n_train = n - n_val - n_test

        train_vids = vids[:n_train]
        val_vids = vids[n_train:n_train + n_val]
        test_vids = vids[n_train + n_val:]

        for v in train_vids:
            split_map[v] = "train"
        for v in val_vids:
            split_map[v] = "val"
        for v in test_vids:
            split_map[v] = "test"

    return split_map


def aggregate_windows(frames_df: pd.DataFrame, window_size: int, stride: int) -> pd.DataFrame:
    if frames_df.empty:
        return pd.DataFrame()

    window_rows: list[dict] = []
    for video_id, g in frames_df.groupby("video_id", sort=True):
        g = g.sort_values("frame_idx").reset_index(drop=True)
        n = len(g)
        if n < window_size:
            continue

        for start in range(0, n - window_size + 1, stride):
            seg = g.iloc[start:start + window_size]

            row = {
                "video_id": video_id,
                "source_dir": seg.iloc[0]["source_dir"],
                "window_start_frame": int(seg.iloc[0]["frame_idx"]),
                "window_end_frame": int(seg.iloc[-1]["frame_idx"]),
                "window_center_sec": float(seg["timestamp_sec"].mean()),
                "window_size": int(window_size),
            }

            # Window-level schema values (compatible with runtime vectorization)
            row["vehicle_count"] = float(seg["vehicle_count"].mean())
            row["proximity_score"] = float(seg["proximity_score"].max())
            row["pedestrian_flag"] = float((seg["pedestrian_flag"] > 0).mean())
            row["mean_flow"] = float(seg["mean_flow"].mean())
            row["flow_variance"] = float(seg["flow_variance"].mean())
            row["braking_flag"] = float((seg["braking_flag"] > 0).mean())
            row["lane_change_flag"] = float((seg["lane_change_flag"] > 0).mean())
            row["braking_flag_ratio"] = row["braking_flag"]
            row["lane_change_flag_ratio"] = row["lane_change_flag"]
            row["proximity_score_mean"] = float(seg["proximity_score"].mean())
            row["mean_flow_mean"] = float(seg["mean_flow"].mean())
            row["timestamp_sec"] = float(seg["timestamp_sec"].mean())

            # Richer window stats (offline quality features; core runtime schema remains unchanged).
            for c in ["vehicle_count", "proximity_score", "mean_flow", "flow_variance"]:
                row[f"{c}_std"] = float(seg[c].std(ddof=0))
                row[f"{c}_min"] = float(seg[c].min())
                row[f"{c}_max"] = float(seg[c].max())

            row["pedestrian_count"] = float((seg["pedestrian_flag"] > 0).sum())
            row["braking_count"] = float((seg["braking_flag"] > 0).sum())
            row["lane_change_count"] = float((seg["lane_change_flag"] > 0).sum())

            road = seg["road_type_id"].replace(-1, np.nan)
            weather = seg["weather_id"].replace(-1, np.nan)
            row["road_type_id"] = _mode_or_default(road, default=0.0)
            row["weather_id"] = _mode_or_default(weather, default=0.0)

            row["eco_score_proxy"] = _proxy_eco_score(row)
            window_rows.append(row)

    out = pd.DataFrame(window_rows)
    if out.empty:
        return out

    # Source-aware deterministic split map to reduce distribution skew and leakage.
    split_map = _balanced_video_split(frames_df)
    out["split"] = out["video_id"].map(split_map).fillna("train")
    return out


def build_dataset(
    video_dirs: list[str],
    sample_every: int,
    window_size: int,
    stride: int,
    max_videos_per_dir: int | None,
    max_frames_per_video: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = iter_videos(video_dirs, max_videos_per_dir=max_videos_per_dir)
    print(f"[video_builder] Videos selected: {len(videos)}")

    frame_dfs = []
    total_rows = 0
    total_errors = 0
    for i, vp in enumerate(videos, start=1):
        print(f"[video_builder] ({i}/{len(videos)}) {vp.parent.name}/{vp.name}")
        df = process_video(vp, sample_every=sample_every, max_frames_per_video=max_frames_per_video)
        if not df.empty:
            total_rows += len(df)
            total_errors += int((df["cv_error"] != "").sum())
            frame_dfs.append(df)

    frames_df = pd.concat(frame_dfs, ignore_index=True) if frame_dfs else pd.DataFrame()

    if total_rows > 0 and total_errors == total_rows:
        raise RuntimeError(
            "All sampled frames failed CV feature extraction. "
            "Install missing CV dependencies (for example: ultralytics) and retry."
        )

    frames_df.to_csv(FRAMES_OUT, index=False)
    print(
        f"[video_builder] Saved frame dataset: {FRAMES_OUT} shape={frames_df.shape} "
        f"cv_errors={total_errors}/{total_rows}"
    )

    windows_df = aggregate_windows(frames_df, window_size=window_size, stride=stride)
    windows_df.to_csv(WINDOWS_OUT, index=False)
    print(f"[video_builder] Saved window dataset: {WINDOWS_OUT} shape={windows_df.shape}")

    return frames_df, windows_df


def main():
    parser = argparse.ArgumentParser(description="Build video-first training datasets")
    parser.add_argument("--video-dirs", nargs="+", default=VIDEO_DIRS_DEFAULT,
                        help="Video folders relative to project root (default: 0009 0010 1001)")
    parser.add_argument("--sample-every", type=int, default=5,
                        help="Sample every Nth frame")
    parser.add_argument("--window-size", type=int, default=20,
                        help="Sliding window size in sampled frames")
    parser.add_argument("--stride", type=int, default=5,
                        help="Sliding window stride")
    parser.add_argument("--max-videos-per-dir", type=int, default=None,
                        help="Limit videos per directory for quick runs")
    parser.add_argument("--max-frames-per-video", type=int, default=None,
                        help="Limit sampled frames per video for quick runs")
    args = parser.parse_args()

    build_dataset(
        video_dirs=args.video_dirs,
        sample_every=max(1, args.sample_every),
        window_size=max(2, args.window_size),
        stride=max(1, args.stride),
        max_videos_per_dir=args.max_videos_per_dir,
        max_frames_per_video=args.max_frames_per_video,
    )


if __name__ == "__main__":
    main()
