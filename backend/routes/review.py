"""
backend/routes/review.py
POST /api/review
Multipart field: video (mp4)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from flask import Blueprint, current_app, jsonify, request

from cv.cv_pipeline import feature_vector_for_xgb
from pipeline.video_dataset_builder import aggregate_windows
from backend.scoring import score_window, score_windows_with_ema, event_to_issue_key

review_bp = Blueprint("review", __name__)
CV_DEBUG = os.environ.get("DRIVEIQ_CV_DEBUG", "0") == "1"
logger = logging.getLogger("driveiq.review")

# Static dictionary removed, now uses centralized rule engine


GREEN_THRESHOLD = 80.0
YELLOW_THRESHOLD = 65.0

def classify_severity(score):
    if score >= GREEN_THRESHOLD:
        return "green"
    elif score >= YELLOW_THRESHOLD:
        return "yellow"
    else:
        return "red"


def _generate_session_summary_gemini(windows: list[dict], duration_sec: float) -> dict:
    """Use Gemini to generate a structured overall session summary with robust truncation recovery."""
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"summary": None, "error": "gemini_unavailable"}

        import google.genai as genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        # Aggregate stats from windows
        scores = [float(w.get("score", 0)) for w in windows]
        avg_score = sum(scores) / max(len(scores), 1)
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 0

        # Count events
        event_counts: dict[str, int] = {}
        for w in windows:
            for ev in w.get("events", []):
                event_counts[ev] = event_counts.get(ev, 0) + 1

        # Aggregate feature averages (from coach_note context)
        feature_keys = ["mean_flow", "flow_variance", "low_motion_ratio", "proximity_score"]
        feature_avgs = {}
        for fk in feature_keys:
            vals = [float(w.get(fk, 0)) for w in windows if fk in w]
            feature_avgs[fk] = round(sum(vals) / max(len(vals), 1), 3) if vals else 0.0

        prompt = (
            "You are a professional driving analyst. Analyze this driving session data and return ONLY valid JSON with no markdown formatting, no code fences, no preamble.\n"
            f"\nSession stats:\n"
            f"- Duration: {duration_sec:.1f} seconds\n"
            f"- Windows analyzed: {len(windows)}\n"
            f"- Average score: {avg_score:.1f} / 100\n"
            f"- Min score: {min_score:.1f}, Max score: {max_score:.1f}\n"
            f"- Events detected: {json.dumps(event_counts) if event_counts else 'None'}\n"
            f"- Avg mean_flow: {feature_avgs.get('mean_flow', 0):.3f}\n"
            f"- Avg flow_variance: {feature_avgs.get('flow_variance', 0):.3f}\n"
            f"- Avg low_motion_ratio: {feature_avgs.get('low_motion_ratio', 0):.3f}\n"
            f"- Avg proximity_score: {feature_avgs.get('proximity_score', 0):.3f}\n"
            "\nReturn JSON with exactly these keys:\n"
            '- "overall_rating": one of "Excellent", "Good", "Needs Improvement", "Poor"\n'
            '- "what_went_well": list of 2-3 SHORT phrases (max 12 words each)\n'
            '- "areas_to_improve": list of 2-3 SHORT actionable phrases (max 12 words each)\n'
            '- "summary_paragraph": 2 sentences max\n'
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,  # FIX: was 512/1000, causing truncation mid-response
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "overall_rating": {
                            "type": "string",
                            # FIX: enum constraint prevents unexpected values and reduces token usage
                            "enum": ["Excellent", "Good", "Needs Improvement", "Poor"]
                        },
                        "what_went_well": {
                            "type": "array",
                            # FIX: maxLength enforced at schema level, not just via prompt instruction
                            "items": {"type": "string", "maxLength": 80},
                            "minItems": 2,
                            "maxItems": 3
                        },
                        "areas_to_improve": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 80},
                            "minItems": 2,
                            "maxItems": 3
                        },
                        "summary_paragraph": {
                            "type": "string",
                            # FIX: hard cap on paragraph length keeps total JSON well under 2048 tokens
                            "maxLength": 300
                        }
                    },
                    "required": ["overall_rating", "what_went_well", "areas_to_improve", "summary_paragraph"]
                }
            )
        )

        raw_text = response.text.strip()

        # Strip any accidental markdown fences Gemini may still emit
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[-2] if "```" in raw_text[3:] else raw_text
            raw_text = raw_text.lstrip("`").lstrip("json").strip()

        try:
            parsed = json.loads(raw_text)
            return {"summary": parsed, "error": None}
        except json.JSONDecodeError:
            # Last-resort: try to recover a truncated JSON object by closing it
            recovered = _try_recover_truncated_json(raw_text)
            if recovered:
                logger.warning("/api/review gemini_summary_recovered via truncation fix")
                return {"summary": recovered, "error": None}

            logger.warning(f"Gemini session summary JSON parse failed. Raw: {raw_text[:200]}...")
            return {"summary": None, "error": "gemini_parse_error", "raw": raw_text}

    except Exception as e:
        logger.error(f"Gemini session summary failed: {e}")
        return {"summary": None, "error": "gemini_unavailable"}


def _try_recover_truncated_json(raw: str) -> dict | None:
    """
    Best-effort recovery for a truncated JSON object from Gemini.
    Tries progressively shorter suffixes to close the object validly.
    Only accepts results that contain all four required keys.
    """
    required_keys = {"overall_rating", "what_went_well", "areas_to_improve", "summary_paragraph"}
    closers = ['"}]}', '"]}', '"}', ']}}', '}}', '}']
    for closer in closers:
        try:
            candidate = raw.rstrip().rstrip(",") + closer
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and required_keys.issubset(parsed.keys()):
                return parsed
        except (json.JSONDecodeError, Exception):
            continue
    return None


def _extract_review_fast_features(
    video_path: Path,
    sample_every: int = 3,
    max_frames_per_video: int | None = None,
) -> pd.DataFrame:
    """Review extractor using the real CV pipeline (YOLO + optical flow)."""
    from cv.cv_pipeline import cv_pipeline
    from cv.optical_flow import reset_flow_state
    reset_flow_state()  # Clear acceleration state between videos

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return pd.DataFrame()

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0

    rows: list[dict] = []
    prev_frame: np.ndarray | None = None
    read_frame_idx = 0
    sampled_count = 0
    prev_ts_sec: float | None = None
    timeline_gap_count = 0
    max_gap_sec = 0.0
    expected_dt = sample_every / fps

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if read_frame_idx % sample_every != 0:
            read_frame_idx += 1
            continue

        actual_ts_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        if actual_ts_sec <= 0.0:
            actual_ts_sec = read_frame_idx / fps

        if prev_ts_sec is not None:
            dt = max(0.0, actual_ts_sec - prev_ts_sec)
            if dt > expected_dt * 1.3:
                timeline_gap_count += 1
                max_gap_sec = max(max_gap_sec, dt)
        prev_ts_sec = actual_ts_sec

        actual_frame_idx = int(round(actual_ts_sec * fps))

        try:
            # Use the same CV pipeline as the live /api/score endpoint
            cv_feats = cv_pipeline(frame, prev_frame, telemetry={})
        except Exception as e:
            print("CV PIPELINE FAILED!!!!", type(e), repr(e))
            logger.error(f"cv_pipeline error: {e}")
            # Fallback to zero features if CV pipeline fails on a frame
            cv_feats = {
                "mean_flow": 0.0, "flow_variance": 0.0,
                "braking_ratio": 0.0, "lane_change_ratio": 0.0,
                "proximity_score": 0.0, "vehicle_density": 0.0,
                "pedestrian_ratio": 0.0, "low_motion_ratio": 1.0,
                "braking_flag": 0.0, "lane_change_flag": 0.0,
                "vehicle_count": 0.0, "pedestrian_flag": 0.0,
            }

        rows.append(
            {
                "video_id": video_path.stem,
                "source_dir": video_path.parent.name,
                "frame_idx": actual_frame_idx,
                "timestamp_sec": actual_ts_sec,
                # 8-feature schema from real CV pipeline
                "mean_flow": float(cv_feats.get("mean_flow", 0.0)),
                "flow_variance": float(cv_feats.get("flow_variance", cv_feats.get("variance", 0.0))),
                "braking_ratio": float(cv_feats.get("braking_ratio", cv_feats.get("braking_flag", 0.0))),
                "lane_change_ratio": float(cv_feats.get("lane_change_ratio", cv_feats.get("lane_change_flag", 0.0))),
                "proximity_score": float(cv_feats.get("proximity_score", 0.0)),
                "vehicle_density": float(cv_feats.get("vehicle_density", cv_feats.get("vehicle_count", 0.0))),
                "pedestrian_ratio": float(cv_feats.get("pedestrian_ratio", cv_feats.get("pedestrian_flag", 0.0))),
                "low_motion_ratio": float(cv_feats.get("low_motion_ratio", 0.0)),
                # Legacy keys
                "vehicle_count": float(cv_feats.get("vehicle_count", cv_feats.get("vehicle_density", 0.0))),
                "braking_flag": float(cv_feats.get("braking_flag", cv_feats.get("braking_ratio", 0.0))),
                "lane_change_flag": float(cv_feats.get("lane_change_flag", cv_feats.get("lane_change_ratio", 0.0))),
                "pedestrian_flag": float(cv_feats.get("pedestrian_flag", cv_feats.get("pedestrian_ratio", 0.0))),
                "road_type_id": 0.0,
                "weather_id": 0.0,
            }
        )

        prev_frame = frame
        sampled_count += 1
        read_frame_idx += 1

        if max_frames_per_video is not None and sampled_count >= max_frames_per_video:
            break

    cap.release()

    if timeline_gap_count > 0:
        logger.warning(
            "/api/review timeline_gaps_detected video=%s count=%s max_gap_sec=%.3f",
            video_path.name,
            timeline_gap_count,
            max_gap_sec,
        )

    return pd.DataFrame(rows)


def _heuristic_score_from_features(features: dict) -> float:
    score = 85.0
    score -= float(features.get("proximity_score", 0.0)) * 30.0
    score -= float(features.get("mean_flow", 0.0)) * 8.0
    score -= float(features.get("flow_variance", 0.0)) * 10.0
    score -= float(features.get("braking_ratio", features.get("braking_flag", 0.0))) * 12.0
    score -= float(features.get("lane_change_ratio", features.get("lane_change_flag", 0.0))) * 10.0
    score -= float(features.get("pedestrian_ratio", features.get("pedestrian_flag", 0.0))) * 5.0
    return max(0.0, min(100.0, score))


def _evaluate_rules(score: float, features: dict, events: list[str]) -> list[str]:
    tips = []
    if "hard_braking" in events:
        tips.append("Reduce hard braking — anticipate stops earlier.")
    if "tailgating" in events:
        tips.append("Increase following distance to reduce collision risk.")
    if "lane_swerving" in events:
        tips.append("Maintain steady lane position — avoid abrupt steering.")
    if "erratic_speed" in events or float(features.get("flow_variance", 0)) > 15:
        tips.append("Erratic speed changes detected — maintain steady throttle.")
    if float(features.get("mean_flow", 0)) > 15:
        tips.append("High optical flow — possible aggressive acceleration.")
    if float(features.get("low_motion_ratio", 0)) > 0.6:
        tips.append("Extended idle detected — consider smoother route planning.")
    if not tips:
        tips.append("Maintain smooth, consistent driving.")
    return tips


@review_bp.route("/api/review", methods=["POST"])
def review():
    t0 = time.perf_counter()
    logger.info("/api/review start")
    video_file = request.files.get("video")
    if video_file is None:
        return jsonify({"error": "missing_video", "message": "Field 'video' is required"}), 400

    if not video_file.filename:
        return jsonify({"error": "empty_video", "message": "Uploaded video is empty"}), 400

    scoring_mode = request.form.get("scoring_mode", "event_rules")  # "event_rules" or "xgboost"

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_path = Path(tmp.name)
            video_file.save(str(temp_path))
        logger.info("/api/review upload_saved path=%s size_bytes=%s", temp_path, temp_path.stat().st_size)

        t_extract = time.perf_counter()
        sample_every = 3
        window_size = 20
        stride = 15

        frames_df = _extract_review_fast_features(temp_path, sample_every=sample_every, max_frames_per_video=None)
        logger.info(
            "/api/review extract_done rows=%s elapsed_ms=%.1f",
            len(frames_df),
            (time.perf_counter() - t_extract) * 1000.0,
        )
        if frames_df.empty:
            return jsonify({"windows": [], "duration_sec": 0.0, "window_count": 0})

        t_windows = time.perf_counter()
        # Reduce overlap from 75% to 25% for more independent window scores.
        windows_df = aggregate_windows(frames_df, window_size=window_size, stride=stride)

        # Skip windows with excessive missing sampled frames.
        # Estimate sampled-frame continuity from actual frame index span.
        if not windows_df.empty:
            span_frames = (windows_df["window_end_frame"] - windows_df["window_start_frame"]).astype(float)
            observed_slots = np.maximum(1.0, np.floor(span_frames / float(sample_every)) + 1.0)
            missing_slots = np.maximum(0.0, observed_slots - float(window_size))
            missing_ratio = np.where(observed_slots > 0.0, missing_slots / observed_slots, 0.0)
            windows_df["missing_frame_ratio"] = missing_ratio

            dropped_windows = windows_df[windows_df["missing_frame_ratio"] > 0.30]
            if not dropped_windows.empty:
                logger.warning(
                    "/api/review skipped_windows_missing_frames count=%s threshold=0.30 max_ratio=%.3f",
                    len(dropped_windows),
                    float(dropped_windows["missing_frame_ratio"].max()),
                )
                windows_df = windows_df[windows_df["missing_frame_ratio"] <= 0.30].reset_index(drop=True)

        logger.info(
            "/api/review windows_done rows=%s elapsed_ms=%.1f",
            len(windows_df),
            (time.perf_counter() - t_windows) * 1000.0,
        )
        if windows_df.empty:
            duration_sec = float(frames_df["timestamp_sec"].max()) if "timestamp_sec" in frames_df.columns else 0.0
            return jsonify({"windows": [], "duration_sec": round(duration_sec, 3), "window_count": 0})
        rows_iter = [r for _, r in windows_df.iterrows()]
        duration_sec = float(frames_df["timestamp_sec"].max()) if "timestamp_sec" in frames_df.columns else 0.0

        models = current_app.config.get("MODELS", {})
        xgb = models.get("xgb")
        scaler = models.get("scaler")
        scoring_path = "xgb" if xgb is not None and scaler is not None else "heuristic"
        logger.info("/api/review scoring_path=%s", scoring_path)

        # Build feature dicts for all windows
        windows_out = []
        t_score = time.perf_counter()
        all_feature_dicts = []
        for row in rows_iter:
            feature_dict = {
                "mean_flow": float(row.get("mean_flow", 0.0)),
                "flow_variance": float(row.get("flow_variance", 0.0)),
                "braking_ratio": float(row.get("braking_ratio", row.get("braking_flag", 0.0))),
                "lane_change_ratio": float(row.get("lane_change_ratio", row.get("lane_change_flag", 0.0))),
                "proximity_score": float(row.get("proximity_score", 0.0)),
                "vehicle_density": float(row.get("vehicle_density", row.get("vehicle_count", 0.0))),
                "pedestrian_ratio": float(row.get("pedestrian_ratio", row.get("pedestrian_flag", 0.0))),
                "low_motion_ratio": float(row.get("low_motion_ratio", 0.0)),
                "braking_flag": float(row.get("braking_flag", row.get("braking_ratio", 0.0))),
                "lane_change_flag": float(row.get("lane_change_flag", row.get("lane_change_ratio", 0.0))),
                "pedestrian_flag": float(row.get("pedestrian_flag", 0.0)),
            }
            all_feature_dicts.append((row, feature_dict))

        # Score all windows based on the requested scoring mode
        use_xgb = scoring_mode == "xgboost" and xgb is not None and scaler is not None
        actual_score_source = "xgboost" if use_xgb else "event_ema"
        logger.info("/api/review scoring_mode_requested=%s actual=%s", scoring_mode, actual_score_source)

        if use_xgb:
            # XGBoost scoring with EMA smoothing
            from cv.cv_pipeline import feature_vector_for_xgb
            prev_smoothed_xgb = 90.0  # BASE_SCORE
            for i, (row, feature_dict) in enumerate(all_feature_dicts):
                # Get events from event engine for timeline
                _, events = score_window(feature_dict)

                # XGBoost prediction
                try:
                    xgb_input = feature_vector_for_xgb(feature_dict, scaler)
                    raw_score = float(xgb.predict(xgb_input)[0])
                    raw_score = max(0.0, min(100.0, raw_score))
                except Exception as e:
                    logger.warning(f"XGBoost predict failed for window {i}: {e}")
                    raw_score, _ = score_window(feature_dict)

                # EMA smoothing
                smoothed = 0.6 * raw_score + 0.4 * prev_smoothed_xgb
                smoothed = max(0.0, min(100.0, smoothed))
                prev_smoothed_xgb = smoothed

                severity = classify_severity(smoothed)
                top_issue = event_to_issue_key(events)
                rule_tips = _evaluate_rules(smoothed, feature_dict, events)
                coach_note = rule_tips[0] if rule_tips else "Maintain smooth, consistent driving."

                windows_out.append({
                    "timestamp_sec": round(float(row.get("timestamp_sec", row.get("window_center_sec", 0.0))), 3),
                    "score": round(smoothed, 2),
                    "severity": severity,
                    "top_issue": top_issue,
                    "coach_note": coach_note,
                    "score_source": "xgboost",
                    "events": events,
                })
        else:
            # Event-based deduction scoring with EMA
            scored = score_windows_with_ema([fd for _, fd in all_feature_dicts])

            for i, (row, feature_dict) in enumerate(all_feature_dicts):
                result = scored[i]
                score_val = result["smoothed_score"]
                events = result["events"]

                severity = classify_severity(score_val)
                top_issue = event_to_issue_key(events)
                rule_tips = _evaluate_rules(score_val, feature_dict, events)
                coach_note = rule_tips[0] if rule_tips else "Maintain smooth, consistent driving."

                windows_out.append({
                    "timestamp_sec": round(float(row.get("timestamp_sec", row.get("window_center_sec", 0.0))), 3),
                    "score": round(score_val, 2),
                    "severity": severity,
                    "top_issue": top_issue,
                    "coach_note": coach_note,
                    "score_source": "event_ema",
                    "events": events,
                })

        logger.info(
            "/api/review score_done rows=%s elapsed_ms=%.1f",
            len(windows_out),
            (time.perf_counter() - t_score) * 1000.0,
        )

        if CV_DEBUG and windows_out:
            logger.info("/api/review sample_window=%s", windows_out[0])

        logger.info("/api/review done elapsed_ms=%.1f", (time.perf_counter() - t0) * 1000.0)

        # Generate Gemini-powered session summary
        session_summary = _generate_session_summary_gemini(windows_out, duration_sec)
        logger.info("/api/review session_summary error=%s", session_summary.get("error"))

        return jsonify(
            {
                "windows": windows_out,
                "duration_sec": round(duration_sec, 3),
                "window_count": len(windows_out),
                "severity_thresholds": {
                    "mode": "static",
                    "yellow_min": YELLOW_THRESHOLD,
                    "green_min": GREEN_THRESHOLD,
                },
                "session_summary": session_summary,
            }
        )
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass