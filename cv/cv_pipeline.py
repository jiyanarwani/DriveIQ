"""
Phase 3: Combined CV Pipeline
Merges YOLO + optical flow features into a single feature vector.

Usage:
    from cv.cv_pipeline import cv_pipeline
    feature_dict = cv_pipeline(curr_frame_bgr, prev_frame_bgr, telemetry_dict)
"""

from __future__ import annotations

import logging
import os
import time
import numpy as np
import cv2


# Must stay in sync exactly with XGBoost inference model features
XGB_FEATURE_SCHEMA = [
    "mean_flow",
    "flow_variance",
    "braking_ratio",
    "lane_change_ratio",
    "proximity_score",
    "vehicle_density",
    "pedestrian_ratio",
    "low_motion_ratio"
]

DEBUG_CV = os.environ.get("DRIVEIQ_CV_DEBUG", "0") == "1"
logger = logging.getLogger("driveiq.cv.pipeline")


def classify_scene(frame_bgr: np.ndarray) -> dict:
    """Return deterministic scene defaults (legacy MobileNet path removed)."""
    _ = frame_bgr
    return {
        "road_type": "unknown",
        "weather": "unknown",
        "road_type_id": -1,
        "weather_id": -1,
    }


def cv_pipeline(
    curr_frame: np.ndarray,
    prev_frame: np.ndarray | None = None,
    telemetry: dict | None = None,
) -> dict:
    """
    Full computer-vision feature extraction pipeline.

    Args:
        curr_frame:  Current video frame (BGR, uint8)
        prev_frame:  Previous frame for optical flow (BGR, uint8), or None
        telemetry:   Dict with keys: speed, rpm, throttle_position, gear, acceleration, fuel_rate

    Returns:
        Combined feature dict ready for model inference.
    """
    t0 = time.perf_counter()
    from cv.yolo_pipeline import extract_yolo_features
    from cv.optical_flow  import extract_flow_features

    # ── YOLO features ────────────────────────────────────────────────────────
    t_yolo = time.perf_counter()
    yolo_feats = extract_yolo_features(curr_frame)
    yolo_ms = (time.perf_counter() - t_yolo) * 1000.0

    # ── Optical flow features ─────────────────────────────────────────────────
    if prev_frame is not None:
        t_flow = time.perf_counter()
        flow_feats = extract_flow_features(prev_frame, curr_frame)
        flow_ms = (time.perf_counter() - t_flow) * 1000.0
    else:
        flow_feats = {
            "mean_flow": 0.0, "variance": 0.0,
            "braking_flag": 0, "lane_change_flag": 0, "erratic_flag": 0,
        }
        flow_ms = 0.0

    # ── Scene classification ──────────────────────────────────────────────────
    scene_feats = classify_scene(curr_frame)

    # ── Telemetry passthrough ─────────────────────────────────────────────────
    tele = telemetry or {}
    pedestrian_count = float(yolo_feats.get("pedestrian_count", yolo_feats.get("pedestrian_flag", 0.0)))
    pedestrian_ratio = min(pedestrian_count / 5.0, 1.0)
    braking_ratio = float(flow_feats["braking_flag"])
    lane_change_ratio = float(flow_feats["lane_change_flag"])
    erratic_flag = float(flow_feats.get("erratic_flag", 0))

    feature_vector = {
        "mean_flow":          float(flow_feats["mean_flow"]),
        "flow_variance":      float(flow_feats["variance"]),
        "erratic_flag":       int(erratic_flag > 0),  # Binary flag for UI
        "braking_flag":       int(braking_ratio > 0.5),  # Keep legacy binary flag for UI
        "braking_ratio":      braking_ratio,
        "lane_change_flag":   int(lane_change_ratio > 0.3),  # Keep legacy binary flag for UI
        "lane_change_ratio":  lane_change_ratio,
        "proximity_score":    float(yolo_feats["proximity_score"]),
        "vehicle_density":    float(yolo_feats["vehicle_count"]),
        "pedestrian_ratio":   float(pedestrian_ratio),
        "low_motion_ratio":   1.0 if float(flow_feats["mean_flow"]) < 0.5 else 0.0,
        
        # Scene
        "road_type":          scene_feats["road_type"],
        "weather":            scene_feats["weather"],
        "road_type_id":       scene_feats["road_type_id"],
        "weather_id":         scene_feats["weather_id"],
        
        # Telemetry
        "speed":              float(tele.get("speed", 0)),
        "rpm":                float(tele.get("rpm", 0)),
        "throttle_position":  float(tele.get("throttle_position", 0)),
        "gear":               tele.get("gear", 1),
        "acceleration":       float(tele.get("acceleration", 0)),
        "fuel_rate":          float(tele.get("fuel_rate", 0)),
    }

    if DEBUG_CV:
        logger.info(
            "cv_pipeline done elapsed_ms=%.1f yolo_ms=%.1f flow_ms=%.1f prev=%s out_keys=%s",
            (time.perf_counter() - t0) * 1000.0,
            yolo_ms,
            flow_ms,
            prev_frame is not None,
            list(feature_vector.keys()),
        )

    return feature_vector


def feature_vector_for_xgb(features: dict, scaler=None) -> np.ndarray:
    """
    Convert the full feature dict into the numeric vector expected by XGBoost.
    Filters explicitly by the exact 8-component array ordered correctly.
    """
    # Defensive casting
    row = {k: float(features.get(k, 0.0)) for k in XGB_FEATURE_SCHEMA}
    
    # Strict 8 ordered vector
    ordered = [row[c] for c in XGB_FEATURE_SCHEMA]
    # Defensive numeric checks: no NaNs, clip ratios [0, 1]
    sanitized = []
    for i, val in enumerate(ordered):
        val = 0.0 if np.isnan(val) or np.isinf(val) else val
        if XGB_FEATURE_SCHEMA[i] in ("braking_ratio", "lane_change_ratio", "proximity_score", "pedestrian_ratio", "low_motion_ratio"):
            val = max(0.0, min(1.0, val))
        sanitized.append(val)

    v = np.array([sanitized], dtype=np.float32)
    
    if scaler is not None:
        try:
            import pandas as pd
            x_df = pd.DataFrame([sanitized], columns=XGB_FEATURE_SCHEMA)
            v = scaler.transform(x_df)
        except Exception:
            v = scaler.transform(v)
            
    return v


# ── Quick sanity test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, cv2

    if len(sys.argv) < 2:
        print("Usage: python cv/cv_pipeline.py path/to/video.mp4")
        sys.exit(0)

    cap = cv2.VideoCapture(sys.argv[1])
    ret, prev = cap.read()
    ret, curr = cap.read()
    cap.release()

    if not ret:
        print("❌ Could not read frames from video")
    else:
        result = cv_pipeline(curr, prev, telemetry={"speed": 60, "rpm": 2000})
        for k, v in result.items():
            print(f"  {k:<25} = {v}")
