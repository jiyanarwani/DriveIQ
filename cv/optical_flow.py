"""
Phase 3: Farneback Optical Flow Feature Extractor
Extracts mean_flow, variance, braking_flag, lane_change_flag from consecutive frames.

Usage (as module):
    from cv.optical_flow import extract_flow_features
    features = extract_flow_features(prev_frame_bgr, curr_frame_bgr)
"""

import cv2
import numpy as np

# Farneback parameters (good balance of speed vs quality)
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)

# Thresholds — tune these based on your video resolution / frame rate
BRAKING_ACCEL_THRESH = 1.5   # y-flow acceleration spike to flag braking
LANE_CHANGE_THRESH   = 2.0   # preserved for compatibility/tuning reference

# ── Module-level state for acceleration-based detection ──────────────────────
_prev_center_y_flow   = 0.0
_prev_x_asymmetry     = 0.0
_last_brake_frame     = -100   # debounce: frame index of last braking flag
_last_erratic_frame   = -100   # debounce: frame index of last erratic flag
_last_lane_frame      = -100   # debounce: frame index of last lane change flag
_frame_counter        = 0      # monotonic frame counter

BRAKE_DEBOUNCE_FRAMES   = 15   # min frames between braking events
ERRATIC_DEBOUNCE_FRAMES = 15   # min frames between erratic speed events
LANE_DEBOUNCE_FRAMES    = 15   # min frames between lane change events
ERRATIC_VARIANCE_THRESH = 20.0 # flow variance must exceed this to flag
LANE_ACCEL_THRESH       = 3.5  # sudden spike in x-asymmetry to flag lane change


def reset_flow_state():
    """Reset module-level state between sessions/videos."""
    global _prev_center_y_flow, _prev_x_asymmetry, _last_brake_frame, _last_erratic_frame, _last_lane_frame, _frame_counter
    _prev_center_y_flow = 0.0
    _prev_x_asymmetry   = 0.0
    _last_brake_frame   = -100
    _last_erratic_frame = -100
    _last_lane_frame    = -100
    _frame_counter      = 0


def extract_flow_features(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
) -> dict:
    """
    Compute Farneback optical flow between two BGR frames.

    Args:
        prev_frame: previous frame (BGR, uint8)
        curr_frame: current frame  (BGR, uint8)

    Returns:
        dict with keys:
            mean_flow         (float)  — mean magnitude of all flow vectors
            variance          (float)  — variance of flow magnitudes
            braking_flag      (float)  — normalized braking intensity ratio [0, 1]
            lane_change_flag  (float)  — normalized lane-change intensity ratio [0, 1]
    """
    global _prev_center_y_flow, _prev_x_asymmetry, _last_brake_frame, _last_erratic_frame, _last_lane_frame, _frame_counter
    _frame_counter += 1

    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, **FARNEBACK_PARAMS
    )  # shape: (H, W, 2) — [x-flow, y-flow]

    fx = flow[..., 0]   # horizontal component
    fy = flow[..., 1]   # vertical component

    # Magnitude of each flow vector
    magnitude = np.sqrt(fx ** 2 + fy ** 2)

    mean_flow = float(np.mean(magnitude))
    variance  = float(np.var(magnitude))

    # ── Braking detection: acceleration-based (sudden spike in y-flow) ────────
    h, w = fy.shape
    central_fy = fy[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    mean_central_fy = float(np.mean(central_fy))

    # Acceleration = change in center y-flow between consecutive frames
    y_acceleration = mean_central_fy - _prev_center_y_flow
    _prev_center_y_flow = mean_central_fy  # store for next frame

    # Only flag on a SHARP positive spike (sudden deceleration pushes
    # the horizon down in the camera frame = positive y-flow jump),
    # AND only if enough frames have passed since the last braking event.
    braking_flag = 0.0
    if y_acceleration > BRAKING_ACCEL_THRESH:
        if (_frame_counter - _last_brake_frame) >= BRAKE_DEBOUNCE_FRAMES:
            braking_flag = min(y_acceleration / 3.0, 1.0)
            _last_brake_frame = _frame_counter

    # ── Erratic speed detection: variance-based with debounce ──────────────────
    erratic_flag = 0.0
    if variance > ERRATIC_VARIANCE_THRESH:
        if (_frame_counter - _last_erratic_frame) >= ERRATIC_DEBOUNCE_FRAMES:
            erratic_flag = min(variance / 40.0, 1.0)  # normalize to [0, 1]
            _last_erratic_frame = _frame_counter

    # ── Lane-change detection: acceleration-based (sudden asymmetry spike) ────
    left_fx  = fx[:, : w // 2]
    right_fx = fx[:, w // 2 :]
    x_asymmetry = abs(float(np.mean(left_fx)) - float(np.mean(right_fx)))

    # Track the CHANGE in asymmetry — steady camera tilt = constant asymmetry
    # (delta ≈ 0), actual lane change = sudden spike in asymmetry (big delta).
    x_accel = abs(x_asymmetry - _prev_x_asymmetry)
    _prev_x_asymmetry = x_asymmetry

    lane_change_flag = 0.0
    if x_accel > LANE_ACCEL_THRESH:
        if (_frame_counter - _last_lane_frame) >= LANE_DEBOUNCE_FRAMES:
            lane_change_flag = min(x_accel / 4.0, 1.0)
            _last_lane_frame = _frame_counter

    return {
        "mean_flow":        round(mean_flow, 4),
        "variance":         round(variance, 4),
        "erratic_flag":     round(float(erratic_flag), 4),
        "braking_flag":     round(float(braking_flag), 4),
        "lane_change_flag": round(float(lane_change_flag), 4),
    }


# ── Quick demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cv/optical_flow.py path/to/video.mp4")
        sys.exit(1)

    cap = cv2.VideoCapture(sys.argv[1])
    ret, prev = cap.read()
    if not ret:
        print("❌ Could not read video")
        sys.exit(1)

    frame_count = 0
    while frame_count < 5:
        ret, curr = cap.read()
        if not ret:
            break
        features = extract_flow_features(prev, curr)
        print(f"Frame {frame_count}: {features}")
        prev = curr
        frame_count += 1

    cap.release()
