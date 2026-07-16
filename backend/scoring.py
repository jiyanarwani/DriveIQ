"""
backend/scoring.py
Event-Based Deduction Scoring Engine

Instead of relying on XGBoost regression (which produces erratic scores from
noisy CV features), this module scores driving using deterministic event
detection with confidence thresholds.

How it works:
  1. Each window starts at a BASE_SCORE (90).
  2. Specific driving infractions are detected by checking CV features
     against safety thresholds. Only values that exceed the threshold
     trigger a deduction.
  3. Sustained clean driving earns a small bonus (up to +5).
  4. For chronological sequences (review mode), an Exponential Moving
     Average (EMA) is applied to smooth the curve.

This matches how production telematics systems (Samsara, Root, Progressive)
actually score driving behavior.
"""

from __future__ import annotations
import logging

logger = logging.getLogger("driveiq.scoring")

# ── Configuration ────────────────────────────────────────────────────────────

BASE_SCORE = 90.0
MIN_SCORE  = 0.0
MAX_SCORE  = 100.0

# Event definitions: (feature_key, threshold, deduction_points, event_label)
# Only values ABOVE the threshold trigger the deduction.
EVENT_RULES = [
    ("braking_ratio",      0.5,  10.0, "hard_braking"),
    ("proximity_score",    0.12, 10.0, "tailgating"),
    ("lane_change_ratio",  0.3,  5.0, "lane_swerving"),
    ("pedestrian_ratio",   0.5,  0.5, "pedestrian_risk"),
    ("flow_variance",      20.0,  3.0, "erratic_speed"),
]

# EMA smoothing factor for review mode (0 = ignore current, 1 = ignore history)
EMA_ALPHA = 0.6

# Bonus for sustained clean driving
CLEAN_WINDOW_BONUS = 0.5  # points added per consecutive clean window
MAX_CLEAN_BONUS    = 5.0  # cap on the total bonus


# ── Core Scoring Function ─────────────────────────────────────────────────────

def score_window(features: dict) -> tuple[float, list[str]]:
    """
    Score a single time window using event-based deductions.

    Args:
        features: dict of CV + telemetry features for this window.

    Returns:
        (score, detected_events) where score is 0-100 and
        detected_events is a list of event labels that fired.
    """
    score = BASE_SCORE
    events = []

    for feat_key, threshold, deduction, event_label in EVENT_RULES:
        val = float(features.get(feat_key, 0.0))
        if val > threshold:
            # Scale deduction by how far above threshold
            excess_ratio = min((val - threshold) / max(threshold, 0.01), 1.2)
            actual_deduction = deduction * excess_ratio
            score -= actual_deduction
            events.append(event_label)

    score = max(MIN_SCORE, min(MAX_SCORE, score))
    return score, events


def score_windows_with_ema(window_features_list: list[dict]) -> list[dict]:
    """
    Score a chronological sequence of windows with EMA smoothing.
    Used by the review (post-drive) pipeline.

    Args:
        window_features_list: list of feature dicts, ordered by time.

    Returns:
        list of dicts with keys: raw_score, smoothed_score, events
    """
    results = []
    prev_smoothed = BASE_SCORE
    consecutive_clean = 0

    for features in window_features_list:
        raw_score, events = score_window(features)

        # Track consecutive clean windows for bonus
        if len(events) == 0:
            consecutive_clean += 1
            bonus = min(consecutive_clean * CLEAN_WINDOW_BONUS, MAX_CLEAN_BONUS)
            raw_score = min(MAX_SCORE, raw_score + bonus)
        else:
            consecutive_clean = 0

        # EMA smoothing: blends current score with previous to prevent jitter
        smoothed = EMA_ALPHA * raw_score + (1.0 - EMA_ALPHA) * prev_smoothed
        smoothed = max(MIN_SCORE, min(MAX_SCORE, smoothed))
        prev_smoothed = smoothed

        results.append({
            "raw_score": round(raw_score, 2),
            "smoothed_score": round(smoothed, 2),
            "events": events,
        })

    return results


def event_to_issue_key(events: list[str]) -> str:
    """Map detected events to the issue key format used by the coaching system."""
    ISSUE_MAP = {
        "hard_braking":    "braking_flag_ratio",
        "tailgating":      "proximity_score_mean",
        "lane_swerving":   "lane_change_flag_ratio",
        "pedestrian_risk": "proximity_score_mean",
        "erratic_speed":   "mean_flow_mean",
    }
    if not events:
        return "smooth_driving"
    # Return the issue key for the first (most severe) event
    return ISSUE_MAP.get(events[0], "smooth_driving")
