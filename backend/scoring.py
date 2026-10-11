"""
backend/scoring.py
Dual-Tier Scoring Engine: XGBoost on 8 Features with Rule-Based Fallback

Architecture:
  1. Primary Scorer: XGBoost Regression Model
     After the 8 core CV and telemetry features are extracted
     (mean_flow, flow_variance, braking_ratio, lane_change_ratio,
      proximity_score, vehicle_density, pedestrian_ratio, low_motion_ratio),
     XGBoost is evaluated first as the primary eco-driving predictor.
  2. Fallback Scorer: Event-Based Deduction Engine
     Only if XGBoost is unavailable, missing, schema-mismatched, or fails
     during inference, the deterministic rule-based event deduction engine
     is used as the safety fallback.
"""

from __future__ import annotations
import logging
import numpy as np

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


def score_features_xgboost(features: dict, xgb, scaler) -> float:
    """
    Score driving features using trained XGBoost model on the 8 features.

    Features used (exact 8 schema):
      mean_flow, flow_variance, braking_ratio, lane_change_ratio,
      proximity_score, vehicle_density, pedestrian_ratio, low_motion_ratio
    """
    from cv.cv_pipeline import feature_vector_for_xgb

    xgb_input = feature_vector_for_xgb(features, scaler)
    raw_score = float(xgb.predict(xgb_input)[0])
    return max(MIN_SCORE, min(MAX_SCORE, raw_score))


def score_window_with_engine(
    features: dict,
    models: dict | None = None,
    scoring_mode: str = "xgboost"
) -> tuple[float, list[str], str]:
    """
    Score a feature window using the dual-tier architecture:
    1. Always detect safety events for UI/timeline.
    2. Primary: Predict eco score using XGBoost on the 8 extracted features.
    3. Fallback: Only if XGBoost is unavailable, schema is invalid, or
       prediction raises an exception, fall back to rule-based event deductions.

    Returns:
        (score, events, score_source)
        where score_source is one of:
          - "xgboost"
          - "event_rules" (if explicitly requested)
          - "event_rules_fallback" (if XGBoost could not run)
    """
    # 1. Rule-based evaluation (always run to detect events)
    rule_score, events = score_window(features)

    # If caller explicitly asked for rule-based, use it directly
    if scoring_mode == "event_rules":
        return rule_score, events, "event_rules"

    # 2. Primary: evaluate XGBoost on the 8 features
    if models:
        xgb = models.get("xgb")
        scaler = models.get("scaler")
        schema_valid = bool(models.get("schema_valid", False))

        if schema_valid and xgb is not None and scaler is not None:
            try:
                raw_score = score_features_xgboost(features, xgb, scaler)
                return raw_score, events, "xgboost"
            except Exception as e:
                logger.warning("XGBoost prediction failed on 8 features (%s); falling back to rule-based.", e)
        else:
            schema_err = models.get("schema_error", "models_not_loaded")
            logger.warning("XGBoost unavailable or schema mismatch (%s); falling back to rule-based.", schema_err)

    # 3. Fallback: only if XGBoost doesn't work
    return rule_score, events, "event_rules_fallback"


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
