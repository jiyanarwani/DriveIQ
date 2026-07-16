"""
backend/routes/coach.py
POST /api/coach
Body: { "score": float, "features": dict, "events": [str] }

Returns: { "message": str, "tips": [str], "severity": str, "source": str }

Uses Flan-T5-Small for dynamic coaching text generation.
Falls back to deterministic rules if the model is unavailable.
"""

from __future__ import annotations
import time
import logging
from flask import Blueprint, request, jsonify

from backend.coach_llm import generate_coaching_tip, is_model_loaded
from backend.scoring import score_window
from backend.db import trip_summaries_collection

logger = logging.getLogger("driveiq.coach")
coach_bp = Blueprint("coach", __name__)

_tip_cache: dict = {}
COACH_CACHE_TTL_SEC = 120
COACH_CACHE_MAX = 256


# ── Cache helpers ────────────────────────────────────────────────────────────

def _cache_key(score: float, features: dict, session_id: str = "") -> str:
    score_bucket = int(round(score / 5.0) * 5)
    braking = int(float(features.get("braking_ratio", features.get("braking_flag", 0))) > 0)
    lane = int(float(features.get("lane_change_ratio", features.get("lane_change_flag", 0))) > 0)
    proximity = min(9, max(0, int(float(features.get("proximity_score", 0.0)) * 10)))
    sid = (session_id or "").strip()
    return f"s{score_bucket}|b{braking}|l{lane}|prox{proximity}|sid{sid}"


def _cache_get(key: str):
    entry = _tip_cache.get(key)
    if not entry:
        return None
    if time.time() - float(entry.get("ts", 0.0)) > COACH_CACHE_TTL_SEC:
        _tip_cache.pop(key, None)
        return None
    payload = dict(entry.get("payload", {}))
    payload["cached"] = True
    return payload


def _cache_set(key: str, payload: dict):
    if len(_tip_cache) >= COACH_CACHE_MAX and key not in _tip_cache:
        oldest_key = min(_tip_cache, key=lambda k: _tip_cache[k].get("ts", 0.0))
        _tip_cache.pop(oldest_key, None)
    _tip_cache[key] = {"ts": time.time(), "payload": payload}


# ── Severity ─────────────────────────────────────────────────────────────────

def _severity_from_score(score: float) -> str:
    if score >= 75:
        return "green"
    if score >= 50:
        return "yellow"
    return "red"


import random

# ── Rule-based fallback tips ────────────────────────────────────────────────

def _evaluate_rules(score: float, features: dict, events: list[str]) -> list[str]:
    """Varied deterministic rule-based tips — used as fallback when LLM is unavailable."""
    tips = []

    if "tailgating" in events or float(features.get("proximity_score", 0.0)) > 0.15:
        tips.append(random.choice([
            "Increase following distance to allow smoother braking.",
            "Leave more space ahead to avoid sudden stops.",
            "Back off slightly from the car in front to improve reaction time."
        ]))

    if "hard_braking" in events or float(features.get("braking_ratio", features.get("braking_flag", 0.0))) > 0.5:
        tips.append(random.choice([
            "Anticipate stops earlier and ease off the accelerator gradually.",
            "Try coasting to a stop instead of braking hard.",
            "Harsh braking uses extra fuel; brake a few seconds earlier."
        ]))

    if "lane_swerving" in events or float(features.get("lane_change_ratio", features.get("lane_change_flag", 0.0))) > 0.3:
        tips.append(random.choice([
            "Minimize unnecessary lane changes — they increase fuel burn by 5-10%.",
            "Hold your lane to maintain a steadier, more efficient speed.",
            "Frequent lane changes disrupt traffic flow and waste fuel."
        ]))

    if "erratic_speed" in events or float(features.get("flow_variance", 0.0)) > 20.0:
        tips.append(random.choice([
            "Maintain a steady speed to improve fuel efficiency.",
            "Fluctuating speeds burn extra gas; try to keep a constant pace.",
            "Smooth out your acceleration for a better eco score."
        ]))

    if "pedestrian_risk" in events:
        tips.append(random.choice([
            "Slow down in pedestrian zones and maintain awareness.",
            "Watch out for crosswalks and ease off the gas.",
            "Pedestrians detected: reduce speed immediately."
        ]))

    # New rules for Optical Flow features (mean_flow and low_motion_ratio)
    mean_flow = float(features.get("mean_flow", 0.0))
    flow_variance = float(features.get("flow_variance", 0.0))
    low_motion_ratio = float(features.get("low_motion_ratio", 0.0))

    if mean_flow > 15.0:
        tips.append("High optical flow detected — possible aggressive acceleration or speeding.")
    elif mean_flow > 8.0 and flow_variance > 15.0:
        tips.append("Erratic speed changes detected — maintain steady throttle.")

    if low_motion_ratio > 0.6:
        tips.append("Extended idle or stop-and-go detected — consider smoother route planning.")

    if not tips:
        tips = [random.choice([
            "Your driving is smooth. Keep maintaining this consistent pace.",
            "Excellent speed control. No issues detected.",
            "Fuel use is highly optimized right now. Great job!"
        ])]

    return tips


# ── Main endpoint ───────────────────────────────────────────────────────────

@coach_bp.route("/api/coach", methods=["POST"])
def coach():
    data = request.get_json(silent=True) or {}
    score = float(data.get("score", 50.0))
    features = data.get("features", {})
    session_id = str(data.get("session_id", "") or request.headers.get("X-Session-Id", ""))
    is_summary = bool(data.get("is_summary"))

    if is_summary and session_id:
        from backend.db import sessions_collection
        db_doc = sessions_collection.find_one({"session_id": session_id})
        if db_doc and isinstance(db_doc.get("tips"), list) and len(db_doc["tips"]) > 0:
            return jsonify({"tips": db_doc["tips"]})

    # Detect events from features if not provided
    events = data.get("events")
    if events is None:
        _, events = score_window(features)

    severity = _severity_from_score(score)

    # Check cache first
    cache_key = _cache_key(score, features, session_id)
    if not is_summary:
        cached = _cache_get(cache_key)
        if cached:
            return jsonify(cached)

    try:
        # Try Flan-T5 first
        llm_tip, source = generate_coaching_tip(score, severity, events, features)

        if llm_tip and source == "gemini":
            tips = [llm_tip]
            # Add one rule-based tip as backup variety
            rule_tips = _evaluate_rules(score, features, events)
            if rule_tips and rule_tips[0] != tips[0]:
                tips.append(rule_tips[0])
        else:
            # LLM unavailable — full rule-based fallback
            tips = _evaluate_rules(score, features, events)
            source = "cv_rules"

        # Ensure at least 1 tip, at most 3
        if not tips:
            tips = ["Your driving is smooth. Keep it up!"]
        tips = tips[:3]

        # Build natural message based on severity
        if severity == "green":
            message = "Great driving! Here are some tips to stay efficient."
        elif severity == "yellow":
            message = "Room for improvement. Focus on the tips below."
        else:
            message = "Your driving needs attention. Follow these tips closely."

        payload = {
            "message": message,
            "tips": tips,
            "severity": severity,
            "source": source,
            "model_loaded": is_model_loaded(),
        }

    except Exception as e:
        logger.error(f"Coach endpoint error: {e}")
        payload = {
            "message": "Coaching engine encountered an issue.",
            "tips": ["Maintain steady speed and smooth braking."],
            "severity": "yellow",
            "source": "fallback_error",
            "model_loaded": is_model_loaded(),
        }

    if is_summary and session_id:
        from backend.db import sessions_collection
        sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": {"tips": payload.get("tips", [])}}
        )
    else:
        _cache_set(cache_key, payload)
        
    return jsonify(payload)
