"""
backend/routes/score.py
POST /api/score
"""

from __future__ import annotations

import base64
import numpy as np
import time
import uuid
import logging

from flask import Blueprint, request, jsonify, current_app
from backend.db import save_session, sessions_collection
from backend.auth import verify_token, token_required
from backend.scoring import score_window, EMA_ALPHA, BASE_SCORE

logger = logging.getLogger("driveiq.routes.score")
score_bp = Blueprint("score", __name__)

INACTIVE_TIMEOUT_SEC = 300  # 5 minutes
EMA_RESET_GAP_SEC = 30  # reset live EMA when stream is idle for this long

# Isolation states mapping to dicts for cleanup
_PREV_FRAME_BY_SESSION: dict[str, dict] = {}
_PREV_SCORE_BY_SESSION: dict[str, float] = {}  # EMA state for live scoring
_LAST_ACTIVITY_BY_SESSION: dict[str, float] = {}
_LAST_SCORE_TS_BY_SESSION: dict[str, float] = {}
_SESSION_START_TOKEN_BY_SESSION: dict[str, str] = {}

def _decode_frame(b64_str: str) -> np.ndarray | None:
    try:
        import cv2
        img_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def _vision_like_features_from_telemetry(telemetry: dict) -> dict:
    speed = float(telemetry.get("speed", 0.0))
    accel = float(telemetry.get("acceleration", 0.0))
    throttle = float(telemetry.get("throttle_position", 0.0))

    proximity = min(max(speed - 40.0, 0.0) / 60.0, 1.0)
    mean_flow = abs(accel) * 2.0 + throttle / 50.0
    flow_variance = abs(accel) * 0.5

    return {
        "vehicle_density": max(0.0, min(12.0, 1 + speed / 30.0)),
        "proximity_score": round(proximity, 4),
        "pedestrian_ratio": 0.0,
        "mean_flow": round(mean_flow, 4),
        "flow_variance": round(flow_variance, 4),
        "braking_ratio": 1.0 if accel < -0.8 else 0.0,
        "lane_change_ratio": 1.0 if abs(accel) > 1.6 else 0.0,
        "low_motion_ratio": 1.0 if mean_flow < 0.5 else 0.0,
        
        # Legacy UI variables matching exactly
        "vehicle_count": int(max(0, min(12, round(1 + speed / 30.0)))),
        "pedestrian_flag": 0,
        "braking_flag": int(accel < -0.8),
        "lane_change_flag": int(abs(accel) > 1.6),
        "road_type_id": 1 if speed > 70 else 0,
        "weather_id": 0,
    }

def _heuristic_score_from_features(features: dict) -> float:
    score = 85.0
    score -= float(features.get("proximity_score", 0.0)) * 30.0
    score -= float(features.get("mean_flow", 0.0)) * 8.0
    score -= float(features.get("flow_variance", 0.0)) * 10.0
    score -= float(features.get("braking_ratio", features.get("braking_flag", 0))) * 12.0
    score -= float(features.get("lane_change_ratio", features.get("lane_change_flag", 0))) * 10.0
    return max(0.0, min(100.0, score))


def _resolve_session(payload: dict) -> str:
    """Explicit session definition using explicit key, falling back explicitly to unique UUID."""
    sid = payload.get("session_id") or request.headers.get("X-Session-Id")
    if not sid:
        sid = f"temp-{uuid.uuid4()}"
    return str(sid)

def _cleanup_inactive_sessions():
    """Buffer memory leak prevention mapping bounds and timeouts."""
    now = time.time()

    # Expire inactive sessions from all state maps.
    for sid, last_ts in list(_LAST_ACTIVITY_BY_SESSION.items()):
        if now - last_ts > INACTIVE_TIMEOUT_SEC:
            _LAST_ACTIVITY_BY_SESSION.pop(sid, None)
            _PREV_FRAME_BY_SESSION.pop(sid, None)
            _PREV_SCORE_BY_SESSION.pop(sid, None)
            _LAST_SCORE_TS_BY_SESSION.pop(sid, None)
            _SESSION_START_TOKEN_BY_SESSION.pop(sid, None)

    # Evict oldest globally if total active sessions exceeds cap.
    MAX_SESSIONS_GLOBAL = 200
    if len(_LAST_ACTIVITY_BY_SESSION) > MAX_SESSIONS_GLOBAL:
        active_sessions = sorted((ts, sid) for sid, ts in _LAST_ACTIVITY_BY_SESSION.items())
        while len(_LAST_ACTIVITY_BY_SESSION) > MAX_SESSIONS_GLOBAL and active_sessions:
            _, sid = active_sessions.pop(0)
            _LAST_ACTIVITY_BY_SESSION.pop(sid, None)
            _PREV_FRAME_BY_SESSION.pop(sid, None)
            _PREV_SCORE_BY_SESSION.pop(sid, None)
            _LAST_SCORE_TS_BY_SESSION.pop(sid, None)
            _SESSION_START_TOKEN_BY_SESSION.pop(sid, None)



@score_bp.route("/api/score", methods=["POST"])
def score():
    try:
        _cleanup_inactive_sessions()
        
        data = request.get_json(silent=True) or {}
        telemetry = data.get("telemetry", {})
        frame_b64 = data.get("frame_b64")
        prev_frame_b64 = data.get("prev_frame_b64")
        scoring_mode = data.get("scoring_mode", "event_rules")  # "event_rules" or "xgboost"
        session_id = _resolve_session(data)
        now_ts = time.time()
        _LAST_ACTIVITY_BY_SESSION[session_id] = now_ts

        # Optional stream/session start token from client.
        # If this token changes while session_id is reused, reset EMA state immediately.
        session_start_token = data.get("session_started_at") or request.headers.get("X-Session-Started-At")
        session_start_token = str(session_start_token) if session_start_token is not None else ""
        existing_start_token = _SESSION_START_TOKEN_BY_SESSION.get(session_id)
        if existing_start_token is None:
            _SESSION_START_TOKEN_BY_SESSION[session_id] = session_start_token
        elif session_start_token and session_start_token != existing_start_token:
            _SESSION_START_TOKEN_BY_SESSION[session_id] = session_start_token
            _PREV_SCORE_BY_SESSION.pop(session_id, None)
            _LAST_SCORE_TS_BY_SESSION.pop(session_id, None)
            _PREV_FRAME_BY_SESSION.pop(session_id, None)
        
        auth_header = request.headers.get("Authorization")
        user_id = None
        auth_failed = False
        auth_error = None
        token_present = bool(auth_header and auth_header.startswith("Bearer "))
        if token_present:
            token = auth_header.split(" ")[1]
            user_id = verify_token(token)
            logger.info(f"Auth check: user_id={user_id}, token_present={auth_header is not None}")
            if user_id is None:
                auth_failed = True
                auth_error = "invalid_or_expired_token"
                logger.warning("Auth failed for /api/score: invalid_or_expired_token")
        elif auth_header:
            auth_failed = True
            auth_error = "malformed_authorization_header"
            logger.warning("Auth failed for /api/score: malformed_authorization_header")

        # Fail fast if runtime schema/model state is invalid.
        models = current_app.config.get("MODELS", {})
        if not bool(models.get("schema_valid", False)):
            return jsonify({
                "error": "schema_mismatch",
                "details": models.get("schema_error"),
            }), 503
        
        cv_features = _vision_like_features_from_telemetry(telemetry)
        if frame_b64:
            try:
                from cv.cv_pipeline import cv_pipeline
                frame = _decode_frame(frame_b64)
                if frame is None:
                    raise ValueError("invalid frame MIME data")

                if prev_frame_b64:
                    prev_frame = _decode_frame(prev_frame_b64)
                else:
                    prev_node = _PREV_FRAME_BY_SESSION.get(session_id)
                    prev_frame = prev_node["frame"] if prev_node else None

                cv_features = cv_pipeline(frame, prev_frame, telemetry)
                _PREV_FRAME_BY_SESSION[session_id] = {"frame": frame, "updated_at": time.time()}
            except Exception as e:
                cv_features = {**cv_features, "cv_error": str(e)}

        feature_dict = {**telemetry, **cv_features}
        
        # (6) Before any inference, replace NaN with 0, clip ratios to [0,1]
        ratio_keys = {"braking_ratio", "lane_change_ratio", "proximity_score", "pedestrian_ratio", "low_motion_ratio"}
        for k, v in feature_dict.items():
            if isinstance(v, (int, float)):
                if np.isnan(v) or np.isinf(v):
                    feature_dict[k] = 0.0
                elif k in ratio_keys:
                    feature_dict[k] = max(0.0, min(1.0, float(v)))
        
        # Always detect events for the timeline (regardless of scoring mode)
        _, events = score_window(feature_dict)

        # Choose scoring path based on client request
        score_source = "event_rules"
        if scoring_mode == "xgboost":
            xgb = models.get("xgb")
            scaler = models.get("scaler")
            if xgb is not None and scaler is not None:
                try:
                    from cv.cv_pipeline import feature_vector_for_xgb
                    xgb_input = feature_vector_for_xgb(feature_dict, scaler)
                    raw_score = float(xgb.predict(xgb_input)[0])
                    raw_score = max(0.0, min(100.0, raw_score))
                    score_source = "xgboost"
                except Exception as e:
                    logger.warning(f"XGBoost prediction failed, falling back to event_rules: {e}")
                    raw_score, _ = score_window(feature_dict)
                    score_source = "event_rules_fallback"
            else:
                logger.warning("XGBoost model not loaded, falling back to event_rules")
                raw_score, _ = score_window(feature_dict)
                score_source = "event_rules_fallback"
        else:
            raw_score, _ = score_window(feature_dict)

        # Apply per-session EMA smoothing for live mode stability.
        # Reset EMA when idle gap is too long (new segment) or when session_id is first seen.
        last_score_ts = _LAST_SCORE_TS_BY_SESSION.get(session_id)
        if last_score_ts is None or (now_ts - last_score_ts) > EMA_RESET_GAP_SEC:
            prev_score = BASE_SCORE
            _PREV_SCORE_BY_SESSION.pop(session_id, None)
            # Reset optical flow acceleration state for the new session
            try:
                from cv.optical_flow import reset_flow_state
                reset_flow_state()
            except Exception:
                pass
        else:
            prev_score = _PREV_SCORE_BY_SESSION.get(session_id, BASE_SCORE)

        score_val = EMA_ALPHA * raw_score + (1.0 - EMA_ALPHA) * prev_score
        score_val = max(0.0, min(100.0, score_val))
        _PREV_SCORE_BY_SESSION[session_id] = score_val
        _LAST_SCORE_TS_BY_SESSION[session_id] = now_ts

        if events:
            logger.info(f"[{score_source}] Events: {events} | raw={raw_score:.1f} smoothed={score_val:.1f}")

        session_saved = False
        session_save_error = None
        if user_id:
            try:
                save_session(user_id, session_id, score_val, score_val, feature_dict, events)
                session_saved = True
            except Exception as e:
                session_save_error = str(e)
                logger.error("Session save failed for /api/score: %s", e)

        # Standard Payload Output Guaranteed Complete explicitly referencing rules
        return jsonify({
            "score": round(score_val, 2),
            "eco_score": round(score_val, 2),
            "features": feature_dict,
            "score_source": score_source,
            "auth_failed": auth_failed,
            "auth_error": auth_error,
            "session_saved": session_saved,
            "session_save_error": session_save_error,
        })
    except Exception as general_err:
        logger.error(f"Score systemic failure -> {general_err}")
        return jsonify({
            "error": "internal_error",
        }), 500  # Changed from 200

@score_bp.route("/api/trips/history", methods=["GET"])
@token_required
def get_trip_history(current_user):
    user_id = current_user["_id"]
    
    trips = list(sessions_collection.find({"user_id": user_id}).sort("start_time", -1).limit(50))
    
    for trip in trips:
        if "_id" in trip:
            del trip["_id"]
        
        trip["date"] = trip.get("start_time")
        trip["frame_count"] = trip.get("frame_count", len(trip.get("frames", [])))
        
        all_events = []
        for f in trip.get("frames", []):
            evs = f.get("events", [])
            if isinstance(evs, list):
                all_events.extend(evs)
            elif isinstance(evs, str):
                all_events.append(evs)
                
        trip["top_event"] = max(set(all_events), key=all_events.count) if all_events else "none"
        trip["total_events"] = len(all_events)
        
        if "frames" in trip:
            del trip["frames"]
        if "start_time" in trip:
            del trip["start_time"]
        if "end_time" in trip:
            del trip["end_time"]
            
    return jsonify(trips), 200

@score_bp.route("/api/trips/<session_id>/timeline", methods=["GET"])
@token_required
def get_trip_timeline(current_user, session_id):
    user_id = current_user["_id"]
    
    trip = sessions_collection.find_one({"user_id": user_id, "session_id": session_id})
    if not trip:
        return jsonify({"error": "trip_not_found"}), 404
        
    frames = trip.get("frames", [])
        
    for f in frames:
        if "timestamp" in f and f["timestamp"]:
            f["timestamp_sec"] = f["timestamp"].timestamp()
            del f["timestamp"]
        
        score = f.get("score", 0)
        if score >= 80:
            f["severity"] = "green"
        elif score >= 65:
            f["severity"] = "yellow"
        else:
            f["severity"] = "red"
            
    return jsonify(frames), 200
