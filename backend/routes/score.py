import base64
import numpy as np
import time
import uuid
import logging
from typing import Any
from fastapi import APIRouter, Depends, Header, HTTPException, status

from backend.db import save_session, sessions_collection
from backend.auth import verify_token, get_current_user
from backend.scoring import score_window, EMA_ALPHA, BASE_SCORE
from backend.model_loader import load_models
from backend.schemas import ScoreRequest, ScoreResponse, TripHistoryItem, TimelineFrame

logger = logging.getLogger("driveiq.routes.score")
score_router = APIRouter(prefix="/api/v1")

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


def _resolve_session(payload: ScoreRequest, x_session_id: str | None = None) -> str:
    """Explicit session definition using explicit key, falling back explicitly to unique UUID."""
    sid = payload.session_id or x_session_id
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


@score_router.post("/score", response_model=ScoreResponse)
def score(
    payload: ScoreRequest,
    authorization: str | None = Header(None),
    x_session_id: str | None = Header(None),
    x_session_started_at: str | None = Header(None)
) -> dict:
    try:
        _cleanup_inactive_sessions()
        
        telemetry = payload.telemetry
        frame_b64 = payload.frame_b64
        prev_frame_b64 = payload.prev_frame_b64
        scoring_mode = payload.scoring_mode
        session_id = _resolve_session(payload, x_session_id)
        now_ts = time.time()
        _LAST_ACTIVITY_BY_SESSION[session_id] = now_ts

        # Optional stream/session start token from client.
        session_start_token = payload.session_started_at or x_session_started_at
        session_start_token = str(session_start_token) if session_start_token is not None else ""
        existing_start_token = _SESSION_START_TOKEN_BY_SESSION.get(session_id)
        if existing_start_token is None:
            _SESSION_START_TOKEN_BY_SESSION[session_id] = session_start_token
        elif session_start_token and session_start_token != existing_start_token:
            _SESSION_START_TOKEN_BY_SESSION[session_id] = session_start_token
            _PREV_SCORE_BY_SESSION.pop(session_id, None)
            _LAST_SCORE_TS_BY_SESSION.pop(session_id, None)
            _PREV_FRAME_BY_SESSION.pop(session_id, None)
        
        user_id = None
        auth_failed = False
        auth_error = None
        token_present = bool(authorization and authorization.startswith("Bearer "))
        if token_present:
            token = authorization.split(" ")[1]
            user_id = verify_token(token)
            logger.info(f"Auth check: user_id={user_id}, token_present=True")
            if user_id is None:
                auth_failed = True
                auth_error = "invalid_or_expired_token"
                logger.warning("Auth failed for /api/v1/score: invalid_or_expired_token")
        elif authorization:
            auth_failed = True
            auth_error = "malformed_authorization_header"
            logger.warning("Auth failed for /api/v1/score: malformed_authorization_header")

        # Fail fast if runtime schema/model state is invalid.
        models = load_models()
        if not bool(models.get("schema_valid", False)):
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "schema_mismatch",
                    "details": models.get("schema_error"),
                }
            )
        
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
        
        ratio_keys = {"braking_ratio", "lane_change_ratio", "proximity_score", "pedestrian_ratio", "low_motion_ratio"}
        for k, v in feature_dict.items():
            if isinstance(v, (int, float)):
                if np.isnan(v) or np.isinf(v):
                    feature_dict[k] = 0.0
                elif k in ratio_keys:
                    feature_dict[k] = max(0.0, min(1.0, float(v)))
        
        # Always detect events for the timeline
        _, events = score_window(feature_dict)

        # Choose scoring path
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
        last_score_ts = _LAST_SCORE_TS_BY_SESSION.get(session_id)
        if last_score_ts is None or (now_ts - last_score_ts) > EMA_RESET_GAP_SEC:
            prev_score = BASE_SCORE
            _PREV_SCORE_BY_SESSION.pop(session_id, None)
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
                logger.error("Session save failed for /api/v1/score: %s", e)

        return {
            "score": round(score_val, 2),
            "eco_score": round(score_val, 2),
            "features": feature_dict,
            "score_source": score_source,
            "auth_failed": auth_failed,
            "auth_error": auth_error,
            "session_saved": session_saved,
            "session_save_error": session_save_error,
        }
    except HTTPException:
        raise
    except Exception as general_err:
        logger.error(f"Score systemic failure -> {general_err}")
        raise HTTPException(status_code=500, detail="Internal server error")


@score_router.get("/trips/history", response_model=list[TripHistoryItem])
def get_trip_history(current_user: dict = Depends(get_current_user)) -> list:
    user_id = current_user["_id"]
    
    trips = list(sessions_collection.find({"user_id": user_id}).sort("start_time", -1).limit(50))
    
    out_trips = []
    for trip in trips:
        trip_data = {}
        trip_data["session_id"] = trip.get("session_id")
        trip_data["user_id"] = str(trip.get("user_id"))
        trip_data["final_score"] = float(trip.get("final_score", 0.0))
        trip_data["frame_count"] = trip.get("frame_count", len(trip.get("frames", [])))
        
        all_events = []
        for f in trip.get("frames", []):
            evs = f.get("events", [])
            if isinstance(evs, list):
                all_events.extend(evs)
            elif isinstance(evs, str):
                all_events.append(evs)
                
        trip_data["top_event"] = max(set(all_events), key=all_events.count) if all_events else "none"
        trip_data["total_events"] = len(all_events)
        
        # Format dates to ISO strings
        start_time = trip.get("start_time")
        if start_time:
            trip_data["date"] = start_time.isoformat()
        else:
            trip_data["date"] = None
            
        out_trips.append(trip_data)
        
    return out_trips


@score_router.get("/trips/{session_id}/timeline", response_model=list[TimelineFrame])
def get_trip_timeline(session_id: str, current_user: dict = Depends(get_current_user)) -> list:
    user_id = current_user["_id"]
    
    trip = sessions_collection.find_one({"user_id": user_id, "session_id": session_id})
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
        
    frames = trip.get("frames", [])
    out_frames = []
        
    for f in frames:
        frame_data = {}
        ts = f.get("timestamp")
        if ts:
            frame_data["timestamp_sec"] = ts.timestamp()
        else:
            frame_data["timestamp_sec"] = 0.0
            
        score_val = float(f.get("score", 0.0))
        frame_data["score"] = score_val
        frame_data["eco_score"] = float(f.get("eco_score", score_val))
        frame_data["features"] = f.get("features", {})
        frame_data["events"] = f.get("events", [])
        
        if score_val >= 80:
            frame_data["severity"] = "green"
        elif score_val >= 65:
            frame_data["severity"] = "yellow"
        else:
            frame_data["severity"] = "red"
            
        out_frames.append(frame_data)
            
    return out_frames
