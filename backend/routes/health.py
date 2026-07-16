"""
backend/routes/health.py
GET /api/health — liveness probe
"""
import os
from flask import Blueprint, jsonify
from flask import current_app

health_bp = Blueprint("health", __name__)


@health_bp.route("/api/health", methods=["GET"])
def health():
    models = current_app.config.get("MODELS", {})
    # Current primary path: XGBoost scoring with schema validation. LSTM predictor is optional.
    core_required = ["xgb", "scaler"]

    core_models_loaded = all(models.get(k) is not None for k in core_required)
    schema_valid = bool(models.get("schema_valid", False))
    predictor_loaded = models.get("predictor") is not None

    models_loaded = core_models_loaded and schema_valid

    # Coach is purely deterministic/rule-based natively now
    coach_ready = True
    score_ready = models_loaded
    overall_ready = score_ready and coach_ready
    degraded = not overall_ready or not predictor_loaded

    return jsonify({
        "status": "ok",
        "service": "DriveIQ API",
        "models_loaded": models_loaded,
        "core_models_loaded": core_models_loaded and schema_valid,
        "schema_valid": schema_valid,
        "schema_error": models.get("schema_error"),
        "predictor_loaded": predictor_loaded,
        "missing_core_models": [k for k in core_required if models.get(k) is None],
        "score_ready": score_ready,
        "coach_ready": coach_ready,
        "coach_status": "rule_based_active",
        "coach_disabled": False,
        "ready": overall_ready,
        "degraded": degraded,
        "version": os.environ.get("DRIVEIQ_API_VERSION", "v1"),
    })
