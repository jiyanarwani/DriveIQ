import time
import logging
from fastapi import APIRouter
from backend.config import settings
from backend.model_loader import load_models
from backend.db import is_mongo_available
from backend.schemas import HealthResponse

# Record startup time for uptime calculation
START_TIME = time.time()

logger = logging.getLogger("driveiq.routes.health")
health_router = APIRouter(prefix="/api/v1/health")

@health_router.get("", response_model=HealthResponse)
def health_check() -> dict:
    models = load_models()
    core_required = ["xgb", "scaler"]

    core_models_loaded = all(models.get(k) is not None for k in core_required)
    schema_valid = bool(models.get("schema_valid", False))
    predictor_loaded = models.get("predictor") is not None

    models_loaded = core_models_loaded and schema_valid

    coach_ready = True
    score_ready = models_loaded
    overall_ready = score_ready and coach_ready
    degraded = not overall_ready or not predictor_loaded

    # Check MongoDB status
    mongo_ok, mongo_error = is_mongo_available()

    # Check GPU availability
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except Exception:
        pass

    uptime = time.time() - START_TIME

    return {
        "status": "ok" if not degraded else "degraded",
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
        "version": settings.api_version,
        "uptime_seconds": round(uptime, 2),
        "mongodb_connected": mongo_ok,
        "gpu_available": gpu_available
    }
