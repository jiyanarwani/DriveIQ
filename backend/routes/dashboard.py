import logging
from fastapi import APIRouter, Depends, HTTPException
from backend.auth import get_current_user
from backend.db import get_dashboard_metrics
from backend.schemas import DashboardMetricsResponse

logger = logging.getLogger("driveiq.routes.dashboard")
dashboard_router = APIRouter(prefix="/api/v1/dashboard")

@dashboard_router.get("/metrics", response_model=DashboardMetricsResponse)
def get_metrics(current_user: dict = Depends(get_current_user)) -> dict:
    try:
        metrics = get_dashboard_metrics(current_user["_id"])
        return metrics
    except Exception as e:
        logger.error(f"Dashboard metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
