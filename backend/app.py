"""
backend/app.py
DriveIQ FastAPI API entry point.

Start with:
    python backend/app.py
"""

import sys
import os

# CRITICAL: Set BEFORE any library imports.
# Prevents macOS Apple Silicon segfault from libomp conflicts between cv2 and torch.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from contextlib import asynccontextmanager
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# CRITICAL: Import torch BEFORE cv2 to ensure torch's libomp loads first.
try:
    import torch
    torch.set_num_threads(1)
except (ImportError, OSError):
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from backend.config import settings
from backend.model_loader import load_models
from backend.routes.health import health_router
from backend.routes.score import score_router
from backend.routes.coach import coach_router
from backend.routes.review import review_router
from backend.routes.auth import auth_router
from backend.routes.dashboard import dashboard_router

# Configure logging at application entry point
log_level = settings.log_level.upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("driveiq.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup phase: Load ML models and pre-warm YOLO
    logger.info("Initializing models on startup...")
    load_models()
    
    try:
        from cv.yolo_pipeline import get_yolo_model
        get_yolo_model()
        logger.info("✅ YOLO model pre-warmed.")
    except Exception as e:
        logger.warning(f"YOLO pre-warm failed (non-fatal): {e}")

    yield

    # Shutdown phase:
    logger.info("Shutting down FastAPI app...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="DriveIQ API",
        version=settings.api_version,
        lifespan=lifespan
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register FastAPI Routers
    app.include_router(health_router)
    app.include_router(score_router)
    app.include_router(coach_router)
    app.include_router(review_router)
    app.include_router(auth_router)
    app.include_router(dashboard_router)

    from fastapi.responses import RedirectResponse
    @app.get("/", include_in_schema=False)
    def redirect_to_docs():
        return RedirectResponse(url="/docs")

    return app


app = create_app()

if __name__ == "__main__":
    port = settings.port
    logger.info(f"Starting DriveIQ FastAPI API on http://localhost:{port}")
    logger.info(f"Open Swagger UI docs at http://localhost:{port}/docs")
    
    # Run the server
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, log_level=log_level.lower())
