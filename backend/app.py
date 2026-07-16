"""
backend/app.py
DriveIQ Flask API entry point.

Start with:
    cd /Users/jatinankushnimje/Documents/Coding/driveiq_practice
    source .venv/bin/activate
    python backend/app.py

Or with gunicorn (production):
    gunicorn -w 2 -b 0.0.0.0:5000 backend.app:app
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
import threading
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# CRITICAL: Import torch BEFORE cv2 to ensure torch's libomp loads first.
# On macOS Apple Silicon, if cv2 loads its libomp first and torch loads a
# different copy later, the process segfaults.
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

from flask import Flask
from flask_cors import CORS

from backend.model_loader import load_models
from backend.routes.health  import health_bp
from backend.routes.score   import score_bp
from backend.routes.coach   import coach_bp
from backend.routes.review  import review_bp
from backend.routes.auth    import auth_bp
from backend.routes.dashboard import dashboard_bp


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, origins=["http://localhost:5173", "http://localhost:3000"])

    # Load ML models once at startup
    app.config["MODELS"] = load_models()

    # Pre-warm YOLO so it doesn't lazy-load mid-request and race with cv2
    try:
        from cv.yolo_pipeline import get_yolo_model
        get_yolo_model()
    except Exception as e:
        logging.getLogger("driveiq.app").warning(f"YOLO pre-warm failed (non-fatal): {e}")

    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(score_bp)
    app.register_blueprint(coach_bp)
    app.register_blueprint(review_bp)
    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")

    return app


app = create_app()

if __name__ == "__main__":
    log_level = os.environ.get("DRIVEIQ_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"\n DriveIQ API running on http://localhost:{port}")
    print("   Endpoints:")
    print("     GET  /api/health")
    print("     POST /api/score")
    print("     POST /api/coach")
    print("     POST /api/review")

    # Production: gunicorn -w 1 -b 0.0.0.0:5000 backend.app:app
    app.run(host="0.0.0.0", port=port, debug=debug)
