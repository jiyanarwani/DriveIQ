"""
run.py
DriveIQ Single-Command Launcher.
Starts the FastAPI server which serves both the REST API and the Frontend Dashboard.

Usage:
    python run.py
"""

import sys
import os
from pathlib import Path

# Fix Windows console encoding for Unicode if needed
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure root directory is in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import settings
import uvicorn

if __name__ == "__main__":
    port = settings.port
    print("\n" + "=" * 60)
    print("  [*] DriveIQ AI Driving Coach & Dashboard")
    print(f"  [>] Dashboard:   http://localhost:{port}")
    print(f"  [>] API Docs:    http://localhost:{port}/docs")
    print("=" * 60 + "\n")

    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, reload=False)
