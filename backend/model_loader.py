"""
backend/model_loader.py
Loads and caches all trained models perfectly ONCE at Flask startup.
"""

from __future__ import annotations

import joblib
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

_cache = {}
logger = logging.getLogger("driveiq.model_loader")

def _validate_runtime_schema(feature_cols: list[str]) -> tuple[bool, str | None, list[str]]:
    """Validate trained feature schema against runtime scorer schema."""
    try:
        from cv.cv_pipeline import XGB_FEATURE_SCHEMA
        runtime_cols = list(XGB_FEATURE_SCHEMA)
    except Exception as e:
        return False, f"Runtime schema import failed: {e}", []

    if not feature_cols:
        return False, "Scaler metadata missing feature_cols", runtime_cols

    if list(feature_cols) != runtime_cols:
        return (
            False,
            f"Schema mismatch. trained={feature_cols} runtime={runtime_cols}",
            runtime_cols,
        )

    return True, None, runtime_cols


def load_models() -> dict:
    """
    Load all DriveIQ models once and return a dict.
    Fallback mechanisms safely default values to None if binaries differ or disappear.

    Returns:
        {
            "xgb":       XGBRegressor | None,
            "scaler":    StandardScaler | None,
            "feature_cols": list[str],
        }
    """
    if _cache:
        return _cache

    logger.info("Loading models into system memory...")

    _cache["schema_valid"] = False
    _cache["schema_error"] = "schema_not_checked"
    _cache["runtime_feature_cols"] = []
    _cache["trained_feature_cols"] = []

    # 1. Load Primary Scoring Engine
    xgb_path = MODELS_DIR / "xgb_scorer.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"

    _cache["xgb"] = None
    if xgb_path.exists():
        try:
            _cache["xgb"] = joblib.load(xgb_path)
            logger.info(f"✅ XGBoost loaded from {xgb_path.name}")
        except Exception as e:
            logger.error(f"Failed to load XGBoost: {e}")

    _cache["scaler"] = None
    _cache["feature_cols"] = []
    if scaler_path.exists():
        try:
            bundle = joblib.load(scaler_path)
            if isinstance(bundle, dict):
                _cache["scaler"] = bundle.get("scaler")
                _cache["feature_cols"] = list(bundle.get("feature_cols", []))
            else:
                _cache["scaler"] = bundle
                _cache["feature_cols"] = list(getattr(bundle, "feature_names_in_", []))
            
            logger.info(f"✅ Scaler loaded — features: {_cache['feature_cols']}")
        except Exception as e:
            logger.error(f"Failed to load Scaler: {e}")

    # Validate ML Schema
    schema_valid, schema_error, runtime_cols = _validate_runtime_schema(_cache["feature_cols"])
    _cache["schema_valid"] = schema_valid
    _cache["schema_error"] = schema_error
    _cache["runtime_feature_cols"] = runtime_cols

    if schema_valid:
        logger.info("✅ Runtime schema perfectly matches trained scaler metadata.")
    else:
        logger.error(f"❌ Runtime schema invalid: {schema_error}")
        _cache["xgb"] = None
        _cache["scaler"] = None

    logger.info("✅ Global ML routing cache fully initialized.")
    return _cache
