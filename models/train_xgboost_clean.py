"""Train XGBoost from clean CV-based dataset.

Expects dataset:
    data/eco_driving_cv_dataset_clean.csv

Run:
    python models/train_xgboost_clean.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("driveiq.train.xgb")
logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "eco_driving_cv_dataset_clean.csv"
OUT = ROOT / "models"
OUT.mkdir(parents=True, exist_ok=True)

# CRITICAL RULE: DO NOT REORDER
FEATURE_COLS = [
    "mean_flow",
    "flow_variance",
    "braking_ratio",
    "lane_change_ratio",
    "proximity_score",
    "vehicle_density",
    "pedestrian_ratio",
    "low_motion_ratio"
]

TARGET_COL = "eco_score"

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Dataset is empty.")

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(float)

    # 70 / 15 / 15 split
    X_train_raw, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    # STRICT SCALER CONSISTENCY
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_raw)
    X_val_s   = scaler.transform(X_val_raw)
    X_test_s  = scaler.transform(X_test_raw)

    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, scaler

def evaluate(name, model, X, y):
    p = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, p))
    mae = mean_absolute_error(y, p)
    r2 = r2_score(y, p)
    logger.info(f"{name:10s} RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
    return rmse, mae, r2

def main():
    try:
        from xgboost import XGBRegressor
    except ImportError as e:
        raise RuntimeError("XGBoost import failed.") from e

    logger.info("Loading and scaling dataset ...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data()

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )

    logger.info("Training XGBoost ...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    logger.info("\nEvaluation")
    evaluate("Train", model, X_train, y_train)
    evaluate("Val", model, X_val, y_val)
    evaluate("Test", model, X_test, y_test)

    model_path = OUT / "xgb_scorer.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved model -> {model_path}")

    scaler_path = OUT / "scaler.pkl"
    joblib.dump(
        {
            "scaler": scaler,
            "feature_cols": FEATURE_COLS,
            "schema": "xgb_v3_clean_cv",
            "target_col": TARGET_COL,
        },
        scaler_path,
    )
    logger.info(f"Saved scaler -> {scaler_path}")
    logger.info("Done.")

if __name__ == "__main__":
    main()
