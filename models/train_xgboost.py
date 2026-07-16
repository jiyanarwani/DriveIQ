"""
Phase 2A: XGBoost Fuel Efficiency Scorer
Trains an XGBRegressor to predict eco score (0–100).
Outputs: models/xgb_scorer.pkl

Run: python models/train_xgboost.py
"""

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT  = ROOT / "models"
OUT.mkdir(parents=True, exist_ok=True)


def load_splits():
    feature_cols = None
    scaler_path = OUT / "scaler.pkl"
    if scaler_path.exists():
        try:
            bundle = joblib.load(scaler_path)
            feature_cols = bundle.get("feature_cols")
        except Exception:
            feature_cols = None

    def _load_x(name: str) -> pd.DataFrame:
        df = pd.read_csv(DATA / f"{name}.csv")
        # If old split files were saved without headers, restore canonical names from scaler metadata.
        if feature_cols and len(feature_cols) == df.shape[1]:
            try:
                numeric_headers = all(str(c).isdigit() for c in df.columns)
                if numeric_headers:
                    df.columns = feature_cols
            except Exception:
                pass
        return df

    X_train = _load_x("X_train")
    X_val   = _load_x("X_val")
    X_test  = _load_x("X_test")
    y_train = pd.read_csv(DATA / "y_train.csv").iloc[:, 0]
    y_val   = pd.read_csv(DATA / "y_val.csv").iloc[:, 0]
    y_test  = pd.read_csv(DATA / "y_test.csv").iloc[:, 0]
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(name, model, X, y):
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae  = mean_absolute_error(y, preds)
    r2   = r2_score(y, preds)
    print(f"[xgboost] {name:10s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return rmse, mae, r2


def main():
    print("[xgboost] Loading data ...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # ── Baseline model ────────────────────────────────────────────────────────
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

    print("[xgboost] Training ...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    print("\n[xgboost] ── Evaluation ──")
    evaluate("Train", model, X_train, y_train)
    val_rmse, _, _ = evaluate("Val",   model, X_val,   y_val)
    evaluate("Test",  model, X_test,  y_test)

    if val_rmse < 8.0:
        print(f"[xgboost] ✅ Target met — Val RMSE {val_rmse:.4f} < 8.0")
    else:
        print(f"[xgboost] ⚠️  Val RMSE {val_rmse:.4f} > 8.0 — consider further tuning")

    # ── Save model ────────────────────────────────────────────────────────────
    out_path = OUT / "xgb_scorer.pkl"
    joblib.dump(model, out_path)
    print(f"\n[xgboost] Saved → {out_path}")

    # ── SHAP analysis ─────────────────────────────────────────────────────────
    print("[xgboost] Generating SHAP summary ...")
    try:
        explainer = shap.TreeExplainer(model)
        x_shap = X_test.iloc[:500]
        shap_values = explainer.shap_values(x_shap)  # sample for speed
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, x_shap, show=False)
        shap_path = OUT / "shap_summary.png"
        plt.savefig(shap_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"[xgboost] SHAP plot saved → {shap_path}")
    except Exception as e:
        print(f"[xgboost] SHAP failed (non-critical): {e}")

    print("[xgboost] ✅ Done.")


if __name__ == "__main__":
    main()
