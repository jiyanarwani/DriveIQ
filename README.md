# DriveIQ — AI Powered Driving Analysis & Coaching Dashboard

DriveIQ is a modern, comprehensive AI-powered platform designed to analyze driving runs, evaluate driver safety and efficiency, and provide real-time, context-aware coaching feedback. It leverages a hybrid system combining Computer Vision (CV), Machine Learning (ML), and Large Language Models (LLMs) to scan driver behavior and offer recommendations.

---

## Key Features

* **Computer Vision Processing**: Motion estimation with Optical Flow (Lucas-Kanade / Farnebäck) and vehicle/obstacle tracking via Ultralytics YOLOv8.
* **Hybrid Scoring Engine (Predictive ML + Deterministic Guardrails)**:
  * **XGBoost Regression**: Predicts an eco-efficiency score (0–100) from an 8-dimensional spatiotemporal telemetry and computer vision feature vector.
  * **Deterministic Event Guardrails**: Evaluates critical driving infractions (tailgating proximity, hard braking, lane swerving, pedestrian risk, flow variance) with confidence thresholds.
  * **Exponential Moving Average (EMA)**: Smooths sequential window scores to eliminate score volatility caused by optical noise and camera vibrations.
* **Generative AI Coaching**: Direct integration with Google Gemini (`gemini-2.5-flash` via the official `google-genai` SDK) translating telemetry vectors and infraction timelines into practical, encouraging driving coaching tips.
* **Async Video Review**: Upload driving footage for background processing (YOLO + Optical Flow + scoring engine) to generate detailed timeline statistics and structured lesson summaries.
* **PDF Report Generation**: Download stylized PDF coaching reports containing journey performance metrics, Gemini feedback, and significant infraction timelines (built with ReportLab).
* **Lightweight Dashboard**: A clean, responsive Plain HTML5, CSS3, and Vanilla JavaScript interface featuring interactive telemetry timelines, active infraction overlays, and native SVG score gauges and trend lines (zero external UI framework or chart library dependencies).
* **User Authentication & Trip History**: Custom secure JWT authentication backed by MongoDB to track historic sessions and lifetime driving metrics.

---

## Technology Stack

### **Frontend**
* **Markup & Structure**: Plain HTML5 & CSS3
* **Scripting**: Pure Vanilla JavaScript (0 runtime dependencies, 0 external UI frameworks, zero React, zero Node.js)
* **Visualizations**: Native Browser SVG and CSS vector graphics (Score Gauge and Trend Line)
* **API Client**: Native Browser `fetch`

### **Backend**
* **Framework**: FastAPI (Python) serving both the REST API and Dashboard static files
* **Database**: MongoDB (via `pymongo`)
* **Machine Learning & CV**:
  * XGBoost & scikit-learn (Scoring model and telemetry feature scaling)
  * PyTorch & Ultralytics YOLOv8 (Vehicle and obstacle detection; 1D-CNN + LSTM sequential architectures)
  * OpenCV (Optical flow / image processing)
  * SHAP (Model interpretability and feature attribution)
* **GenAI / LLMs**: Google GenAI SDK (Gemini 2.5 Flash API for coaching and trip synthesis)
* **Security**: JWT (`PyJWT`), `bcrypt`
* **Report Generation**: ReportLab (PDF)

---

## Getting Started

### Prerequisites
* Python 3.10+
* MongoDB (optional for local guest mode; required for persistent trip history and JWT auth)

---

### Quick Start (Single Command)

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables (optional)**:
   Create a `.env` file in the root directory:
   ```env
   MONGO_URI=mongodb://localhost:27017/DriveIQ
   JWT_SECRET=your_secret_key_at_least_32_characters_long
   GEMINI_API_KEY=your_gemini_api_key_here
   PORT=5000
   ```

3. **Train / refresh the ML model (optional)**:
   A baseline XGBoost model and scaler are pre-compiled under `models/`. To retrain from scratch:
   ```bash
   python models/train_xgboost_clean.py
   ```

4. **Start the application**:
   ```bash
   python run.py
   ```
   * **Dashboard UI**: `http://localhost:5000`
   * **Interactive API Docs**: `http://localhost:5000/docs`

---

## Scoring Architecture

DriveIQ uses a production-inspired **two-tier hybrid scoring strategy**:

```text
  [ Dashcam Video / Telemetry ]
                │
        ┌───────┴───────┐
        ▼               ▼
 [ YOLOv8 Detections ] [ Optical Flow Field ]
        └───────┬───────┘
                ▼
  [ 8-D CV Feature Vector ]
   (mean_flow, flow_variance, braking_ratio,
    lane_change_ratio, proximity_score,
    vehicle_density, pedestrian_ratio, low_motion_ratio)
                │
        ┌───────┴──────────────────────────┐
        ▼                                  ▼
[ Tier 1: XGBoost Regressor ]   [ Tier 2: Event Deductions ]
  • Predicts base eco-score       • Deterministic infraction checks
  • SHAP-interpretable            • Guaranteed safety penalty bounds
        └───────┬──────────────────────────┘
                ▼
    [ EMA Temporal Smoothing ] (α = 0.6)
                │
                ▼
  [ Final Score + Gemini AI Coaching ]
```

* **Fault-Tolerant Runtime**: If ML artifacts are missing or unscaled, the backend automatically fails over to the deterministic rule engine without degrading API availability.

---

## Repository Structure

```text
├── backend/                # FastAPI REST API implementation
│   ├── routes/             # API Endpoints (auth, health, score, review, dashboard, coach)
│   ├── app.py              # API server entrypoint & static mount
│   ├── auth.py             # User JWT/auth helper functions
│   ├── config.py           # Central configuration using Pydantic Settings
│   ├── db.py               # MongoDB database manager
│   ├── model_loader.py     # Loader & schema validator for pretrained ML models
│   ├── schemas.py          # Unified Pydantic schema validation models
│   ├── scoring.py          # Event-based deduction engine & EMA smoothing
│   └── coach_llm.py        # Google Gemini API connector for coaching tips
├── frontend/               # Plain HTML5 + CSS3 + Vanilla JS client-side dashboard
│   ├── index.html          # Main HTML entrypoint
│   ├── index.css           # Styling rules
│   ├── app.js              # Application logic, visual renderers, and event management
│   └── favicon.svg         # Dashboard icon
├── cv/                     # Computer Vision pipelines
│   ├── cv_pipeline.py      # Combines optical flow & YOLO tracking into XGB feature vectors
│   ├── optical_flow.py     # Dense/Sparse motion analysis
│   └── yolo_pipeline.py    # YOLOv8 object detection wrapper
├── models/                 # ML scoring model training and inference
│   ├── predictor.py        # PyTorch 1D-CNN + LSTM sequential model architecture
│   ├── train_xgboost.py    # XGBoost training script with SHAP analysis
│   ├── train_xgboost_clean.py # Clean CV-dataset XGBoost training & scaler serializer
│   ├── xgb_scorer.pkl      # Pretrained XGBoost regression model artifact
│   └── scaler.pkl          # Feature scaler and schema metadata bundle
├── pipeline/               # Data ingestion & dataset creation scripts
│   └── video_dataset_builder.py # Extracts frame windows and builds training datasets
├── run.py                  # Single-command application launcher
├── requirements.txt        # Clean Python pip dependencies
└── README.md               # Project documentation
```
