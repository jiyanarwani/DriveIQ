# DriveIQ — AI Powered Driving Analysis & Coaching Dashboard

DriveIQ is a modern, comprehensive AI powered platform designed to analyze driving runs, evaluate driver safety and efficiency, and provide real-time, context aware coaching feedback. It leverages a hybrid system combining Computer Vision (CV), Machine Learning (ML) and Large Language Models (LLMs) to scan driver behavior and offer recommendations.

---

## Key Features

* **Computer Vision Processing**: Motion estimation with Optical Flow and vehicle/obstacle tracking via YOLOv8.
* **Predictive ML Scoring**: An XGBoost model classifies the severity of driving runs and generates safety scores based on telemetry features.
* **Generative AI Coaching**: Integration with Google Gemini (`gemini-2.5-flash` using the official `google-genai` SDK) translating telemetry analysis into practical, encouraging driving feedback.
* **Async Video Review**: Upload driving videos for background processing (YOLO + Optical Flow + XGBoost) to generate detailed timeline stats.
* **PDF Report Generation**: Download stylized PDF coaching reports containing journey performance metrics, Gemini feedback and significant infraction timelines (built with ReportLab).
* **Immersive Dashboard**: A React-based web interface showing interactive telemetry timelines, active infraction overlays, custom SVG gauges and performance statistics.
* **User Authentication & Trip History**: Custom secure JWT authentication backed by MongoDB to track historic sessions.

---

## Technology Stack

### **Frontend**
* **Markup & Structure**: Plain HTML5 & CSS3
* **Scripting**: Pure Vanilla JavaScript (zero UI runtime dependencies, zero React)
* **Visualizations**: Native SVG and CSS telemetry gauges and trend charts
* **API Client**: Native Browser `fetch`

### **Backend**
* **Framework**: FastAPI (Python) with CORS
* **Database**: MongoDB (via `pymongo`)
* **Machine Learning & CV**:
  * XGBoost & scikit-learn (Scoring model)
  * PyTorch & Ultralytics YOLOv8 (Vehicle detection)
  * OpenCV (Optical flow / image processing)
  * SHAP (Model interpretability visualization)
* **GenAI / LLMs**: Google GenAI SDK (Gemini API) and Anthropic SDK (Claude API integration ready)
* **Security**: JWT (`PyJWT`), `bcrypt`
* **Report Generation**: ReportLab (PDF)

---

## Repository Structure

```text
├── backend/                # FastAPI REST API implementation
│   ├── routes/             # API Endpoints (auth, health, score, review, dashboard, coach)
│   ├── app.py              # API server entrypoint
│   ├── auth.py             # User JWT/auth helper functions
│   ├── config.py           # Central configuration using Pydantic Settings
│   ├── db.py               # MongoDB database manager
│   ├── model_loader.py     # Loader for pretrained ML models
│   ├── schemas.py          # Unified Pydantic schema validation models
│   ├── scoring.py          # Processing metrics and scoring engine
│   └── coach_llm.py        # Google Gemini API connector for coaching
├── frontend/               # Plain HTML5 + CSS3 + Vanilla JS client-side dashboard
│   ├── index.html          # Main HTML entrypoint
│   ├── index.css           # Styling rules
│   ├── app.js              # Application logic and event management
│   └── package.json        # Frontend NPM configurations
├── cv/                     # Computer Vision pipelines
│   ├── cv_pipeline.py      # Combines optical flow & YOLO tracking
│   ├── optical_flow.py     # Dense/Sparse motion analysis
│   └── yolo_pipeline.py    # YOLOv8 object detection wrapper
├── models/                 # ML scoring model training and inference
│   ├── predictor.py        # XGBoost scoring inference script
│   └── train_xgboost.py    # Training & evaluation script
├── pipeline/               # Data ingestion & dataset creation scripts
├── requirements.txt        # Python pip dependencies
└── README.md               # Project documentation (this file)
```

