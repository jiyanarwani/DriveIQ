# DriveIQ — AI Powered Driving Analysis & Coaching Dashboard

DriveIQ is a modern, comprehensive AI powered platform designed to analyze driving runs, evaluate driver safety and efficiency, and provide real-time, context-aware coaching feedback. It leverages a hybrid system combining Computer Vision (CV), Machine Learning (ML), and Large Language Models (LLMs) to scan driver behavior and offer recommendations.

---

## Key Features

* **Computer Vision Processing**: Motion estimation with Optical Flow and vehicle/obstacle tracking via YOLOv8.
* **Predictive ML Scoring**: An XGBoost model classifies the severity of driving runs and generates safety scores based on telemetry features.
* **Generative AI Coaching**: Integration with Google Gemini (`gemini-2.5-flash` using the official `google-genai` SDK) translating telemetry analysis into practical, encouraging driving feedback.
* **Async Video Review**: Upload driving videos for background processing (YOLO + Optical Flow + XGBoost) to generate detailed timeline stats and structured lesson modules.
* **PDF Report Generation**: Download stylized PDF coaching reports containing journey performance metrics, Gemini feedback, and significant infraction timelines (built with ReportLab).
* **Lightweight Dashboard**: A clean, responsive Plain HTML5, CSS3, and Vanilla JavaScript interface featuring interactive telemetry timelines, active infraction overlays, and native SVG score gauges and trend lines (zero external UI framework or chart library dependencies).
* **User Authentication & Trip History**: Custom secure JWT authentication backed by MongoDB to track historic sessions and lifetime driving metrics.

---

## Technology Stack

### **Frontend**
* **Markup & Structure**: Plain HTML5 & CSS3
* **Scripting**: Pure Vanilla JavaScript (0 runtime dependencies, 0 external UI frameworks, zero React)
* **Visualizations**: Native Browser SVG and CSS vector graphics (Score Gauge and Trend Line)
* **Dev Server & Bundling**: Vite (for local dev server and reverse proxying `/api` to the backend)
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

## Getting Started

### Prerequisites
* Python 3.10+
* Node.js 18+ (for running the frontend dev server)
* MongoDB (optional for local guest mode; required for persistent trip history and JWT auth)

---

### Backend Setup

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

3. **Start the FastAPI backend server**:
   ```bash
   python backend/app.py
   ```
   * API Server will run at: `http://localhost:5000`
   * Interactive Swagger documentation: `http://localhost:5000/docs`

---

### Frontend Setup

1. **Install dev dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Start the frontend server**:
   ```bash
   npm run dev
   ```
   * Dashboard will run at: `http://localhost:5173`
   * API calls to `/api` are automatically proxied to the backend at `http://localhost:5000`.

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
│   ├── app.js              # Application logic, visual renderers, and event management
│   ├── vite.config.js      # Dev server & reverse proxy configuration
│   └── package.json        # Frontend configuration (zero runtime dependencies)
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
