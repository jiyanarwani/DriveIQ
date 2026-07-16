# DriveIQ — AI-Powered Driving Analysis & Coaching Dashboard

DriveIQ is a modern, comprehensive AI-powered platform designed to analyze driving runs, evaluate driver safety and efficiency, and provide real-time, context-aware coaching feedback. It leverages a hybrid system combining Computer Vision (CV), Machine Learning (ML), and Large Language Models (LLMs) to scan driver behavior and offer recommendations.

---

## 🚀 Key Features

* **Computer Vision Processing**: Motion estimation with Optical Flow and vehicle/obstacle tracking via YOLOv8.
* **Predictive ML scoring**: An XGBoost model classifies the severity of driving runs and generates safety scores based on telemetry features.
* **Generative AI Coaching**: Real-time integration with Google Gemini (`gemini-2.5-flash`) that translates telemetry analysis into practical, encouraging driving feedback.
* **Immersive Dashboard**: A React-based web interface showing interactive telemetry timelines, 3D visualizations, and performance statistics.
* **User Authentication & Trip History**: Custom secure JWT authentication backed by MongoDB to track historic sessions.

---

## 🛠️ Technology Stack

### **Frontend**
* **Framework**: React.js with Vite
* **Styling**: Vanilla CSS
* **Visualizations**: 3D rendering (React Three Fiber, Drei, Three.js) and 2D charts (Chart.js, React-Chartjs-2)
* **API Client**: Axios

### **Backend**
* **Framework**: Flask (Python) with CORS
* **Database**: MongoDB (via `pymongo`)
* **Machine Learning & CV**:
  * XGBoost & scikit-learn (Scoring model)
  * PyTorch & Ultralytics YOLOv8 (Vehicle detection)
  * OpenCV (Optical flow / image processing)
  * SHAP (Model interpretability visualization)
* **GenAI / LLMs**: Google GenAI SDK (Gemini API) and Anthropic SDK (Claude API integration ready)
* **Security**: JWT (`PyJWT`), `bcrypt`

---

## 📁 Repository Structure

```text
├── backend/                # Flask REST API implementation
│   ├── routes/             # API Endpoints (auth, health, score, review, dashboard, coach)
│   ├── app.py              # API server entrypoint
│   ├── auth.py             # User JWT/auth helper functions
│   ├── db.py               # MongoDB database manager
│   ├── model_loader.py     # Loader for pretrained ML models
│   ├── scoring.py          # Processing metrics and scoring engine
│   └── coach_llm.py        # Google Gemini API connector for coaching
├── frontend/               # React + Vite client-side dashboard
│   ├── src/                # Component logic, views, hooks, and context
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

---

## ⚙️ Installation & Local Setup

### **1. Prerequisites**
Ensure you have the following installed:
* [Python 3.9+](https://www.python.org/downloads/)
* [Node.js (LTS)](https://nodejs.org/)
* [MongoDB](https://www.mongodb.com/) (Local or Atlas cloud database)

---

### **2. Backend Setup**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jiyanarwani/DriveIQ.git
   cd DriveIQ
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python -m venv .venv
   # Activate on Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # Activate on macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   MONGO_URI=mongodb://localhost:27017/DriveIQ
   GEMINI_API_KEY=your_google_gemini_api_key_here
   JWT_SECRET=your_jwt_secret_key_here
   FLASK_DEBUG=1
   ```

5. **Start Flask Server**:
   ```bash
   python backend/app.py
   ```
   The backend will be available at `http://localhost:5000`.

---

### **3. Frontend Setup**

1. **Navigate to the frontend folder**:
   ```bash
   cd frontend
   ```

2. **Install node dependencies**:
   ```bash
   npm install
   ```

3. **Start the Vite development server**:
   ```bash
   npm run dev
   ```
   The web dashboard will start at `http://localhost:5173`.

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :--- |
| **GET** | `/api/health` | Service status health check | No |
| **POST** | `/api/auth/register` | Register a new driver profile | No |
| **POST** | `/api/auth/login` | Login to receive a JWT token | No |
| **POST** | `/api/score` | Score a telemetry data frame | Yes |
| **POST** | `/api/coach` | Fetch LLM-generated coaching recommendations | Yes |
| **GET** | `/api/dashboard/metrics` | Retrieve user stats & weekly trip summaries | Yes |

---

## 🛡️ License
Distributed under the MIT License.
