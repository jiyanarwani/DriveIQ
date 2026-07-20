from pydantic import BaseModel, Field

# ── Health Schemas ────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    service: str
    models_loaded: bool
    core_models_loaded: bool
    schema_valid: bool
    schema_error: str | None = None
    predictor_loaded: bool
    missing_core_models: list[str]
    score_ready: bool
    coach_ready: bool
    coach_status: str
    coach_disabled: bool
    ready: bool
    degraded: bool
    version: str
    uptime_seconds: float
    mongodb_connected: bool
    gpu_available: bool

# ── Auth Schemas ──────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    token: str

# ── Dashboard Schemas ─────────────────────────────────────────────────────────
class DashboardMetricsResponse(BaseModel):
    mean_eco_score: int
    lowest_eco_score: int
    total_trips: int
    trips_this_week: int

# ── Coach Schemas ─────────────────────────────────────────────────────────────
class CoachRequest(BaseModel):
    score: float = 50.0
    features: dict = {}
    session_id: str | None = None
    is_summary: bool = False
    events: list[str] | None = None

class CoachResponse(BaseModel):
    message: str
    tips: list[str]
    severity: str
    source: str
    model_loaded: bool

# ── Score Schemas ─────────────────────────────────────────────────────────────
class ScoreRequest(BaseModel):
    telemetry: dict = {}
    frame_b64: str | None = None
    prev_frame_b64: str | None = None
    scoring_mode: str = "event_rules"
    session_id: str | None = None
    session_started_at: str | None = None

class ScoreResponse(BaseModel):
    score: float
    eco_score: float
    features: dict
    score_source: str
    auth_failed: bool
    auth_error: str | None = None
    session_saved: bool
    session_save_error: str | None = None

class TripHistoryItem(BaseModel):
    session_id: str
    user_id: str
    final_score: float
    frame_count: int
    top_event: str
    total_events: int
    date: str | None = None

class TimelineFrame(BaseModel):
    timestamp_sec: float
    score: float
    eco_score: float
    features: dict
    events: list[str]
    severity: str

# ── Review Schemas ────────────────────────────────────────────────────────────
class ReviewInitResponse(BaseModel):
    task_id: str
    status: str

class WindowOut(BaseModel):
    timestamp_sec: float
    score: float
    severity: str
    top_issue: str
    coach_note: str
    score_source: str
    events: list[str]

class SeverityThresholds(BaseModel):
    mode: str
    yellow_min: float
    green_min: float

class GeminiSummaryFields(BaseModel):
    overall_rating: str
    what_went_well: list[str]
    areas_to_improve: list[str]
    summary_paragraph: str

class GeminiSummary(BaseModel):
    summary: GeminiSummaryFields | None = None
    error: str | None = None
    raw: str | None = None

class ReviewResult(BaseModel):
    windows: list[WindowOut]
    duration_sec: float
    window_count: int
    severity_thresholds: SeverityThresholds
    session_summary: GeminiSummary

class ReviewStatusResponse(BaseModel):
    task_id: str
    status: str
    error: str | None = None
    result: ReviewResult | None = None
