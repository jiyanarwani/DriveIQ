import json
import logging
import os
import tempfile
import time
import uuid
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, UploadFile, Form, BackgroundTasks, HTTPException, status

from cv.cv_pipeline import feature_vector_for_xgb
from pipeline.video_dataset_builder import aggregate_windows
from backend.scoring import score_window, score_windows_with_ema, event_to_issue_key
from backend.model_loader import load_models
from backend.schemas import ReviewInitResponse, ReviewStatusResponse

review_router = APIRouter(prefix="/api/v1")
CV_DEBUG = os.environ.get("DRIVEIQ_CV_DEBUG", "0") == "1"
logger = logging.getLogger("driveiq.review")

GREEN_THRESHOLD = 80.0
YELLOW_THRESHOLD = 65.0

# Global dictionary to track background review tasks
_review_tasks: dict[str, dict[str, Any]] = {}


def classify_severity(score):
    if score >= GREEN_THRESHOLD:
        return "green"
    elif score >= YELLOW_THRESHOLD:
        return "yellow"
    else:
        return "red"


def _generate_session_summary_gemini(windows: list[dict], duration_sec: float) -> dict:
    """Use Gemini to generate a structured overall session summary with robust truncation recovery."""
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"summary": None, "error": "gemini_unavailable"}

        import google.genai as genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        scores = [float(w.get("score", 0)) for w in windows]
        avg_score = sum(scores) / max(len(scores), 1)
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 0

        event_counts: dict[str, int] = {}
        for w in windows:
            for ev in w.get("events", []):
                event_counts[ev] = event_counts.get(ev, 0) + 1

        feature_keys = ["mean_flow", "flow_variance", "low_motion_ratio", "proximity_score"]
        feature_avgs = {}
        for fk in feature_keys:
            vals = [float(w.get(fk, 0)) for w in windows if fk in w]
            feature_avgs[fk] = round(sum(vals) / max(len(vals), 1), 3) if vals else 0.0

        prompt = (
            "You are a professional driving analyst. Analyze this driving session data and return ONLY valid JSON with no markdown formatting, no code fences, no preamble.\n"
            f"\nSession stats:\n"
            f"- Duration: {duration_sec:.1f} seconds\n"
            f"- Windows analyzed: {len(windows)}\n"
            f"- Average score: {avg_score:.1f} / 100\n"
            f"- Min score: {min_score:.1f}, Max score: {max_score:.1f}\n"
            f"- Events detected: {json.dumps(event_counts) if event_counts else 'None'}\n"
            f"- Avg mean_flow: {feature_avgs.get('mean_flow', 0):.3f}\n"
            f"- Avg flow_variance: {feature_avgs.get('flow_variance', 0):.3f}\n"
            f"- Avg low_motion_ratio: {feature_avgs.get('low_motion_ratio', 0):.3f}\n"
            f"- Avg proximity_score: {feature_avgs.get('proximity_score', 0):.3f}\n"
            "\nReturn JSON with exactly these keys:\n"
            '- "overall_rating": one of "Excellent", "Good", "Needs Improvement", "Poor"\n'
            '- "what_went_well": list of 2-3 SHORT phrases (max 12 words each)\n'
            '- "areas_to_improve": list of 2-3 SHORT actionable phrases (max 12 words each)\n'
            '- "summary_paragraph": 2 sentences max\n'
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "overall_rating": {
                            "type": "string",
                            "enum": ["Excellent", "Good", "Needs Improvement", "Poor"]
                        },
                        "what_went_well": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 80},
                            "minItems": 2,
                            "maxItems": 3
                        },
                        "areas_to_improve": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 80},
                            "minItems": 2,
                            "maxItems": 3
                        },
                        "summary_paragraph": {
                            "type": "string",
                            "maxLength": 300
                        }
                    },
                    "required": ["overall_rating", "what_went_well", "areas_to_improve", "summary_paragraph"]
                }
            )
        )

        raw_text = response.text.strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[-2] if "```" in raw_text[3:] else raw_text
            raw_text = raw_text.lstrip("`").lstrip("json").strip()

        try:
            parsed = json.loads(raw_text)
            return {"summary": parsed, "error": None}
        except json.JSONDecodeError:
            recovered = _try_recover_truncated_json(raw_text)
            if recovered:
                logger.warning("/api/v1/review gemini_summary_recovered via truncation fix")
                return {"summary": recovered, "error": None}

            logger.warning(f"Gemini session summary JSON parse failed. Raw: {raw_text[:200]}...")
            return {"summary": None, "error": "gemini_parse_error", "raw": raw_text}

    except Exception as e:
        logger.error(f"Gemini session summary failed: {e}")
        return {"summary": None, "error": "gemini_unavailable"}


def _try_recover_truncated_json(raw: str) -> dict | None:
    required_keys = {"overall_rating", "what_went_well", "areas_to_improve", "summary_paragraph"}
    closers = ['"}]}', '"]}', '"}', ']}}', '}}', '}']
    for closer in closers:
        try:
            candidate = raw.rstrip().rstrip(",") + closer
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and required_keys.issubset(parsed.keys()):
                return parsed
        except (json.JSONDecodeError, Exception):
            continue
    return None


def _extract_review_fast_features(
    video_path: Path,
    sample_every: int = 3,
    max_frames_per_video: int | None = None,
) -> pd.DataFrame:
    """Review extractor using the real CV pipeline (YOLO + optical flow)."""
    from cv.cv_pipeline import cv_pipeline
    from cv.optical_flow import reset_flow_state
    reset_flow_state()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return pd.DataFrame()

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0

    rows: list[dict] = []
    prev_frame: np.ndarray | None = None
    read_frame_idx = 0
    sampled_count = 0
    prev_ts_sec: float | None = None
    timeline_gap_count = 0
    max_gap_sec = 0.0
    expected_dt = sample_every / fps

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if read_frame_idx % sample_every != 0:
            read_frame_idx += 1
            continue

        actual_ts_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        if actual_ts_sec <= 0.0:
            actual_ts_sec = read_frame_idx / fps

        if prev_ts_sec is not None:
            dt = max(0.0, actual_ts_sec - prev_ts_sec)
            if dt > expected_dt * 1.3:
                timeline_gap_count += 1
                max_gap_sec = max(max_gap_sec, dt)
        prev_ts_sec = actual_ts_sec

        actual_frame_idx = int(round(actual_ts_sec * fps))

        try:
            cv_feats = cv_pipeline(frame, prev_frame, telemetry={})
        except Exception as e:
            logger.error(f"cv_pipeline error: {e}")
            cv_feats = {
                "mean_flow": 0.0, "flow_variance": 0.0,
                "braking_ratio": 0.0, "lane_change_ratio": 0.0,
                "proximity_score": 0.0, "vehicle_density": 0.0,
                "pedestrian_ratio": 0.0, "low_motion_ratio": 1.0,
                "braking_flag": 0.0, "lane_change_flag": 0.0,
                "vehicle_count": 0.0, "pedestrian_flag": 0.0,
            }

        rows.append(
            {
                "video_id": video_path.stem,
                "source_dir": video_path.parent.name,
                "frame_idx": actual_frame_idx,
                "timestamp_sec": actual_ts_sec,
                "mean_flow": float(cv_feats.get("mean_flow", 0.0)),
                "flow_variance": float(cv_feats.get("flow_variance", cv_feats.get("variance", 0.0))),
                "braking_ratio": float(cv_feats.get("braking_ratio", cv_feats.get("braking_flag", 0.0))),
                "lane_change_ratio": float(cv_feats.get("lane_change_ratio", cv_feats.get("lane_change_flag", 0.0))),
                "proximity_score": float(cv_feats.get("proximity_score", 0.0)),
                "vehicle_density": float(cv_feats.get("vehicle_density", cv_feats.get("vehicle_count", 0.0))),
                "pedestrian_ratio": float(cv_feats.get("pedestrian_ratio", cv_feats.get("pedestrian_flag", 0.0))),
                "low_motion_ratio": float(cv_feats.get("low_motion_ratio", 0.0)),
                "vehicle_count": float(cv_feats.get("vehicle_count", cv_feats.get("vehicle_density", 0.0))),
                "braking_flag": float(cv_feats.get("braking_flag", cv_feats.get("braking_ratio", 0.0))),
                "lane_change_flag": float(cv_feats.get("lane_change_flag", cv_feats.get("lane_change_ratio", 0.0))),
                "pedestrian_flag": float(cv_feats.get("pedestrian_flag", cv_feats.get("pedestrian_ratio", 0.0))),
                "road_type_id": 0.0,
                "weather_id": 0.0,
            }
        )

        prev_frame = frame
        sampled_count += 1
        read_frame_idx += 1

        if max_frames_per_video is not None and sampled_count >= max_frames_per_video:
            break

    cap.release()

    if timeline_gap_count > 0:
        logger.warning(
            "/api/v1/review timeline_gaps_detected video=%s count=%s max_gap_sec=%.3f",
            video_path.name,
            timeline_gap_count,
            max_gap_sec,
        )

    return pd.DataFrame(rows)


def _heuristic_score_from_features(features: dict) -> float:
    score = 85.0
    score -= float(features.get("proximity_score", 0.0)) * 30.0
    score -= float(features.get("mean_flow", 0.0)) * 8.0
    score -= float(features.get("flow_variance", 0.0)) * 10.0
    score -= float(features.get("braking_ratio", features.get("braking_flag", 0.0))) * 12.0
    score -= float(features.get("lane_change_ratio", features.get("lane_change_flag", 0.0))) * 10.0
    score -= float(features.get("pedestrian_ratio", features.get("pedestrian_flag", 0.0))) * 5.0
    return max(0.0, min(100.0, score))


def _evaluate_rules(score: float, features: dict, events: list[str]) -> list[str]:
    tips = []
    if "hard_braking" in events:
        tips.append("Reduce hard braking — anticipate stops earlier.")
    if "tailgating" in events:
        tips.append("Increase following distance to reduce collision risk.")
    if "lane_swerving" in events:
        tips.append("Maintain steady lane position — avoid abrupt steering.")
    if "erratic_speed" in events or float(features.get("flow_variance", 0)) > 15:
        tips.append("Erratic speed changes detected — maintain steady throttle.")
    if float(features.get("mean_flow", 0)) > 15:
        tips.append("High optical flow — possible aggressive acceleration.")
    if float(features.get("low_motion_ratio", 0)) > 0.6:
        tips.append("Extended idle detected — consider smoother route planning.")
    if not tips:
        tips.append("Maintain smooth, consistent driving.")
    return tips


# ── Background Task processing ──────────────────────────────────────────────

def _process_video_task(task_id: str, temp_path: Path, scoring_mode: str) -> None:
    t0 = time.perf_counter()
    logger.info(f"Background task {task_id} started processing")

    try:
        sample_every = 3
        window_size = 20
        stride = 15

        frames_df = _extract_review_fast_features(temp_path, sample_every=sample_every, max_frames_per_video=None)
        logger.info(
            "Task %s extract_done rows=%s elapsed_ms=%.1f",
            task_id,
            len(frames_df),
            (time.perf_counter() - t0) * 1000.0,
        )
        if frames_df.empty:
            _review_tasks[task_id] = {
                "status": "completed",
                "error": None,
                "result": {
                    "windows": [],
                    "duration_sec": 0.0,
                    "window_count": 0,
                    "severity_thresholds": {
                        "mode": "static",
                        "yellow_min": YELLOW_THRESHOLD,
                        "green_min": GREEN_THRESHOLD,
                    },
                    "session_summary": {"summary": None, "error": "empty_video"},
                }
            }
            return

        t_windows = time.perf_counter()
        windows_df = aggregate_windows(frames_df, window_size=window_size, stride=stride)

        if not windows_df.empty:
            span_frames = (windows_df["window_end_frame"] - windows_df["window_start_frame"]).astype(float)
            observed_slots = np.maximum(1.0, np.floor(span_frames / float(sample_every)) + 1.0)
            missing_slots = np.maximum(0.0, observed_slots - float(window_size))
            missing_ratio = np.where(observed_slots > 0.0, missing_slots / observed_slots, 0.0)
            windows_df["missing_frame_ratio"] = missing_ratio

            dropped_windows = windows_df[windows_df["missing_frame_ratio"] > 0.30]
            if not dropped_windows.empty:
                logger.warning(
                    "Task %s skipped_windows_missing_frames count=%s threshold=0.30 max_ratio=%.3f",
                    task_id,
                    len(dropped_windows),
                    float(dropped_windows["missing_frame_ratio"].max()),
                )
                windows_df = windows_df[windows_df["missing_frame_ratio"] <= 0.30].reset_index(drop=True)

        logger.info(
            "Task %s windows_done rows=%s elapsed_ms=%.1f",
            task_id,
            len(windows_df),
            (time.perf_counter() - t_windows) * 1000.0,
        )
        
        if windows_df.empty:
            duration_sec = float(frames_df["timestamp_sec"].max()) if "timestamp_sec" in frames_df.columns else 0.0
            _review_tasks[task_id] = {
                "status": "completed",
                "error": None,
                "result": {
                    "windows": [],
                    "duration_sec": round(duration_sec, 3),
                    "window_count": 0,
                    "severity_thresholds": {
                        "mode": "static",
                        "yellow_min": YELLOW_THRESHOLD,
                        "green_min": GREEN_THRESHOLD,
                    },
                    "session_summary": {"summary": None, "error": "empty_windows"},
                }
            }
            return

        rows_iter = [r for _, r in windows_df.iterrows()]
        duration_sec = float(frames_df["timestamp_sec"].max()) if "timestamp_sec" in frames_df.columns else 0.0

        models = load_models()
        xgb = models.get("xgb")
        scaler = models.get("scaler")
        scoring_path = "xgb" if xgb is not None and scaler is not None else "heuristic"
        logger.info("Task %s scoring_path=%s", task_id, scoring_path)

        windows_out = []
        t_score = time.perf_counter()
        all_feature_dicts = []
        for row in rows_iter:
            feature_dict = {
                "mean_flow": float(row.get("mean_flow", 0.0)),
                "flow_variance": float(row.get("flow_variance", 0.0)),
                "braking_ratio": float(row.get("braking_ratio", row.get("braking_flag", 0.0))),
                "lane_change_ratio": float(row.get("lane_change_ratio", row.get("lane_change_flag", 0.0))),
                "proximity_score": float(row.get("proximity_score", 0.0)),
                "vehicle_density": float(row.get("vehicle_density", row.get("vehicle_count", 0.0))),
                "pedestrian_ratio": float(row.get("pedestrian_ratio", row.get("pedestrian_flag", 0.0))),
                "low_motion_ratio": float(row.get("low_motion_ratio", 0.0)),
                "braking_flag": float(row.get("braking_flag", row.get("braking_ratio", 0.0))),
                "lane_change_flag": float(row.get("lane_change_flag", row.get("lane_change_ratio", 0.0))),
                "pedestrian_flag": float(row.get("pedestrian_flag", 0.0)),
            }
            all_feature_dicts.append((row, feature_dict))

        use_xgb = scoring_mode == "xgboost" and xgb is not None and scaler is not None
        actual_score_source = "xgboost" if use_xgb else "event_ema"

        if use_xgb:
            from cv.cv_pipeline import feature_vector_for_xgb
            prev_smoothed_xgb = 90.0
            for i, (row, feature_dict) in enumerate(all_feature_dicts):
                _, events = score_window(feature_dict)
                try:
                    xgb_input = feature_vector_for_xgb(feature_dict, scaler)
                    raw_score = float(xgb.predict(xgb_input)[0])
                    raw_score = max(0.0, min(100.0, raw_score))
                except Exception as e:
                    logger.warning(f"XGBoost predict failed for window {i}: {e}")
                    raw_score, _ = score_window(feature_dict)

                smoothed = 0.6 * raw_score + 0.4 * prev_smoothed_xgb
                smoothed = max(0.0, min(100.0, smoothed))
                prev_smoothed_xgb = smoothed

                severity = classify_severity(smoothed)
                top_issue = event_to_issue_key(events)
                rule_tips = _evaluate_rules(smoothed, feature_dict, events)
                coach_note = rule_tips[0] if rule_tips else "Maintain smooth, consistent driving."

                windows_out.append({
                    "timestamp_sec": round(float(row.get("timestamp_sec", row.get("window_center_sec", 0.0))), 3),
                    "score": round(smoothed, 2),
                    "severity": severity,
                    "top_issue": top_issue,
                    "coach_note": coach_note,
                    "score_source": "xgboost",
                    "events": events,
                })
        else:
            scored = score_windows_with_ema([fd for _, fd in all_feature_dicts])
            for i, (row, feature_dict) in enumerate(all_feature_dicts):
                result = scored[i]
                score_val = result["smoothed_score"]
                events = result["events"]

                severity = classify_severity(score_val)
                top_issue = event_to_issue_key(events)
                rule_tips = _evaluate_rules(score_val, feature_dict, events)
                coach_note = rule_tips[0] if rule_tips else "Maintain smooth, consistent driving."

                windows_out.append({
                    "timestamp_sec": round(float(row.get("timestamp_sec", row.get("window_center_sec", 0.0))), 3),
                    "score": round(score_val, 2),
                    "severity": severity,
                    "top_issue": top_issue,
                    "coach_note": coach_note,
                    "score_source": "event_ema",
                    "events": events,
                })

        logger.info(
            "Task %s score_done rows=%s elapsed_ms=%.1f",
            task_id,
            len(windows_out),
            (time.perf_counter() - t_score) * 1000.0,
        )

        session_summary = _generate_session_summary_gemini(windows_out, duration_sec)
        logger.info("Task %s session_summary error=%s", task_id, session_summary.get("error"))

        _review_tasks[task_id] = {
            "status": "completed",
            "error": None,
            "result": {
                "windows": windows_out,
                "duration_sec": round(duration_sec, 3),
                "window_count": len(windows_out),
                "severity_thresholds": {
                    "mode": "static",
                    "yellow_min": YELLOW_THRESHOLD,
                    "green_min": GREEN_THRESHOLD,
                },
                "session_summary": session_summary,
            }
        }
        logger.info(f"Background task {task_id} completed successfully in {(time.perf_counter() - t0):.1f}s")
    except Exception as ex:
        logger.exception(f"Background task {task_id} failed: {ex}")
        _review_tasks[task_id] = {
            "status": "failed",
            "error": str(ex),
            "result": None
        }
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
                logger.info(f"Temporary file {temp_path} deleted for task {task_id}")
            except Exception as clean_ex:
                logger.error(f"Error cleaning up temporary file {temp_path}: {clean_ex}")


@review_router.post("/review", response_model=ReviewInitResponse)
async def review_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    scoring_mode: str = Form("event_rules")
) -> dict:
    logger.info("Multipart review upload received.")
    
    if not video.filename:
        raise HTTPException(status_code=400, detail="Uploaded video is empty")

    task_id = str(uuid.uuid4())
    _review_tasks[task_id] = {
        "status": "processing",
        "error": None,
        "result": None
    }

    try:
        # Create a persistent temporary file on disk (deleted after execution completes)
        temp_fd, temp_path_str = tempfile.mkstemp(suffix=".mp4")
        os.close(temp_fd)  # close descriptor, write with shutil
        
        temp_path = Path(temp_path_str)
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
            
        logger.info("Saved review video to temporary file: %s (%s bytes)", temp_path, temp_path.stat().st_size)
    except Exception as io_err:
        logger.error(f"Failed to save uploaded video file: {io_err}")
        _review_tasks[task_id] = {
            "status": "failed",
            "error": f"Failed to save uploaded video: {io_err}",
            "result": None
        }
        return {"task_id": task_id, "status": "failed"}

    # Enqueue background task
    background_tasks.add_task(_process_video_task, task_id, temp_path, scoring_mode)
    
    return {"task_id": task_id, "status": "processing"}


@review_router.get("/review/status/{task_id}", response_model=ReviewStatusResponse)
def get_review_status(task_id: str) -> dict:
    task = _review_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
        
    return {
        "task_id": task_id,
        "status": task["status"],
        "error": task["error"],
        "result": task["result"]
    }


@review_router.get("/review/report/{task_id}")
def export_review_report(task_id: str):
    from fastapi.responses import StreamingResponse
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    task = _review_tasks.get(task_id)
    if not task or task["status"] != "completed":
        raise HTTPException(status_code=404, detail="Task not found or not completed")
    
    result = task["result"]
    windows = result["windows"]
    duration_sec = result["duration_sec"]
    session_summary = result["session_summary"]
    
    scores = [w["score"] for w in windows]
    avg_score = sum(scores) / len(scores) if scores else 0
    min_score = min(scores) if scores else 0
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4A5568'),
        spaceAfter=25
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=15,
        spaceAfter=10
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=10
    )
    
    story = []
    
    story.append(Paragraph("DriveIQ — Driver Coaching & Performance Report", title_style))
    story.append(Paragraph(f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')} | Session ID: {task_id}", subtitle_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("1. Journey Performance Metrics", section_heading))
    data = [
        ["Metric", "Value"],
        ["Overall Average Score", f"{avg_score:.1f} / 100"],
        ["Lowest Window Score", f"{min_score:.1f} / 100"],
        ["Total Duration Evaluated", f"{duration_sec:.1f} seconds"],
        ["Total Evaluation Windows", f"{len(windows)} segments"],
    ]
    t = Table(data, colWidths=[200, 250])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('TOPPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F7FAFC')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('TOPPADDING', (0,1), (-1,-1), 6),
        ('BOTTOMPADDING', (0,1), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("2. AI Coaching Insights & Feedback", section_heading))
    if session_summary and session_summary.get("summary"):
        summary_data = session_summary["summary"]
        rating = summary_data.get("overall_rating", "N/A")
        para = summary_data.get("summary_paragraph", "")
        
        story.append(Paragraph(f"<b>Overall Rating:</b> {rating}", body_style))
        story.append(Paragraph(f"<b>Executive Summary:</b> {para}", body_style))
        story.append(Spacer(1, 5))
        
        story.append(Paragraph("<b>What Went Well:</b>", body_style))
        for item in summary_data.get("what_went_well", []):
            story.append(Paragraph(f"• {item}", body_style))
        story.append(Spacer(1, 5))
            
        story.append(Paragraph("<b>Areas to Improve:</b>", body_style))
        for item in summary_data.get("areas_to_improve", []):
            story.append(Paragraph(f"• {item}", body_style))
    else:
        story.append(Paragraph("AI Coaching Summary is unavailable (fallback rule-based active).", body_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("3. Significant Segment & Infraction Timeline", section_heading))
    timeline_data = [["Timestamp (sec)", "Score", "Dominant Event / Issue", "Coaching Suggestion"]]
    
    critical_windows = [w for w in windows if w["score"] < 80 or w.get("events")]
    if not critical_windows:
        critical_windows = windows[:15]
    else:
        critical_windows = critical_windows[:25]
        
    for w in critical_windows:
        time_str = f"{w['timestamp_sec']:.1f}s"
        score_str = f"{w['score']:.1f}"
        top_issue = w.get("top_issue", "None") or "None"
        note = w.get("coach_note", "Maintain smooth driving.")
        timeline_data.append([time_str, score_str, top_issue.replace('_', ' '), note])
        
    t_timeline = Table(timeline_data, colWidths=[80, 50, 120, 200])
    t_timeline.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4A5568')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_timeline)
    
    doc.build(story)
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=driveiq_report_{task_id}.pdf"}
    )