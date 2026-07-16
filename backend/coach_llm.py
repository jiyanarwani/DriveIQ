"""
backend/coach_llm.py
Google Gemini API inference for dynamic driving coaching tips.

Uses gemini-1.5-flash to generate highly contextual, varied coaching
advice instantly without burning local memory.
"""

from __future__ import annotations

import logging
import time
import os
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

logger = logging.getLogger("driveiq.coach_llm")

# Initialize and configure Gemini
_configured = False
_client = None

def _init_gemini():
    """Load API key and configure Google Generative AI."""
    global _configured, _client
    if _configured:
        return True

    # Ensure .env is loaded to grab GEMINI_API_KEY
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in environment. Coaching will fallback to rules.")
        return False
        
    try:
        _client = genai.Client(api_key=api_key)
        _configured = True
        logger.info("✅ Gemini API fully configured for coaching.")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to configure Gemini: {e}")
        return False


def build_coaching_prompt(score: float, severity: str, events: list[str], features: dict) -> str:
    """Build a natural instruction prompt for Gemini."""

    if not events:
        event_desc = "The driving was flawless with no issues detected."
    else:
        event_names = [e.replace("_", " ") for e in events]
        event_desc = "Specific issues detected: " + ", ".join(event_names) + "."

    # Adding context makes Gemini highly intelligent
    context_parts = []
    proximity = float(features.get("proximity_score", 0.0))
    braking = float(features.get("braking_ratio", features.get("braking_flag", 0.0)))

    if proximity > 0.15:
        context_parts.append(f"driver is tailgating with very high risk (proximity: {proximity:.2f})")
    if braking > 0.3:
        context_parts.append("driver slams the brakes constantly instead of coasting")

    context = (" Telemetry shows: " + "; ".join(context_parts) + ".") if context_parts else ""

    prompt = (
        f"You are a professional, encouraging driving instructor. "
        f"A driver just finished a segment and scored {score:.0f} out of 100 (severity zone: {severity}).\n"
        f"{event_desc}{context}\n"
        f"Based on this data, provide a highly detailed, practical coaching tip (3-4 sentences minimum) "
        f"to improve their driving safety or fuel economy. Explain the 'why' behind your advice. Don't mention the score itself."
    )
    return prompt


def generate_coaching_tip(
    score: float,
    severity: str,
    events: list[str],
    features: dict
) -> tuple[str | None, str]:
    """
    Generate a dynamic coaching tip using Google Gemini 2.5 Flash.

    Returns:
        (tip_text, source) 
        source is "gemini" or "unavailable"
    """
    if not _init_gemini():
        return None, "unavailable"

    prompt = build_coaching_prompt(score, severity, events, features)

    try:
        t0 = time.perf_counter()
        
        # Generation config specifically tailored for direct, non-chatty responses
        response = _client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=300,
            )
        )
        
        tip = response.text.strip()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"Gemini generated tip in {elapsed_ms:.0f}ms")
        
        return tip, "gemini"

    except Exception as e:
        logger.warning(f"Gemini generation failed: {e}")
        return None, "unavailable"


def is_model_loaded() -> bool:
    """Check if Gemini API is configured and ready."""
    return _init_gemini()
