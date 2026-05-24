"""
{{PROJECT_NAME}} Gemini API integration.

OPTIONAL MODULE — remove this file and src/prompts.py if not using Gemini.
Also remove google-generativeai from pyproject.toml and app/requirements.txt.

Four features:
  1. explain_prediction()  - plain English explanation for {{PERSONA}}
  2. chat_with_user()      - conversational Q&A about a prediction
  3. detect_patterns()     - narrative pattern detection across a batch
  4. generate_report()     - executive summary for stakeholders

IMPORTANT: Gemini is advisory only. It explains model decisions.
It does not make, modify, or override predictions.

Owner(s):
"""

import os
import logging

import google.generativeai as genai
from dotenv import load_dotenv

from src.prompts import (
    explanation_prompt,
    system_prompt_chat,
    pattern_detection_prompt,
    report_prompt,
)

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-1.5-flash"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not set. Gemini features will not work.")


def _get_model(system_instruction: str = None) -> genai.GenerativeModel:
    kwargs = {"model_name": MODEL_NAME}
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    return genai.GenerativeModel(**kwargs)


def _require_key():
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not configured. Add it to your .env file "
            "or remove gemini.py if this project does not use Gemini."
        )


def explain_prediction(
    record: dict,
    shap_df,
    stream: bool = True,
):
    """
    Feature 1: plain English explanation of a prediction for {{PERSONA}}.

    record: dict of input fields for a single row
    shap_df: DataFrame from src.explain.waterfall_data() for this row
    stream: if True, returns a generator for streaming into Streamlit

    TODO: update the call to explanation_prompt() in src/prompts.py
    with the fields relevant to {{PROJECT_NAME}}.
    """
    _require_key()
    prompt = explanation_prompt(record=record, shap_df=shap_df)
    model = _get_model()
    try:
        if stream:
            return model.generate_content(prompt, stream=True)
        return model.generate_content(prompt).text
    except Exception as e:
        logger.error("explain_prediction failed: %s", e)
        return "Explanation unavailable. Please review the details manually."


def chat_with_user(
    user_message: str,
    conversation_history: list[dict],
    record_context: dict,
    related_history: list[dict],
) -> str:
    """
    Feature 2: conversational Q&A for {{PERSONA}} reviewing a prediction.

    user_message: latest message from the user
    conversation_history: prior messages [{"role": ..., "parts": [...]}]
    record_context: dict of the current record under review
    related_history: list of related records for context (e.g. prior transactions)
    """
    _require_key()
    system = system_prompt_chat(record_context, related_history)
    model = _get_model(system_instruction=system)
    chat = model.start_chat(history=conversation_history)
    try:
        return chat.send_message(user_message).text
    except Exception as e:
        logger.error("chat_with_user failed: %s", e)
        return "Unable to answer. Please try rephrasing."


def detect_patterns(records: list[dict]) -> str:
    """
    Feature 3: identify narrative patterns across a batch of flagged records.

    records: list of summarised record dicts from the current session
    """
    _require_key()
    if not records:
        return "No flagged records to analyse."
    prompt = pattern_detection_prompt(records)
    try:
        return _get_model().generate_content(prompt).text
    except Exception as e:
        logger.error("detect_patterns failed: %s", e)
        return "Pattern analysis unavailable."


def generate_report(
    metrics: dict,
    flagged_records: list[dict],
    date_range: str = "today",
) -> str:
    """
    Feature 4: generate a stakeholder executive summary report.

    metrics: output dict from src.evaluate.compute_business_metrics()
    flagged_records: list of flagged record dicts for the session
    """
    _require_key()
    patterns = detect_patterns(flagged_records)
    prompt = report_prompt(metrics, patterns, date_range)
    try:
        return _get_model().generate_content(prompt).text
    except Exception as e:
        logger.error("generate_report failed: %s", e)
        return "Report generation unavailable."
