"""
{{PROJECT_NAME}} Gemini prompt templates.

OPTIONAL MODULE — remove if not using Gemini integration.

All prompts live here so {{TEAM_DS}} can iterate on prompt quality
independently from the API integration code in gemini.py.

Rules for all prompts in this file:
  - Never expose ML jargon (SHAP values, probabilities, model names) to the end user
  - Write for {{PERSONA}} — someone who needs to act, not a data scientist
  - If a prompt cannot be answered from the available data, instruct Gemini to say so
  - Keep outputs concise: 2 to 4 sentences unless the prompt explicitly asks for more

Owner(s):
"""


def explanation_prompt(record: dict, shap_df) -> str:
    """
    Prompt for Feature 1: plain English prediction explanation.

    TODO: replace the placeholder fields with the actual fields for {{PROJECT_NAME}}.
    Include the most informative fields for {{PERSONA}} and the top SHAP features.
    Instruct Gemini not to mention SHAP, probabilities, or model scores directly.
    """
    shap_lines = "\n".join([
        f"  - {row['feature']} = {row['feature_value']} ({row['direction']})"
        for _, row in shap_df.head(3).iterrows()
    ])

    # TODO: update this prompt for {{PROJECT_NAME}}
    return f"""You are an assistant helping a {{PERSONA}} at {{COMPANY_NAME}}. A model has flagged the following record. Write 2 to 3 sentences explaining why in plain language the analyst can act on immediately. Do not mention model scores, probabilities, or SHAP values.

Record details:
  TODO: add relevant fields from the record dict

Signals from the model (use these to inform your explanation but do not quote directly):
{shap_lines}

Write your explanation now."""


def system_prompt_chat(record_context: dict, related_history: list[dict]) -> str:
    """
    System prompt for Feature 2: analyst chat.

    Injected once at the start of a session. Sets context for all
    subsequent questions about this record.

    TODO: update with the fields and history format for {{PROJECT_NAME}}.
    """
    history_lines = "\n".join([
        f"  {i+1}. {str(r)[:120]}"
        for i, r in enumerate(related_history)
    ])

    # TODO: update this prompt for {{PROJECT_NAME}}
    return f"""You are an assistant for a {{PERSONA}} at {{COMPANY_NAME}}. Answer questions about the record under review concisely and factually. If you cannot answer from available data, say so clearly.

Current record:
  TODO: add key fields from record_context

Related history ({len(related_history)} records):
{history_lines}"""


def pattern_detection_prompt(records: list[dict]) -> str:
    """
    Prompt for Feature 3: narrative pattern detection across a batch.

    TODO: update with the fields most useful for detecting patterns
    in {{PROJECT_NAME}}. Focus on what a senior {{PERSONA}} would notice
    when looking across many flagged records at once.
    """
    lines = "\n".join([
        f"  {i+1}. {str(r)[:150]}"
        for i, r in enumerate(records)
    ])

    # TODO: update this prompt for {{PROJECT_NAME}}
    return f"""You are a senior analyst at {{COMPANY_NAME}}. Below are {len(records)} records flagged as high risk. Identify 3 to 5 patterns that suggest systemic issues an individual review would miss. Be specific. Reference actual values. If no clear patterns exist, say so.

Flagged records:
{lines}

Write your analysis as a numbered list."""


def report_prompt(metrics: dict, patterns: str, date_range: str) -> str:
    """
    Prompt for Feature 4: stakeholder executive report.

    TODO: update metric keys and the report framing for {{PROJECT_NAME}}.
    The report should answer: what did the system accomplish for {{COMPANY_NAME}}?
    """
    # TODO: update metric references for {{PROJECT_NAME}}
    return f"""You are the analytics team at {{COMPANY_NAME}}. Write a concise executive summary for the period {date_range}. Use business language only. No ML jargon. Structure: (1) executive summary, (2) key numbers, (3) patterns identified, (4) recommended actions.

Key metrics:
  TODO: add metrics relevant to {{PROJECT_NAME}}

Patterns identified:
{patterns}

Write the report now. Tone: confident, factual, brief."""
