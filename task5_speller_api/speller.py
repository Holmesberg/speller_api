"""Core predictive speller logic.

Public contract (stable for the Unity UI team):

    predict_words(prefix: str, context: str = "",
                  mental_state: str | None = None) -> list[str]

Returns exactly three lowercase English words. See INTEGRATION.md for the full
integration guide (error model, expected latency, mocking for Unity tests).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from ._client import get_client

logger = logging.getLogger(__name__)

_MODEL = "gpt-4o-mini"
_MAX_TOKENS = 20
_TEMPERATURE = 0.3
_N_PREDICTIONS = 3
_PREFIX_PATTERN = re.compile(r"^[A-Za-z]{1,3}$")
_FALLBACK_WORDS = ("the", "and", "of")

# NOTE(Mohamed, M1): these placeholder prompt strings are Ali's stub so the
# end-to-end pipeline runs today. Replace them with the final prompt from the
# M1 prompt-design package (system prompt + few-shot examples + user template).
_SYSTEM_PROMPT = (
    "You are a BCI predictive speller assistant. Given a 1-3 letter prefix "
    "and an optional context, return exactly three common English words that "
    "start with the prefix, lowercased, as a JSON object of the shape "
    '{"predictions": ["w1", "w2", "w3"]}. Prefer high-frequency everyday '
    "words. Do not include rare, archaic, or proper-noun vocabulary. Do not "
    "add commentary. Return only the JSON object."
)

_USER_TEMPLATE = "prefix: {prefix}\ncontext: {context}"


def predict_words(
    prefix: str,
    context: str = "",
    mental_state: Optional[str] = None,
) -> list[str]:
    """Predict three complete words from a 1-3 letter prefix.

    Args:
        prefix: 1-3 alphabetic characters. Case is normalised to lowercase
            before the prompt is built.
        context: Free-text scenario description, e.g. "writing an email to my
            professor". Empty string is allowed and falls back to a neutral
            default inside the prompt.
        mental_state: Reserved for WBS 4.1 (Merge ML Context with LLM Prompt).
            Currently **ignored** — the parameter exists in the v0.1 signature
            so downstream callers do not need to change their code when
            mental-state merging lands. See DESIGN_NOTES.md.

    Returns:
        A list of exactly three lowercase English words. On OpenAI error or
        unparseable response, returns a documented fallback list (see
        `_FALLBACK_WORDS`). Callers should treat the fallback as degraded
        mode — the words are NOT guaranteed to start with the prefix in that
        case; this is deliberate, see INTEGRATION.md §"Error model".

    Raises:
        ValueError: `prefix` is not 1-3 alphabetic characters.
        RuntimeError: `OPENAI_API_KEY` is not configured. See KEY_SETUP.md.
    """
    if not isinstance(prefix, str) or not _PREFIX_PATTERN.match(prefix):
        raise ValueError(
            f"prefix must be 1-3 alphabetic characters; got {prefix!r}"
        )
    prefix_norm = prefix.lower()

    user_message = _USER_TEMPLATE.format(
        prefix=prefix_norm,
        context=context or "(no context provided)",
    )

    # Let RuntimeError from a missing key propagate so the UI learns at
    # startup, not via a silent fallback list. Only network / API errors
    # from the completion call itself are converted to degraded mode.
    client = get_client()

    try:
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
        )
    except Exception as exc:  # noqa: BLE001 — any transport/API error -> degraded mode
        logger.warning("OpenAI call failed (%s); returning fallback list", exc)
        return list(_FALLBACK_WORDS)

    raw = response.choices[0].message.content
    return _parse_predictions(raw)


def _parse_predictions(raw: str | None) -> list[str]:
    if not raw:
        logger.warning("Empty response from model; returning fallback list")
        return list(_FALLBACK_WORDS)

    try:
        payload = json.loads(raw)
        words = payload.get("predictions", [])
        if not isinstance(words, list):
            raise ValueError("'predictions' was not a list")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Could not parse model output %r: %s", raw, exc)
        return list(_FALLBACK_WORDS)

    cleaned = [str(w).strip().lower() for w in words if str(w).strip()]
    cleaned = cleaned[:_N_PREDICTIONS]

    for fallback in _FALLBACK_WORDS:
        if len(cleaned) >= _N_PREDICTIONS:
            break
        if fallback not in cleaned:
            cleaned.append(fallback)

    return cleaned[:_N_PREDICTIONS]
