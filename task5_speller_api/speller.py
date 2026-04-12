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

from ._client import get_client, get_response

logger = logging.getLogger(__name__)

# _MODEL = "gpt-4o-mini"
# _MAX_TOKENS = 20
# _TEMPERATURE = 0.3
# _PREFIX_PATTERN = re.compile(r"^[A-Za-z]{1,3}$")

# NOTE(Mohamed, M1): these placeholder prompt strings are Ali's stub so the
# end-to-end pipeline runs today. Replace them with the final prompt from the
# M1 prompt-design package (system prompt + few-shot examples + user template).


class API:
    _N_PREDICTIONS = 3
    _FALLBACK_WORDS = ("the", "and", "of")
    _SYSTEM_PROMPT = """
    You are a next-word predictor.
    You will be given context and a prefix of a word, 
    and you will predict the next word in the sentence based on the context,
    or a common English word if the prefix is too vague.
    Your response should be the most likely next words in the sentence formed by the prefix.
    Input format: prediction_count = <prediction_count>, context = <context>, prefix = <prefix>, sentence = <sentence>
    Only return a json object with a "predictions" field containing a list of three words, like this:
    {"predictions": ["word1", "word2", "word3"]}
    """

    _USER_TEMPLATE = "prediction_count = {prediction_count}, context = {context}, prefix = {prefix}, sentence = {sentence}"

    def __init__(self, prediction_count: int = _N_PREDICTIONS):
        self.client = get_client()
        self._N_PREDICTIONS = prediction_count

    def _parse_predictions(self, raw: str | None) -> list[str]:
        if not raw:
            logger.warning("Empty response from model; returning fallback list")
            return list(self._FALLBACK_WORDS)

        try:
            payload = json.loads(raw)
            words = payload.get("predictions", [])
            if not isinstance(words, list):
                raise ValueError("'predictions' was not a list")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Could not parse model output %r: %s", raw, exc)
            return list(self._FALLBACK_WORDS)
        except RuntimeError as exc:
            logger.warning("Error during model response parsing: %s", exc)
            return list(self._FALLBACK_WORDS)

        cleaned = [str(w).strip().lower() for w in words if str(w).strip()]
        cleaned = cleaned[: self._N_PREDICTIONS]

        for fallback in self._FALLBACK_WORDS:
            if len(cleaned) >= self._N_PREDICTIONS:
                break
            if fallback not in cleaned:
                cleaned.append(fallback)

        return cleaned[: self._N_PREDICTIONS]

    def predict_words(
        self,
        context: str,
        prefix: str,
        sentence: str = "",
        # mental_state: Optional[str] = None,
    ) -> list[str]:
        """Predict three complete words from a 1-3 letter prefix.

        Args:
            context: Free-text scenario description, e.g. "writing an email to my
                professor". Empty string is allowed and falls back to a neutral
                default inside the prompt.
            prefix: 1-3 alphabetic characters. Case is normalised to lowercase
                before the prompt is built.
            sentence: the current sentence being composed, up to the prefix.
            This is optional and may be empty if the prefix is not part of an
            in-progress sentence.
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
        if not isinstance(context, str):
            raise ValueError(f"context must be a string; got {context!r}")
        user_message = self._USER_TEMPLATE.format(
            prediction_count=self._N_PREDICTIONS,
            context=context or "(no context provided)",
            prefix=(
                prefix.lower() if isinstance(prefix, str) else "(no prefix provided)"
            ),
            sentence=sentence or "(no sentence provided)",
        )
        try:
            response = get_response(
                self.client,
                system_prompt=self._SYSTEM_PROMPT,
                user_message=user_message,
            )
        except (
            ValueError,
            RuntimeError,
        ) as exc:  # noqa: BLE001 — any transport/API error -> degraded mode
            logger.warning(exc)
            return list(self._FALLBACK_WORDS)

        return self._parse_predictions(response)


# for single file testing
if __name__ == "__main__":
    api = API()
    for word in api.predict_words(
        context="technology",
        prefix="br",  # bright
        sentence="The future of AI is ",
    ):
        print(word)
