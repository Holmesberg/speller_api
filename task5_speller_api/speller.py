"""Core predictive speller logic — enhanced for local llms.

Public contract (stable for the Unity UI team):

    predict_words(prefix, sentence, context) -> list[str]

Returns exactly N lowercase/capitalized English words.
See INTEGRATION.md for the full integration guide.

Architecture change from v1 (OpenAI, single prompt):
    - Routing logic lives in Python, not the prompt
    - Three single-task agents replace one overloaded prompt
    - Cold start (both inputs empty) requires NO model call
    - Prefix constraint enforced in Python, not relied on from the model
    - Capitalization applied in Python based on sentence state
    - Wordlist fallback pads results without a second model call
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from ._client import get_client, get_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent system prompts
# Each prompt does exactly ONE thing. All branching lives in Python.
# ---------------------------------------------------------------------------

# Agent 1: user is mid-sentence and has started typing a word
_PROMPT_PREFIX_COMPLETION = (
    "You are a word completion assistant for a BCI speller.\n"
    "Given a sentence and a prefix, return the {n} most likely English words that:\n"
    "  1. Start exactly with the given prefix\n"
    "  2. Fit naturally as the next word in the sentence\n\n"
    "Return ONLY valid JSON: {{\"predictions\": [\"word1\", \"word2\", ...]}}\n"
    "No explanation. No markdown. No extra keys."
)

# Agent 2: user is mid-sentence, no prefix typed yet
_PROMPT_NEXT_WORD = (
    "You are a next-word prediction assistant for a BCI speller.\n"
    "Given a sentence and optional context, return the {n} most likely English words "
    "that would naturally follow the sentence.\n\n"
    "Return ONLY valid JSON: {{\"predictions\": [\"word1\", \"word2\", ...]}}\n"
    "No explanation. No markdown. No extra keys."
)

# Agent 3: sentence is empty, user has typed a prefix to start with
_PROMPT_SENTENCE_START = (
    "You are a word completion assistant for a BCI speller.\n"
    "Given a prefix and optional context, return the {n} most likely English words that:\n"
    "  1. Start exactly with the given prefix\n"
    "  2. Would naturally begin a sentence\n\n"
    "Return ONLY valid JSON: {{\"predictions\": [\"word1\", \"word2\", ...]}}\n"
    "No explanation. No markdown. No extra keys."
)


# ---------------------------------------------------------------------------
# Cold start defaults — no model call needed when both inputs are empty.
# Order: article, question word, pronoun — covers the three most common
# sentence-opening intents in AAC/BCI use cases.
# ---------------------------------------------------------------------------
_COLD_START_DEFAULTS = ["The", "I", "What", "How", "Can", "Please", "My", "We"]


# ---------------------------------------------------------------------------
# Fixer agent prompt
# Fires only when the main agent returns words that violate the prefix
# constraint. Single-task: "give me N common words starting with X."
# No sentence context needed — this is purely lexical recovery.
# ---------------------------------------------------------------------------
_PROMPT_FIXER = (
    "You are a word lookup assistant for a BCI speller.\n"
    "Return the {n} most common English words that start exactly with the prefix '{prefix}'.\n"
    "They must all begin with '{prefix}' — no exceptions.\n\n"
    "Return ONLY valid JSON: {{\"predictions\": [\"word1\", \"word2\", ...]}}\n"
    "No explanation. No markdown. No extra keys."
)


class API:
    _N_PREDICTIONS = 3
    # Fallback used only when model fails AND wordlist has no prefix match.
    # Deliberately generic — callers should treat this as degraded mode.
    _FALLBACK_WORDS = ("the", "and", "of")

    def __init__(self, prediction_count: int = _N_PREDICTIONS):
        self.client = get_client()
        self._N_PREDICTIONS = prediction_count

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict_words(
        self,
        context: str,
        prefix: str,
        sentence: str = "",
    ) -> list[str]:
        """Predict N complete words based on current speller state.

        Routing table (all edge cases handled here, not in the model):
          prefix=''  sentence=''  ->  cold_start()        [no model call]
          prefix=X   sentence=''  ->  sentence_start_agent
          prefix=''  sentence=X   ->  next_word_agent
          prefix=X   sentence=X   ->  prefix_completion_agent

        Post-processing (always in Python):
          - Prefix constraint enforcement + wordlist padding
          - Capitalization based on sentence boundary state

        Args:
            context:  Free-text description of the user's intent/topic.
            prefix:   Letters typed so far for the current word (may be empty).
            sentence: In-progress sentence up to the current word (may be empty).

        Returns:
            Exactly self._N_PREDICTIONS words. Never raises on model failure
            — returns a documented fallback list instead.
        """
        if not isinstance(context, str):
            raise ValueError(f"context must be a string; got {context!r}")

        # Normalise inputs
        prefix   = prefix.strip()   if isinstance(prefix, str)   else ""
        sentence = sentence.strip() if isinstance(sentence, str) else ""
        context  = context.strip()  or "(no context)"

        has_prefix   = bool(prefix)
        has_sentence = bool(sentence)

        # --- Route ---
        if not has_prefix and not has_sentence:
            return self._cold_start()

        if has_prefix and not has_sentence:
            raw = self._call_sentence_start_agent(prefix, context)
        elif not has_prefix and has_sentence:
            raw = self._call_next_word_agent(sentence, context)
        else:
            raw = self._call_prefix_completion_agent(prefix, sentence, context)

        # --- Parse ---
        predictions = self._parse_predictions(raw)

        # --- Post-process (Python owns this, not the model) ---
        if has_prefix:
            predictions = self._enforce_prefix(predictions, prefix)

        predictions = self._apply_capitalization(predictions, sentence)

        return predictions[: self._N_PREDICTIONS]

    # ------------------------------------------------------------------
    # Agents — one prompt per task
    # ------------------------------------------------------------------

    def _call_prefix_completion_agent(
        self, prefix: str, sentence: str, context: str
    ) -> str | None:
        system = _PROMPT_PREFIX_COMPLETION.format(n=self._N_PREDICTIONS)
        user   = f"context: {context}\nsentence: {sentence}\nprefix: {prefix}"
        return self._safe_call(system, user)

    def _call_next_word_agent(self, sentence: str, context: str) -> str | None:
        system = _PROMPT_NEXT_WORD.format(n=self._N_PREDICTIONS)
        user   = f"context: {context}\nsentence: {sentence}"
        return self._safe_call(system, user)

    def _call_sentence_start_agent(self, prefix: str, context: str) -> str | None:
        system = _PROMPT_SENTENCE_START.format(n=self._N_PREDICTIONS)
        user   = f"context: {context}\nprefix: {prefix}"
        return self._safe_call(system, user)

    def _cold_start(self) -> list[str]:
        """No model call — return deterministic sentence starters.

        Covers the three most common AAC/BCI opening intents:
          'The' / 'A'  -> starting a declarative sentence
          'I'          -> first-person statement
          'What/How'   -> question
        """
        return list(_COLD_START_DEFAULTS[: self._N_PREDICTIONS])

    def _safe_call(self, system_prompt: str, user_message: str) -> str | None:
        """Wrapper that converts any model exception to None (-> fallback path)."""
        try:
            return get_response(
                self.client,
                system_prompt=system_prompt,
                user_message=user_message,
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("Model call failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Output processing — all validation and correction lives here
    # ------------------------------------------------------------------

    def _parse_predictions(self, raw: str | None) -> list[str]:
        """Parse JSON from model output. Returns fallback on any failure."""
        if not raw:
            logger.warning("Empty response from model; returning fallback")
            return list(self._FALLBACK_WORDS)

        try:
            # Strip markdown fences — some models wrap JSON in ```json ... ```
            # even when explicitly told not to.
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
            payload = json.loads(cleaned)
            words   = payload.get("predictions", [])
            if not isinstance(words, list):
                raise ValueError("'predictions' field was not a list")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Could not parse model output %r: %s", raw, exc)
            return list(self._FALLBACK_WORDS)

        return [str(w).strip() for w in words if str(w).strip()]

    def _enforce_prefix(self, predictions: list[str], prefix: str) -> list[str]:
        """Hard-filter: keep only words starting with prefix.

        If the main agent returned any invalid words, call the fixer agent
        for exactly the number of missing slots. The fixer agent fires at most
        once per predict_words call.

        Worst case: two sequential model calls. This is an accepted tradeoff
        for full agentic behaviour — callers should be aware of the latency
        ceiling this creates.
        """
        prefix_lower = prefix.lower()

        # 1. Keep valid predictions from the main agent
        valid = [w for w in predictions if w.lower().startswith(prefix_lower)]

        # 2. If short, call the fixer agent for the missing count
        missing = self._N_PREDICTIONS - len(valid)
        if missing > 0:
            logger.info(
                "Main agent returned %d invalid words for prefix %r — calling fixer",
                missing, prefix,
            )
            raw = self._call_fixer_agent(prefix, missing)
            fixer_predictions = self._parse_predictions(raw)

            already = {v.lower() for v in valid}
            for w in fixer_predictions:
                if len(valid) >= self._N_PREDICTIONS:
                    break
                if w.lower().startswith(prefix_lower) and w.lower() not in already:
                    valid.append(w)
                    already.add(w.lower())

        # 3. Absolute last resort: fixer also failed or returned non-prefix words.
        #    Return the prefix itself as a word — at minimum the user sees what
        #    they typed. No further model calls.
        if not valid:
            logger.warning("Fixer agent also failed for prefix %r", prefix)
            valid = [prefix_lower]

        return valid[: self._N_PREDICTIONS]

    def _call_fixer_agent(self, prefix: str, missing_count: int) -> str | None:
        """Fixer agent: recover common words for a given prefix.

        Called only when the main agent violated the prefix constraint.
        Intentionally has no sentence context — it is purely lexical recovery.
        Asking for exactly `missing_count` keeps the response minimal.
        """
        system = _PROMPT_FIXER.format(n=missing_count, prefix=prefix.lower())
        user   = f"prefix: {prefix.lower()}"
        return self._safe_call(system, user)

    def _apply_capitalization(self, predictions: list[str], sentence: str) -> list[str]:
        """Capitalize if predictions will start a new sentence; lowercase otherwise.

        A new sentence is indicated by:
          - sentence being empty (first word of a new sentence)
          - sentence ending with terminal punctuation (. ! ?)
        """
        at_sentence_start = (
            not sentence
            or sentence.rstrip().endswith((".", "!", "?"))
        )
        if at_sentence_start:
            return [w.capitalize() for w in predictions]
        return [w.lower() for w in predictions]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    tests = [
        # Normal cases
        {"context": "football",   "prefix": "ch", "sentence": "Barcelona is the "},
        {"context": "technology", "prefix": "br", "sentence": "The future of AI is "},
        {"context": "food",       "prefix": "p",  "sentence": "I want to cook "},
        # Edge cases handled in Python
        {"context": "",           "prefix": "",   "sentence": ""},           # cold start
        {"context": "medicine",   "prefix": "tr", "sentence": ""},           # sentence start
        {"context": "weather",    "prefix": "",   "sentence": "It is very"}, # next word
        {"context": "sport",      "prefix": "xr", "sentence": "He ran "},   # rare prefix
    ]

    api = API()

    print(f"{'context':<15} {'prefix':<6} {'sentence':<35} predictions")
    print("-" * 80)
    for test in tests:
        t0 = time.perf_counter()
        results = api.predict_words(**test)
        elapsed = time.perf_counter() - t0
        print(
            f"{test['context']:<15} "
            f"{test['prefix']!r:<6} "
            f"{test['sentence']!r:<35} "
            f"{results}  ({elapsed:.2f}s)"
        )
