"""Contract tests for task5_speller_api — updated to the multi-agent rewrite.

The OpenAI client is mocked, no network calls are made, no API key required.
These tests verify the public contract that callers (Unity UI, Proj2
WebSocket backend, future LyraOS bridge) depend on.

Post-rewrite contract (see `speller.py` module docstring):
  * Cold start (empty prefix + empty sentence) → static defaults, zero model
    calls.
  * Happy path → words that all start with the given prefix.
  * Capitalization is applied by Python, based on whether a new sentence is
    beginning. Capitalized at sentence start (empty sentence or sentence
    ending in .!?), lowercased otherwise.
  * Model failure (exception / bad JSON / empty response) → documented
    fallback list _FALLBACK_WORDS = ("the","and","of"). Then the prefix
    constraint is enforced: if none of the fallback words start with the
    prefix, a fixer agent is called; if that also fails, the method returns
    [prefix] as a final safety net.
  * Missing / placeholder OPENAI_API_KEY → RuntimeError at API() time.

Run: pytest tests/test_speller.py
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def _mock_llm_response(content: str | None) -> MagicMock:
    """Build a stand-in for an OpenAI ChatCompletion response object."""
    resp = MagicMock()
    resp.choices = [MagicMock()] if content is not None else []
    if content is not None:
        resp.choices[0].message.content = content
    return resp


@pytest.fixture(autouse=True)
def _patch_env_and_client(monkeypatch):
    """Fresh OpenAI client mock + valid fake key for every test."""
    monkeypatch.setenv("OPENAI_API_KEY", "gsk-test-key-for-unit-tests")
    monkeypatch.setenv("OPENAI_API_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    from task5_speller_api import _client, speller

    _client.get_client.cache_clear()
    speller._DEFAULT_API = None

    fake_client = MagicMock()
    with patch("task5_speller_api._client.OpenAI", return_value=fake_client):
        yield fake_client


# --- Basic surface ----------------------------------------------------------

def test_imports():
    from task5_speller_api import predict_words, API
    assert callable(predict_words)
    assert callable(API)


# --- Cold start (the interesting new behaviour) ----------------------------

def test_cold_start_uses_static_defaults_no_model_call(_patch_env_and_client):
    """Both inputs empty → deterministic defaults, zero LLM calls."""
    from task5_speller_api import predict_words

    result = predict_words(prefix="", context="", sentence="")

    assert len(result) == 3
    assert all(isinstance(w, str) for w in result)
    assert not _patch_env_and_client.chat.completions.create.called, (
        "cold start must not hit the model"
    )


# --- Happy path + prefix enforcement ---------------------------------------

def test_happy_path_all_words_match_prefix(_patch_env_and_client):
    """All returned words must start with the given prefix."""
    from task5_speller_api import predict_words

    # All three must share the prefix — otherwise _enforce_prefix drops the
    # non-matching ones and we'd need a separate fixer-agent mock.
    _patch_env_and_client.chat.completions.create.return_value = _mock_llm_response(
        json.dumps({"predictions": ["hello", "help", "here"]})
    )

    result = predict_words(prefix="he", context="email", sentence="")

    assert len(result) == 3
    assert all(w.lower().startswith("he") for w in result)


def test_model_returns_too_many_gets_truncated(_patch_env_and_client):
    from task5_speller_api import predict_words

    _patch_env_and_client.chat.completions.create.return_value = _mock_llm_response(
        json.dumps({"predictions": ["apple", "ant", "axe", "arrow", "aim"]})
    )

    result = predict_words(prefix="a", context="ctx", sentence="I see an ")

    assert len(result) == 3


# --- Capitalization contract -----------------------------------------------

def test_sentence_start_capitalizes(_patch_env_and_client):
    """Empty sentence = start of new sentence → capitalize."""
    from task5_speller_api import predict_words

    _patch_env_and_client.chat.completions.create.return_value = _mock_llm_response(
        json.dumps({"predictions": ["hello", "hope", "help"]})
    )

    result = predict_words(prefix="he", context="ctx", sentence="")

    assert all(w[0].isupper() for w in result), f"expected capitalized, got {result}"


def test_mid_sentence_lowercases(_patch_env_and_client):
    """Sentence present and not terminated → mid-sentence → lowercase."""
    from task5_speller_api import predict_words

    _patch_env_and_client.chat.completions.create.return_value = _mock_llm_response(
        json.dumps({"predictions": ["Hello", "Hope", "Help"]})
    )

    result = predict_words(prefix="he", context="ctx", sentence="I said ")

    assert all(w == w.lower() for w in result), f"expected lowercase, got {result}"


def test_after_terminator_capitalizes(_patch_env_and_client):
    """Sentence ending in . ! ? → next word starts a new sentence → capitalize."""
    from task5_speller_api import predict_words

    _patch_env_and_client.chat.completions.create.return_value = _mock_llm_response(
        json.dumps({"predictions": ["hello", "hope", "help"]})
    )

    result = predict_words(prefix="he", context="ctx", sentence="I said hi. ")

    assert all(w[0].isupper() for w in result), f"expected capitalized, got {result}"


# --- Fallback paths --------------------------------------------------------

def test_bad_json_no_prefix_returns_fallback_words(_patch_env_and_client):
    """Unparseable JSON + no prefix to enforce → documented fallback list."""
    from task5_speller_api import predict_words

    _patch_env_and_client.chat.completions.create.return_value = _mock_llm_response(
        "not valid json at all"
    )

    result = predict_words(prefix="", context="ctx", sentence="I went to ")

    assert len(result) == 3
    # fallback list is ["the","and","of"] lowercased mid-sentence
    assert [w.lower() for w in result] == ["the", "and", "of"]


def test_api_error_with_unmatched_prefix_falls_back_to_prefix(_patch_env_and_client):
    """LLM raises + prefix doesn't match fallback words + fixer fails too
    → final safety net returns [prefix] so the user at least sees what they
    typed."""
    from task5_speller_api import predict_words

    _patch_env_and_client.chat.completions.create.side_effect = RuntimeError(
        "simulated transport failure"
    )

    result = predict_words(prefix="xy", context="ctx", sentence="I see ")

    assert len(result) >= 1
    # The prefix itself must appear (lowercase mid-sentence).
    assert any(w.lower() == "xy" for w in result), (
        f"final fallback should preserve the prefix; got {result}"
    )


def test_empty_response_matching_prefix_recovers(_patch_env_and_client):
    """Empty response, but fallback word starts with prefix → recovers valid words."""
    from task5_speller_api import predict_words

    _patch_env_and_client.chat.completions.create.return_value = _mock_llm_response(None)

    # Prefix "th" matches fallback "the" → enforce_prefix keeps "the".
    result = predict_words(prefix="th", context="ctx", sentence="I went to ")

    assert len(result) >= 1
    # Must all start with "th"
    assert all(w.lower().startswith("th") for w in result)


# --- Input validation ------------------------------------------------------

def test_non_string_context_raises(_patch_env_and_client):
    from task5_speller_api import predict_words

    with pytest.raises(ValueError, match="context must be a string"):
        predict_words(prefix="he", context=123)  # type: ignore[arg-type]


def test_non_string_prefix_is_coerced_not_raised(_patch_env_and_client):
    """Relaxed validation: non-string prefix is coerced to '' (BCI noise tolerance)."""
    from task5_speller_api import predict_words

    result = predict_words(prefix=None, context="", sentence="")  # type: ignore[arg-type]
    # None prefix + empty sentence ≡ cold start (both treated as empty)
    assert len(result) == 3


# --- Key loading -----------------------------------------------------------

def test_missing_key_raises_runtime_error(monkeypatch):
    from task5_speller_api import _client, speller

    monkeypatch.setenv("OPENAI_API_KEY", "")
    _client.get_client.cache_clear()
    speller._DEFAULT_API = None

    from task5_speller_api import predict_words

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        predict_words(prefix="he")


def test_placeholder_key_raises_runtime_error(monkeypatch):
    from task5_speller_api import _client, speller

    monkeypatch.setenv("OPENAI_API_KEY", "gsk-replace-me")
    _client.get_client.cache_clear()
    speller._DEFAULT_API = None

    from task5_speller_api import predict_words

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        predict_words(prefix="he")
