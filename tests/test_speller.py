"""Contract tests for task5_speller_api.

These are pure unit tests — the OpenAI client is mocked, no network calls are
made, no API key required. They verify the public contract that callers
(Unity UI, Proj2 WebSocket backend, future LyraOS bridge) depend on.

Run with: pytest tests/test_speller.py
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
    """Give every test a fresh OpenAI client mock and a valid fake key.

    The `get_client()` lru_cache in _client.py would otherwise reuse a real
    client built during an earlier import. We force-clear it and patch the
    OpenAI constructor so no real HTTP is attempted.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "gsk-test-key-for-unit-tests")
    monkeypatch.setenv("OPENAI_API_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    from task5_speller_api import _client, speller

    _client.get_client.cache_clear()
    # Reset the module-level singleton so each test gets a fresh API
    speller._DEFAULT_API = None

    fake_client = MagicMock()
    with patch("task5_speller_api._client.OpenAI", return_value=fake_client):
        yield fake_client


def test_imports():
    """Both public entry points resolve from the package root."""
    from task5_speller_api import predict_words, API
    assert callable(predict_words)
    assert callable(API)


def test_predict_words_returns_three_lowercase(_patch_env_and_client):
    """Happy path: 3 lowercase words, exactly."""
    from task5_speller_api import predict_words

    _patch_env_and_client.chat.completions.create.return_value = _mock_llm_response(
        json.dumps({"predictions": ["Hello", "Hope", "Help"]})
    )

    result = predict_words(prefix="he", context="writing an email")

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(w, str) for w in result)
    assert all(w == w.lower() for w in result), "all words must be lowercased"


def test_predict_words_truncates_long_response(_patch_env_and_client):
    """Model returned >3 words → truncate."""
    from task5_speller_api import predict_words

    _patch_env_and_client.chat.completions.create.return_value = _mock_llm_response(
        json.dumps({"predictions": ["a", "b", "c", "d", "e"]})
    )

    result = predict_words(prefix="x")
    assert result == ["a", "b", "c"]


def test_predict_words_pads_short_response(_patch_env_and_client):
    """Model returned <3 words → pad with fallbacks."""
    from task5_speller_api import predict_words

    _patch_env_and_client.chat.completions.create.return_value = _mock_llm_response(
        json.dumps({"predictions": ["only"]})
    )

    result = predict_words(prefix="on")
    assert len(result) == 3
    assert result[0] == "only"
    # remaining slots filled from _FALLBACK_WORDS = ("the", "and", "of")
    assert all(w in {"only", "the", "and", "of"} for w in result)


def test_predict_words_fallback_on_bad_json(_patch_env_and_client):
    """Unparseable JSON → documented fallback list."""
    from task5_speller_api import predict_words

    _patch_env_and_client.chat.completions.create.return_value = _mock_llm_response(
        "not valid json at all"
    )

    result = predict_words(prefix="he")
    assert result == ["the", "and", "of"]


def test_predict_words_fallback_on_api_error(_patch_env_and_client):
    """LLM raises → degraded-mode fallback, no exception escapes."""
    from task5_speller_api import predict_words

    _patch_env_and_client.chat.completions.create.side_effect = RuntimeError(
        "simulated transport failure"
    )

    result = predict_words(prefix="he")
    assert result == ["the", "and", "of"]


def test_predict_words_fallback_on_empty_response(_patch_env_and_client):
    """Empty choices list (documented edge case) → fallback."""
    from task5_speller_api import predict_words

    _patch_env_and_client.chat.completions.create.return_value = _mock_llm_response(None)

    result = predict_words(prefix="he")
    assert result == ["the", "and", "of"]


def test_predict_words_raises_on_non_string_context(_patch_env_and_client):
    """Programmer-error validation: non-string context → ValueError."""
    from task5_speller_api import predict_words

    with pytest.raises(ValueError, match="context must be a string"):
        predict_words(prefix="he", context=123)  # type: ignore[arg-type]


def test_predict_words_accepts_empty_prefix(_patch_env_and_client):
    """BCI noise tolerance: empty prefix must not raise (INTEGRATION.md §4)."""
    from task5_speller_api import predict_words

    _patch_env_and_client.chat.completions.create.return_value = _mock_llm_response(
        json.dumps({"predictions": ["the", "a", "what"]})
    )
    result = predict_words(prefix="", sentence="")
    assert len(result) == 3


def test_missing_key_raises_runtime_error(monkeypatch):
    """No OPENAI_API_KEY → RuntimeError on first call (fail fast, not fallback)."""
    from task5_speller_api import _client, speller

    monkeypatch.setenv("OPENAI_API_KEY", "")  # empty
    _client.get_client.cache_clear()
    speller._DEFAULT_API = None

    from task5_speller_api import predict_words

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        predict_words(prefix="he")


def test_placeholder_key_raises_runtime_error(monkeypatch):
    """Key still set to the .env.example placeholder → RuntimeError."""
    from task5_speller_api import _client, speller

    monkeypatch.setenv("OPENAI_API_KEY", "gsk-replace-me")
    _client.get_client.cache_clear()
    speller._DEFAULT_API = None

    from task5_speller_api import predict_words

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        predict_words(prefix="he")
