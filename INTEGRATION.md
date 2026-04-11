# INTEGRATION.md — UI team integration guide

How to call `task5_speller_api` from the Unity speller UI (or any other Python-capable host). v0.1 is an **in-process Python function**, not an HTTP service — the UI process is expected to import and call it directly. An HTTP/WebSocket wrapper is out of scope for this milestone; see `DESIGN_NOTES.md` for the v0.2+ sketch.

## 1. Install

```bash
git clone https://github.com/Holmesberg/speller_api.git
cd speller_api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit .env and paste your OpenAI key
```

Full OpenAI account / key / spending-cap walkthrough: see `KEY_SETUP.md`.

## 2. Import path

```python
from task5_speller_api import predict_words
```

Importing the package does **not** read the API key or make network calls — the client is built lazily on the first call so Unity-side unit tests can monkeypatch `predict_words` without setting up a key.

## 3. Function signature — stable contract

```python
def predict_words(
    prefix: str,
    context: str = "",
    mental_state: str | None = None,
) -> list[str]: ...
```

| Parameter       | Type       | v0.1 behaviour                                                                                  |
|-----------------|------------|--------------------------------------------------------------------------------------------------|
| `prefix`        | `str`      | 1–3 alphabetic characters. Lowercased internally. Anything else → `ValueError`.                 |
| `context`       | `str`      | Free-text scenario description. Empty string allowed.                                            |
| `mental_state`  | `str\|None`| **Reserved, currently ignored.** Part of the v0.1 signature so callers do not need to change their code when WBS 4.1 (Merge ML Context with LLM Prompt) lands. |

**Return shape:** `list[str]` of length **exactly 3**, all lowercase. This is a hard invariant — the function truncates over-long model outputs and pads short ones with a documented fallback list, so the UI can always index `[0]`, `[1]`, `[2]` without defensive checks.

## 4. Error model

| Situation                              | Behaviour                                                                                  |
|----------------------------------------|--------------------------------------------------------------------------------------------|
| `prefix` not 1–3 alphabetic chars      | Raises `ValueError` — the UI should never send this; treat as a programmer error.          |
| `OPENAI_API_KEY` not set or still `sk-replace-me` | Raises `RuntimeError` with setup instructions — the UI should surface this at startup, not at first keypress. |
| OpenAI transport / API / timeout error | Returns the documented fallback list (`["the", "and", "of"]`). **The fallback words are NOT guaranteed to start with the prefix** — this is a deliberate degraded-mode signal. If the UI wants to detect this explicitly, compare the first word's prefix to the input. |
| Model returned unparseable JSON        | Same as above — fallback list returned, raw response logged at `WARNING` level.            |
| Model returned fewer than 3 words      | Padded with fallback words so length is always 3.                                          |
| Model returned more than 3 words       | Truncated to 3.                                                                            |

No exceptions are raised for network/parse errors — the UI never needs a `try/except` around the function call for runtime failures, only for input-validation (`ValueError`) at development time.

## 5. Latency envelope

| Percentile | Expected (gpt-4o-mini, warm client) |
|-----------|--------------------------------------|
| p50       | ~600–900 ms                          |
| p95       | ~1.5–2.0 s                           |
| hard cap  | 5 s (client timeout → fallback list) |

Measured empirically during QA — see `LATENCY.md` for a list of ways to push these down further. If the UI cannot tolerate a 5 s worst case, wrap the call in an `asyncio.to_thread` and race it against a shorter deadline on your side.

## 6. Sequence (Unity → Python → OpenAI → Unity)

```
Unity speller UI           task5_speller_api             OpenAI API
     |                            |                            |
     |  predict_words("he", ctx)  |                            |
     |--------------------------->|                            |
     |                            |  chat.completions.create   |
     |                            |--------------------------->|
     |                            |                            |
     |                            |  JSON: {"predictions":...} |
     |                            |<---------------------------|
     |  ["hello","hope","help"]   |                            |
     |<---------------------------|                            |
     |                                                          |
  render 3 words                                                |
  as SSVEP targets                                              |
```

## 7. End-to-end example

```python
from task5_speller_api import predict_words

# Call once per letter the user commits on the grid.
words = predict_words("he", context="writing an email to my professor")
assert len(words) == 3
top, second, third = words
# Render each as an SSVEP-tagged button in Unity.
```

## 8. What the UI team should NOT assume

- **No streaming.** The function is blocking. The UI must call it on a worker thread if it needs to keep the render loop responsive.
- **No auth / multi-tenant.** There is one process, one key, one user.
- **No batching.** One call = one prefix. Do not pack multiple prefixes into a single call.
- **No persistent session state.** Each call is independent. If you want session memory (previous sentences, personal vocabulary), pass it through `context`.
- **No rate-limit back-off.** If you hammer the function faster than OpenAI allows, you will get the fallback list back and see warnings in the log.

## 9. Mocking in Unity-side tests (do not burn API credit)

For CI or local test runs, stub the function at import time so no network call is made:

```python
# conftest.py or test bootstrap
import task5_speller_api

def _fake_predict(prefix, context="", mental_state=None):
    return ["hello", "hope", "help"]

task5_speller_api.predict_words = _fake_predict
```

Alternatively, point `OPENAI_API_KEY` at a dummy value and patch `task5_speller_api._client.get_client` to return a mock — but the function-level stub above is simpler and sufficient for UI tests that just want deterministic words.
