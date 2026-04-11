# DESIGN_NOTES.md — forward-looking design

**Not in Assignment 1 scope.** This file exists so the v0.1 we ship Sunday does not need a rewrite when the rest of the Brain↔ChatGPT project starts plugging in. It is documentation, not a commitment.

## 1. The reserved `mental_state` parameter

`predict_words` takes a third parameter today:

```python
def predict_words(prefix, context="", mental_state: str | None = None) -> list[str]:
```

It is **unused in v0.1**. It exists because of WBS 4.1 — *Merge ML Context with LLM Prompt* — which is the direct extension of this task. In 4.1:

- EEG bandpower features will classify the user's mental state into `Focused` / `Neutral` / `Fatigued`.
- That label will be injected into the LLM prompt so the model can adapt tone and complexity (shorter/simpler words when Fatigued, richer vocabulary when Focused).

Reserving the parameter now means the UI team's call sites do not need to change when 4.1 lands — they can start passing `mental_state="Focused"` as soon as the ML classifier is ready, and 4.1 is a pure backend-side change to `speller.py`.

**Important for latency (see `LATENCY.md` §4):** when 4.1 wires `mental_state` into the prompt, it must go in the **user** message, not the **system** message. Otherwise every mental-state change invalidates the OpenAI prompt cache and we lose the prefill savings.

## 2. Future HTTP / FastAPI endpoint sketch

v0.1 is in-process Python-only. When the architecture grows to support a non-Python UI (or multiple UIs), the natural next step is a thin FastAPI wrapper. Sketch only:

```python
# future: task5_speller_api/server.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from . import predict_words

app = FastAPI()

class PredictRequest(BaseModel):
    prefix: str = Field(..., min_length=1, max_length=3, pattern=r"^[A-Za-z]+$")
    context: str = ""
    mental_state: str | None = None

class PredictResponse(BaseModel):
    predictions: list[str]

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    return PredictResponse(predictions=predict_words(**req.model_dump()))
```

## 3. Future WebSocket variant

If the UI starts issuing many predictions per second (e.g. speculating on the next letter before the user commits), HTTP per-request overhead becomes visible. A WebSocket endpoint lets the UI hold a single persistent connection and stream `{prefix, context}` frames in, `{predictions}` frames out.

This pairs naturally with the `AsyncOpenAI` switch in `LATENCY.md` §7.

## 4. Future `ask_chat` sibling

Task 5 is only the speller. The broader Brain↔ChatGPT project also needs a full conversational reply (not just 3-word predictions). That will live as a sibling function:

```python
# future
def ask_chat(message: str, mental_state: str | None = None) -> str: ...
```

Same client, same key, same environment — so it will sit alongside `predict_words` in this package, not in a new package.

## 5. Prompt parts: static vs. dynamic

For prompt caching to keep working as the project grows, we need to be strict about what can change call-to-call. Current layout:

| Part                         | Static / Dynamic | Where it lives       |
|------------------------------|------------------|----------------------|
| System prompt (role framing) | **Static**       | `_SYSTEM_PROMPT`     |
| Few-shot examples (M1)       | **Static**       | `_SYSTEM_PROMPT`     |
| JSON format instructions     | **Static**       | `_SYSTEM_PROMPT`     |
| Prefix                       | Dynamic          | user message         |
| Context                      | Dynamic          | user message         |
| Mental-state label (WBS 4.1) | Dynamic          | user message (NOT system) |
| Session memory (M3/v0.2)     | Dynamic          | user message         |

Rule of thumb: **everything dynamic goes in the user message**. The system message is cache-friendly and must stay byte-stable.

## 6. Things we were tempted to build this weekend but did NOT

Per the out-of-scope list in the work breakdown:

- ~~HTTP server / FastAPI endpoint~~ → sketched in §2 above.
- ~~WebSocket streaming~~ → sketched in §3 above.
- ~~Mental-state context merging~~ → WBS 4.1, reserved parameter in §1.
- ~~Full conversational chat reply~~ → sibling `ask_chat` in §4.
- ~~User authentication, multi-tenant key handling~~ → not needed for single-user BCI demo.
- ~~Custom fine-tuned model~~ → wildly out of scope for a 24-hour sprint.
- ~~pytest harness beyond a JSON-driven test bench~~ → `tests/test_cases.json` is the format.

If any of these become tempting again, add the temptation to this file and move on.
