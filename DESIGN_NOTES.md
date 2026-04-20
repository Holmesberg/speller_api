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

## 6. Lyra Secretary integration — task-governed vocabulary bias

**Status:** design only. No code in v0.1. This section is the contract that `tests/test_cases.json` (v2) already asserts against.

### 6.1. Goal

The speller is noisy (BCI letter selection) and the target host is **Lyra Secretary** — a task-scheduling backend with a closed command vocabulary. Left to itself an LLM will return generic English for ambiguous prefixes (`t → the`, `sc → school`, `st → study`). We want those same prefixes to resolve against **Lyra's current task state** instead (`t → task`, `sc → schedule`, `st → start` when idle, `st → stop` when a stopwatch is running).

The mechanism is a two-layer bias: (1) a fast deterministic lookup against Lyra's closed vocabulary and runtime "hot set", and (2) LLM fallback with that same vocabulary injected into the prompt. Layer 1 is a hashmap; layer 2 is Mohamed's existing prompt with one extra dynamic block.

### 6.2. What Lyra exposes

Canonical lists — the speller mirrors these locally as `task5_speller_api/lyra_vocab.py` to avoid an HTTP call per prediction:

| Layer          | Source in Lyra                                 | Example words                          |
|----------------|-----------------------------------------------|----------------------------------------|
| Command verbs  | `backend/app/api/v1/endpoints/tasks.py`        | create, schedule, reschedule, delete, skip, swap, start, stop, pause, resume, clear, sync, query, undo |
| Entity nouns   | `backend/app/db/models.py`                     | task, stopwatch, session, category, schedule, reflection |
| State words    | `backend/app/db/models.py:24–32`               | planned, executing, paused, executed, skipped, abandoned |
| Categories     | `backend/app/db/seed.py:14–82`                 | fitness, academic, study, development, meeting, prayer, work, … |

Runtime context — polled every 5–10 min by the speller's `LyraContext` poller:

- `GET /v1/stopwatch/status` → `{active, task_title, paused, elapsed_minutes}` — gives us "is a session live, and what's its title/category?"
- `GET /v1/tasks/query?date=<today>&state=planned` → pending tasks with category tags — gives us "what categories are hot today?"

From those two endpoints the poller derives a `hot_verbs` list: if a stopwatch is running, hot = {stop, pause, resume}; if it's paused, hot = {resume, stop}; if planning, hot = {create, schedule, reschedule, delete, skip, swap}.

### 6.3. New module: `task5_speller_api/lyra_context.py`

```python
# sketch — not implemented in v0.1
from dataclasses import dataclass, field
from functools import lru_cache
import time

LYRA_COMMAND_VERBS = frozenset({"create", "schedule", "reschedule", "delete",
    "skip", "void", "swap", "start", "stop", "pause", "resume", "clear",
    "sync", "query", "undo"})
LYRA_ENTITY_NOUNS = frozenset({"task", "stopwatch", "session", "category",
    "schedule", "reflection", "notification"})
LYRA_STATE_WORDS = frozenset({"planned", "executing", "paused", "executed",
    "skipped", "abandoned"})
LYRA_CATEGORIES = frozenset({"fitness", "academic", "study", "development",
    "meeting", "prayer", "network", "personal", "health", "work"})

@dataclass(frozen=True)
class LyraContext:
    active_stopwatch: dict | None = None     # {task_title, category, paused}
    pending_categories: tuple[str, ...] = ()
    hot_verbs: tuple[str, ...] = ()
    fetched_at: float = field(default_factory=time.time)

    def is_fresh(self, ttl_seconds: float = 300.0) -> bool:
        return time.time() - self.fetched_at < ttl_seconds

def fetch_context(base_url: str, timeout: float = 2.0) -> LyraContext: ...
```

### 6.4. Two-layer bias inside `predict_words`

```
predict_words(prefix, context, sentence, lyra_context=None)
  │
  ├── Layer 1: fast lookup (no network)
  │     ├── Union the closed vocabularies with lyra_context.hot_verbs
  │     ├── Filter to words starting with `prefix` (case-insensitive)
  │     ├── Rank by: hot_verbs > command_verbs > entity_nouns > state_words > categories
  │     └── If ≥3 matches → return top 3. DONE.
  │
  └── Layer 2: LLM fallback (existing path)
        ├── Build user message — add a "Preferred vocabulary" block:
        │     "Prefer these words when they fit the prefix:
        │      hot=[resume,stop], verbs=[...], categories=[fitness,work]"
        ├── Inject into USER message only (not system — see §5 cache rule)
        └── Parse + top-up with Layer-1 matches to guarantee len==3
```

Layer 1 is deterministic and sub-millisecond. Layer 2 only fires when the prefix is prose-mode (e.g. composing a task title) or doesn't match enough Lyra words.

### 6.5. Public API delta

```python
class API:
    def __init__(self, prediction_count=3, lyra_context: LyraContext | None = None): ...
    def predict_words(self, context, prefix, sentence="",
                      lyra_context: LyraContext | None = None) -> list[str]: ...

def predict_words(prefix="", context="", sentence="",
                  mental_state=None,
                  lyra_context: LyraContext | None = None) -> list[str]: ...
```

`lyra_context=None` means prose-mode: skip Layer 1, pure LLM. This preserves backwards-compatibility for callers that are not Lyra-aware.

### 6.6. Integration surface on the Lyra side

Three touchpoints, least to most intrusive:

- **Option A (preferred for v0.2):** The **OpenClaw agent** polls speller + Lyra. Agent calls `predict_words(prefix, lyra_context=await fetch_context(...))`, displays the top-3 as a menu, user selects, agent calls `POST /v1/create` (or whichever verb). No Lyra backend changes.
- **Option B (later):** Add `GET /v1/context` to Lyra — one endpoint returning `{active_stopwatch, pending_categories, hot_verbs}` pre-computed. Saves the speller one HTTP round-trip.
- **Option C (far future):** Speller streams predictions over WebSocket and consumes a `/v1/context/stream` SSE feed. Pairs with `LATENCY.md` §7. Out of scope until v0.3+.

### 6.7. Test governance — how `tests/test_cases.json` enforces this

v2 schema (already written) carries three pieces per case: `lyra_state` (the synthetic context handed to the speller), `expected_top_word`, and `acceptable_top_3`. Harness flow:

```python
for case in cases:
    ctx = LyraContext(**case["lyra_state"]) if case["lyra_state"] else None
    top3 = predict_words(prefix=case["prefix"], context=case["context"],
                        sentence=case["sentence"], lyra_context=ctx)
    assert top3[0] == case["expected_top_word"] or top3[0] in case["acceptable_top_3"]
```

The "st → start vs st → stop" pair is the load-bearing test: same prefix, different `lyra_state`, different expected top word. If that pair passes, the state-dependent bias works.

Later — when the speller is wired to a live Lyra — a second harness can run the same cases against a *live* `fetch_context()` by spinning up synthetic Lyra states via `POST /v1/create` + `POST /v1/stopwatch/start`. That's how the bench moves from "synthetic JSON fixtures" to "end-to-end integration tests".

### 6.8. Open questions / deferred

- **Category vs verb tie-break.** `sc` fits both `schedule` (verb) and — if we added a `science` category — a category. Current rule: verbs win. Revisit when categories grow.
- **Stale context.** `LyraContext.is_fresh()` defaults to 5 min. If the user starts a stopwatch and immediately speaks, the speller's cached context may say "planning" instead of "executing". Accept this for v0.2; cadence can tighten to 60 s if misses are visible.
- **Prompt-cache impact.** Adding the vocabulary block to the user message is correct per §5, but it makes the user message longer — each call that fires Layer 2 still hits the cached system prefix, but the user-message tokens grow. Measure before/after in `LATENCY.md`.
- **Who owns `lyra_vocab.py`.** It mirrors Lyra. Decide ownership: either a generated file (script pulls from Lyra at build time) or a hand-maintained copy with a test that diffs it against Lyra's sources.

## 7. BR41N.IO 2026 Scenario B integration (hackathon, 5-day build)

**Status:** local-only plan. GitHub state stays as Scenario D (speller only). Nothing in this section gets pushed. Source: `LYRA_Unified_Pitch_v2.pdf` + `Scenarios.pdf` (Scenario B, recommended).

### 7.1 Goal

One EEG signal drives two reactions simultaneously:

- (a) **Lyra Secretary** reschedules, starts its stopwatch, or logs a cognitive session — per the Neuro-Logic → Action matrix.
- (b) **Speller** fills the reserved `mental_state` parameter into the LLM prompt, so ChatGPT's tone and word complexity adapt to the user's state.

Judge-facing line: *"One signal. One pipeline. The speller adapts what you say. The calendar adapts when you work."*

### 7.2 Classifier choice — band-power rule, not EEGNet

We do not train a learned classifier in the 5-day window. Instead, we compute α/θ/β band powers from BrainFlow FFT on the pitched channels (α from Oz, PO7, PO8; θ from Fz, Cz; β from C3, C4) and apply the thresholds from the Unified Pitch §2.1 Neuro-Logic Matrix.

Why:

- EEGNet takes raw EEG (time × channels), not precomputed band powers. It needs per-user fine-tuning with labelled cognitive-state data.
- Public EEG datasets named in Scenario B (BCI Competition IV-2a, PhysioNet MMI) are **motor-imagery** labelled, not cognitive-workload labelled. Transferring to focus/overload would need a workload-labelled dataset (STEW, MATB-II, DEAP-arousal) and a fine-tune cycle we do not have.
- Rule-based band-power mapping is what the pitch's Neuro-Logic Matrix literally describes. Interpretable, demo-friendly, zero training.
- Per-user adaptation collapses to a 2-minute baseline — record α/θ/β means + variances at rest, trigger states on z-score departures instead of absolute thresholds.

EEGNet stays on the post-hackathon roadmap if we can curate a workload-labelled dataset.

### 7.3 New modules (local-only)

Home: a new `bci/` sub-package inside `speller_api` **on a local-only branch** (never pushed). Keeping `bci/` under `speller_api` avoids a second repo for a 5-day sprint; if the work lives past the hackathon we can split it out.

```
speller_api/
└── bci/
    ├── pipeline.py     # BrainFlow session + filtering + quality gate
    ├── classifier.py   # Neuro-Logic rule-based state emitter
    └── bridge.py       # Dispatch state to speller mental_state + Lyra API
```

**`bci/pipeline.py`** — signal processing:

- BrainFlow session (Unicorn, 8 ch, 250 Hz).
- Notch 50 Hz (Egypt mains) + bandpass 1–40 Hz via BrainFlow one-liners.
- Sliding 2 s window, 250 ms stride.
- FFT → α (8–13 Hz), θ (4–8 Hz), β (13–30 Hz) per channel.
- Quality gate: reject windows whose variance is >4× baseline variance, or line-noise ratio >0.3. Rejected windows emit `unknown`, not a state.

**`bci/classifier.py`** — rule-based emitter:

- `NeuroLogicClassifier(baseline)` holds per-user α/θ/β means + σ from the 2-min calibration.
- `classify(features) -> CognitiveState` returns one of `focus | neutral | overload | fatigue | stress | exhaustion | unknown`.
- Thresholds expressed as z-score multiples of baseline σ per the Unified Pitch matrix (α↑ on Oz/PO7/PO8 → fatigue, θ↑ on Fz/Cz → overload, β↑ on C3/C4 → stress, α↓ stable → focus, α:θ < 1.0 → exhaustion).

**`bci/bridge.py`** — dispatch:

- Subscribe to classifier output stream.
- On every state emission that passes the quality gate: POST `/v1/cognitive/log`.
- Every 30 s if a Lyra stopwatch is active: POST `/v1/cognitive/check` with `{task_id, elapsed_minutes, cognitive_state}` → act on returned intervention.
- On state transition (e.g. neutral → overload): fire the one-shot Lyra action from the matrix (`/v1/tasks/reschedule`, `/v1/stopwatch/start`, etc.).
- On every state emission: write `mental_state` into a process-shared cache so the next `predict_words` call carries it.

### 7.4 Lyra Secretary additions (cognitive layer, local-only branch)

- Alembic migration: new `cognitive_session` table + task-table columns (`cognitive_state_at_start`, `eeg_confidence_at_start`, `overload_duration_minutes`, `focus_duration_minutes`, `overload_onset_minutes`). Schema per Unified Pitch §7.2.
- `POST /v1/cognitive/log` — write a row; update the current task's running counters.
- `POST /v1/cognitive/check` — implement the elapsed-aware intervention logic from Unified Pitch §3.3.

### 7.5 Speller additions (mental_state finally does something)

Ali's reserved parameter gets wired:

- `_USER_TEMPLATE` gains a `mental_state` field (dynamic, so it lives in the **user** message, not system — §5 cache rule).
- System-prompt gets one extra paragraph: "If `mental_state` is `overload` or `fatigue`, prefer shorter, simpler words and break sentences. If `focus`, richer vocabulary is fine. If `stress`, calmer tone."
- `predict_words` already accepts `mental_state=None`; the only code change is reading the bridge's shared cache when the caller doesn't pass one.

### 7.6 Live calibration (hackathon opening sequence, ~5 minutes)

- 2 min eyes-open relaxed baseline → per-user α/θ/β means + σ.
- 60 s mental-math under time pressure → confirm θ(Fz/Cz) rises above baseline + σ threshold.
- 60 s rest → confirm return to baseline.
- 60 s buffer: instantiate `NeuroLogicClassifier(baseline)`, start the bridge, both consumers go live.

### 7.7 Demo choreography — pre-scripted, deterministic

Rehearse the Scenarios.pdf §B 5-minute sequence as a fixed script:

1. Baseline shown, both systems "Ready".
2. Mental-math overload trigger → θ spikes → Lyra defers next 2 tasks 30 min + speller switches to shorter words.
3. Rest → recovery arc visible.
4. Focus task → α drops → speller enters deep-work prompt mode + Lyra auto-starts stopwatch.
5. Hand headset to judge; both systems react to a stranger's EEG.

Fallbacks baked into rehearsal:

- **Notion flake:** render Lyra's own `/today` frontend on the big screen. Removes the external dependency.
- **Unity ERP/SSVEP UI not ready:** degrade speller half to tone-adaptation only (prefix typed via keyboard; demo focus shifts to "cognitive state changed ChatGPT's tone" rather than "brain selected letters"). Still satisfies the Brain↔ChatGPT template.
- **Mid-demo signal loss:** quality gate suppresses false states; demo holds on last-known-good state until signal recovers.

### 7.8 Five-day phase plan

Today = **2026-04-20**. Hackathon opens **2026-04-25**. Team of 5.

| Day | Date       | Focus                              | Deliverable |
|-----|------------|------------------------------------|-------------|
| D1  | 2026-04-20 | Lyra cognitive layer               | Alembic migration; `cognitive_session` table + task columns; `POST /v1/cognitive/log` + `POST /v1/cognitive/check` endpoints; unit tests. |
| D2  | 2026-04-21 | Signal pipeline + classifier       | `bci/pipeline.py` (BrainFlow notch/bandpass/quality gate). `bci/classifier.py` (Neuro-Logic rules, z-score thresholds). Dry run on recorded or synthetic EEG. |
| D3  | 2026-04-22 | Bridge + speller wiring + first hw test | `bci/bridge.py`. `mental_state` wired into speller user-message template. First full path on live Unicorn: scalp → pipeline → classifier → bridge → Lyra API + speller prompt update. |
| D4  | 2026-04-23 | End-to-end demo rehearsal + calibration script | Live 5-min demo sequence runs end-to-end. Calibration script emits per-user baseline in 2 min. Notion sync tested under demo conditions; fallback renderer verified. |
| D5  | 2026-04-24 | Buffer + rehearse + sleep          | Fix anything that broke in D4. Full run ≥3× clean. Finalise screen layout, timing, judge handoff. |

**Suggested ownership split (5 people):**

- **Signal pipeline + classifier** — needs BrainFlow + DSP familiarity.
- **Lyra cognitive endpoints** — FastAPI + Alembic.
- **Speller `mental_state` wiring + LLM prompt tuning** — owns `speller_api`.
- **Unity UI (`omar4a/NeuroTechASU`) + demo screen composition** — owns the visible half of the demo.
- **Rehearsal lead + fallback/quality watcher** — runs the deterministic script, watches for signal/network degradation, triggers fallback renderers.

### 7.9 What we are explicitly NOT building

- EEGNet or any learned classifier.
- ICA stage in the pipeline. Too brittle online for a 5-day window.
- Scenario A's 4-week personal-data prophecy narrative.
- Full ERP + SSVEP paradigm if the Unity UI is not ready by D4 — we degrade to tone-adaptation.
- `GET /v1/context` consolidation on Lyra (§6.6 option B) — speller's task-vocabulary bias (§6) is orthogonal and stays on the `openai` branch, not the hackathon branch.
- Any PRs into the public `speller_api` or `Lyra Secretary` GitHub repos carrying the cognitive layer.

### 7.10 Open risks / post-hackathon

- **Notion sync under demo conditions** — untested. Test on D4; fallback renderer is the mitigation.
- **Unity ERP grid maturity** — `omar4a/NeuroTechASU` is under construction. Tone-adaptation fallback is the contingency.
- **Live classifier noise** — rule-based classifier may flutter at state boundaries. Add a 2-sample hysteresis (require 2 consecutive windows agreeing before emitting a transition).
- **Post-event cleanup** — if the demo wins, decide whether to clean the cognitive-layer branches and open-source as v0.2, or keep them private until a research paper (§10.5 of the pitch) is drafted.

## 8. Things we were tempted to build this weekend but did NOT

Per the out-of-scope list in the work breakdown:

- ~~HTTP server / FastAPI endpoint~~ → sketched in §2 above.
- ~~WebSocket streaming~~ → sketched in §3 above.
- ~~Mental-state context merging~~ → WBS 4.1, reserved parameter in §1.
- ~~Full conversational chat reply~~ → sibling `ask_chat` in §4.
- ~~User authentication, multi-tenant key handling~~ → not needed for single-user BCI demo.
- ~~Custom fine-tuned model~~ → wildly out of scope for a 24-hour sprint.
- ~~pytest harness beyond a JSON-driven test bench~~ → `tests/test_cases.json` is the format.

If any of these become tempting again, add the temptation to this file and move on.
