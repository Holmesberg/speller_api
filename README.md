# task5_speller_api

Backend Python API for the Brain↔ChatGPT BCI project. Takes a 1–3 letter prefix
from the Unity speller grid and returns three predicted complete English words
via the OpenAI API, ready to be rendered as SSVEP-selectable targets.

**Scope:** BR41N.IO Hackathon Prep · Assignment 1 · Task 5. v0.1 ships as a
callable Python function plus a CLI smoke-test. A FastAPI/WebSocket endpoint is
explicitly out of scope for this milestone — see `DESIGN_NOTES.md`.

Owners: **Ali Nasser** (repo, core script, integration/latency docs) · **Mohamed Nady** (prompt design, test bench, accuracy writeup, key-setup guide).

## Layout

```
speller_api/
├── task5_speller_api/      # Python package
│   ├── __init__.py         # re-exports predict_words
│   ├── __main__.py         # CLI entry (python -m task5_speller_api)
│   ├── _client.py          # OpenAI client + API-key loader
│   └── speller.py          # predict_words() — public contract
├── tests/
│   └── test_cases.json     # M2 test bench (Mohamed)
├── data/                   # M6 unigram frequency data (Mohamed)
├── .env.example            # copy to .env and fill in your key
├── .gitignore
├── requirements.txt
├── README.md               # this file
├── INTEGRATION.md          # UI team integration contract (Ali, A5)
├── LATENCY.md              # latency suggestions writeup (Ali, A6)
├── DESIGN_NOTES.md         # forward-looking v0.2+ notes (Ali, A8)
├── KEY_SETUP.md            # OpenAI key walkthrough (Mohamed, M7)
└── ACCURACY.md             # accuracy suggestions writeup (Mohamed, M3)
```

## Install

```bash
git clone https://github.com/Holmesberg/speller_api.git
cd speller_api
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then edit .env and paste your key
```

See **`KEY_SETUP.md`** for the end-to-end OpenAI account + key walkthrough.

### Provider setup (OpenAI or Google AI Studio)

- **OpenAI (default):** set `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`).
- **Google AI Studio (Gemini):** set `OPENAI_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`, put your Gemini key into `OPENAI_API_KEY`, and set `OPENAI_MODEL` to a compatible Gemini model (e.g. `gemini-3-flash-preview`).

Upstream docs (kept current by the providers):

- OpenAI API keys: https://platform.openai.com/api-keys
- OpenAI API quickstart: https://developers.openai.com/api/docs/quickstart
- Gemini API keys (Google AI Studio): https://aistudio.google.com/app/apikey
- Gemini OpenAI compatibility: https://ai.google.dev/gemini-api/docs/openai

## Run

```bash
python -m task5_speller_api he --context "writing an email to my professor"
# → ["hello", "hope", "help"]
```

Or in Python:

```python
from task5_speller_api import predict_words

predict_words("he", context="writing an email to my professor")
# → ['hello', 'hope', 'help']
```

## Docs

| Doc               | Audience           | What's in it                                                                       |
| ----------------- | ------------------ | ---------------------------------------------------------------------------------- |
| `INTEGRATION.md`  | Unity UI team      | Function signature, error model, latency envelope, sequence diagram, mocking guide |
| `KEY_SETUP.md`    | Anyone on the team | Step-by-step OpenAI account / key / spending-cap walkthrough                       |
| `LATENCY.md`      | Future-us (v0.2+)  | Ten concrete ways to reduce per-call latency                                       |
| `ACCURACY.md`     | Future-us (v0.2+)  | Ten concrete ways to improve prediction quality                                    |
| `DESIGN_NOTES.md` | Whole project team | Forward-looking design (WBS 4.1 hook, FastAPI sketch)                              |
