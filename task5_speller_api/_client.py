"""OpenAI client construction and API-key loading.

The client is built lazily and cached, so importing this module does not
require a key (tests can stub it) but every `predict_words` call after the
first reuses a single warm TCP/TLS connection. See LATENCY.md §"Persistent
client + connection pool".
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_CLIENT_TIMEOUT_SECONDS = 5.0


def _load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key or key.startswith("sk-replace"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and paste "
            "your OpenAI key into it. See KEY_SETUP.md for a step-by-step "
            "walkthrough including the $10 monthly spending cap."
        )
    return key


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI(api_key=_load_api_key(), timeout=_CLIENT_TIMEOUT_SECONDS)
