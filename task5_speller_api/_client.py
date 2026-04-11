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


def _load_base_url() -> str | None:
    """Optional base URL override.

    Supports:
    - `OPENAI_API_BASE_URL` (repo convention)
    - `OPENAI_BASE_URL` (OpenAI SDK convention)

    Leave unset to use the OpenAI SDK default (OpenAI).
    """
    base_url = (
        os.environ.get("OPENAI_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or ""
    ).strip()
    return base_url or None


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    base_url = _load_base_url()
    if base_url:
        return OpenAI(
            api_key=_load_api_key(),
            base_url=base_url,
            timeout=_CLIENT_TIMEOUT_SECONDS,
        )
    return OpenAI(api_key=_load_api_key(), timeout=_CLIENT_TIMEOUT_SECONDS)


def get_response(client, system_prompt: str, user_message: str) -> str:
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    if not response.choices or not response.choices[0].message.content:
        raise ValueError(f"Received empty response: {response.output_text}")
    return response.choices[0].message.content
