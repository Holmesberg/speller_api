# LATENCY.md — ways to make `predict_words` faster

Ten concrete suggestions, each tagged with expected savings and implementation effort. v0.1 already applies the first five; the last five are v0.2+ work, written down so we do not forget them when we cross the latency floor.

Ordering: biggest bang-for-buck first.

## 1. Model selection — **biggest single lever**
- **Savings:** 500–2000 ms per call. `gpt-4o-mini` is typically 2–4× faster than `gpt-4o` for short-output tasks.
- **Effort:** 1 line (change `_MODEL` in `speller.py`).
- **Status (v0.1):** ✅ Default is `gpt-4o-mini`. Benchmark `gpt-4.1-mini` only if Mohamed's accuracy writeup flags a quality gap.
- **Trade-off:** marginal accuracy loss on ambiguous prefixes; mitigated by few-shot examples in the system prompt.

## 2. Tight `max_tokens`
- **Savings:** 100–400 ms. Prevents the model from over-generating before hitting a natural stop.
- **Effort:** 1 line.
- **Status (v0.1):** ✅ `max_tokens=20`. Three short words rarely exceed ~15 tokens plus JSON overhead.

## 3. Structured output (JSON mode)
- **Savings:** 50–150 ms of post-processing + eliminates a whole class of parse retries.
- **Effort:** 1 parameter (`response_format={"type": "json_object"}`).
- **Status (v0.1):** ✅ Enabled. No regex post-processing needed.

## 4. Cached system prompt (OpenAI prompt caching)
- **Savings:** 100–500 ms on cache hits. The system prompt is byte-identical across every call, so OpenAI's prompt cache makes the prefill nearly free after the first hit.
- **Effort:** 0 — it's automatic as long as the system prompt does not change call-to-call.
- **Status (v0.1):** ✅ Our system prompt is a module-level constant, not rebuilt per call. **Action item for v0.2:** when mental-state context lands (WBS 4.1), keep the dynamic part in the *user* message, not the system prompt, so the system prompt keeps caching.

## 5. Persistent client + warm connection
- **Savings:** 100–300 ms on the first call after startup, avoided on every call thereafter.
- **Effort:** 1 line (use `lru_cache` on the client factory).
- **Status (v0.1):** ✅ `_client.get_client` is `lru_cache(maxsize=1)`. One OpenAI instance, one TCP/TLS handshake.
- **Next step:** pre-warm the connection at UI startup by calling `get_client()` on the main thread.

## 6. LRU cache on `(prefix, context)` — **v0.2**
- **Savings:** ~950 ms on cache hits (network round-trip eliminated entirely).
- **Effort:** wrap `predict_words` with `functools.lru_cache(maxsize=256)`.
- **Trade-off:** cache must be cleared when the conversation topic shifts. Keying on `(prefix, context)` and clearing whenever the UI resets context handles this naturally.

## 7. `AsyncOpenAI` client — **v0.2**
- **Savings:** parallelism, not per-call latency. Enables the UI to issue predictions for multiple candidate prefixes in flight (e.g. speculate on the next letter before the user commits).
- **Effort:** swap `OpenAI` → `AsyncOpenAI`, make `predict_words` async.
- **When:** once the Unity side has an async/threaded caller. Not worth it for v0.1.

## 8. Streaming the first token — **v0.2**
- **Savings:** first-token latency (~300–500 ms) is what the user perceives as "response time", not total latency.
- **Effort:** medium — add `stream=True`, parse JSON incrementally, yield words as they land.
- **Catch:** JSON mode + streaming is messier; may need to switch to raw streaming + custom parser.

## 9. Local n-gram fallback — **v0.2+**
- **Savings:** sub-5 ms response when the network is slow or down.
- **Effort:** ~50 lines. Load Mohamed's `data/unigrams.txt` (M6), index by prefix, return the top-3 by frequency. Serve immediately, then swap in the LLM result when it arrives.
- **Benefit beyond latency:** the system stays usable even when OpenAI is down or the key is rate-limited.

## 10. Geographic routing — **v0.2+**
- **Savings:** 50–300 ms depending on demo location. Use the OpenAI region closest to the venue.
- **Effort:** zero (just pick the right base URL / project region).
- **Action:** confirm with the team where the hackathon demo runs — EU vs US — and set the region before the demo.

## Measurement plan for v0.2

Before pushing any of the v0.2 items above, establish a baseline:

1. Add a tiny timing harness that calls `predict_words` 20 times across varied prefixes and contexts.
2. Record p50, p95, and max.
3. Log values in this file so future-us can see what each change actually bought.
