# ACCURACY.md — ways to improve prediction quality

**Owner:** Mohamed Nady (work package M3). **Status:** stub — to be filled Sunday per the Task 5 timeline.

Target outline (from the work breakdown):

1. Few-shot anchoring — inline examples in the system prompt (biggest lever for short-output tasks).
2. Temperature tuning — start at `0.3` for stable high-frequency words.
3. `top_p` tuning — `0.9` starting point.
4. Model upgrade path — cost/quality trade for `gpt-4o-mini` vs `gpt-4o` vs `gpt-4.1`.
5. Re-ranking by unigram frequency using `data/unigrams.txt` (M6).
6. Session memory — feed last few sentences back as context.
7. Personalised vocabulary — per-user cache of confirmed selections.
8. Topic detection — cheap classifier per session, label kept in the prompt.

Each suggestion should include a copy-paste-ready Python snippet where applicable (especially #5 — `rerank_by_frequency(words)`).
