# KEY_SETUP.md — OpenAI API key walkthrough

**Owner:** Mohamed Nady (work package M7). **Status:** stub — to be filled in Sat evening per the Task 5 timeline.

This file will walk any teammate through getting a working OpenAI key and running the CLI in under 5 minutes. Target outline:

1. Create an OpenAI account at https://platform.openai.com
2. Create a project scoped to the hackathon.
3. Generate an API key (Settings → API keys → Create new secret key).
4. **Set a monthly spending cap** — $10 is the team default so a runaway loop cannot drain credit. (Settings → Billing → Usage limits.)
5. Copy `.env.example` to `.env` in the repo root.
6. Paste the key after `OPENAI_API_KEY=` in `.env`.
7. Run `python -m task5_speller_api he --context "testing"` to verify.

TODO(Mohamed): expand each step with screenshots where time allows. Keep `.env.example` and this file in sync with Ali.
