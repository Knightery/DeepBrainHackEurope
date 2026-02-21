# Quant Pitch Evaluator

High-level overview of the project. For implementation contracts and detailed rules, see [MVP_SPEC.md](MVP_SPEC.md).

## Mission

Quant Pitch Evaluator helps strong independent quants submit stock ideas and get evaluated with a transparent, structured process that can lead to capital allocation.

## What the product does

- Collects a pitch through an interactive chat workflow.
- Standardizes core inputs (thesis, horizon, stocks, methodology, sources).
- Runs parallel evaluators to assess data quality, methodology risk, and performance metrics.
- Produces a final score, recommendation, and allocation amount.
- Supports final human reviewer approval or rejection.

## User experience (MVP)

1. User starts chat onboarding.
2. Clarifier agent guides completion of a checklist.
3. User provides:
   - thesis,
   - time horizon (`days`, `weeks`, `months`, `years`),
   - stock ticker(s) (just symbols like `AAPL, MSFT`),
   - methodology summary,
   - source URL(s) for submitted data.
4. User can upload supporting files.
5. Evaluation runs when all mandatory checklist items are complete.
6. Evaluation outcome:
   - fabricated/cheating signal -> `Goodbye.`
   - missing/unclear validation aspects -> clarification loop and `/validate`
   - clean validation -> ready for final review with `Congrats!`
7. User receives score + report + allocation recommendation once ready for final review.

## Why this structure

- Keeps onboarding simple for users.
- Preserves minimum standardization for reliable quant evaluation.
- Enforces data provenance (source URLs required).
- Makes missing fields explicit and actionable.

## Current app status

- Interactive Chainlit app is live.
- Local scoring and reporting are implemented.
- Checklist-style readiness and mandatory gates are implemented.
- Some advanced fetcher/auditor integrations remain MVP+ work.

## Key docs

- Product setup and local run: [README_CHAINLIT.md](README_CHAINLIT.md)
- Detailed product and engineering spec: [MVP_SPEC.md](MVP_SPEC.md)
- Chainlit user-facing instructions: [chainlit.md](chainlit.md)

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
chainlit run app.py
```

Model configuration is environment-driven:

- App + evaluators read `GEMINI_MODEL` from `.env`.
- CUA fetcher reads `ANTHROPIC_MODEL` from `.env`.

## Optional: CUA data fetcher (Docker)

The `cua/` folder contains an optional Computer Use Agent runner based on Anthropic's
official `computer-use-demo` image.

Use Docker Compose from inside `cua/`:

```powershell
cd cua
docker compose build
docker compose run --rm --remove-orphans data-fetcher "https://www.netflix.com/tudum/top10" "Fetch global top 10 lists"
```

Watch the live desktop at `http://localhost:6080`.

Model selection:

- Default is `claude-sonnet-4-5-20250929` (recommended for stronger web navigation reliability).
- For cheaper testing, set `ANTHROPIC_MODEL=claude-haiku-4-5-20251001`.
- Tool version is auto-paired in `cua/data_fetcher.py`, or can be overridden with `CUA_TOOL_VERSION`.

Important: `docker run build` is not a valid build command. Use `docker compose build` (or `docker build`).

## Principles

- Keep user inputs minimal but meaningful.
- Keep scoring deterministic in MVP.
- Prefer clarity and auditability over complexity.
