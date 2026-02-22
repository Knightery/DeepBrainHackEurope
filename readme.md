# OpenQuant

*Alpha can come from anyone.*

High-level overview of the project. For implementation contracts and detailed rules, see [MVP_SPEC.md](MVP_SPEC.md).

For a comprehensive discussion of the theoretical foundations, modelling assumptions, and experimental validation, see the [Thesis](https://openquant-thesis.tiiny.site).

## Mission

OpenQuant helps strong independent quants submit stock ideas and get evaluated with a transparent, structured process that can lead to capital allocation.

## What the product does

- Collects a pitch through an interactive chat workflow.
- Standardizes core inputs (thesis, horizon, stocks, sources, supporting notes).
- Runs parallel evaluators to assess data quality, coding risk, and performance metrics.
- **Runs a Claude backtest agent** to generate and execute a standardized runner from the user's strategy script, computing all scored metrics automatically.
- Uses `strategy_scorer.py` as the canonical scoring engine for non-one-shot pitches.
- Produces a final score, recommendation, and allocation amount.
- Supports final human reviewer approval or rejection.

## User experience (MVP)

1. User starts chat onboarding.
2. Clarifier agent guides completion of a checklist.
3. User provides:
   - thesis,
   - time horizon (`days`, `weeks`, `months`, `years`),
   - stock ticker(s) (just symbols like `AAPL, MSFT`),
   - source URL(s) when supporting data files are submitted.
4. User uploads files:
    - At least one strategy/signal file is required for evaluation.
    - Preferred for backtest: **strategy script** (`.py` or `.ipynb`).
    - Supporting CSV/TSV data files are optional.
    - Price CSV uploads are optional and never required for strategy notebooks/scripts (prices are fetched internally via Alpaca during backtesting).
5. Once intake is complete, evaluation auto-runs (chat-style; commands are optional).
6. Before scoring, CUA automatically validates every uploaded data file against submitted source URLs.
7. For non-one-shot pitches, backtest is required before final review submission.
8. For `.py`/`.ipynb` strategy submissions, Fabrication Detector and Pipeline Auditor always run in parallel.
9. For event-driven non-repeatable theses, user can use `/oneshot on` or include `one_shot_mode=true` in chat.
10. Evaluation outcome:
   - fabricated/cheating signal -> `Goodbye.`
   - missing/unclear validation aspects -> clarification loop and `/evaluate`
   - clean validation -> ready for final review with `Congrats!`
11. User receives score + report + allocation recommendation once ready for final review.

## Why this structure

- Keeps onboarding simple for users.
- Preserves minimum standardization for reliable quant evaluation.
- Enforces data provenance when supporting datasets are submitted.
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
- Backtest agent reads `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` from `.env` (same key as CUA, but used directly in-process — no Docker required).
- Paid usage tracking reads:
  - `PAID_API_KEY` (required to send usage signals)
  - `PAID_EVENT_NAME` (optional, defaults to `eva_by_anyquant`)
  - `PAID_EXTERNAL_PRODUCT_ID` (optional, defaults to `quant_pitch_evaluator`)
  - `PAID_EXTERNAL_CUSTOMER_ID` (optional global override)
  - `PAID_USAGE_ENABLED` (`true`/`false`, defaults to enabled when `PAID_API_KEY` is set)

## Optional: CUA data fetcher (Docker)

The `cua/` folder contains an optional Computer Use Agent runner based on Anthropic's
official `computer-use-demo` image.

Use Docker Compose from inside `cua/`:

```powershell
cd cua
docker compose build
docker compose run --rm --remove-orphans data-fetcher "https://www.netflix.com/tudum/top10" "Validate downloaded data against reference file" "reference_user_file.csv"
```

Multiple CUA containers run **simultaneously** when a pitch has more than one data file — each file gets its own isolated container. Host VNC/noVNC ports are **not** statically bound, which allows concurrent runs without port conflicts. To watch a specific container's desktop, use `--service-ports` to get dynamically assigned host ports:

```powershell
docker compose run --rm --service-ports data-fetcher ...
# then check `docker ps` for the assigned host port → open http://localhost:<port>
```

Model selection:

- Default is `claude-sonnet-4-6` (recommended for stronger web navigation reliability).
- For cheaper testing, set `ANTHROPIC_MODEL=claude-haiku-4-5-20251001`.
- Tool version is auto-paired in `cua/data_fetcher.py`, or can be overridden with `CUA_TOOL_VERSION`.

CUA fetch behavior:

- The agent is CUA-first in `cua/data_fetcher.py`: it is strongly encouraged to navigate and click in-browser controls.
- Bash/terminal/script fallback (`bash`, `curl`, `wget`, inline Python) is allowed when it is the best practical path.
- The 3rd CLI arg is a reference filename already present in `~/Downloads`; CUA inspects it first, then searches source pages.
- Artifact checks are dynamic (no hardcoded filename/type requirement); matching quality is decided from CUA reasoning + downstream validators.
- After each CUA run, the main agent performs an LLM match review between reference and downloaded candidates.
- If mismatch is detected, the system automatically re-runs CUA with corrective guidance (configurable by `CUA_MAX_ATTEMPTS`, default `3`).

Important: `docker run build` is not a valid build command. Use `docker compose build` (or `docker build`).

## Principles

- Keep user inputs minimal but meaningful.
- Use `strategy_scorer.py` as the canonical scoring system in MVP.
- Prefer clarity and auditability over complexity.

## One-shot strategy mode

For single-event strategies (where Sharpe/win-rate are not meaningful), use:

```powershell
/oneshot on
```

Then run `/evaluate`. The system returns a binary recommendation (`VALID` / `NOT_VALID`) and does not assign USD allocation.

**Write naturally — no magic keywords required.** An LLM extraction agent reads your thesis, supporting notes, and CSV column names/samples to automatically infer the event type, map columns to the right statistical roles, and parse numeric assumptions from free-form text (e.g. "I think there's a 65% chance the deal closes").

Minimum one-shot inputs:
- **Node 1** (causal chain only): CSV with ≥30 rows containing your causal driver series and the target asset return side-by-side. Column names can be anything.
- **Node 2**: CSV with ≥20 rows of probability estimates (0–1) and binary realized outcomes (0 or 1). Column names can be anything.
- **Node 3** (causal chain only): CSV with ≥8 historical episodes showing driver intensity and resulting price change. Column names can be anything.
- **Node 4**: Your probability estimate, market-implied probability, and upside/downside payoffs — described conversationally in your supporting notes.

The event type (causal chain, binary event, or deal spread) is inferred automatically from your thesis. If extraction confidence is low, the system widens uncertainty and asks a clarifying question instead of failing silently.

Event variants:
- **causal_chain** (default): uses Nodes 1–4 + Monte Carlo.
- **binary_event**: uses Node 2 + Node 4 + Monte Carlo (skips Nodes 1 and 3).
- **deal_spread**: uses Node 2 + deal-pricing node + Monte Carlo.
  - Describe the current price, acquisition/break prices, and your close probability conversationally.

**Legacy key=value format still accepted** as a fallback for backward compatibility.

## Test scripts

Validation test cases:

```powershell
python validator_cases/run_validator_cases.py
```

CUA Netflix fetch test (GUI flow with context "click the download button"):

```powershell
python validator_cases/run_cua_tests.py
```

Basic direct CUA smoke test (Anthropic path only, bypasses `pitch_engine` matching):

```powershell
python validator_cases/run_cua_basic_test.py
```

Optional env overrides for CUA test:

- `CUA_TEST_REFERENCE_FILE` (defaults to `2026-02-21_country_weekly.tsv` in repo root)
- `CUA_TEST_NOTES` (defaults to `click the download button`)
- `CUA_BASIC_TEST_URL` (defaults to `https://www.netflix.com/tudum/top10`)
- `CUA_BASIC_TEST_NOTES` (defaults to `click the download button`)
- `CUA_BASIC_TEST_TIMEOUT_SECONDS` (defaults to `240`)
