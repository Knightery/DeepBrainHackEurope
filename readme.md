# Quant Pitch Evaluator

High-level overview of the project. For implementation contracts and detailed rules, see [MVP_SPEC.md](MVP_SPEC.md).

## Mission

Quant Pitch Evaluator helps strong independent quants submit stock ideas and get evaluated with a transparent, structured process that can lead to capital allocation.

## What the product does

- Collects a pitch through an interactive chat workflow.
- Standardizes core inputs (thesis, horizon, stocks, methodology, sources).
- Runs parallel evaluators to assess data quality, methodology risk, and performance metrics.
- **Runs a Claude backtest agent** to generate and execute a standardized runner from the user's strategy script, computing all scored metrics automatically.
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
4. User uploads supporting files:
   - **strategy script** (`.py` or `.ipynb`) — required for backtest agent scoring.
   - **price data CSV** — used by both the backtest agent and CSV-based fallback.
   - benchmark CSV (optional, filename must contain `benchmark`/`market`/`spy`) — auto-detected.
5. Once intake is complete, evaluation auto-runs (chat-style; commands are optional).
6. Before scoring, CUA automatically validates every uploaded data file against submitted source URLs.
7. User can still run `/backtest` as an optional standalone pre-check.
8. For event-driven non-repeatable theses, user can use `/oneshot on` or include `one_shot_mode=true` in chat.
9. Evaluation outcome:
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
- Backtest agent reads `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` from `.env` (same key as CUA, but used directly in-process — no Docker required).

## Optional: CUA data fetcher (Docker)

The `cua/` folder contains an optional Computer Use Agent runner based on Anthropic's
official `computer-use-demo` image.

Use Docker Compose from inside `cua/`:

```powershell
cd cua
docker compose build
docker compose run --rm --remove-orphans data-fetcher "https://www.netflix.com/tudum/top10" "Validate downloaded data against reference file" "reference_user_file.csv"
```

Watch the live desktop at `http://localhost:6080`.

Model selection:

- Default is `claude-sonnet-4-5-20250929` (recommended for stronger web navigation reliability).
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
- Keep scoring deterministic in MVP.
- Prefer clarity and auditability over complexity.

## One-shot strategy mode

For single-event strategies (where Sharpe/win-rate are not meaningful), use:

```powershell
/oneshot on
```

Then run `/evaluate`. The system returns a binary recommendation (`VALID` / `NOT_VALID`) and does not assign USD allocation.

Minimum one-shot inputs:
- Node 1 relationship history CSV (driver and asset-return columns, >=30 rows)
- Node 2 forecast calibration CSV (`forecast_prob`, `outcome`, >=20 rows)
- Node 3 magnitude history CSV (`drought_severity`, `wheat_change`, >=8 rows)
- Node 4 assumptions in methodology text:
  - `p_true=...`
  - `p_market=...`
  - `payoff_up=...`
  - `payoff_down=...`
  - optional `transaction_cost=...`

Variant shortcuts (easiest implemented):
- `one_shot_event_type=causal_chain` (default): uses Nodes 1-4 + Monte Carlo.
- `one_shot_event_type=binary_event`: uses Node 2 + Node 4 + Monte Carlo (skips Nodes 1 and 3).
- `one_shot_event_type=deal_spread`: uses Node 2 + deal-pricing node + Monte Carlo.
  - Required methodology keys:
    - `p_close=...`
    - `current_price=...`
    - `price_if_close=...`
    - `price_if_break=...`
    - optional `transaction_cost=...`

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
