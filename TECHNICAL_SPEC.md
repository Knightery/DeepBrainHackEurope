# OpenQuant — Full Technical Specification

_Last updated: 2026-02-22_

This document explains what this repository does, how data moves through it, and how decisions are made.
It is written for both:
- **non-technical readers** (what the product does, why outcomes happen), and
- **technical readers** (exact modules, contracts, and runtime behavior).

---

## 1) What this product is

OpenQuant is a **chat-driven evaluator** for quantitative stock strategy pitches.

A user submits a strategy idea and files (typically a `.py` or `.ipynb` strategy script, sometimes supporting CSV/TSV datasets). The system then runs a multi-agent validation pipeline that:

1. checks intake completeness,
2. validates source provenance of uploaded data (via CUA browser automation),
3. backtests strategy code (via Claude-generated runner),
4. checks for fabricated data and methodology errors,
5. computes a final score and recommendation.

Core principle: **structured, auditable evaluation with anti-scam gates**.

---

## 2) Repository map (what each top-level area is for)

### Primary app runtime
- `app.py` — Chainlit chat app, user flow orchestration, command handling.
- `pitch_engine.py` — core evaluation engine (validators + scoring composition + result contract).
- `backtest_agent.py` — Claude 3-phase backtest loop (create/run/review).
- `one_shot_validator.py` — validator for event-driven one-shot ideas.
- `strategy_scorer.py` — deterministic strategy metric scoring and hard gates.
- `app_ui.py` — chart rendering and live CUA log streaming helpers.

### Web/API runtime
- `server.py` — FastAPI host that serves landing page, dashboard, and read-only APIs; mounts Chainlit at `/app`.
- `run_server.py` — uvicorn launcher for `server.py`.
- `pitch_db.py` — SQLite persistence (pitch snapshots and message logs).

### Data provenance subsystem (CUA)
- `cua/data_fetcher.py` — Anthropic Computer Use runner executed in Docker.
- `cua/docker-compose.yml` — CUA service definition; supports parallel containers.
- `cua/downloads/` — shared artifacts and staged reference files for CUA runs.

### Integrations and utility modules
- `agent_skills/` — data-fetch skills (e.g., Alpaca historical bars, Solana xStocks bars).
- `paid_usage.py` — paid usage signal tracking.

### Tests and validation assets
- `tests/` — unit/integration tests for key modules.
- `validator_cases/` — scripted validator and CUA test cases.

### Product docs
- `readme.md`, `MVP_SPEC.md`, `README_CHAINLIT.md`, `chainlit.md`.

---

## 3) Runtime architecture (high-level)

### User-facing surfaces
1. **Chainlit chat app** (`app.py`) for pitch intake and evaluation commands.
2. **FastAPI server** (`server.py`) for:
   - landing page (`/`, `/home`),
   - dashboard (`/dashboard`),
   - read-only APIs (`/api/pitches*`),
   - mounted Chainlit app (`/app`).

### Internal engines
- `pitch_engine.evaluate_pitch(...)` is the central evaluator.
- It delegates to:
  - CUA validation output (from `validate_data_with_cua`),
  - backtest agent (`run_backtest_agent`),
  - one-shot validator (`evaluate_one_shot_strategy`),
  - strategy scorer (`score_strategy`),
  - Gemini-based validator pair (fabrication + coding errors).

### Storage
- File artifacts: `data/pitches/{pitch_id}/...`
- Relational state: SQLite DB at `data/pitches/pitches.db` (unless `PITCH_DB_PATH` set).

---

## 4) End-to-end user flow

### 4.1 Chat session start
On chat start (`@cl.on_chat_start`), system creates a new pitch draft and asks:
1. pitch type (`Recurring strategy` or `One-shot trade`),
2. time horizon (`days`, `weeks`, `months`, `years`).

Then user submits thesis, tickers, files, and optional notes.

### 4.2 Intake requirements (blocking)
Evaluation is blocked until required fields are complete:
- thesis,
- time horizon,
- tickers,
- uploaded strategy file (`.py`/`.ipynb`) for standard mode,
- source URLs if supporting tabular datasets are uploaded.

### 4.3 Clarifier agent behavior
A Gemini clarifier agent:
- extracts structured pitch fields from natural language,
- can request safe read tools for uploaded files (`read_uploaded_file`, `read_notebook_cells`),
- emits orchestrator actions (`run_backtest`, `run_validate_data`, `run_evaluate`),
- auto-advances when intake is sufficient.

### 4.4 Commands
Supported commands in chat:
- `/help`
- `/status`
- `/checklist`
- `/oneshot on|off|status`
- `/evaluate`
- `/backtest`
- `/validate_data "file" "notes"`
- `/reset`

Deprecated command:
- `/validate` (explicitly removed; user is directed to `/evaluate`).

---

## 5) Evaluation pipeline (what runs during `/evaluate`)

The orchestration function is `_handle_evaluate_command` in `app.py`.

### Step 1 — Mandatory CUA provenance checks
Before scoring, all relevant uploaded CSV/TSV files are validated with CUA.

Important runtime behavior:
- validations run **in parallel** (`asyncio.gather`), one container flow per file,
- each run stages the uploaded file as reference (`reference_{pitch_id}_{filename}`),
- CUA result is post-reviewed by Gemini for semantic match quality,
- mismatch can trigger automatic retry up to `CUA_MAX_ATTEMPTS`.

If CUA finds unresolved issues, evaluation moves to clarification path.

### Step 2 — `evaluate_pitch(...)` monolithic evaluation
`pitch_engine.evaluate_pitch` then computes final result by combining:
- file role classification,
- backtest outcome,
- one-shot validator output (if enabled),
- data quality + methodology + provenance checks,
- Gemini validator pair outputs,
- strategy scoring formula.

### Step 3 — Outcome routing
Possible high-level outcomes:
- `blocked_fabrication` → immediate rejection message: `Goodbye.`
- `needs_clarification` → follow-up questions and retry loop
- `ready_for_final_review` → score card + allocation

---

## 6) Core data contracts

### 6.1 Pitch draft model
`PitchDraft` (in `pitch_engine.py`) includes:
- `pitch_id`, `created_at`, `status`
- `thesis`, `time_horizon`, `tickers`, `source_urls`, `supporting_notes`
- `one_shot_mode`
- `uploaded_files[]`

Readiness check: `draft.ready_for_evaluation()`.

### 6.2 Uploaded file model
`UploadedFile` includes:
- `file_id`, `name`, `path`, `mime_type`, `size_bytes`, `sha256`.

### 6.3 Agent output envelope
Outputs are normalized with fields like:
- `agent`, `status`, `confidence`, `summary`, `flags[]`, `artifacts`, `latency_ms`.

### 6.4 Final evaluation model
`EvaluationResult` includes:
- `overall_score`, `allocation_usd`, `decision`, `validation_outcome`,
- `validation_summary`, `validation_questions`,
- `hard_reject_reasons`, `agent_outputs`, `report_markdown`.

---

## 7) File role classification

Function: `_detect_file_roles(...)` in `pitch_engine.py`.

Purpose: classify uploaded files into:
- strategy scripts (`.py`, `.ipynb`),
- primary strategy data (`.csv`, `.tsv`),
- benchmark/reference files.

Behavior:
- if one tabular file exists, it is treated as strategy data,
- with multiple tabular files, Gemini classification is used (with heuristic fallback),
- if classification is uncertain, safety fallbacks prevent empty strategy-data assignment.

---

## 8) Backtest subsystem (Claude loop)

Primary function: `run_backtest_agent(...)` in `backtest_agent.py`.

### 3-phase loop (up to `BACKTEST_MAX_ATTEMPTS`)
1. **Create/Fix**: Claude writes or repairs a self-contained runner script.
2. **Run**: runner executes in isolated temp directory (`subprocess.run`, timeout enforced).
3. **Review**: Claude validates output JSON and chooses termination verdict.

### Termination statuses
- `success`
- `user_action_required`
- `agent_fault`

### Required output metrics
Runner must output a single JSON with full strategy metrics (CAGR, Sharpe, drawdown, alpha, capture ratios, trade stats, etc.).

### Data source behavior
Backtest supports benchmark/market data fetch from:
- Alpaca bars,
- Solana xStocks (Birdeye) path for xStock tickers.

### Validation after backtest
Metrics are validated against `strategy_scorer.validate_and_load(...)`.
If invalid, scorer contribution is not accepted.

---

## 9) One-shot mode subsystem

Entry: `evaluate_one_shot_strategy(...)` in `one_shot_validator.py`.

Used for event-driven single-trade theses where repeated-trade metrics are less relevant.

### Phase 0 — LLM extraction
Gemini infers:
- event type (`causal_chain`, `binary_event`, `deal_spread`),
- column mappings from uploaded CSV profiles,
- numeric assumptions from natural language notes.

### Statistical nodes
Depending on event type:
- Node 1: causal relationship significance/stability (causal chain only)
- Node 2: forecast calibration (Brier/BSS + calibration gap)
- Node 3: magnitude estimate significance (causal chain only)
- Node 4: market mispricing/deal-pricing edge
- Monte Carlo: uncertainty-aware expected value edge

### One-shot decision
Output recommendation is binary:
- `VALID`
- `NOT_VALID`

In one-shot mode:
- `allocation_usd = 0`
- no standard USD capital allocation is produced.

---

## 10) Scoring and allocation logic

### 10.1 Strategy-level hard gates (`strategy_scorer.py`)
A strategy is disqualified if any gate fails, including:
- non-positive excess return,
- non-positive alpha,
- non-positive Sharpe,
- profit factor `<= 1`,
- max drawdown worse than `-50%`.

### 10.2 Weighted strategy composite
If hard gates pass, weighted component scoring produces a composite in `[0, 100]`.
Components cover:
- benchmark-relative outperformance,
- risk-adjusted returns,
- trade quality and consistency.

### 10.3 Global final score in evaluator
For standard mode, `evaluate_pitch` computes:

`overall_score = 0.65 * strategy_scorer_composite + 15 * data_quality_score + 10 * methodology_score + 10 * match_rate`

If strategy scorer is unavailable (non-one-shot), score is forced to 0 and clarification is required.

### 10.4 Allocation policy
When ready for final review (and not one-shot), allocation is computed using `_compute_allocation(overall_score, time_horizon)` in `pitch_engine.py`.

---

## 11) CUA provenance validation details

Entry point in evaluator: `validate_data_with_cua(...)` (`pitch_engine.py`).
Container behavior implemented in `cua/data_fetcher.py`.

### Key design choices
- CUA-first GUI navigation (screenshots/clicks/page text).
- Bash fallback allowed when GUI path is blocked.
- Downloaded candidates are semantically compared to staged reference via Gemini review.
- Severe mismatch emits `SOURCE_MISMATCH_SEVERE` and can fail the run.

### Typical failure/flag codes
- `MISSING_SOURCE_URLS`
- `NO_NEW_DOWNLOAD`
- `CUA_OUTPUT_PARSE_FAILED`
- `CUA_RUN_TIMEOUT`
- `SOURCE_MISMATCH_SEVERE`

### Parallelism
Full evaluation runs multiple CUA validations simultaneously for multiple files.
`cua/docker-compose.yml` intentionally avoids fixed host port bindings so concurrent containers do not conflict.

---

## 12) Validator pair (fabrication + coding errors)

`pitch_engine` runs two Gemini validators concurrently:
1. **Fabrication Detector** — focuses on intentional manipulation evidence.
2. **Coding Errors Detector** — focuses on methodological mistakes (look-ahead, leakage, overfit, etc.).

Both can emit:
- status + confidence,
- flags,
- clarification questions that feed back into chat loop.

Special hard-block rule:
- fabrication verdict blocks only when verdict is `fabrication` and confidence `>= 0.8`.

---

## 13) Persistence model

### 13.1 Filesystem artifacts
Per pitch directory:
- `data/pitches/{pitch_id}/pitch.json`
- `data/pitches/{pitch_id}/clarifier_history.jsonl`
- `data/pitches/{pitch_id}/agent_outputs/*.json`
- `data/pitches/{pitch_id}/result.json`
- `data/pitches/{pitch_id}/uploads/*`

### 13.2 SQLite schema (`pitch_db.py`)
Tables:
- `pitches`
- `pitch_messages`

Stored fields include status, thesis, tickers/source URLs JSON, one-shot mode, score/allocation/decision, and serialized `result_json`.

---

## 14) API and dashboard contract

FastAPI endpoints in `server.py`:
- `GET /` → redirect to `/home`
- `GET /home` → static landing page
- `GET /dashboard` → built-in HTML dashboard
- `GET /api/theme`
- `GET /api/pitches`
- `GET /api/pitches/{pitch_id}`
- `GET /api/pitches/{pitch_id}/messages`

Dashboard behavior:
- shows ranked completed pitches,
- clicking a pitch loads metadata + full chat history.

---

## 15) Environment and dependencies

### 15.1 Main dependencies (`requirements.txt`)
- Chainlit, FastAPI, Uvicorn
- `google-genai`, `anthropic`
- `pandas`, `numpy`, `scikit-learn`, `plotly`
- `pytest`
- `paid-python`

### 15.2 Required/important env vars
- `GEMINI_API_KEY` (clarifier + validators + extraction + match review)
- `ANTHROPIC_API_KEY` (backtest agent + CUA)
- `GEMINI_MODEL`, `ANTHROPIC_MODEL`
- `BACKTEST_TIMEOUT_SECONDS`, `BACKTEST_MAX_ATTEMPTS`
- `CUA_MAX_ATTEMPTS`, `CUA_RUN_TIMEOUT_SECONDS`
- paid usage variables: `PAID_API_KEY`, `PAID_USAGE_ENABLED`, etc.

If Gemini key is absent, app startup intentionally fails because clarifier is mandatory.

---

## 16) Security and robustness controls

Implemented safeguards include:
- safe upload filename sanitization and path traversal prevention,
- SHA256 hashing of uploaded files,
- bounded retries/timeouts for networked LLM calls,
- explicit handling for non-finite metrics (`NaN`/`Inf`) in backtest outputs,
- clear separation of user-action-required vs. agent-fault failures,
- deterministic storage of all key artifacts for auditability.

---

## 17) Testing posture

The repo includes tests for:
- strategy scorer validation,
- one-shot validator scenarios,
- paid usage behavior,
- alpaca/solana skill paths,
- backtest agent markers,
- validator retries and security hardening.

Representative test files:
- `tests/test_strategy_scorer.py`
- `tests/test_one_shot_validator.py`
- `tests/test_security_hardening.py`
- `tests/test_backtest_agent_alpaca_marker.py`

---

## 18) Operational runbook (practical)

### Run chat app only
```powershell
chainlit run app.py
```

### Run unified server (home + dashboard + API + mounted Chainlit)
```powershell
python run_server.py
```

### Build CUA image
```powershell
docker compose -f cua/docker-compose.yml build data-fetcher
```

### Typical evaluation sequence
1. user uploads strategy + required fields,
2. CUA validates all relevant datasets,
3. backtest and validators run,
4. user receives either:
   - `Goodbye.` (hard fabrication block),
   - clarification questions, or
   - final score + allocation recommendation.

---

## 19) Current known product posture (as implemented)

- MVP-grade system with strong anti-scam emphasis and transparent reports.
- Read-only ops dashboard is available; reviewer approval workflow is not a separate API feature yet.
- Scoring for normal mode depends on successful backtest strategy metrics; CSV-only approximation fallback is disabled.
- One-shot mode uses a separate statistical decision path and outputs binary validity without USD allocation.

---

## 20) Glossary (plain language)

- **Pitch**: one user submission (thesis + files + metadata).
- **Clarifier**: chat agent that turns free text into structured fields.
- **CUA**: browser-automation agent used to verify data provenance.
- **Fabrication Detector**: checks for likely intentional data manipulation.
- **Pipeline Auditor**: checks strategy methodology quality (non-fraud errors).
- **Strategy scorer**: deterministic engine that scores validated backtest metrics.
- **One-shot mode**: alternate validation for event-driven, non-repeatable theses.
