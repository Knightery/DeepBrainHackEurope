# OpenQuant - MVP Spec (Detailed Overview)

This is the implementation contract for MVP behavior, data requirements, and service responsibilities.

## 1) Objective

Ship a working end-to-end system that:
- accepts one quant pitch,
- runs parallel evaluators,
- returns one aggregate report + allocation recommendation,
- supports final human review.

## 2) Scope

### In scope (MVP)
- Chainlit intake flow with clarifier interactions.
- File upload and metadata persistence.
- Checklist-style onboarding and readiness gates.
- Deterministic strategy-scorer-based scoring and allocation.
- Parallel evaluator outputs with unified envelope.
- Read-only dashboard/API views for pitch history and rankings.

### Out of scope (MVP)
- Brokerage execution and real-money movement.
- Production auth/RBAC and multi-tenant scaling.
- Dedicated reviewer approve/reject workflow API.
- Advanced anti-fraud ML systems.

## 3) User Intake Contract (Mandatory)

Evaluation is blocked until all mandatory items are provided.

1. **Thesis**: concise statement of what is mispriced and why.
2. **Time horizon**: one of `days`, `weeks`, `months`, `years`.
3. **Stock tickers**: one or more symbols (e.g., `AAPL`, `MSFT`).
4. **Source URLs**: required only for uploaded supporting CSV/TSV datasets (provenance requirement).
5. **Supporting notes** (optional): any additional assumptions/context for validators.

Notes:
- Supporting file uploads are optional but recommended.
- Price CSV uploads are never mandatory for `.py`/`.ipynb` strategy backtests; market prices are fetched internally via Alpaca.
- If data is missing or unclear, clarifier asks follow-ups before evaluation.

## 4) State Machine

Pitch statuses:
- `draft` -> `ready` -> `running` -> `needs_clarification|ready_for_final_review|rejected`
- failure path: `running` -> `failed`

Rules:
- `ready` only when mandatory checklist passes.
- `/evaluate` must hard-block when mandatory fields are missing.
- `/evaluate` is also used to re-run validation after user clarifications.

## 5) Canonical Pitch Entity

```json
{
  "pitch_id": "pit_01HZY...",
  "created_at": "2026-02-21T13:20:00Z",
  "status": "draft",
  "submitter": {
    "display_name": "optional",
    "country": "IN"
  },
  "thesis": "string",
  "time_horizon": "days|weeks|months|years",
  "tickers": ["AAPL", "MSFT"],
  "source_urls": ["https://..."],
  "supporting_notes": "string",
  "uploaded_files": [
    {
      "file_id": "fil_...",
      "name": "data.csv",
      "mime_type": "text/csv",
      "size_bytes": 12345,
      "sha256": "hex"
    }
  ]
}
```

## 6) Canonical Timeseries Schema (Internal)

Required columns (or mappable equivalents):
- `timestamp_utc` (ISO8601 datetime)
- `symbol` (string)
- `close` (float)

Optional columns:
- `open`, `high`, `low`, `volume`, `adj_close`
- exogenous features (`feature_name`, `feature_value`)
- `source_url`

Validation rules:
- monotonic timestamps per symbol,
- no duplicate `(symbol, timestamp_utc)`,
- numeric columns parseable and finite,
- minimum 30 valid rows for scoring.

## 7) Agent Output Envelope (Required)

All evaluators return:

```json
{
  "agent": "data_fetcher|data_validator|pipeline_auditor|scoring",
  "status": "ok|warn|fail",
  "confidence": 0.0,
  "summary": "string",
  "flags": [
    {
      "code": "LOOKAHEAD_BIAS",
      "message": "string"
    }
  ],
  "artifacts": {},
  "latency_ms": 0
}
```

## 8) Evaluator Responsibilities

### 8.0 File Role Classifier (internal)
Runs automatically before any evaluation to classify uploaded CSV/TSV files.

Inputs:
- Column headers and first 5 sample rows of each uploaded CSV/TSV.
- Pitch `tickers` and `thesis` as context.

Outputs:
- `strategy_data` â€” main price/signal data files for the pitched strategy.
- `benchmark` â€” reference market/index/macro data.
- `other` â€” unclassified.

Runtime contract:
- If only one CSV/TSV is uploaded, it is always `strategy_data` (no LLM call).
- For multiple CSVs, Gemini classifies based on content, not filename keywords.
- Falls back to keyword heuristics if no API key is available.
- If LLM marks everything as benchmark, all files are promoted to `strategy_data` as a safety net.
- Classification result is used by the backtest agent and data validation logic.

### 8.1 Data Fetcher
Inputs:
- `source_urls`, uploaded files metadata, thesis.

Outputs:
- fetched artifacts,
- source/provenance log,
- submitted-vs-source match indicators.

Runtime contract (CUA mode):
- Browser interaction is preferred via GUI-driven computer-use actions (CUA-first behavior).
- Terminal/script fallbacks are allowed when GUI flow is blocked or clearly inferior.
- CUA first inspects the submitted reference file, then navigates source pages to find matching downloadable data.
- Matching is dynamic (schema/entity/date-range reasoning), not tied to fixed filename or extension rules.
- Post-download file matching is reviewed by an LLM against the reference file profile.
- If mismatch is detected, CUA is re-run with explicit retry guidance until match or retry limit is reached.
- During full pitch evaluation, CUA validation is mandatory for every uploaded CSV/TSV data file before scoring.
- All per-file CUA containers are launched **simultaneously** via `asyncio.gather`; each container is isolated with no fixed host port binding, so multiple containers can run without port conflicts.
- If any required data file lacks a clean CUA validation result, evaluation is routed to clarification (anti-scam gate).

Critical examples:
- `UNREACHABLE_SOURCE_ALL`
- `SOURCE_MISMATCH_SEVERE`
- `MISSING_SOURCE_URLS`
- `BASH_FALLBACK_USED`
- `NO_NEW_DOWNLOAD`

### 8.2 Data Validator (Fabrication Detector)
Checks:
- parse integrity,
- missingness and duplicates,
- suspicious/fabricated series patterns,
- schema sufficiency for scoring.

Runtime outcomes:
- `blocked_fabrication`: terminate user flow with `Goodbye.`
- `needs_clarification`: emit concise issue summary + clarification questions for user loop
- `ready_for_final_review`: no blocking concerns; proceed to final review

Question generation:
- When `verdict=unclear` or `verdict=fabrication`, the LLM populates `artifacts.questions` with
  1-3 specific, direct questions targeting the suspicious patterns found.
- These questions are surfaced to the user via the Orchestrator's "Validation Follow-up" message
  and stored in `validation_context` so the Clarifier Agent can work through them conversationally
  across subsequent chat turns.

Critical flag examples:
- `INSUFFICIENT_DATA`
- `MISSING_PRICE_COLUMN`
- `MISSING_TICKERS`

### 8.3 Pipeline Auditor (Coding Errors Detector)
Checks:
- methodology clarity and consistency,
- leakage/overfit risk signs,
- evidence of robust validation framing.

Runtime contract:
- For pitches that include `.py`/`.ipynb` strategy files, Pipeline Auditor and Data Validator run concurrently (mandatory parallel execution).

Question generation:
- When `verdict=errors_found`, the LLM populates `artifacts.questions` with 1-4 specific, actionable
  questions pointing at each detected issue (e.g. rolling window anchoring, transaction cost assumptions).
- These questions flow to the user via the same clarification loop as the Fabrication Detector.

### 8.4 Scoring
Computes:
- Sharpe,
- max drawdown,
- risk score,
- time to return (nullable),
- final score + allocation via deterministic formula.

### 8.5 Backtest Agent (`backtest_agent.py`)

Triggered automatically during `/evaluate` (and manually via `/backtest`) when the user uploads a `.py` or `.ipynb` strategy file.

**File role detection** (LLM-powered, auto-classifies all uploaded files):
- `.py`/`.ipynb` â†’ strategy scripts (no LLM needed)
- Single CSV/TSV â†’ always `strategy_data` (no LLM needed)
- Multiple CSV/TSVs â†’ Gemini reads column headers + 5 sample rows for each file and classifies
  using pitch `tickers` and `thesis` as context, producing `strategy_data` or `benchmark`.
- Heuristic keyword fallback is used when no Gemini API key is available.

**3-attempt loop per run:**
- **Phase 1 â€“ CREATE/FIX**: Claude (`ANTHROPIC_MODEL`) generates a self-contained Python runner that loads the user's strategy, runs it, fetches strategy/benchmark prices via Alpaca REST bars when needed (including notebooks that originally used other data APIs), computes all required metrics, and prints a JSON object to stdout.
- **Phase 2 â€" RUN**: `subprocess.run` executes the generated script in an isolated temp directory (timeout: `BACKTEST_TIMEOUT_SECONDS`, default 1200s). The generated runner may `pip install` any packages required by the strategy.
- **Phase 3 â€“ REVIEW**: Claude validates the JSON output and decides the termination verdict.

**Termination statuses:**
- `success` â€” all required fields present and valid; `strategy_scorer` composite is used for scoring.
- `agent_fault` â€” 3 attempts exhausted with self-introduced bugs; evaluation fails and requires rerun after fix.
- `user_action_required` â€” Claude determines the user's script is broken or missing inputs; emits a `high` flag and user-facing message.

Price-data policy:
- Do not require users to upload price CSV files for `.py`/`.ipynb` strategy backtests.
- Use Alpaca-fetched bars as the default market data source when local price files are absent.

**Scoring override (when `success`):**
```
overall_score = 0.65 Ã— strategy_scorer_composite
              + 15 Ã— data_quality_score
              + 10 Ã— methodology_score
              + 10 Ã— match_rate
```

`agent_outputs["scoring"]["artifacts"]["source"]` is set to `"strategy_scorer"` (or `"one_shot"` in one-shot mode) for auditability.

**Required output fields computed by the agent:**
`backtest_start`, `backtest_end`, `cagr`, `total_return`, `volatility`, `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `max_drawdown`, `max_drawdown_duration`, `total_trades`, `win_rate`, `avg_win`, `avg_loss`, `profit_factor`, `expectancy`, `benchmark_cagr`, `benchmark_max_drawdown`, `benchmark_total_return`, `alpha`, `information_ratio`, `excess_return`, `up_capture`, `down_capture`.

`name` and `ticker` are injected from the pitch context.

### 8.6 One-shot Strategy Validator (`one_shot_validator.py`)

Triggered only when the user explicitly enables one-shot mode via `/oneshot on`.

Purpose:
- Validate event-driven single-trade theses where repeated-trade metrics (Sharpe/win-rate) are not the primary evidence.

#### Phase 0 â€” LLM Extraction Agent

Before any statistical node runs, `_extract_one_shot_params(draft)` is called. It sends the thesis, supporting notes, ticker list, and a profile of each uploaded CSV (column names + 5 sample rows) to Gemini. The model returns a JSON payload with:
- `event_type`: inferred variant (`causal_chain`, `binary_event`, or `deal_spread`)
- `event_type_reasoning`: short explanation
- `column_mappings`: semantic role â†’ `{"file": ..., "column": ...}` for each CSV column role
- `numeric_params`: free-form text â†’ extracted key/value pairs
- `extraction_questions`: clarification questions if confidence is low
- `extraction_confidence`: 0â€“1 float

If `GEMINI_API_KEY` is absent or the call fails, `extraction_available=False` and all nodes fall back to legacy column-name heuristics and regex parsing.

Extraction sub-dict is included in `OneShotResult.artifacts["extraction"]` for observability.

#### Node checks

- `causal_chain` (default): Node 1 + Node 2 + Node 3 + Node 4 + Monte Carlo
- `binary_event`: Node 2 + Node 4 + Monte Carlo
- `deal_spread`: Node 2 + deal-pricing node + Monte Carlo

Event type is inferred by the extraction agent from the thesis. Legacy key `one_shot_event_type=...` in methodology text is still accepted as a fallback.

#### Column resolution

Each statistical node calls `_find_pair_with_fallback(tables, llm_mapping_left, llm_mapping_right, legacy_candidates_left, legacy_candidates_right)`:
1. If the LLM mapping is confident, resolve the named column from the named file.
2. Otherwise fall back to the legacy candidate name list (e.g. `("wheat_price", "driver_value", ...)`).

#### Numeric parameter resolution

`_resolve_numeric(extraction, key, method_text)` resolves numeric params:
1. Use `extraction.numeric_params[key]` if extraction is available and confidence â‰¥ 0.6.
2. Fall back to `_parse_numeric_from_text(method_text, key)` (regex `key=value`).

Users may write values conversationally (e.g. "I think there's a 65% chance"); the extraction agent handles parsing.

#### Decision output

- Binary recommendation only: `VALID` or `NOT_VALID`
- No USD allocation is computed in one-shot mode (`allocation_usd = 0`)
- Missing node inputs produce plain-English clarification questions and widened Monte Carlo uncertainty
- Clarification questions from Phase 0 are surfaced alongside per-node questions

## 9) Scoring & Allocation Policy

### 9.1 Composite score (0-100, non-one-shot)

```text
overall_score = 0.65 * strategy_scorer_composite
              + 15   * data_quality_score
              + 10   * methodology_score
              + 10   * match_rate
```

Runtime rules:
- `strategy_scorer` output from backtest is mandatory for `.py`/`.ipynb` strategy pitches.
- CSV approximation scoring is disabled.
- If strategy-scorer output is unavailable, evaluation routes to failure/clarification rather than fallback scoring.

### 9.2 Hard reject rules
Allocation is forced to `0` when any critical condition occurs, including:
- missing mandatory intake fields,
- fewer than 30 valid rows,
- critical validator/fetcher/auditor flags.

Clarification-loop rule:
- If outcome is `needs_clarification`, allocation remains `0` until user resolves issues and passes `/evaluate`.

### 9.3 Allocation ladder (USD)
If not hard-rejected:
- `<55`: `0`
- `55-64.9`: `1000`
- `65-74.9`: `2500`
- `75-84.9`: `5000`
- `85-92.9`: `10000`
- `>=93`: `15000`

Horizon multiplier:
- `days`: `0.8`
- `weeks`: `1.0`
- `months`: `1.2`
- `years`: `1.4`

Final:
- `allocation = min(20000, round_to_100(base * multiplier))`

## 10) Runtime Interfaces (current MVP)

### Primary interaction surface

Chainlit chat app mounted at `/app` with command-driven orchestration:
- `/status`, `/checklist`
- `/oneshot on|off|status`
- `/evaluate`
- `/backtest`
- `/validate_data "file" "notes"`
- `/reset`

### Dashboard/API surface

FastAPI read endpoints:
- `GET /api/theme`
- `GET /api/pitches`
- `GET /api/pitches/{pitch_id}`
- `GET /api/pitches/{pitch_id}/messages`

Landing/dashboard routes:
- `GET /home` (landing)
- `GET /dashboard` (ranked pitch view)

Validation loop behavior:
- `/evaluate` runs both validation agents.
- If non-fabrication issues exist, user gets a compact summary + follow-up questions.
- User can answer in chat, then run `/evaluate` to re-run only from summarized context.
- Only concise validation summaries are fed back into the user-facing clarifier context.

## 11) Storage Layout

```text
data/
  pitches/
    {pitch_id}/
      pitch.json
      clarifier_history.jsonl
      uploads/
      agent_outputs/
        data_fetcher.json
        backtest_agent.json
      result.json
```

Notes:
- `data_validator`, `pipeline_auditor`, and `scoring` outputs are persisted inside `result.json` under `agent_outputs`.

## 12) Reliability and Security Baseline

- Limit upload types and size caps.
- For MVP, notebooks may be executed as compiled linear scripts in isolated runtime.
- Isolate browser/data-fetch runtime.
- Log all external URL access attempts.
- Redact secrets from logs.

## 13) Acceptance Criteria

MVP is complete when:
1. User can complete onboarding checklist in chat.
2. `/evaluate` blocks until mandatory intake fields are complete.
3. Evaluators produce normalized output envelopes.
4. Result includes score, allocation, decision, and report.
5. Hard-reject rules consistently force zero allocation.
6. Pitch history and messages are queryable via `/api/pitches*`.

## 14) Build Priorities

1. Keep intake UX simple and strict on mandatory fields.
2. Keep scoring deterministic and debuggable.
3. Keep all outputs auditable and recoverable from storage.
4. Add advanced fetch/validation intelligence after core reliability is stable.

