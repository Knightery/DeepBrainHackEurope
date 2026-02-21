# Quant Pitch Evaluator - MVP Spec (Detailed Overview)

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
- Deterministic v0 scoring and allocation.
- Parallel evaluator outputs with unified envelope.
- Reviewer approve/reject action with persisted result.

### Out of scope (MVP)
- Brokerage execution and real-money movement.
- Production auth/RBAC and multi-tenant scaling.
- Full backtesting platform.
- Advanced anti-fraud ML systems.

## 3) User Intake Contract (Mandatory)

Evaluation is blocked until all mandatory items are provided.

1. **Thesis**: concise statement of what is mispriced and why.
2. **Time horizon**: one of `days`, `weeks`, `months`, `years`.
3. **Stock tickers**: one or more symbols (e.g., `AAPL`, `MSFT`).
4. **Methodology summary**: data used, signal idea, and validation approach.
5. **Source URLs**: source URL(s) for submitted data (provenance requirement).

Notes:
- Supporting file uploads are optional but recommended.
- If data is missing or unclear, clarifier asks follow-ups before evaluation.

## 4) State Machine

Pitch statuses:
- `draft` -> `ready` -> `running` -> `needs_clarification|ready_for_final_review|rejected`
- reviewer path: `ready_for_final_review` -> `approved|rejected`
- failure path: `running` -> `failed`

Rules:
- `ready` only when mandatory checklist passes.
- `/evaluate` must hard-block when mandatory fields are missing.
- `/validate` re-runs validation after user clarifications.

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
  "methodology_summary": "string",
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
      "severity": "low|medium|high|critical",
      "message": "string"
    }
  ],
  "artifacts": {},
  "latency_ms": 0
}
```

## 8) Evaluator Responsibilities

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

Critical examples:
- `UNREACHABLE_SOURCE_ALL`
- `SOURCE_MISMATCH_SEVERE`
- `MISSING_SOURCE_URLS`
- `BASH_FALLBACK_USED`
- `NO_NEW_DOWNLOAD`

### 8.2 Data Validator
Checks:
- parse integrity,
- missingness and duplicates,
- suspicious/fabricated series patterns,
- schema sufficiency for scoring.

Runtime outcomes:
- `blocked_fabrication`: terminate user flow with `Goodbye.`
- `needs_clarification`: emit concise issue summary + clarification questions for user loop
- `ready_for_final_review`: no blocking concerns; proceed to final review

Critical examples:
- `INSUFFICIENT_DATA`
- `MISSING_PRICE_COLUMN`
- `MISSING_TICKERS`

### 8.3 Pipeline Auditor
Checks:
- methodology clarity and consistency,
- leakage/overfit risk signs,
- evidence of robust validation framing.

### 8.4 Scoring
Computes:
- Sharpe,
- max drawdown,
- risk score,
- time to return (nullable),
- final score + allocation via deterministic formula.

### 8.5 Backtest Agent (`backtest_agent.py`)

Triggered automatically during `/evaluate` (and manually via `/backtest`) when the user uploads a `.py` strategy file.

**File role detection** (auto-classifies all uploaded files):
- `.py` → strategy scripts
- `.csv`/`.tsv` with `benchmark`/`market`/`index`/`spy` in name → benchmark files
- other `.csv`/`.tsv` → price data files

**3-attempt loop per run:**
- **Phase 1 – CREATE/FIX**: Claude (`ANTHROPIC_MODEL`) generates a self-contained Python runner that loads the user's strategy, runs it, fetches benchmark data via `yfinance`, computes all required metrics, and prints a JSON object to stdout.
- **Phase 2 – RUN**: `subprocess.run` executes the generated script in an isolated temp directory (timeout: `BACKTEST_TIMEOUT_SECONDS`, default 120s).
- **Phase 3 – REVIEW**: Claude validates the JSON output and decides the termination verdict.

**Termination statuses:**
- `success` — all required fields present and valid; `strategy_scorer` composite replaces CSV-based scoring.
- `agent_fault` — 3 attempts exhausted with self-introduced bugs; falls back to CSV scoring with a `medium` flag.
- `user_action_required` — Claude determines the user's script is broken or missing inputs; emits a `high` flag and user-facing message.

**Scoring override (when `success`):**
```
overall_score = 0.65 × strategy_scorer_composite
              + 15 × data_quality_score
              + 10 × methodology_score
              + 10 × match_rate
```

`agent_outputs["scoring"]["artifacts"]["source"]` is set to `"strategy_scorer"` or `"csv_approximation"` for auditability.

**Required output fields computed by the agent:**
`backtest_start`, `backtest_end`, `cagr`, `total_return`, `volatility`, `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `max_drawdown`, `max_drawdown_duration`, `total_trades`, `win_rate`, `avg_win`, `avg_loss`, `profit_factor`, `expectancy`, `benchmark_cagr`, `benchmark_max_drawdown`, `benchmark_total_return`, `alpha`, `information_ratio`, `excess_return`, `up_capture`, `down_capture`.

`name` and `ticker` are injected from the pitch context.

## 9) Scoring & Allocation Policy (v0)

### 9.1 Composite score (0-100)

```text
score = 100 * (
  0.30 * sharpe_n +
  0.20 * drawdown_n +
  0.15 * risk_n +
  0.15 * data_quality_score +
  0.10 * methodology_score +
  0.10 * match_rate
)
```

### 9.2 Hard reject rules
Allocation is forced to `0` when any critical condition occurs, including:
- missing mandatory intake fields,
- fewer than 30 valid rows,
- critical validator/fetcher/auditor flags.

Clarification-loop rule:
- If outcome is `needs_clarification`, allocation remains `0` until user resolves issues and passes `/validate`.

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

## 10) APIs (MVP shape)

Base path: `/v1`

- `POST /pitches`: create draft with metadata/files.
- `GET /pitches/{pitch_id}`: fetch pitch + status.
- `POST /pitches/{pitch_id}/clarifier/messages`: chat turn + missing fields.
- `POST /pitches/{pitch_id}/evaluate`: trigger run (idempotent while running).
- `GET /pitches/{pitch_id}/result`: fetch result when complete.
- `POST /pitches/{pitch_id}/review`: approve/reject with note.

Clarifier response must expose:
- `assistant_message`
- `missing_fields`
- `ready_for_evaluation`

Validation loop behavior:
- `/evaluate` runs both validation agents.
- If non-fabrication issues exist, user gets a compact summary + follow-up questions.
- User can answer in chat, then run `/validate` to re-run only from summarized context.
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
        data_validator.json
        pipeline_auditor.json
        scoring.json
      result.json
      review.json
```

## 12) Reliability and Security Baseline

- Limit upload types and size caps.
- Never execute uploaded notebooks.
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
6. Reviewer decision is stored and queryable.

## 14) Build Priorities

1. Keep intake UX simple and strict on mandatory fields.
2. Keep scoring deterministic and debuggable.
3. Keep all outputs auditable and recoverable from storage.
4. Add advanced fetch/validation intelligence after core reliability is stable.
