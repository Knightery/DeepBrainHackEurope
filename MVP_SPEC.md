# Quant Pitch Evaluator - MVP Spec (v0)

## 1) Goal

Ship a hackathon-grade end-to-end system that:
- accepts one quant pitch with files + source URLs,
- runs 4 async evaluators (Data Fetcher, Data Validator, Pipeline Auditor, Scoring),
- produces one aggregate report + allocation recommendation,
- supports final human approval or rejection.

This document defines contracts that are concrete enough to implement immediately.

## 2) MVP Scope and Non-Goals

In scope:
- FastAPI backend with REST endpoints.
- File upload + basic metadata ingestion.
- Clarifier chat endpoint (single model, persisted history).
- Parallel evaluator execution with timeout/retry.
- Deterministic v0 scoring and allocation policy.
- File-based persistence and audit logs.
- Minimal reviewer actions (approve/reject with note).

Out of scope for MVP:
- Production auth/RBAC.
- Brokerage execution and real money movement.
- Full historical backtest platform.
- Advanced anti-fraud ML models.
- Multi-tenant scaling and HA.

## 3) Architecture (MVP)

Flow:
1. User creates pitch and uploads files/metadata.
2. User exchanges clarifier messages until ready.
3. User triggers evaluation.
4. Orchestrator runs evaluators in parallel.
5. Aggregator computes final score/allocation and writes report.
6. Human reviewer approves or rejects.

Core services:
- `api`: FastAPI routes + request validation.
- `orchestrator`: fan-out/fan-in async execution.
- `agents`: wrappers around Claude tools/prompts.
- `scoring`: deterministic formula library.
- `storage`: local filesystem repository.

## 4) Canonical Data Contracts

### 4.1 Pitch Entity

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
  "time_horizon": "days|weeks|months",
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

### 4.2 Canonical Timeseries Schema (internal)

Required columns:
- `timestamp_utc` (ISO8601 datetime)
- `symbol` (string)
- `close` (float)

Optional columns:
- `open`, `high`, `low`, `volume`, `adj_close` (float)
- `feature_name` (string), `feature_value` (float) for exogenous features
- `source_url` (string)

Validation rules:
- timestamps must be monotonic per symbol,
- no duplicate `(symbol, timestamp_utc)`,
- numeric columns parseable and finite,
- minimum 30 rows total for scoring.

### 4.3 Agent Output Envelope

All agents must return this envelope:

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

## 5) API Spec (FastAPI)

Base path: `/v1`

### 5.1 Create Pitch

- `POST /v1/pitches`
- Content type: `multipart/form-data`
- Fields:
  - `metadata` (JSON string with thesis, horizon, methodology, source_urls, submitter)
  - `files[]` (0..10 files, each max 25 MB)
- Response `201`:

```json
{
  "pitch_id": "pit_...",
  "status": "draft"
}
```

### 5.2 Get Pitch

- `GET /v1/pitches/{pitch_id}`
- Response `200`: full pitch record + current status.

### 5.3 Clarifier Chat Turn

- `POST /v1/pitches/{pitch_id}/clarifier/messages`
- Request:

```json
{
  "message": "user text",
  "stream": true
}
```

- Response `200`:

```json
{
  "assistant_message": "string",
  "missing_fields": ["methodology_summary", "source_urls"],
  "ready_for_evaluation": false
}
```

### 5.4 Trigger Evaluation

- `POST /v1/pitches/{pitch_id}/evaluate`
- Idempotent: if already `running`, return current run id.
- Response `202`:

```json
{
  "pitch_id": "pit_...",
  "run_id": "run_...",
  "status": "running"
}
```

### 5.5 Get Evaluation Result

- `GET /v1/pitches/{pitch_id}/result`
- Response `200` when complete:

```json
{
  "pitch_id": "pit_...",
  "status": "completed",
  "overall_score": 78.4,
  "allocation_usd": 5000,
  "decision": "recommend_allocate",
  "agent_outputs": {},
  "report_markdown": "..."
}
```

If incomplete, return `202` with status.

### 5.6 Reviewer Decision

- `POST /v1/pitches/{pitch_id}/review`
- Request:

```json
{
  "reviewer_id": "rev_1",
  "decision": "approve|reject",
  "note": "string"
}
```

- Response `200` with final status.

## 6) Orchestration and State Machine

Pitch statuses:
- `draft` -> `ready` -> `running` -> `completed` -> `approved|rejected`
- terminal failure path: `running` -> `failed`

Evaluator runtime policy:
- run in parallel: fetcher, validator, auditor, scoring.
- per-agent timeout: 8 minutes.
- retry policy: 1 retry on transport/tool error only.
- no retry on deterministic validation failures.

Fan-in aggregation:
- wait for all agent completions or hard timeout.
- if any agent returns `fail` with critical flag, mark decision as reject candidate.
- always produce a final report, even on partial failure.

## 7) Agent-Specific Contracts

### 7.1 Data Fetcher

Inputs:
- `source_urls`, submitted files metadata, pitch thesis.

Outputs in `artifacts`:
- `fetched_files[]` with sha256,
- `match_rate` in `[0,1]` comparing fetched vs submitted overlap,
- `provenance_log[]` (url, timestamp, action).

Critical flags:
- `UNREACHABLE_SOURCE_ALL`
- `SOURCE_MISMATCH_SEVERE`

### 7.2 Data Validator

Checks:
- parse validity, missingness, outlier structure,
- fabricated-series heuristics (flat segments, repeated blocks),
- look-ahead and survivorship indicators.

Outputs in `artifacts`:
- `data_quality_score` in `[0,1]`,
- `row_count`, `symbol_count`,
- `anomaly_summary`.

Critical flags:
- `DATA_FABRICATION_SUSPECTED`
- `INSUFFICIENT_DATA`

### 7.3 Pipeline Auditor

Checks:
- methodology consistency vs data/time horizon,
- leakage risk, overfitting signals, p-hacking signs,
- presence of out-of-sample or walk-forward narrative.

Outputs in `artifacts`:
- `methodology_score` in `[0,1]`,
- `required_followups[]`.

Critical flags:
- `LEAKAGE_HIGH_RISK`

### 7.4 Scoring Agent

Computes raw metrics from canonical data:
- `sharpe`,
- `max_drawdown`,
- `risk_score` in `[0,1]` (higher is riskier),
- `time_to_return_days` (nullable).

Returns strict JSON only.

## 8) v0 Scoring and Allocation Policy

## 8.1 Normalizations

Given:
- `sharpe`
- `max_drawdown` (negative value, example `-0.25`)
- `risk_score` in `[0,1]`
- `data_quality_score` in `[0,1]`
- `methodology_score` in `[0,1]`
- `match_rate` in `[0,1]`

Normalize:
- `sharpe_n = clamp((sharpe + 1.0) / 3.0, 0, 1)` (maps -1..2 to 0..1)
- `drawdown_n = 1 - clamp(abs(max_drawdown) / 0.60, 0, 1)`
- `risk_n = 1 - risk_score`

## 8.2 Composite Score (0-100)

```
score = 100 * (
  0.30 * sharpe_n +
  0.20 * drawdown_n +
  0.15 * risk_n +
  0.15 * data_quality_score +
  0.10 * methodology_score +
  0.10 * match_rate
)
```

Round to 1 decimal.

## 8.3 Hard Reject Rules

Allocation is forced to `0` if any:
- critical flag from validator/auditor/fetcher,
- fewer than 30 valid rows,
- no valid source URL and no uploaded dataset.

## 8.4 Allocation Ladder (USD)

If not hard-rejected:
- `< 55`: `0` (reject)
- `55-64.9`: `1000`
- `65-74.9`: `2500`
- `75-84.9`: `5000`
- `85-92.9`: `10000`
- `>= 93`: `15000`

Horizon multiplier:
- `days`: `0.8`
- `weeks`: `1.0`
- `months`: `1.2`

Final allocation:
- `allocation = min(20000, round_to_100(base * multiplier))`

## 9) Storage Layout (Filesystem)

```
data/
  pitches/
    {pitch_id}/
      pitch.json
      clarifier_history.jsonl
      uploads/
      fetched/
      agent_outputs/
        data_fetcher.json
        data_validator.json
        pipeline_auditor.json
        scoring.json
      result.json
      review.json
      audit.log
```

Audit log line format:
- `timestamp_utc | level | pitch_id | component | message | context_json`

## 10) Security and Reliability Baseline

- File allowlist: `.csv`, `.tsv`, `.json`, `.txt`, `.ipynb`, `.pdf`.
- Reject executable uploads.
- Enforce max total upload size per pitch: 100 MB.
- Do not execute uploaded notebooks.
- Computer Use runs in isolated sandbox; no host filesystem mount.
- Redact secrets from logs.
- Record all external URLs accessed by fetcher.

## 11) Acceptance Criteria

MVP is done when:
1. A user can submit a pitch with at least one CSV and one source URL.
2. Clarifier returns missing-field prompts until ready.
3. Evaluation endpoint triggers all 4 agents and stores outputs.
4. Aggregator always returns a result payload with score and decision.
5. Hard reject rules override allocation to zero.
6. Reviewer can approve/reject and decision is persisted.
7. A demo pitch run completes in under 10 minutes end-to-end.

## 12) Two-Day Build Sequence

Day 1:
1. Scaffold FastAPI app, models, filesystem repo, pitch CRUD.
2. Implement `/pitches`, `/clarifier/messages`, `/evaluate`, `/result`.
3. Add orchestrator with stubbed agent adapters and state transitions.
4. Implement scoring/allocation library + unit tests for formulas.

Day 2:
1. Wire real Claude calls for clarifier + scoring agent first.
2. Implement fetcher/validator/auditor minimal prompts and output schema checks.
3. Add aggregation report generator and reviewer endpoint.
4. Run one scripted demo pitch and patch contract mismatches.

## 13) Open Items to Revisit After MVP

- Replace file storage with Postgres + object storage.
- Add auth, reviewer roles, and tamper-evident audit trails.
- Expand anti-fraud models (time-series anomaly models).
- Move from fixed ladder to portfolio-aware allocation optimizer.
