# Quant Pitch Evaluator

**Thesis:** Democratize quantitative trading. Let anyone — regardless of geography, pedigree, or access — submit a quantitative stock pitch and receive real capital allocation.

> "I am, somehow, less interested in the weight and convolutions of Einstein's brain than in the near certainty that people of equal talent have lived and died in cotton fields and sweatshops."

---

## What is a Quant Pitch?

A data-driven argument that a stock is mispriced, backed by quantitative evidence.

**Example:** A user discovers a correlation between wheat futures in India and McDonald's stock price. They see a drought forecast in 4 days that hasn't been priced in. They submit their data, methodology, and thesis — and our system evaluates whether the pitch is sound.

---

## Who is this for?

- Sharp traders in developing countries missed by traditional quant firm recruiting pipelines
- Quantitative thinkers who would otherwise lose money retail trading or on prop firm forex
- Anyone with a data-driven insight and no institutional access

---

## How it works

### User Flow
1. User opens the web form and talks with an interactive agent
2. The agent guides them through submitting their pitch:
   - Strategy description / thesis
   - Supporting data (CSVs, notebooks, any format)
   - Data sources and methodology
   - Time horizon (user picks: days, weeks, months)
3. Agents evaluate the pitch **in parallel**
4. Results: a score, a report, and a dollar allocation amount
5. Final step: human review

### Agent Pipeline

| Agent | Role | Runs when |
|-------|------|-----------|
| **Clarifier Agent** | Interactive Q&A — guides pitch submission, asks about data sources, clarifies assumptions, fills in gaps | During submission (interactive) |
| **Data Fetcher** | Uses Claude **Computer Use** to open a real browser, navigate to the data source URL the user provided, and download/scrape the actual data. Verifies that the user's submitted data matches the source. | After submission (async) |
| **Data Validator** | Is the data real? Check for fabricated price series, survivorship bias, look-ahead bias, data integrity | Parallel evaluation |
| **Pipeline Auditor** | Any problems with the ML/analysis pipeline? Check for overfitting, data leakage, lack of walk-forward validation, p-hacking. Uses extended thinking. | Parallel evaluation |
| **Scoring Agent** | Computes metrics: Sharpe ratio, max drawdown, risk assessment, time to return. Returns structured JSON with dollar allocation amount. | Parallel evaluation |

### Output
- Per-pitch score across key metrics
- Written evaluation report with flagged issues
- Dollar amount of capital allocated (or rejection with reasons)
- Passed to human reviewer for final decision

---

## Metrics (in progress)

- Sharpe ratio
- Max drawdown
- Risk score
- Time to return
- (More being defined by team)

---

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Backend | Python (FastAPI) | Standard, fast, async-friendly |
| Agent framework | Anthropic Agent SDK | Official multi-agent orchestration — handles handoffs, tool use, guardrails |
| Agent LLM | Claude API with **tool use + structured outputs + extended thinking** | Agents call real Python functions, return typed JSON, and reason deeply |
| Data Fetcher | Claude **Computer Use** (beta) | Agent navigates real browsers to fetch data from user-provided sources |
| Data handling | Pandas, file-based storage | |
| Frontend | Web form UI (interactive agent chat) | |
| Database | File-based / in-memory (hackathon) | |

### Claude API features we use

| Feature | Where | What it does |
|---------|-------|-------------|
| **Tool use** | All agents | Claude calls Python functions — validate CSVs, compute Sharpe, query APIs. The agent decides when to call what. |
| **Structured outputs** | Scoring Agent | Forces JSON: `{"sharpe": 1.4, "max_drawdown": -0.12, "allocation_usd": 5000}`. No parsing needed. |
| **Extended thinking** | Pipeline Auditor, Data Validator | Claude reasons step-by-step before responding. Critical for catching overfitting and data leakage. |
| **Computer Use** | Data Fetcher Agent | Claude controls a real browser — navigates to the data source URL the user provides, finds the data, downloads it. |
| **Multi-turn streaming** | Clarifier Agent | Real-time chat in the web form. |
| **Prompt caching** | All agents | Agent system prompts are long and reused. Caching cuts cost and latency. |

### Sponsor tech worth integrating

| Sponsor | How we could use it |
|---------|-------------------|
| **Anthropic (Claude)** | Core LLM for all agents + Computer Use for data fetching |
| **Hugging Face** | Pre-trained models for data validation, anomaly detection in submitted datasets |
| **Stripe / Paid** | Payment rails for capital allocation; Paid's outcome-based billing fits our model |
| **Lovable** | Rapid UI generation for the web form / pitch submission interface |
| **SIG (Susquehanna)** | Domain alignment — our product directly serves their world (quant trading) |

---

## Architecture

```
User (web form)
    │
    ▼
Clarifier Agent (interactive chat, multi-turn streaming)
    │  collects: thesis, data files, data source URLs, methodology, time horizon
    │
    ▼
┌──────────────────────────────────────────────────┐
│              Parallel Evaluation                  │
│                                                   │
│  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Data Fetcher │  │    Data Validator         │  │
│  │ (Computer    │  │    (checks integrity,     │  │
│  │  Use — real  │  │     bias, fabrication)    │  │
│  │  browser)    │  │                           │  │
│  └──────┬───────┘  └──────────────────────────┘  │
│         │                                         │
│         │ fetched data compared                   │
│         ▼ against submitted data                  │
│  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Pipeline    │  │    Scoring Agent          │  │
│  │  Auditor     │  │    (Sharpe, drawdown,     │  │
│  │  (extended   │  │     risk → $ allocation)  │  │
│  │   thinking)  │  │    [structured JSON]      │  │
│  └──────────────┘  └──────────────────────────┘  │
└──────────────────────────────────────────────────┘
    │
    ▼
Aggregated Report + Dollar Allocation
    │
    ▼
Human Review
```

---

## What needs to be built

### Phase 1: Core Backend
- [ ] FastAPI server with pitch submission endpoint
- [ ] Data ingestion — accept CSVs and other file formats, standardize into internal format
- [ ] Clarifier agent — interactive multi-turn chat that guides user through pitch submission
- [ ] Data Fetcher agent — Computer Use browser automation to fetch data from user-provided source URLs
- [ ] Data Validator agent — checks data integrity, bias, fabrication
- [ ] Pipeline Auditor agent — checks methodology for overfitting, leakage, etc. (extended thinking)
- [ ] Scoring agent — structured JSON output with Sharpe, drawdown, risk, allocation
- [ ] Aggregator — combines agent outputs into final report + allocation

### Phase 2: Frontend
- [ ] Interactive web form with agent chat interface (streaming)
- [ ] File upload (CSV, notebooks, docs)
- [ ] Results display — score, report, allocation amount

### Phase 3: Polish
- [ ] Data standardization pipeline (handle messy real-world uploads)
- [ ] Human review dashboard
- [ ] Example demo pitch (wheat/India/McDonald's scenario)
- [ ] Data Fetcher cross-checks submitted data against source data

---

## Open Questions

- Exact scoring formula and metric weights
- Data standardization format (what's the canonical internal schema?)
- Capital allocation formula (score → dollar amount mapping)
- Computer Use environment: Docker container vs. cloud VM for the browser sandbox
