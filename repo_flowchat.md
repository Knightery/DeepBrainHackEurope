# ✨ OpenQuant Repository Flowchart

> A simple, visual guide to how this repo works end-to-end.

---

## 🧭 What this shows

This page explains the **full product flow** in plain English:

1. How a user submits an idea
2. How the system validates and tests it
3. How a final decision is produced
4. Where results are stored and shown

---

## 🗺️ High-level phases

| Phase | What happens | Why it matters |
|---|---|---|
| 1. Intake | User enters thesis, tickers, horizon, and uploads files | Ensures minimum required information |
| 2. Validation | CUA checks uploaded data against public sources | Prevents fake or mismatched datasets |
| 3. Analysis | Backtest + fraud check + methodology review | Measures quality and catches risks |
| 4. Decision | Score/recommendation is produced | Creates a clear final outcome |
| 5. Persistence | Results saved to DB/files and shown in dashboard | Keeps history auditable and visible |

---

## 🔁 End-to-end flow diagram

```mermaid
flowchart TD
    U[👤 User / Analyst] --> UI[💬 Chat App Interface]
    U --> DASH[📊 Read-only Dashboard & API]

    UI --> START[🚀 Session starts]
    START --> ONBOARD[🧩 2-step onboarding\n1) Strategy vs One-shot\n2) Time horizon]
    ONBOARD --> CLAR[🤖 Clarifier Agent\nExtracts thesis, tickers, URLs, notes]
    CLAR --> FILES[📎 User uploads files\n.py / .ipynb / .csv / .tsv]

    FILES --> CHECK{✅ All required inputs complete?}
    CHECK -- No --> FOLLOWUP[📝 Ask follow-up questions\nand update checklist]
    FOLLOWUP --> CLAR
    CHECK -- Yes --> AUTO[⚙️ Auto-run evaluation\n(or user runs commands)]

    AUTO --> CMD{🧭 Entry path}
    CMD -->|/evaluate or auto| EVAL[🧪 Run full evaluation pipeline]
    CMD -->|/backtest| BT_ONLY[📈 Run backtest agent only]
    CMD -->|/validate_data| CUA_ONLY[🌐 Run CUA check for one file]

    CUA_ONLY --> CUA_SINGLE[🖥️ CUA Docker run\nBrowser-first source validation]
    CUA_SINGLE --> SAVE_CUA[💾 Save data-fetcher output]

    BT_ONLY --> ROLE_BT[🗂️ Classify uploaded files\n(strategy vs data vs benchmark)]
    ROLE_BT --> BT_AGENT[🧠 Backtest Agent (3-attempt loop)\nCreate/Fix -> Run -> Review]
    BT_AGENT --> BT_OUT{📍 Backtest status}
    BT_OUT -->|success| BT_METRICS[✅ Metrics + strategy_scorer-ready output]
    BT_OUT -->|user_action_required| BT_FIX[🛠️ Ask user to fix strategy/data]
    BT_OUT -->|agent_fault| BT_FAIL[❌ Backtest failed]

    EVAL --> CUA_ALL[🛡️ Mandatory CUA checks\nfor every uploaded CSV/TSV\nin parallel containers]
    CUA_ALL --> CUA_MERGE[🧷 Merge provenance results\ninto one data-fetcher output]

    CUA_MERGE --> CORE[🧰 Core evaluator engine]
    CORE --> ROLES[🧠 File role classifier\n(content-aware)]

    ROLES --> BACKTEST{📄 Strategy script present?}
    BACKTEST -- Yes --> BT_PIPE[📈 Backtest Agent + strategy_scorer]
    BACKTEST -- No --> NO_BT[↪️ Skip backtest\n(use available data context)]

    ROLES --> VALS[⚖️ Run two validators in parallel]
    VALS --> FAB[🕵️ Fabrication Detector\n(intentional manipulation check)]
    VALS --> AUD[🔍 Pipeline Auditor\n(methodology/coding risk check)]

    ROLES --> ONESHOT{⚡ One-shot mode enabled?}
    ONESHOT -- Yes --> OSV[🎯 One-shot Validator\n(event-based statistical checks)]
    ONESHOT -- No --> STD[📐 Standard scoring path]

    BT_PIPE --> SCORE
    NO_BT --> SCORE
    FAB --> SCORE
    AUD --> SCORE
    OSV --> SCORE
    STD --> SCORE

    SCORE[🏁 Scoring & decision engine\nCombine strategy score + data quality + method + source match]
    SCORE --> DECIDE{📣 Validation outcome}

    DECIDE -->|blocked_fabrication| REJECT[⛔ Reject pitch\nUser message: Goodbye.]
    DECIDE -->|needs_clarification| ASK[❓ Return clear issues/questions\nfor user clarification]
    DECIDE -->|ready_for_final_review| READY[✅ Show scorecard\nDecision + allocation]

    ASK --> CLAR

    REJECT --> STORE
    READY --> STORE
    ASK --> STORE

    STORE[🗃️ Persist artifacts\nPitch snapshot, agent outputs, result report]
    STORE --> DB[(🧱 SQLite pitch database)]
    STORE --> FS[(📁 Pitch files on disk)]

    DB --> DASH

    TRACK[📡 Paid usage tracker\nasync telemetry events] -.-> EXT_PAID[(Paid API)]
    UI -. usage events .-> TRACK

    GEM[Gemini models\nClarifier + validators + classification\n+ one-shot extraction + match review]
    CLAUDE[Anthropic models\nBacktest generation/review + CUA computer use]
    MARKET[Market data providers\nAlpaca + Solana xStocks skill]

    CLAR -.-> GEM
    FAB -.-> GEM
    AUD -.-> GEM
    ROLES -.-> GEM
    OSV -.-> GEM
    CUA_MERGE -.-> GEM

    BT_AGENT -.-> CLAUDE
    CUA_SINGLE -.-> CLAUDE
    CUA_ALL -.-> CLAUDE

    BT_AGENT -. fetch prices .-> MARKET

    subgraph Ops_and_Dev_Tools[🛠️ Ops & Developer Workflows]
      TESTS[Automated tests]
      CASES[Validator case runners\n& CUA smoke scripts]
      NOTEBOOKS[Research notebooks\n(backtesting/ML experiments)]
      TESTS --> CORE
      CASES --> CORE
      NOTEBOOKS --> CORE
    end
```

---

## 🧩 Quick legend

- **Solid arrows** = main workflow path
- **Dashed arrows** = external services or telemetry links
- **Diamonds** = decision points
- **Cylinders** = storage layers (database/files)

---

## ✅ Outcome summary (plain English)

- The system helps users provide complete investment ideas.
- It automatically checks data authenticity and technical quality.
- It runs strategy backtests when needed.
- It returns one clear outcome: reject, clarify, or ready for final review.
- Everything is saved for transparency and shown in the dashboard/API.
