# OpenQuant

*Alpha can come from anyone.*

Submit a quantitative trading strategy for structured, multi-agent evaluation.

---

## What you need

- **Thesis** — what is mispriced and why
- **Time horizon** — `days`, `weeks`, `months`, or `years`
- **Tickers** — e.g. `AAPL, MSFT`
- **Strategy file** — upload your `.py` or `.ipynb` strategy script
- **Source URLs** *(only required when uploading supporting CSV/TSV data)*

Price data is fetched internally via Alpaca — no OHLCV uploads needed.

---

## Commands

| Command | Description |
|---|---|
| `/evaluate` | Run the full validation & scoring pipeline |
| `/backtest` | Run the backtest agent on your uploaded strategy |
| `/validate_data "file" "notes"` | CUA browser validation against source URLs |
| `/status` | Show pitch completeness |
| `/checklist` | Show onboarding checklist |
| `/reset` | Start a new pitch |
| `/help` | Show all commands |

---

## Evaluation pipeline

Each run executes these agents in sequence (expandable in chat):

1. **Clarifier** — parses your pitch into structured fields
2. **Backtest Agent** — generates and executes a standardised backtest runner
3. **Fabrication Detector** — checks uploaded data for manipulation
4. **Pipeline Auditor** — reviews for look-ahead bias, leakage, and overfitting
5. **CUA Data Fetcher** — validates files against source URLs via browser automation
6. **Scoring Engine** — computes a composite score (0–100) and capital allocation (USD)

Use the attachment button to upload strategy files or supporting data.
