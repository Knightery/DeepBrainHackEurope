# Quant Pitch Evaluator

Submit a quantitative trading strategy for structured, multi-agent evaluation.

## Required Fields

- **Thesis** -- what is mispriced and why
- **Time horizon** -- `days`, `weeks`, `months`, or `years`
- **Stock tickers** -- e.g. `AAPL, MSFT`
- **Source URLs** -- provenance links for all submitted data
- **Strategy files** -- upload your `.py` or `.ipynb` strategy script and/or `.csv` price data

## Commands

- `/status` -- show pitch completeness
- `/checklist` -- show onboarding checklist
- `/evaluate` -- run full validation and scoring pipeline (also used to re-run after clarifications)
- `/backtest` -- run Claude backtest agent on uploaded `.py` or `.ipynb` strategy
- `/validate_data "file" "notes"` -- run CUA browser validation against source URLs
- `/reset` -- start a new pitch
- `/help` -- show commands

## Pipeline Agents

Each evaluation runs these agents (visible as expandable steps in the chat):

1. **Clarifier Agent** -- parses your natural-language pitch into structured fields
2. **Backtest Agent** -- generates and executes a standardised backtest runner for `.py` or `.ipynb` strategies
3. **Fabrication Detector** -- checks uploaded data for manipulation or fabrication
4. **Pipeline Auditor** -- reviews methodology for look-ahead bias, leakage, overfitting
5. **CUA Data Fetcher** -- validates uploaded files against source URLs via browser automation
6. **Scoring Engine** -- computes composite score (0-100) and capital allocation (USD)

Upload files using the attachment button in the chat input.
