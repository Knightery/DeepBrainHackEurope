# Chainlit App Setup

## 1) Install dependencies

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Configure environment

`.env` must include:

```env
GEMINI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
PAID_API_KEY=your_key_here
# optional
GEMINI_MODEL=gemini-3.1-pro-preview
ANTHROPIC_MODEL=claude-opus-4-5-20251101
BACKTEST_TIMEOUT_SECONDS=1200
BACKTEST_MAX_ATTEMPTS=3
PAID_EVENT_NAME=eva_by_anyquant
PAID_EXTERNAL_PRODUCT_ID=quant_pitch_evaluator
PAID_USAGE_ENABLED=true
```

> `ANTHROPIC_API_KEY` is required for both the backtest agent (in-process) and the CUA data fetcher (Docker). If absent, `/backtest` will be unavailable and strategy scoring cannot complete.
> `PAID_API_KEY` enables Paid usage tracking; when it is missing, usage signal emission is skipped.

## 3) Run the app

```powershell
chainlit run app.py
```

Then open the local URL shown by Chainlit.

## Notes

- `cua/` is intentionally not touched by this app, still uses Anthropic credentials, and uses CUA-first fetch validation in `cua/data_fetcher.py`.
- Uploaded files and session artifacts are written under `data/pitches/{pitch_id}`.
- Files are uploaded from the normal chat attachment UI (no `/upload` command).
- Evaluation requires: thesis, time horizon (`days|weeks|months|years`), stock tickers, and an uploaded strategy file (`.py` or `.ipynb`).
- Source URL(s) are required only when you upload supporting CSV/TSV data files.
- Price CSV uploads are not required for strategy notebooks/scripts; the backtest agent fetches market prices via Alpaca.
- `/evaluate` uses strategy-scorer-based scoring from `strategy_scorer.py` (no CSV approximation fallback).
- For `.py`/`.ipynb` strategy pitches, Fabrication Detector and Pipeline Auditor are executed in parallel.
- `/backtest` runs the Claude backtest agent on any uploaded `.py` or `.ipynb` strategy file. Notebook uploads are compiled into linear `.py` scripts first. Claude then generates a standardised runner, executes it in a temp directory, and validates the output over up to 3 attempts. Results are saved to `data/pitches/{pitch_id}/agent_outputs/backtest_agent.json`.
- `/validate_data "file_to_validate" "notes"` runs CUA against source URLs: CUA opens the staged reference file first, then navigates and downloads with GUI-first behavior (fallback scripts allowed when needed).
- Downloaded candidates are reviewed by an LLM against the reference file; when mismatch is detected, CUA is retried automatically with mismatch feedback.
