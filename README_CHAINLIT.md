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
# optional
GEMINI_MODEL=gemini-3.1-pro-preview
```

## 3) Run the app

```powershell
chainlit run app.py
```

Then open the local URL shown by Chainlit.

## Notes

- `cua/` is intentionally not touched by this app, still uses Anthropic credentials, and uses CUA-first fetch validation in `cua/data_fetcher.py`.
- Uploaded files and session artifacts are written under `data/pitches/{pitch_id}`.
- Files are uploaded from the normal chat attachment UI (no `/upload` command).
- Evaluation requires: thesis, time horizon (`days|weeks|months|years`), stock tickers, methodology summary, and source URL(s).
- `/evaluate` runs local v0 scoring rules from `MVP_SPEC.md`.
- `/validate_data "file_to_validate" "notes"` runs CUA against source URLs: CUA opens the staged reference file first, then navigates and downloads with GUI-first behavior (fallback scripts allowed when needed).
- Downloaded candidates are reviewed by an LLM against the reference file; when mismatch is detected, CUA is retried automatically with mismatch feedback.
