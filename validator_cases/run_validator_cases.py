from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pitch_engine import PitchDraft, UploadedFile, evaluate_pitch, file_sha256, validate_data_with_cua

DATA_DIR = ROOT / "data"


def make_uploaded_file(path: Path) -> UploadedFile:
    return UploadedFile(
        file_id=f"fil_{path.stem}",
        name=path.name,
        path=str(path),
        mime_type="text/csv",
        size_bytes=path.stat().st_size,
        sha256=file_sha256(path),
    )


def build_cases() -> list[PitchDraft]:
    malicious_csv = DATA_DIR / "malicious_prices.csv"
    common_csv = DATA_DIR / "common_error_prices.csv"
    clean_csv = DATA_DIR / "clean_momentum_prices.csv"

    malicious = PitchDraft(
        pitch_id="pit_case_malicious",
        created_at="2026-02-21T00:00:00Z",
        status="ready",
        thesis="Short-term directional edge from engineered return-pattern signal.",
        time_horizon="days",
        tickers=["AAPL"],
        source_urls=["https://example.com/aapl-history"],
        methodology_summary=(
            "We denoise the signal by creating a direction feature from one-step shifted returns, "
            "then trade with a one-day lag. Validation is based on historical holdout windows."
        ),
        uploaded_files=[make_uploaded_file(malicious_csv)],
    )

    common_error = PitchDraft(
        pitch_id="pit_case_common_error",
        created_at="2026-02-21T00:00:00Z",
        status="ready",
        thesis="Predict next-day direction using momentum and realized volatility features.",
        time_horizon="weeks",
        tickers=["MSFT"],
        source_urls=["https://example.com/msft-history"],
        methodology_summary=(
            "Features (mom_5, mom_10, vol_10) are standardized using StandardScaler then fed into "
            "logistic regression. We use an 80/20 time-ordered split; accuracy and Sharpe are "
            "computed on the held-out 20%."
        ),
        uploaded_files=[make_uploaded_file(common_csv)],
    )
    clean = PitchDraft(
        pitch_id="pit_case_clean",
        created_at="2026-02-21T00:00:00Z",
        status="ready",
        thesis=(
            "Daily momentum signal on a simulated ETF using 21-day and 63-day return signals. "
            "Logistic regression predicts next-day direction; long/flat based on predicted probability. "
            "1bp transaction cost assumed per round-trip."
        ),
        time_horizon="days",
        tickers=["BACKTST"],
        source_urls=["https://example.com/backtst-simulated-history"],
        methodology_summary=(
            "Simulated ETF (BACKTST), ~2 years of daily data. Features: 21-day momentum (mom_21), "
            "63-day momentum (mom_63), and 21-day realized volatility (rvol_21), all computed from "
            "lagged closes (t-1 anchor, no look-ahead). First 63 warmup rows dropped before fitting. "
            "StandardScaler fitted on training rows only (75% time-ordered split) then applied to "
            "the held-out test set. Logistic regression (C=1.0) trained on training window; "
            "next-day direction predicted and position sized ±1 on each trading day. "
            "Annualized Sharpe and accuracy reported on the 25% OOS walk-forward window. "
            "Transaction cost: 1bp per round-trip on each daily trade."
        ),
        uploaded_files=[make_uploaded_file(clean_csv)],
    )
    return [malicious, common_error, clean]


def summarize(result: dict) -> dict:
    agent_outputs = result["agent_outputs"]
    return {
        "pitch_id": result["pitch_id"],
        "validation_outcome": result["validation_outcome"],
        "decision": result["decision"],
        "data_fetcher_status": agent_outputs["data_fetcher"]["status"],
        "data_fetcher_flags": agent_outputs["data_fetcher"]["flags"],
        "data_validator_flags": agent_outputs["data_validator"]["flags"],
        "pipeline_auditor_flags": agent_outputs["pipeline_auditor"]["flags"],
        "validation_questions": result["validation_questions"],
    }


def _stub_cua_output(case: PitchDraft) -> dict:
    match_rate_raw = os.getenv("VALIDATOR_CASES_STUB_MATCH_RATE", "0.9")
    try:
        match_rate = float(match_rate_raw)
    except ValueError:
        match_rate = 0.9
    match_rate = max(0.0, min(1.0, match_rate))

    return {
        "agent": "data_fetcher",
        "status": "ok",
        "confidence": 0.9,
        "summary": "Stubbed CUA output for validator-case pipeline tests.",
        "flags": [],
        "artifacts": {
            "match_rate": match_rate,
            "source": "validator_cases_stub",
            "validated_file_names": [entry.name for entry in case.uploaded_files],
        },
        "latency_ms": 1,
    }


def evaluate_case(case: PitchDraft):
    mode = os.getenv("VALIDATOR_CASES_CUA_MODE", "stub").strip().lower()
    if mode == "none":
        return evaluate_pitch(case)

    if mode == "real":
        csv_like = [f for f in case.uploaded_files if Path(f.path).suffix.lower() in {".csv", ".tsv"}]
        if not csv_like:
            fetcher_output = {
                "agent": "data_fetcher",
                "status": "fail",
                "confidence": 0.0,
                "summary": "No CSV/TSV uploaded for CUA validation.",
                "flags": [{"code": "CUA_DATA_FILES_MISSING", "message": "No CSV/TSV files to validate."}],
                "artifacts": {"match_rate": 0.0},
                "latency_ms": 0,
            }
        else:
            fetcher_output = validate_data_with_cua(
                draft=case,
                file_to_validate=csv_like[0].name,
                notes="validator_cases real CUA run",
            )
        return evaluate_pitch(case, data_fetcher_output=fetcher_output)

    return evaluate_pitch(case, data_fetcher_output=_stub_cua_output(case))


def main() -> int:
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is missing. Add it to .env, then rerun this script.")
        return 1

    for case in build_cases():
        print(f"\n=== Running {case.pitch_id} ===")
        result = evaluate_case(case)
        print(json.dumps(summarize(result.to_dict()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
