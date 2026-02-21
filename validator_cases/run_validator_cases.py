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

from pitch_engine import PitchDraft, UploadedFile, evaluate_pitch, file_sha256

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
            "We normalize all features before splitting into train/test and then train logistic regression. "
            "The split is 80/20 on time order, with accuracy measured on the final segment."
        ),
        uploaded_files=[make_uploaded_file(common_csv)],
    )
    return [malicious, common_error]


def summarize(result: dict) -> dict:
    agent_outputs = result["agent_outputs"]
    return {
        "pitch_id": result["pitch_id"],
        "validation_outcome": result["validation_outcome"],
        "decision": result["decision"],
        "data_validator_flags": agent_outputs["data_validator"]["flags"],
        "pipeline_auditor_flags": agent_outputs["pipeline_auditor"]["flags"],
        "validation_questions": result["validation_questions"],
    }


def main() -> int:
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is missing. Add it to .env, then rerun this script.")
        return 1

    for case in build_cases():
        print(f"\n=== Running {case.pitch_id} ===")
        result = evaluate_pitch(case)
        print(json.dumps(summarize(result.to_dict()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

