from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pitch_engine import PitchDraft, UploadedFile, file_sha256, validate_data_with_cua


LOGGER = logging.getLogger("cua_test")


def _setup_logging() -> Path:
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_name = os.getenv("CUA_TEST_LOG_FILE", "run_cua_tests.log").strip() or "run_cua_tests.log"
    log_path = log_dir / log_name

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    return log_path


def _resolve_reference_file() -> Path:
    raw = os.getenv("CUA_TEST_REFERENCE_FILE", "").strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path
    return REPO_ROOT / "fed_canada_gdp.csv"


def _build_uploaded_file(path: Path) -> UploadedFile:
    return UploadedFile(
        file_id=f"fil_{path.stem}",
        name=path.name,
        path=str(path),
        mime_type="text/csv",
        size_bytes=path.stat().st_size,
        sha256=file_sha256(path),
    )


def _build_fred_canada_gdp_case(reference_file: Path) -> tuple[PitchDraft, str, str]:
    pitch = PitchDraft(
        pitch_id="pit_case_cua_fred_canada_gdp",
        created_at="2026-02-21T00:00:00Z",
        status="ready",
        thesis="Validate Canada Real GDP (NGDPRXDCCAA) source data from FRED against submitted CSV.",
        time_horizon="years",
        tickers=["CAD"],
        source_urls=["https://fred.stlouisfed.org/series/NGDPRXDCCAA"],
        supporting_notes=(
            "Use Computer Use Agent to open the FRED series page for Canada Real GDP, "
            "click the CSV download button, and compare the downloaded file against the "
            "submitted reference CSV."
        ),
        uploaded_files=[_build_uploaded_file(reference_file)],
    )
    file_to_validate = reference_file.name
    notes = os.getenv("CUA_TEST_NOTES", "click the download button, it is the csv").strip() or "click the download button, it is the csv"
    return pitch, file_to_validate, notes


def _print_result(result: dict) -> None:
    summary = {
        "agent": result.get("agent"),
        "status": result.get("status"),
        "confidence": result.get("confidence"),
        "summary": result.get("summary"),
        "flags": result.get("flags"),
        "latency_ms": result.get("latency_ms"),
        "downloaded_files": (result.get("artifacts") or {}).get("downloaded_files", []),
        "match_review": (result.get("artifacts") or {}).get("match_review", {}),
        "attempt_history": (result.get("artifacts") or {}).get("attempt_history", []),
    }
    print(json.dumps(summary, indent=2))
    LOGGER.info("Result summary: %s", json.dumps(summary, ensure_ascii=True))


def _assertions(result: dict) -> tuple[bool, list[str]]:
    checks: list[str] = []
    passed = True

    status = str(result.get("status", "")).lower()
    checks.append(f"status={status}")
    allow_warn = os.getenv("CUA_TEST_ALLOW_WARN", "0").strip().lower() in {"1", "true", "yes", "on"}
    allowed_statuses = {"ok", "warn"} if allow_warn else {"ok"}
    if status not in allowed_statuses:
        passed = False
        checks.append(
            f"unexpected status from data_fetcher output (expected one of {sorted(allowed_statuses)})"
        )

    artifacts = result.get("artifacts", {}) if isinstance(result.get("artifacts"), dict) else {}
    attempts = artifacts.get("attempt_history", []) if isinstance(artifacts.get("attempt_history"), list) else []
    checks.append(f"attempts={len(attempts)}")
    if len(attempts) == 0:
        passed = False
        checks.append("no CUA attempt history captured")

    downloaded = artifacts.get("downloaded_files", []) if isinstance(artifacts.get("downloaded_files"), list) else []
    checks.append(f"downloaded_files={len(downloaded)}")
    if len(downloaded) == 0:
        passed = False
        checks.append("CUA did not download any source artifacts")

    staged_reference = str(artifacts.get("staged_reference", "")).strip().lower()
    match_review = artifacts.get("match_review", {}) if isinstance(artifacts.get("match_review"), dict) else {}
    verdict = str(match_review.get("verdict", "")).strip().lower()
    checks.append(f"match_verdict={verdict or '(missing)'}")
    if verdict != "match":
        passed = False
        checks.append("match_review verdict is not 'match'")

    best_candidate = str(match_review.get("best_candidate", "")).strip().lower()
    if staged_reference and best_candidate and best_candidate == staged_reference:
        passed = False
        checks.append("best candidate equals staged reference file (invalid self-match)")

    return passed, checks


def main() -> int:
    log_path = _setup_logging()
    LOGGER.info("Starting CUA test runner")
    LOGGER.info("Log file: %s", log_path)

    load_dotenv()
    LOGGER.info("Loaded .env")

    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        LOGGER.error("ANTHROPIC_API_KEY is missing")
        print("ANTHROPIC_API_KEY is missing. Add it to .env, then rerun this script.")
        return 1

    reference_file = _resolve_reference_file()
    LOGGER.info("Resolved reference file: %s", reference_file)
    if not reference_file.exists():
        LOGGER.error("Reference file does not exist: %s", reference_file)
        print(
            "Reference file is missing. Set CUA_TEST_REFERENCE_FILE or place the file at: "
            f"{reference_file}"
        )
        return 1

    pitch, file_to_validate, notes = _build_fred_canada_gdp_case(reference_file)
    LOGGER.info("Built test pitch: %s", pitch.pitch_id)
    LOGGER.info("Source URL: %s", pitch.source_urls[0])
    LOGGER.info("Notes: %s", notes)
    LOGGER.info("File to validate: %s", file_to_validate)

    print("\n=== Running CUA FRED Canada GDP test case ===")
    print(f"reference_file={reference_file}")
    print(f"source_url={pitch.source_urls[0]}")
    print(f"notes={notes}")

    LOGGER.info("Invoking validate_data_with_cua")
    result = validate_data_with_cua(
        draft=pitch,
        file_to_validate=file_to_validate,
        notes=notes,
    )
    LOGGER.info("validate_data_with_cua completed with status=%s", result.get("status"))

    _print_result(result)
    passed, checks = _assertions(result)
    LOGGER.info("Assertions completed. passed=%s checks=%s", passed, json.dumps(checks, ensure_ascii=True))

    print("\n=== Checks ===")
    for check in checks:
        print(f"- {check}")

    if not passed:
        LOGGER.error("CUA TEST RESULT: FAIL")
        print("\nCUA TEST RESULT: FAIL")
        return 2

    LOGGER.info("CUA TEST RESULT: PASS")
    print("\nCUA TEST RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

