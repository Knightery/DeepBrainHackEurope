"""Run a single named test case from run_validator_cases.py."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from pitch_engine import evaluate_pitch
from run_validator_cases import build_cases, summarize


def main() -> int:
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY missing")
        return 1

    case_id = sys.argv[1] if len(sys.argv) > 1 else "pit_case_clean"
    cases = {c.pitch_id: c for c in build_cases()}
    case = cases.get(case_id)
    if not case:
        print(f"Unknown case. Available: {list(cases)}")
        return 1

    print(f"\n=== Running {case.pitch_id} ===")
    result = evaluate_pitch(case)
    print(json.dumps(summarize(result.to_dict()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
