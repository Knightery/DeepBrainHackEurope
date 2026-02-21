from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CUA_DIR = REPO_ROOT / "cua"


def _extract_json_after_separator(raw_output: str) -> dict | None:
    if not raw_output.strip():
        return None

    candidate = raw_output
    if "=" * 20 in raw_output:
        candidate = raw_output.split("=" * 20)[-1]

    candidate = candidate.strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def main() -> int:
    load_dotenv()

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not anthropic_key:
        print("ANTHROPIC_API_KEY is missing. Add it to .env, then rerun this script.")
        return 1

    anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929").strip()
    if not anthropic_model:
        anthropic_model = "claude-sonnet-4-5-20250929"

    url = os.getenv("CUA_BASIC_TEST_URL", "https://www.netflix.com/tudum/top10").strip()
    notes = os.getenv("CUA_BASIC_TEST_NOTES", "click the download button").strip() or "click the download button"
    timeout_seconds = int(os.getenv("CUA_BASIC_TEST_TIMEOUT_SECONDS", "240"))

    print("=== Running basic direct CUA test (Anthropic) ===")
    print(f"url={url}")
    print(f"notes={notes}")
    print(f"anthropic_model={anthropic_model}")
    print(f"cua_dir={CUA_DIR}")

    env = os.environ.copy()
    env["ANTHROPIC_MODEL"] = anthropic_model

    command = [
        "docker",
        "compose",
        "run",
        "--rm",
        "--remove-orphans",
        "data-fetcher",
        url,
        notes,
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=CUA_DIR,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(60, timeout_seconds),
            env=env,
        )
    except subprocess.TimeoutExpired:
        print(f"FAIL: CUA test timed out after {timeout_seconds}s")
        return 2
    except Exception as exc:
        print(f"FAIL: Could not run Docker CUA test: {exc.__class__.__name__}")
        return 2

    raw_output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    parsed = _extract_json_after_separator(raw_output)

    model_line_detected = "[CUA] Model:" in raw_output
    claude_detected = "claude-" in raw_output.lower()

    print("\n=== Docker invocation ===")
    print(f"exit_code={completed.returncode}")
    print(f"model_line_detected={model_line_detected}")
    print(f"claude_detected={claude_detected}")

    if not parsed:
        print("FAIL: Could not parse JSON result from CUA output.")
        tail = raw_output[-2500:]
        print("--- output tail ---")
        print(tail)
        return 2

    validation = parsed.get("validation", {}) if isinstance(parsed.get("validation"), dict) else {}
    tool_usage = validation.get("tool_usage", {}) if isinstance(validation.get("tool_usage"), dict) else {}
    tool_count = int(tool_usage.get("tool_count", 0) or 0)
    used_gui_tool = bool(tool_usage.get("used_gui_tool", False))
    used_bash = bool(tool_usage.get("used_bash", False))

    summary = {
        "status": parsed.get("status"),
        "downloaded_files": parsed.get("downloaded_files", []),
        "tool_usage": tool_usage,
    }

    print("\n=== Parsed CUA summary ===")
    print(json.dumps(summary, indent=2))

    checks: list[str] = []
    passed = True

    if completed.returncode != 0:
        passed = False
        checks.append(f"docker compose exit_code != 0 ({completed.returncode})")

    if not model_line_detected:
        passed = False
        checks.append("missing [CUA] Model line in output")

    if not claude_detected:
        passed = False
        checks.append("no claude model identifier detected in output")

    if tool_count <= 0:
        passed = False
        checks.append("no tool activity detected (tool_count <= 0)")

    if not used_gui_tool and not used_bash:
        passed = False
        checks.append("neither GUI nor bash tools were used")

    print("\n=== Checks ===")
    print(f"- tool_count={tool_count}")
    print(f"- used_gui_tool={used_gui_tool}")
    print(f"- used_bash={used_bash}")
    if checks:
        for item in checks:
            print(f"- {item}")

    if not passed:
        print("\nBASIC CUA TEST RESULT: FAIL")
        return 2

    print("\nBASIC CUA TEST RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
