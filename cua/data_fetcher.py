"""
Data Fetcher Agent — uses Anthropic's official Computer Use demo
to navigate a real browser, go to a URL, and fetch/download data.

Runs inside Anthropic's official computer-use-demo Docker image.
Uses their tool implementations (screenshot, click, type, etc.)
so we don't maintain any of that ourselves.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from computer_use_demo.loop import sampling_loop, APIProvider
from computer_use_demo.tools import ToolResult

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

DOWNLOADS_DIR = Path("/home/computeruse/Downloads")

ALLOWED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".zip"}
DISALLOWED_EXTENSIONS = {".html", ".htm"}

MODEL_TO_TOOL_VERSION = {
    "claude-sonnet-4-5-20250929": "computer_use_20250124",
    "claude-haiku-4-5-20251001": "computer_use_20250124",
}

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

SYSTEM_PROMPT = """You are a Data Fetcher agent for a quant pitch evaluation platform.
Your job is to use COMPUTER-USE GUI actions to fetch the exact downloadable source file.

Hard constraints:
1. GUI-only execution: use browser/computer actions only.
2. Do NOT use bash, curl, wget, python, or any terminal/script workaround.
3. Navigate the real page UI, find the download control, and click it.
4. Save downloaded files to ~/Downloads/.
5. Prefer canonical data artifacts (.csv, .tsv, .xlsx, .xls, .json, .zip).
6. If an exact filename is requested, only finish after that exact filename appears in ~/Downloads/.
7. If blocked by cookies/modals, dismiss them in the GUI and continue.

Completion response format (plain text):
- Download status: success|failed
- Exact filename(s) downloaded
- Short trace of GUI steps you performed
"""


async def run_data_fetcher(url: str, description: str = "", expected_filename: str = "") -> dict:
    """
    Main entry point. Takes a URL and optional description,
    uses Anthropic's sampling_loop to drive Computer Use.
    """
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    tool_version = os.environ.get("CUA_TOOL_VERSION", MODEL_TO_TOOL_VERSION.get(model, "computer_use_20250124"))

    task = f"Navigate to this URL and fetch the data via GUI-only browser actions:\n\nURL: {url}\n"
    if description:
        task += f"Description: {description}\n"
    if expected_filename:
        task += f"Expected exact filename: {expected_filename}\n"
    task += (
        "\nRequirements:\n"
        "- Do not use bash/terminal/python/curl/wget.\n"
        "- Click through the page UI and download the real source artifact.\n"
        "- If expected filename is provided, only complete when that exact file exists in ~/Downloads/.\n"
    )

    existing_files = {f.resolve() for f in DOWNLOADS_DIR.glob("*") if f.is_file()}

    messages = [{"role": "user", "content": task}]

    print(f"[CUA] Starting data fetch from: {url}")
    print(f"[CUA] Model: {model} | Tool version: {tool_version}")
    if expected_filename:
        print(f"[CUA] Expected filename: {expected_filename}")

    # Use Anthropic's official sampling loop — it handles
    # screenshots, clicking, typing, bash, text editor, coordinate
    # scaling, and the full agent loop automatically.
    response_messages = await sampling_loop(
        model=model,
        provider=APIProvider.ANTHROPIC,
        system_prompt_suffix=SYSTEM_PROMPT,
        messages=messages,
        output_callback=lambda msg: print(f"[CUA] {_format_output(msg)}"),
        tool_output_callback=lambda result, id: print(f"[CUA] Tool result: {_format_tool_result(result)}"),
        api_response_callback=lambda request, response, error: None,
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        max_tokens=4096,
        tool_version=tool_version,
    )

    # Extract final text from Claude's last message
    final_text = ""
    for msg in reversed(response_messages):
        if msg["role"] == "assistant":
            if isinstance(msg["content"], str):
                final_text = msg["content"]
            elif isinstance(msg["content"], list):
                for block in msg["content"]:
                    if hasattr(block, "text"):
                        final_text += block.text
                    elif isinstance(block, dict) and block.get("type") == "text":
                        final_text += block.get("text", "")
            break

    tool_events = _extract_tool_events(response_messages)
    downloaded = [f for f in DOWNLOADS_DIR.glob("*") if f.is_file()]
    new_downloaded = [f for f in downloaded if f.resolve() not in existing_files]

    validation = _validate_run(
        tool_events=tool_events,
        downloaded_files=new_downloaded,
        expected_filename=expected_filename,
    )

    return {
        "status": validation["status"],
        "summary": final_text,
        "downloaded_files": [str(f) for f in new_downloaded],
        "validation": validation,
    }


def _format_output(msg) -> str:
    """Format a message for logging."""
    if hasattr(msg, "text"):
        return f"Claude: {msg.text[:200]}"
    if hasattr(msg, "type") and msg.type == "tool_use":
        return f"Action: {msg.name} {json.dumps(msg.input)[:100]}"
    return str(msg)[:200]


def _format_tool_result(result: ToolResult) -> str:
    """Format tool result for logging."""
    if result.error:
        return f"ERROR: {result.error[:200]}"
    if result.output:
        return result.output[:200]
    if result.base64_image:
        return "[screenshot captured]"
    return "[empty result]"


def _extract_tool_events(response_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for msg in response_messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue

        for block in content:
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")
            if block_type != "tool_use":
                continue

            name = getattr(block, "name", None)
            if name is None and isinstance(block, dict):
                name = block.get("name")

            tool_input = getattr(block, "input", None)
            if tool_input is None and isinstance(block, dict):
                tool_input = block.get("input", {})

            events.append(
                {
                    "name": str(name or ""),
                    "input": tool_input if isinstance(tool_input, dict) else {},
                }
            )
    return events


def _validate_run(
    tool_events: list[dict[str, Any]],
    downloaded_files: list[Path],
    expected_filename: str,
) -> dict[str, Any]:
    used_bash = any(event.get("name") == "bash" for event in tool_events)
    used_gui_tool = any(event.get("name") == "computer" for event in tool_events)
    clicked = any(
        event.get("name") == "computer" and event.get("input", {}).get("action") in {"left_click", "double_click"}
        for event in tool_events
    )

    allowed_ext = _parse_extension_set_from_env("CUA_ALLOWED_EXTENSIONS", ALLOWED_EXTENSIONS)
    disallowed_ext = _parse_extension_set_from_env("CUA_DISALLOWED_EXTENSIONS", DISALLOWED_EXTENSIONS)

    allowed_downloads = [
        file_path
        for file_path in downloaded_files
        if file_path.suffix.lower() in allowed_ext and file_path.suffix.lower() not in disallowed_ext
    ]

    issues: list[dict[str, str]] = []

    if used_bash:
        issues.append({"code": "GUI_ONLY_VIOLATION", "message": "Run used bash/terminal tools, which are forbidden."})

    if not used_gui_tool:
        issues.append({"code": "NO_GUI_ACTIONS", "message": "No computer-use GUI actions were detected."})

    if not clicked:
        issues.append({"code": "NO_CLICK_ACTION", "message": "No GUI click action detected; download click was not verified."})

    if not downloaded_files:
        issues.append({"code": "NO_NEW_DOWNLOAD", "message": "No new file was downloaded into ~/Downloads/."})

    if downloaded_files and not allowed_downloads:
        issues.append(
            {
                "code": "INVALID_FILE_TYPE",
                "message": "Downloaded files do not match allowed data artifact extensions.",
            }
        )

    expected_name = expected_filename.strip()
    matched_expected = None
    if expected_name:
        for file_path in allowed_downloads:
            if file_path.name == expected_name:
                matched_expected = file_path
                break
        if matched_expected is None:
            issues.append(
                {
                    "code": "EXPECTED_FILE_NOT_FOUND",
                    "message": f"Expected exact filename '{expected_name}' was not downloaded.",
                }
            )

    status = "success" if not issues else "fail"
    selected_files = [str(matched_expected)] if matched_expected else [str(path) for path in allowed_downloads]

    return {
        "status": status,
        "issues": issues,
        "allowed_extensions": sorted(allowed_ext),
        "disallowed_extensions": sorted(disallowed_ext),
        "selected_files": selected_files,
        "tool_usage": {
            "used_bash": used_bash,
            "used_gui_tool": used_gui_tool,
            "click_detected": clicked,
            "tool_count": len(tool_events),
        },
    }


def _parse_extension_set_from_env(key: str, default: set[str]) -> set[str]:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return set(default)
    parsed: set[str] = set()
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if not token.startswith("."):
            token = f".{token}"
        parsed.add(token)
    return parsed or set(default)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_fetcher.py <url> [description] [expected_filename]")
        print(
            "Example: python data_fetcher.py "
            "'https://www.netflix.com/tudum/top10' "
            "'Fetch global top 10 lists from download tab' "
            "'NetflixTop10.xlsx'"
        )
        sys.exit(1)

    url = sys.argv[1]
    desc = sys.argv[2] if len(sys.argv) > 2 else ""
    expected = sys.argv[3] if len(sys.argv) > 3 else ""

    result = asyncio.run(run_data_fetcher(url, desc, expected))
    print(f"\n{'=' * 60}")
    print(json.dumps(result, indent=2))
