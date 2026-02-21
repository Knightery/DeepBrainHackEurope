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
from dotenv import load_dotenv

load_dotenv()

DOWNLOADS_DIR = Path("/home/computeruse/Downloads")

MODEL_TO_TOOL_VERSION = {
    # Claude 4.6 / 4.5 Opus — require computer-use-2025-11-24 beta header
    "claude-sonnet-4-6": "computer_use_20251124",
    "claude-opus-4-6": "computer_use_20251124",
    "claude-opus-4-5-20251101": "computer_use_20251124",
    # Claude Sonnet / Haiku 4.5
    "claude-sonnet-4-5-20250929": "computer_use_20250124",
    "claude-haiku-4-5-20251001": "computer_use_20250124",
}

DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a Data Fetcher agent running inside a real Linux desktop environment (Anthropic computer-use-demo image).
You have full computer control. The exact tools available to you are:

  computer (use for ALL GUI actions):
    screenshot                        — capture the current screen state; use this liberally after EVERY action to see what changed
    left_click   {x, y}               — click a button, link, or UI element
    double_click {x, y}               — double-click a file or icon
    right_click  {x, y}               — open a context menu
    type         {text}               — type text into the focused input field
    key          {key}                — press a key or combo: Return, ctrl+t, ctrl+l, ctrl+w, Escape, Tab, etc.
    scroll       {x, y, direction}    — scroll up/down/left/right
    mouse_move   {x, y}               — hover without clicking
    get_page_text                     — extract ALL visible rendered text from the current browser page;
                                        use this to read page content, find download links, confirm data, and
                                        avoid guessing from a screenshot when text is what matters

  bash                                — run shell commands (cat, head, wget, curl, python3, ls, mv, etc.)
                                        use only as a last resort when the GUI path is truly blocked

  text_editor                         — read and write local files directly

Required workflow — follow this order:
1. Take a screenshot first to see the current state of the desktop.
2. Inspect the reference file: bash → head -20 ~/Downloads/<reference_file>
   Understand its columns, date range, granularity, and entity names.
3. Open Firefox (it is already running). Focus the address bar with key ctrl+l,
   type the target URL, press key Return.
4. Take a screenshot to confirm the page loaded.
5. Use get_page_text to read the full page text and find the download or export link/button.
6. Click the download button or link. Take a screenshot to confirm the file download started.
7. Wait for the download to complete; verify the file appeared in ~/Downloads/ with bash if needed.
8. Confirm the downloaded file matches the reference schema (columns, granularity, entities).
9. If a cookie banner or modal blocks the page, dismiss it first, then continue from step 5.
10. Only fall back to bash/curl/wget if the GUI path is clearly impossible after trying.

Completion response format (plain text):
- Download status: success|failed
- Downloaded filename(s)
- Matching rationale versus the reference file
- Short trace of GUI steps you performed
"""


async def run_data_fetcher(start_url: str, description: str = "", reference_file_name: str = "") -> dict:
    """
    Main entry point. Takes a URL and optional description,
    uses Anthropic's sampling_loop to drive Computer Use.
    """
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    tool_version = os.environ.get("CUA_TOOL_VERSION", MODEL_TO_TOOL_VERSION.get(model, "computer_use_20250124"))

    task = f"Start from this URL and fetch matching data with CUA-first behavior:\n\nURL: {start_url}\n"
    if description:
        task += f"Description: {description}\n"
    if reference_file_name:
        task += f"Reference file in Downloads: {reference_file_name}\n"
    task += (
        "\nRequirements:\n"
        "- Prefer browser GUI steps first (CUA-first).\n"
        "- Use bash/terminal/python/curl/wget only when it is the best available fallback.\n"
        "- Open and inspect the reference file first to understand what to match.\n"
        "- Then use browser UI to locate source downloads and retrieve matching artifacts.\n"
        "- After each download, compare against the reference file.\n"
        "- If mismatch or uncertain, continue searching and download a better candidate.\n"
    )

    existing_files = {f.resolve() for f in DOWNLOADS_DIR.glob("*") if f.is_file()}

    messages = [{"role": "user", "content": task}]

    print(f"[CUA] Starting data fetch from: {start_url}")
    print(f"[CUA] Model: {model} | Tool version: {tool_version}")
    if reference_file_name:
        print(f"[CUA] Reference file: {reference_file_name}")
    print(f"[CUA] Task length: {len(task)} chars")
    print("[CUA] --- task begin ---")
    print(task)
    print("[CUA] --- task end ---")

    def _on_output(msg: Any) -> None:
        print(f"[CUA:out] {_format_output(msg)}")

    def _on_tool_output(result: ToolResult, tool_id: str) -> None:
        print(f"[CUA:tool_result id={tool_id}] {_format_tool_result(result)}")

    def _on_api_response(request: Any, response: Any, error: Exception | None) -> None:
        if error:
            print(f"[CUA:api_error] {error.__class__.__name__}: {error}")
            return
        if response is not None:
            usage = getattr(response, "usage", None)
            if usage:
                print(
                    f"[CUA:api] input_tokens={getattr(usage, 'input_tokens', '?')} "
                    f"output_tokens={getattr(usage, 'output_tokens', '?')}"
                )

    # Use Anthropic's official sampling loop — it handles
    # screenshots, clicking, typing, get_page_text, bash, text editor,
    # coordinate scaling, and the full agent loop automatically.
    response_messages = await sampling_loop(
        model=model,
        provider=APIProvider.ANTHROPIC,
        system_prompt_suffix=SYSTEM_PROMPT,
        messages=messages,
        output_callback=_on_output,
        tool_output_callback=_on_tool_output,
        api_response_callback=_on_api_response,
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

    # Print tool-call summary so the run is auditable in logs
    action_counts: dict[str, int] = {}
    for event in tool_events:
        name = event.get("name", "unknown")
        action = event.get("input", {}).get("action", "") if name == "computer" else ""
        key = f"{name}:{action}" if action else name
        action_counts[key] = action_counts.get(key, 0) + 1
    print(f"[CUA:summary] total_tool_calls={len(tool_events)}")
    for key, count in sorted(action_counts.items()):
        print(f"[CUA:summary]   {key} x{count}")
    print(f"[CUA:summary] new_files={[f.name for f in new_downloaded]}")

    validation = _validate_run(
        tool_events=tool_events,
        downloaded_files=new_downloaded,
    )

    return {
        "status": validation["status"],
        "summary": final_text,
        "downloaded_files": [str(f) for f in new_downloaded],
        "validation": validation,
    }


def _format_output(msg: Any) -> str:
    """Format a Claude message block for logging. Handles both SDK objects and plain dicts."""
    # Normalise to a simple type string + payload, whether msg is a dict or an SDK object
    msg_type = msg.get("type") if isinstance(msg, dict) else getattr(msg, "type", None)

    if msg_type == "tool_use":
        name = msg.get("name") if isinstance(msg, dict) else getattr(msg, "name", "?")
        raw_input = msg.get("input", {}) if isinstance(msg, dict) else getattr(msg, "input", {})
        input_str = json.dumps(raw_input, ensure_ascii=False)
        preview = input_str[:500] + (" ..." if len(input_str) > 500 else "")
        return f"Tool call [{name}]: {preview}"

    if msg_type == "text":
        text = (msg.get("text", "") if isinstance(msg, dict) else getattr(msg, "text", "") or "").strip()
        preview = text[:400] + (f" ... [{len(text)} chars]" if len(text) > 400 else "")
        return f"Claude: {preview}"

    if msg_type == "thinking":
        thinking = (msg.get("thinking", "") if isinstance(msg, dict) else getattr(msg, "thinking", "") or "").strip()
        preview = thinking[:200] + (" ..." if len(thinking) > 200 else "")
        return f"[thinking]: {preview}"

    return str(msg)[:400]


def _format_tool_result(result: ToolResult) -> str:
    """Format a tool result for logging."""
    parts: list[str] = []
    if result.error:
        parts.append(f"ERROR: {result.error[:500]}")
    if result.output:
        text = (result.output or "").strip()
        preview = text[:500] + (f" ... [{len(text)} chars]" if len(text) > 500 else "")
        parts.append(f"output: {preview}")
    if result.base64_image:
        size_kb = len(result.base64_image) * 3 // 4 // 1024
        parts.append(f"[screenshot ~{size_kb}KB]")
    return " | ".join(parts) if parts else "[empty result]"


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
) -> dict[str, Any]:
    used_bash = any(event.get("name") == "bash" for event in tool_events)
    used_gui_tool = any(event.get("name") == "computer" for event in tool_events)
    clicked = any(
        event.get("name") == "computer" and event.get("input", {}).get("action") in {"left_click", "double_click"}
        for event in tool_events
    )
    took_screenshot = any(
        event.get("name") == "computer" and event.get("input", {}).get("action") == "screenshot"
        for event in tool_events
    )
    used_get_page_text = any(
        event.get("name") == "computer" and event.get("input", {}).get("action") == "get_page_text"
        for event in tool_events
    )

    issues: list[dict[str, str]] = []
    advisories: list[dict[str, str]] = []

    if used_bash:
        advisories.append(
            {
                "code": "BASH_FALLBACK_USED",
                "message": "Bash/terminal fallback was used. Prefer GUI-first when practical.",
            }
        )

    if not used_gui_tool:
        advisories.append(
            {
                "code": "NO_GUI_ACTIONS",
                "message": "No computer-use GUI actions were detected in this run.",
            }
        )

    if used_gui_tool and not clicked:
        advisories.append(
            {
                "code": "NO_CLICK_ACTION",
                "message": "GUI actions occurred but no click was detected.",
            }
        )

    if not took_screenshot:
        advisories.append(
            {
                "code": "NO_SCREENSHOT",
                "message": "No screenshot action was taken; agent may not have observed screen state.",
            }
        )

    if not used_get_page_text:
        advisories.append(
            {
                "code": "NO_GET_PAGE_TEXT",
                "message": "get_page_text was never used; agent may have missed readable page content.",
            }
        )

    if not downloaded_files:
        issues.append({"code": "NO_NEW_DOWNLOAD", "message": "No new file was downloaded into ~/Downloads/."})

    status = "success" if not issues else "fail"

    return {
        "status": status,
        "issues": issues,
        "advisories": advisories,
        "selected_files": [str(path) for path in downloaded_files],
        "tool_usage": {
            "used_bash": used_bash,
            "used_gui_tool": used_gui_tool,
            "click_detected": clicked,
            "screenshot_taken": took_screenshot,
            "get_page_text_used": used_get_page_text,
            "tool_count": len(tool_events),
        },
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_fetcher.py <start_url> [description] [reference_file_name]")
        print(
            "Example: python data_fetcher.py "
            "'https://www.netflix.com/tudum/top10' "
            "'Fetch global top 10 lists from download tab' "
            "'submitted_user_file.csv'"
        )
        sys.exit(1)

    url = sys.argv[1]
    desc = sys.argv[2] if len(sys.argv) > 2 else ""
    reference_file = sys.argv[3] if len(sys.argv) > 3 else ""

    result = asyncio.run(run_data_fetcher(url, desc, reference_file))
    print(f"\n{'=' * 60}")
    print(json.dumps(result, indent=2))
