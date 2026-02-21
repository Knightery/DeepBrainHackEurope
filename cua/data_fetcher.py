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

from computer_use_demo.loop import sampling_loop, APIProvider
from computer_use_demo.tools import ToolResult

DOWNLOADS_DIR = Path("/home/computeruse/Downloads")

SYSTEM_PROMPT = """You are a Data Fetcher agent for a quant pitch evaluation platform.
Your job is to navigate to a URL and extract data from it.

Instructions:
1. Open the URL in Firefox (Ctrl+L for address bar).
2. Find the data — tables, CSV/Excel download links, datasets, API responses.
3. If there's a downloadable file (CSV, Excel, JSON), click the download link.
4. If data is in a table on the page, use bash to save it:
   - You can use `curl` or `wget` to download files directly if you find the URL.
   - You can use Python to scrape table data if needed.
5. Save everything to ~/Downloads/
6. When done, give a concise summary of what data you found and where it was saved.

Tips:
- Use keyboard shortcuts (Ctrl+L, Enter, Ctrl+A, Tab).
- Dismiss cookie banners by clicking Accept/OK.
- If a page needs scrolling, scroll down to find all the data.
- If a table is too large, try finding a download/export button instead.
"""


async def run_data_fetcher(url: str, description: str = "") -> dict:
    """
    Main entry point. Takes a URL and optional description,
    uses Anthropic's sampling_loop to drive Computer Use.
    """
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    task = f"Navigate to this URL and fetch the data:\n\nURL: {url}\n"
    if description:
        task += f"Description: {description}\n"
    task += "\nSave any downloaded files to ~/Downloads/"

    messages = [{"role": "user", "content": task}]

    print(f"[CUA] Starting data fetch from: {url}")

    # Use Anthropic's official sampling loop — it handles
    # screenshots, clicking, typing, bash, text editor, coordinate
    # scaling, and the full agent loop automatically.
    response_messages = await sampling_loop(
        model="claude-sonnet-4-6",
        provider=APIProvider.ANTHROPIC,
        system_prompt_suffix=SYSTEM_PROMPT,
        messages=messages,
        output_callback=lambda msg: print(f"[CUA] {_format_output(msg)}"),
        tool_output_callback=lambda result, id: print(f"[CUA] Tool result: {_format_tool_result(result)}"),
        api_response_callback=lambda resp, msgs: None,
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        max_tokens=4096,
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

    downloaded = list(DOWNLOADS_DIR.glob("*"))

    return {
        "status": "success",
        "summary": final_text,
        "downloaded_files": [str(f) for f in downloaded],
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_fetcher.py <url> [description]")
        print("Example: python data_fetcher.py 'https://finance.yahoo.com/quote/MCD' 'McDonalds stock price'")
        sys.exit(1)

    url = sys.argv[1]
    desc = sys.argv[2] if len(sys.argv) > 2 else ""

    result = asyncio.run(run_data_fetcher(url, desc))
    print(f"\n{'=' * 60}")
    print(json.dumps(result, indent=2))
