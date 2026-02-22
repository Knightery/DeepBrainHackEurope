from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chainlit as cl
from dotenv import load_dotenv
from google import genai
from google.genai import types

from pitch_engine import (
    PitchDraft,
    UploadedFile,
    _detect_file_roles,
    _load_strategy_source_for_backtest,
    evaluate_pitch,
    extract_tagged_json,
    file_sha256,
    heuristic_update_from_user_text,
    strip_tagged_json,
    validate_data_with_cua,
)
from pitch_db import append_pitch_message, save_pitch_result, upsert_pitch_snapshot
from paid_usage import PaidUsageTracker

try:
    from backtest_agent import run_backtest_agent as _run_backtest_agent
    _BACKTEST_AVAILABLE = True
except ImportError:
    _BACKTEST_AVAILABLE = False

load_dotenv()

import os as _os
BACKTEST_TIMEOUT_SECONDS = int(_os.getenv("BACKTEST_TIMEOUT_SECONDS", "120"))
_DEFAULT_PAID_EXTERNAL_PRODUCT_ID = "quant_pitch_evaluator"
_PAID_TRACKER = PaidUsageTracker.from_env(default_event_name="eva_by_anyquant")
if _PAID_TRACKER.external_product_id is None:
    _PAID_TRACKER.external_product_id = _DEFAULT_PAID_EXTERNAL_PRODUCT_ID

DATA_ROOT = Path("data/pitches")

SYSTEM_PROMPT = """
You are the Clarifier Agent for a quant pitch intake flow.
You are interviewer-led: you control the flow and actively test the pitch for weaknesses.

Your goals:
1) Help the user produce a clear investment thesis.
2) Confirm target stock tickers and time horizon.
3) Request source URLs only when the user uploads supporting data files.
4) Remind the user to upload their strategy files (.py or .ipynb) if not yet attached.
5) Keep and show a practical checklist of required submissions:
   - thesis
   - time_horizon (days|weeks|months|years)
   - tickers
   - at least one uploaded strategy/signal file
   - source_urls only when supporting data files are submitted
   - supporting data files only when needed for the strategy or evidence
6) Ask probing, high-signal follow-up questions to find holes in assumptions, evidence quality, implementation realism, and risk controls.
7) Act autonomously: once enough information is available, trigger the next pipeline actions yourself without asking the user for approval.
8) Keep responses concise, practical, and direct.

At the end of every response, include these XML blocks with strict JSON:
<pitch_state>{"thesis": "...", "time_horizon": "days|weeks|months|years|null", "tickers": [], "source_urls": [], "one_shot_mode": false, "ready_for_evaluation": false}</pitch_state>
<orchestrator_actions>{"actions":[{"action":"run_evaluate|run_backtest|run_validate_data","file_name":"optional","notes":"optional","reason":"short reason"}]}</orchestrator_actions>

When you need file context before replying, request tools with:
<file_tools>{"calls":[{"tool":"read_uploaded_file|read_notebook_cells","file_name":"...","start_line":1,"max_lines":200,"start_cell":0,"cell_count":3,"max_chars":12000}]}</file_tools>
After tool results are returned, continue the conversation and then output `<pitch_state>` and `<orchestrator_actions>`.

Rules:
- If you are uncertain about a field, return the current best value or empty string.
- `tickers` must always be a JSON array of stock tickers (e.g., ["AAPL", "MSFT"]).
- `source_urls` must always be a JSON array.
- Keep conversational text before the XML block.
- Include exactly one `<pitch_state>` block and exactly one `<orchestrator_actions>` block.
- If you emit `<file_tools>`, do not emit `<pitch_state>` or `<orchestrator_actions>` in that same response.
- If no action is needed, return: `<orchestrator_actions>{"actions":[]}</orchestrator_actions>`.
- You may emit multiple actions in one turn when it improves workflow; order them as they should run.
- Evaluate-only flow: do not reference or request `/validate`.
- Intake is ready when ALL are true:
  a) thesis is present,
  b) time_horizon is present,
  c) tickers has at least one symbol,
  d) at least one strategy/signal file is uploaded,
  e) if supporting data files exist, source_urls must be present.
- If supporting data files exist and source URLs are present, emit one `run_validate_data` action per relevant file with `file_name`.
- A backtest is mandatory before any submission to final review for non-one-shot strategies.
- For non-one-shot strategies, emit `run_backtest` before `run_evaluate` whenever final-review submission is the intent.
- The agent may submit to final review at any time by emitting `run_evaluate` when it judges evidence is sufficient.
""".strip()

COMMANDS_TEXT = """
Optional commands:
- `/status` show current pitch completeness
- `/checklist` show onboarding checklist
- `/oneshot on|off|status` explicitly toggle one-shot validation mode
- `/evaluate` run validation and scoring
- `/backtest` run Claude backtest agent on uploaded strategy script (.py or .ipynb)
- `/validate_data "file_to_validate" "notes"` run CUA data-source validation for one uploaded file
- `/reset` start a new pitch
- `/help` show commands
""".strip()

MAX_CLARIFIER_TOOL_ROUNDS = 4
MAX_TOOL_CALLS_PER_ROUND = 4
MAX_TOOL_CHARS = 12000
MAX_TOOL_LINES = 300
MAX_TOOL_CELLS = 8
_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backtest flavor text
# ---------------------------------------------------------------------------

_BACKTEST_FLAVOR = [
    "Interrogating the Sharpe ratio...",
    "Cross-referencing alpha with Jensen’s model...",
    "Auditing for look-ahead bias...",
    "Bribing the benchmark...",
    "Calibrating max drawdown thresholds...",
    "Challenging the win rate assumptions...",
    "Computing Monte Carlo tail risk...",
    "Reviewing trade execution slippage...",
    "Stress-testing regime-change sensitivity...",
    "Auditing position sizing logic...",
    "Negotiating with the Sortino ratio...",
    "Validating signal generation pipeline...",
    "Decomposing factor exposures...",
    "Checking for survivorship bias...",
    "Verifying annualisation assumptions...",
]


async def _backtest_flavor_task(msg: cl.Message) -> None:
    """Cycle quant-flavoured status text while the backtest worker runs."""
    import itertools
    texts = itertools.cycle(_BACKTEST_FLAVOR)
    while True:
        await asyncio.sleep(2.2)
        msg.content = f"⏳ **Backtest running…** {next(texts)}"
        try:
            await msg.update()
        except Exception:
            break



def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _gemini_model() -> str:
    import os

    return os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")


def _get_client() -> genai.Client | None:
    import os

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _pitch_dir(pitch_id: str) -> Path:
    path = DATA_ROOT / pitch_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _save_pitch_snapshot(draft: PitchDraft) -> None:
    draft_path = _pitch_dir(draft.pitch_id) / "pitch.json"
    payload = draft.to_dict()
    _write_json(draft_path, payload)
    upsert_pitch_snapshot(payload, _now_iso())


def _append_chat_event(draft: PitchDraft, role: str, content: str) -> None:
    now_iso = _now_iso()
    history_path = _pitch_dir(draft.pitch_id) / "clarifier_history.jsonl"
    _append_jsonl(
        history_path,
        {
            "timestamp_utc": now_iso,
            "role": role,
            "content": content,
        },
    )
    append_pitch_message(draft.pitch_id, now_iso, role, content)


def _new_pitch() -> PitchDraft:
    return PitchDraft(
        pitch_id=f"pit_{uuid.uuid4().hex[:12]}",
        created_at=_now_iso(),
        status="draft",
    )


def _session_draft() -> PitchDraft:
    draft: PitchDraft = cl.user_session.get("draft")
    return draft


def _set_session_draft(draft: PitchDraft) -> None:
    cl.user_session.set("draft", draft)


def _session_history() -> list[dict[str, str]]:
    history: list[dict[str, str]] = cl.user_session.get("llm_history", [])
    return history


def _set_session_history(history: list[dict[str, str]]) -> None:
    cl.user_session.set("llm_history", history[-16:])


def _session_validation_context() -> list[str]:
    return cl.user_session.get("validation_context", [])


def _set_session_validation_context(items: list[str]) -> None:
    cl.user_session.set("validation_context", items[-3:])


def _session_data_fetcher_output() -> dict[str, Any] | None:
    output = cl.user_session.get("data_fetcher_output")
    return output if isinstance(output, dict) else None


def _set_session_data_fetcher_output(output: dict[str, Any] | None) -> None:
    cl.user_session.set("data_fetcher_output", output)


def _non_empty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _session_external_customer_id() -> str | None:
    env_override = _non_empty_str(_os.getenv("PAID_EXTERNAL_CUSTOMER_ID", ""))
    if env_override:
        return env_override

    for key in ("external_customer_id", "customer_id", "user_id", "id"):
        candidate = _non_empty_str(cl.user_session.get(key))
        if candidate:
            return candidate

    user_obj = cl.user_session.get("user")
    if isinstance(user_obj, dict):
        for key in ("identifier", "id", "email", "name"):
            candidate = _non_empty_str(user_obj.get(key))
            if candidate:
                return candidate
    elif user_obj is not None:
        for attr in ("identifier", "id", "email", "name"):
            candidate = _non_empty_str(getattr(user_obj, attr, None))
            if candidate:
                return candidate

    return None


def _track_paid_usage(draft: PitchDraft | None, usage_action: str, **data: Any) -> None:
    if not _PAID_TRACKER.enabled:
        return

    payload: dict[str, Any] = {"usage_action": usage_action}
    payload.update(data)
    if draft is not None:
        payload.setdefault("pitch_id", draft.pitch_id)
        payload.setdefault("pitch_status", draft.status)
        payload.setdefault("one_shot_mode", bool(draft.one_shot_mode))

    external_customer_id = _session_external_customer_id() or (draft.pitch_id if draft else None)

    async def _emit() -> None:
        try:
            await _PAID_TRACKER.send_usage_record_async(
                external_customer_id=external_customer_id,
                data=payload,
            )
        except Exception as exc:
            _LOGGER.warning("Paid usage task failed: %s: %s", exc.__class__.__name__, exc)

    asyncio.create_task(_emit())


def _apply_clarification_update(draft: PitchDraft, user_text: str) -> None:
    if draft.status != "needs_clarification":
        return
    text = user_text.strip()
    if not text:
        return
    # Preserve concise clarification evidence for validator reruns.
    if draft.supporting_notes:
        draft.supporting_notes = f"{draft.supporting_notes}\n\nClarification:\n{text}"
    else:
        draft.supporting_notes = text


def _status_markdown(draft: PitchDraft) -> str:
    missing = draft.missing_fields()
    files = ", ".join(file.name for file in draft.uploaded_files) if draft.uploaded_files else "(none)"
    tickers = ", ".join(draft.tickers) if draft.tickers else "(none)"
    urls = ", ".join(draft.source_urls) if draft.source_urls else "(none)"
    return (
        f"## Pitch Status\n"
        f"- Pitch ID: `{draft.pitch_id}`\n"
        f"- Status: `{draft.status}`\n"
        f"- Ready for evaluation: `{draft.ready_for_evaluation()}`\n"
        f"- Thesis: `{draft.thesis or '(missing)'}`\n"
        f"- One-shot mode: `{draft.one_shot_mode}`\n"
        f"- Time horizon: `{draft.time_horizon or '(missing)'}`\n"
        f"- Tickers: {tickers}\n"
        f"- Source URLs: {urls}\n"
        f"- Uploaded files: {files}\n"
        f"- Missing fields: `{', '.join(missing) if missing else 'none'}`"
    )


def _checklist_markdown(draft: PitchDraft) -> str:
    has_tabular_data = any(Path(file.path).suffix.lower() in {".csv", ".tsv"} for file in draft.uploaded_files)
    has_strategy_script = any(Path(file.path).suffix.lower() in {".py", ".ipynb"} for file in draft.uploaded_files)
    checks = [
        ("Strategy mode explicitly set (`/oneshot on` for one-shot, otherwise standard)", True),
        ("Thesis (1-3 sentences)", bool(draft.thesis.strip())),
        ("Time horizon (`days`, `weeks`, `months`, `years`)", bool(draft.time_horizon)),
        ("Stock tickers (e.g., `AAPL, MSFT`)", len(draft.tickers) > 0),
        (
            "Source URL(s) for uploaded supporting data files (CSV/TSV)",
            (not has_tabular_data) or len(draft.source_urls) > 0,
        ),
        (
            "Strategy file uploaded (`.py`/`.ipynb`)",
            has_strategy_script if not draft.one_shot_mode else len(draft.uploaded_files) > 0,
        ),
    ]
    lines = ["## Onboarding Checklist"]
    for label, done in checks:
        lines.append(f"- {chr(0x2705) if done else chr(0x2B1C)} {label}")
    missing = draft.missing_fields()
    lines.append("")
    lines.append(f"- Ready: `{draft.ready_for_evaluation()}`")
    lines.append(f"- Missing: `{', '.join(missing) if missing else 'none'}`")
    lines.append("- Upload your strategy `.py` or `.ipynb` file using the attachment button.")
    return "\n".join(lines)


def _local_clarifier_reply(draft: PitchDraft) -> str:
    missing = draft.missing_fields()
    if not missing:
        return "Your pitch looks complete. I will automatically run validation now."

    upload_prompt = (
        "Upload your strategy file (`.py` or `.ipynb`) using the attachment button below."
        if not draft.one_shot_mode
        else "Upload your supporting one-shot data/evidence file(s) (`.csv`, `.tsv`, `.py`, or `.ipynb`) using the attachment button below."
    )
    prompt_map = {
        "thesis": "State the thesis in one sentence: what is mispriced and why now?",
        "time_horizon": "Pick a horizon: days, weeks, months, or years.",
        "tickers": "Share the stock ticker(s), for example: AAPL or AAPL, MSFT.",
        "source_urls": "Share source URL(s) for any uploaded supporting data file so we can verify provenance.",
        "uploaded_files": upload_prompt,
    }
    first_missing = missing[0]
    return f"I captured your latest details. Next: {prompt_map.get(first_missing, first_missing)}"


def _resolve_uploaded_file(draft: PitchDraft, file_name: str) -> UploadedFile | None:
    target = file_name.strip()
    if not target:
        return None
    for item in draft.uploaded_files:
        if item.name == target:
            return item
    lowered = target.lower()
    for item in draft.uploaded_files:
        if item.name.lower() == lowered:
            return item
    return None


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def _read_uploaded_file_lines(
    draft: PitchDraft,
    file_name: str,
    start_line: int = 1,
    max_lines: int = 200,
    max_chars: int = MAX_TOOL_CHARS,
) -> dict[str, Any]:
    uploaded = _resolve_uploaded_file(draft, file_name)
    if uploaded is None:
        return {"ok": False, "error": f"Uploaded file not found: {file_name}"}

    path = Path(uploaded.path)
    if not path.exists():
        return {"ok": False, "error": f"Stored file missing on disk: {uploaded.name}"}

    suffix = path.suffix.lower()
    if suffix not in {".py", ".md", ".txt", ".csv", ".tsv", ".json", ".yaml", ".yml", ".ipynb"}:
        return {"ok": False, "error": f"Unsupported text file type for read_uploaded_file: {suffix or '(none)'}"}

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    line_count = len(lines)
    safe_start = max(1, int(start_line or 1))
    safe_max_lines = max(1, min(int(max_lines or 200), MAX_TOOL_LINES))
    safe_max_chars = max(500, min(int(max_chars or MAX_TOOL_CHARS), MAX_TOOL_CHARS))

    start_idx = min(safe_start - 1, line_count)
    end_idx = min(start_idx + safe_max_lines, line_count)
    numbered = [f"{idx + 1}|{lines[idx]}" for idx in range(start_idx, end_idx)]
    snippet = _truncate_text("\n".join(numbered), safe_max_chars)

    return {
        "ok": True,
        "file_name": uploaded.name,
        "line_start": safe_start,
        "line_end": end_idx,
        "total_lines": line_count,
        "content": snippet,
    }


def _read_uploaded_notebook_cells(
    draft: PitchDraft,
    file_name: str,
    start_cell: int = 0,
    cell_count: int = 3,
    max_chars: int = MAX_TOOL_CHARS,
) -> dict[str, Any]:
    uploaded = _resolve_uploaded_file(draft, file_name)
    if uploaded is None:
        return {"ok": False, "error": f"Uploaded file not found: {file_name}"}

    path = Path(uploaded.path)
    if path.suffix.lower() != ".ipynb":
        return {"ok": False, "error": f"File is not a notebook: {uploaded.name}"}
    if not path.exists():
        return {"ok": False, "error": f"Stored file missing on disk: {uploaded.name}"}

    try:
        notebook = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"Could not parse notebook JSON: {exc}"}

    cells = notebook.get("cells", [])
    if not isinstance(cells, list):
        return {"ok": False, "error": "Notebook has no readable cells array."}

    safe_start = max(0, int(start_cell or 0))
    safe_count = max(1, min(int(cell_count or 3), MAX_TOOL_CELLS))
    safe_max_chars = max(500, min(int(max_chars or MAX_TOOL_CHARS), MAX_TOOL_CHARS))

    selected = cells[safe_start:safe_start + safe_count]
    rendered: list[str] = []
    for idx, cell in enumerate(selected, start=safe_start):
        cell_type = str(cell.get("cell_type", "unknown"))
        source = cell.get("source", "")
        if isinstance(source, list):
            source_text = "".join(str(part) for part in source)
        else:
            source_text = str(source)
        rendered.append(f"Cell {idx} ({cell_type}):\n{source_text}")

    return {
        "ok": True,
        "file_name": uploaded.name,
        "cell_start": safe_start,
        "cell_end_exclusive": safe_start + len(selected),
        "total_cells": len(cells),
        "content": _truncate_text("\n\n".join(rendered), safe_max_chars),
    }


def _execute_clarifier_file_tools(draft: PitchDraft, raw_text: str) -> tuple[list[dict[str, Any]], bool]:
    request = extract_tagged_json(raw_text, "file_tools")
    if not isinstance(request, dict):
        return [], False

    calls = request.get("calls")
    if not isinstance(calls, list):
        return [{"ok": False, "error": "file_tools.calls must be a JSON array."}], True

    results: list[dict[str, Any]] = []
    for call in calls[:MAX_TOOL_CALLS_PER_ROUND]:
        if not isinstance(call, dict):
            results.append({"ok": False, "error": "Tool call entry must be a JSON object."})
            continue
        tool = str(call.get("tool", "")).strip()
        file_name = str(call.get("file_name", "")).strip()
        if not file_name:
            results.append({"ok": False, "tool": tool, "error": "file_name is required."})
            continue

        if tool == "read_uploaded_file":
            results.append(
                _read_uploaded_file_lines(
                    draft=draft,
                    file_name=file_name,
                    start_line=int(call.get("start_line", 1) or 1),
                    max_lines=int(call.get("max_lines", 200) or 200),
                    max_chars=int(call.get("max_chars", MAX_TOOL_CHARS) or MAX_TOOL_CHARS),
                )
            )
            continue

        if tool == "read_notebook_cells":
            results.append(
                _read_uploaded_notebook_cells(
                    draft=draft,
                    file_name=file_name,
                    start_cell=int(call.get("start_cell", 0) or 0),
                    cell_count=int(call.get("cell_count", 3) or 3),
                    max_chars=int(call.get("max_chars", MAX_TOOL_CHARS) or MAX_TOOL_CHARS),
                )
            )
            continue

        results.append(
            {
                "ok": False,
                "tool": tool,
                "error": "Unsupported tool. Use read_uploaded_file or read_notebook_cells.",
            }
        )

    return results, True


def _extract_orchestrator_actions(raw_text: str) -> list[dict[str, str]]:
    parsed = extract_tagged_json(raw_text, "orchestrator_actions")
    if not isinstance(parsed, dict):
        return []
    actions = parsed.get("actions")
    if not isinstance(actions, list):
        return []

    normalized: list[dict[str, str]] = []
    allowed = {"run_evaluate", "run_backtest", "run_validate_data"}
    for item in actions:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip()
        if action not in allowed:
            continue
        entry: dict[str, str] = {"action": action}
        file_name = item.get("file_name")
        notes = item.get("notes")
        reason = item.get("reason")
        if isinstance(file_name, str) and file_name.strip():
            entry["file_name"] = file_name.strip()
        if isinstance(notes, str) and notes.strip():
            entry["notes"] = notes.strip()
        if isinstance(reason, str) and reason.strip():
            entry["reason"] = reason.strip()
        normalized.append(entry)

    return normalized[:8]


async def _run_clarifier_turn(draft: PitchDraft, user_text: str) -> tuple[str, list[dict[str, str]]]:
    client = _get_client()
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not set. Clarifier cannot run without Gemini.")

    context = draft.to_llm_context()
    context["validation_context"] = _session_validation_context()
    context_payload = json.dumps(context, ensure_ascii=True)
    history = _session_history()
    history.append({"role": "user", "content": user_text})
    history_text = "\n".join(f"{item['role'].upper()}: {item['content']}" for item in history[-10:])
    prompt = (
        "You are in a running clarifier conversation. Use current structured context and recent dialogue.\n\n"
        f"CURRENT_PITCH_CONTEXT={context_payload}\n\n"
        f"RECENT_TURNS=\n{history_text}"
    )

    raw_text = ""
    tool_context: list[dict[str, Any]] = []
    async with cl.Step(name="Clarifier Agent", type="llm") as step:
        step.input = f"Analyzing user input and extracting pitch fields..."
        for round_num in range(MAX_CLARIFIER_TOOL_ROUNDS):
            prompt_with_tools = prompt
            if tool_context:
                tool_payload = json.dumps(tool_context, ensure_ascii=True)
                prompt_with_tools = f"{prompt}\n\nTOOL_RESULTS={tool_payload}"

            response = await cl.make_async(client.models.generate_content)(
                model=_gemini_model(),
                contents=prompt_with_tools,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=10000,
                    system_instruction=SYSTEM_PROMPT,
                ),
            )
            raw_text = (getattr(response, "text", "") or "").strip()

            # Extract the tool request payload BEFORE executing, for display
            file_tools_request = extract_tagged_json(raw_text, "file_tools")

            tool_results, has_tool_request = _execute_clarifier_file_tools(draft, raw_text)
            if not has_tool_request:
                break

            # Render each tool call as a nested expandable step
            if isinstance(file_tools_request, dict):
                calls = file_tools_request.get("calls", []) or []
                for call_idx, call in enumerate(calls[:MAX_TOOL_CALLS_PER_ROUND]):
                    if not isinstance(call, dict):
                        continue
                    tool_name = call.get("tool", "tool")
                    file_name = call.get("file_name", "?")
                    result_data: dict[str, Any] = tool_results[call_idx] if call_idx < len(tool_results) else {}

                    # Input: show params minus the tool name itself
                    input_params = {k: v for k, v in call.items() if k != "tool"}
                    input_summary = json.dumps(input_params, indent=2)

                    # Output: readable summary
                    if result_data.get("ok"):
                        content_preview = str(result_data.get("content", ""))[:600]
                        line_start = result_data.get("line_start") or result_data.get("cell_start", "?")
                        line_end = result_data.get("line_end") or result_data.get("cell_end_exclusive", "?")
                        total = result_data.get("total_lines") or result_data.get("total_cells", "?")
                        output_summary = (
                            f"**Read** `{file_name}` — rows {line_start}–{line_end} of {total}\n\n"
                            f"```\n{content_preview}\n```"
                        )
                    else:
                        output_summary = f"**Error:** {result_data.get('error', 'unknown error')}"

                    async with cl.Step(
                        name=f"{tool_name}({file_name})",
                        type="tool",
                    ) as tool_step:
                        tool_step.input = input_summary
                        tool_step.output = output_summary

            tool_context.append({"request_results": tool_results})

        extracted = extract_tagged_json(raw_text, "pitch_state")
        if extracted:
            draft.merge_structured_update(extracted)
            step.output = f"Extracted fields: {json.dumps(extracted, indent=2)}"
        else:
            step.output = "No structured fields extracted this turn."

    actions = _extract_orchestrator_actions(raw_text)
    text_wo_tools = strip_tagged_json(raw_text, "file_tools")
    text_wo_state = strip_tagged_json(text_wo_tools, "pitch_state")
    assistant_text = strip_tagged_json(text_wo_state, "orchestrator_actions") or _local_clarifier_reply(draft)
    history.append({"role": "assistant", "content": assistant_text})
    _set_session_history(history)
    return assistant_text, actions


def _copy_to_pitch_uploads(draft: PitchDraft, src_path: Path, original_name: str, mime_type: str, size: int) -> UploadedFile:
    upload_dir = _pitch_dir(draft.pitch_id) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    raw_name = str(original_name or "").strip().replace("\x00", "")
    leaf_name = raw_name.replace("\\", "/").split("/")[-1]
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", leaf_name).strip("._")
    if not safe_name:
        safe_name = f"upload_{uuid.uuid4().hex[:8]}"

    destination = (upload_dir / safe_name).resolve()
    upload_root = upload_dir.resolve()
    try:
        destination.relative_to(upload_root)
    except ValueError as exc:
        raise ValueError(f"Unsafe upload filename: {original_name!r}") from exc

    if destination.exists():
        destination = (upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}").resolve()
        destination.relative_to(upload_root)

    shutil.copy2(src_path, destination)
    return UploadedFile(
        file_id=f"fil_{uuid.uuid4().hex[:12]}",
        name=safe_name,
        path=str(destination),
        mime_type=mime_type or "",
        size_bytes=size if size > 0 else destination.stat().st_size,
        sha256=file_sha256(destination),
    )


def _ingest_message_files(draft: PitchDraft, message: cl.Message) -> list[str]:
    elements = getattr(message, "elements", None) or []
    added: list[str] = []
    for element in elements:
        path_value = getattr(element, "path", None)
        if not path_value:
            continue
        src_path = Path(path_value)
        if not src_path.exists():
            continue
        name = getattr(element, "name", src_path.name)
        mime_type = getattr(element, "mime", "") or getattr(element, "type", "") or ""
        size = int(getattr(element, "size", 0) or 0)
        stored = _copy_to_pitch_uploads(draft, src_path, name, mime_type, size)
        draft.uploaded_files.append(stored)
        added.append(stored.name)
    return added


def _files_requiring_cua(draft: PitchDraft) -> list[UploadedFile]:
    roles = _detect_file_roles(draft.uploaded_files, tickers=draft.tickers, thesis=draft.thesis)
    candidates = roles["data_files"] + roles["benchmark_files"]
    if candidates:
        # Keep deterministic order and avoid duplicates by file_id.
        seen: set[str] = set()
        ordered: list[UploadedFile] = []
        for item in candidates:
            if item.file_id in seen:
                continue
            seen.add(item.file_id)
            ordered.append(item)
        return ordered
    # Fallback: any uploaded CSV/TSV should be provenance-validated.
    fallback: list[UploadedFile] = []
    for item in draft.uploaded_files:
        name = (item.name or "").lower()
        if name.endswith(".csv") or name.endswith(".tsv"):
            fallback.append(item)
    return fallback


def _parse_source_fetch_hints(text: str) -> dict[str, Any]:
    file_notes: dict[str, str] = {}
    file_urls: dict[str, str] = {}
    global_notes: list[str] = []
    global_urls: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        match = re.match(r"^(source_note|source_url)\[(.+?)\]\s*=\s*(.+)$", line, flags=re.IGNORECASE)
        if not match:
            continue
        key = match.group(1).strip().lower()
        target = match.group(2).strip().lower()
        value = match.group(3).strip()
        if not value:
            continue
        if target in {"*", "all", "global"}:
            if key == "source_note":
                global_notes.append(value)
            else:
                global_urls.append(value)
            continue
        if key == "source_note":
            file_notes[target] = value
        else:
            file_urls[target] = value
    return {
        "file_notes": file_notes,
        "file_urls": file_urls,
        "global_notes": global_notes,
        "global_urls": global_urls,
    }


def _build_cua_context_for_file(draft: PitchDraft, file_entry: UploadedFile, reason: str) -> tuple[str, list[str], str]:
    hints = _parse_source_fetch_hints(draft.supporting_notes or "")
    file_key = (file_entry.name or "").strip().lower()
    file_note = hints["file_notes"].get(file_key, "")
    file_url = hints["file_urls"].get(file_key, "").strip()
    global_notes = [note for note in hints["global_notes"] if note]
    global_urls = [url for url in hints["global_urls"] if url]

    expected_format = Path(file_entry.name or "").suffix.lower().lstrip(".")
    if not expected_format:
        expected_format = "unknown"

    scoped_urls: list[str] = []
    if file_url:
        scoped_urls.append(file_url)
    if global_urls:
        scoped_urls.extend(global_urls)
    if not scoped_urls:
        scoped_urls.extend(draft.source_urls)
    scoped_urls = [url for idx, url in enumerate(scoped_urls) if url and url not in scoped_urls[:idx]]

    note_parts = [
        f"automatic full-pitch validation ({reason})",
        f"expected_reference_file={file_entry.name}",
        "goal: confirm the data on the source page(s) is consistent with what the user submitted — same entity, same rough date range, same order of magnitude for prices/values",
        "do NOT fail because of missing columns or format differences — the user may have cleaned or subsetted the data",
        "flag only if the source page shows clearly different data, a different asset, or the page is unreachable",
    ]
    if scoped_urls:
        note_parts.append("expected_source_urls=" + ", ".join(scoped_urls))
    if file_note:
        note_parts.append(f"source_fetch_note={file_note}")
    for item in global_notes:
        note_parts.append(f"global_source_note={item}")

    return "\n".join(note_parts), scoped_urls, expected_format


def _merge_cua_outputs(per_file_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not per_file_outputs:
        return {
            "agent": "data_fetcher",
            "status": "ok",
            "confidence": 1.0,
            "summary": "No uploaded CSV/TSV files required CUA validation; price history is fetched internally for backtesting.",
            "flags": [],
            "artifacts": {"validated_file_names": [], "match_rate": 1.0, "per_file": {}},
            "latency_ms": 0,
        }

    merged_flags: list[dict[str, str]] = []
    total_latency = 0
    confidence_sum = 0.0
    match_rates: list[float] = []
    failed_files: list[str] = []

    for file_name, output in per_file_outputs.items():
        total_latency += int(output.get("latency_ms", 0) or 0)
        confidence_sum += float(output.get("confidence", 0.0) or 0.0)
        file_status = str(output.get("status", "warn")).lower()

        artifacts = output.get("artifacts", {})
        if isinstance(artifacts, dict):
            match = artifacts.get("match_rate")
            if isinstance(match, (int, float)):
                match_rates.append(float(match))

        raw_flags = output.get("flags", [])
        if isinstance(raw_flags, list):
            for raw in raw_flags:
                if not isinstance(raw, dict):
                    continue
                merged_flags.append(
                    {
                        "code": str(raw.get("code", "CUA_FLAG")).upper(),
                        "message": f"[{file_name}] {str(raw.get('message', '')).strip()}",
                    }
                )

        if file_status != "ok":
            failed_files.append(file_name)

    total = len(per_file_outputs)
    match_rate = sum(match_rates) / len(match_rates) if match_rates else 0.0
    confidence = confidence_sum / total if total else 0.0
    all_ok = len(failed_files) == 0
    status = "ok" if all_ok else "warn"
    summary = (
        f"CUA validated all {total} data file(s): no provenance issues found."
        if all_ok
        else f"CUA found potential issues in {len(failed_files)} of {total} data file(s): {', '.join(failed_files)}."
    )
    return {
        "agent": "data_fetcher",
        "status": status,
        "confidence": confidence,
        "summary": summary,
        "flags": merged_flags,
        "artifacts": {
            "validated_file_names": sorted(per_file_outputs.keys()),
            "match_rate": match_rate,
            "per_file": per_file_outputs,
        },
        "latency_ms": total_latency,
    }


async def _auto_validate_all_data_files_with_cua(draft: PitchDraft, reason: str) -> dict[str, Any]:
    files = _files_requiring_cua(draft)
    if not files:
        return _merge_cua_outputs({})

    await cl.Message(
        content=(
            f"Running mandatory CUA provenance checks for {len(files)} data file(s) "
            "before scoring."
        ),
        author="CUA Data Fetcher",
    ).send()

    _auto_cua_loop = asyncio.get_running_loop()

    async def _validate_one(file_entry: UploadedFile) -> tuple[str, dict[str, Any]]:
        notes, source_urls_override, expected_format = _build_cua_context_for_file(draft, file_entry, reason)

        # Per-file live log stream
        _lines: list[str] = []
        _log_msg: list[cl.Message | None] = [None]
        _last_t: list[float] = [0.0]

        async def _flush() -> None:
            txt = "```\n" + "\n".join(_lines[-25:]) + "\n```"
            if _log_msg[0] is None:
                _log_msg[0] = cl.Message(
                    content=txt, author=f"CUA Live — {file_entry.name}"
                )
                await _log_msg[0].send()
            else:
                _log_msg[0].content = txt
                await _log_msg[0].update()

        def _cb(line: str) -> None:
            s = line.strip()
            if not s:
                return
            _lines.append(s)
            now = time.time()
            if now - _last_t[0] > 0.7:
                _last_t[0] = now
                asyncio.run_coroutine_threadsafe(_flush(), _auto_cua_loop)

        try:
            output = await cl.make_async(validate_data_with_cua)(
                draft,
                file_entry.name,
                notes,
                source_urls_override,
                log_callback=_cb,
            )
            if _lines:
                await _flush()
        except Exception as exc:
            output = {
                "agent": "data_fetcher",
                "status": "fail",
                "confidence": 0.0,
                "summary": f"CUA execution failed for {file_entry.name}.",
                "flags": [
                    {
                        "code": "CUA_RUNTIME_ERROR",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                ],
                "artifacts": {"requested_file": file_entry.name, "match_rate": 0.0},
                "latency_ms": 0,
            }
        artifacts = output.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
        artifacts["expected_file_format"] = expected_format
        artifacts["expected_source_urls"] = source_urls_override
        artifacts["expected_fetch_notes"] = notes
        output["artifacts"] = artifacts
        return file_entry.name, output

    outputs: dict[str, dict[str, Any]] = {}
    async with cl.Step(name="CUA Data Fetcher", type="tool") as step:
        step.input = (
            f"Auto-validating {len(files)} file(s) against submitted source URLs "
            "(running all CUAs simultaneously)."
        )
        step.output = f"Running {len(files)} CUA container(s) in parallel..."
        results = await asyncio.gather(*[_validate_one(f) for f in files])
        for name, output in results:
            outputs[name] = output
        step.output = "Completed CUA checks for all required data files."

    combined = _merge_cua_outputs(outputs)
    _set_session_data_fetcher_output(combined)
    output_path = _pitch_dir(draft.pitch_id) / "agent_outputs" / "data_fetcher.json"
    _write_json(output_path, combined)
    _append_chat_event(
        draft,
        "system",
        f"auto_cua status={combined.get('status', 'unknown')} files={','.join(sorted(outputs.keys()))}",
    )
    return combined


def _validation_followup_markdown(summary: str, questions: list[str]) -> str:
    lines = [
        "## Validation Follow-up",
        f"- Summary: {summary}",
    ]
    if questions:
        lines.append("- Please clarify the following:")
        for idx, question in enumerate(questions, start=1):
            lines.append(f"  {idx}. {question}")
    lines.append("- After updates, run `/evaluate` to re-run this loop.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent display names and descriptions
# ---------------------------------------------------------------------------

AGENT_DISPLAY = {
    "data_fetcher": {
        "name": "CUA Data Fetcher",
        "desc": "Browser-based validation of uploaded data against source URLs.",
    },
    "data_validator": {
        "name": "Fabrication Detector",
        "desc": "Checks for intentional data manipulation and fabricated series.",
    },
    "pipeline_auditor": {
        "name": "Pipeline Auditor",
        "desc": "Reviews methodology for leakage, overfitting, and coding errors.",
    },
    "one_shot_validator": {
        "name": "One-Shot Validator",
        "desc": "Evaluates event-driven one-shot causal-chain validity and Monte Carlo edge.",
    },
    "scoring": {
        "name": "Scoring Engine",
        "desc": "Computes composite score, allocation, and final decision.",
    },
}

def _format_flags_md(flags: list[dict[str, str]]) -> str:
    """Render a list of agent flags as markdown lines."""
    if not flags:
        return "No flags."
    lines = []
    for flag in flags:
        lines.append(f"- `{flag.get('code', '?')}` {flag.get('message', '')}")
    return "\n".join(lines)


def _status_icon(status: str) -> str:
    return {"ok": "[OK]", "warn": "[WARN]", "fail": "[FAIL]"}.get(status, "[?]")


async def _render_agent_step(agent_key: str, agent_output: dict[str, Any]) -> None:
    """Create a cl.Step + authored cl.Message for one agent's output."""
    display = AGENT_DISPLAY.get(agent_key, {"name": agent_key, "desc": ""})
    name = display["name"]
    status = agent_output.get("status", "ok")
    confidence = agent_output.get("confidence", 0.0)
    summary = agent_output.get("summary", "")
    flags = agent_output.get("flags", [])
    latency = agent_output.get("latency_ms", 0)

    step_output_lines = [
        f"{_status_icon(status)} **{status.upper()}**  |  Confidence: `{confidence:.0%}`  |  Latency: `{latency}ms`",
        "",
        f"**Summary:** {summary}",
        "",
        "**Flags:**",
        _format_flags_md(flags),
    ]

    async with cl.Step(name=name, type="tool") as step:
        step.input = display["desc"]
        step.output = "\n".join(step_output_lines)

    # Authored message with key findings
    if flags:
        flag_summary = _format_flags_md(flags[:4])
    else:
        flag_summary = "No issues detected."

    await cl.Message(
        content=(
            f"**{_status_icon(status)}** {summary}\n\n"
            f"{flag_summary}"
        ),
        author=name,
    ).send()


def _render_score_card(result: Any) -> str:
    """Build a markdown score card for the final evaluation result."""
    scoring = result.agent_outputs.get("scoring", {})
    artifacts = scoring.get("artifacts", {})
    source = artifacts.get("source", "strategy_scorer_unavailable")
    one_shot_mode = bool(artifacts.get("one_shot_mode"))
    one_shot_recommendation = artifacts.get("one_shot_recommendation")

    lines = [
        "---",
        "## Evaluation Result",
        "",
        f"| | |",
        f"|---|---|",
        f"| **Score** | **{result.overall_score}** / 100 |",
        f"| **Allocation** | **{'N/A (one-shot mode)' if one_shot_mode else f'${result.allocation_usd:,} USD'}** |",
        f"| **Decision** | `{result.decision}` |",
        f"| **Scoring Source** | `{source}` |",
        f"| **Outcome** | `{result.validation_outcome}` |",
        "",
    ]

    if one_shot_mode:
        lines.extend(
            [
                "### One-Shot Recommendation",
                "",
                f"- Recommendation: **{one_shot_recommendation or 'NOT_VALID'}**",
                "",
            ]
        )

    # Add strategy backtest metrics if available
    backtest_metrics = artifacts.get("backtest_metrics")
    if backtest_metrics and source == "strategy_scorer":
        m = backtest_metrics
        lines.extend([
            "### Strategy Backtest Metrics",
            "",
            "| Metric | Strategy | Benchmark |",
            "|--------|----------|-----------|",
            f"| CAGR | `{float(m.get('cagr', 0)):.2%}` | `{float(m.get('benchmark_cagr', 0)):.2%}` |",
            f"| Total Return | `{float(m.get('total_return', 0)):.2%}` | `{float(m.get('benchmark_total_return', 0)):.2%}` |",
            f"| Max Drawdown | `{float(m.get('max_drawdown', 0)):.2%}` | `{float(m.get('benchmark_max_drawdown', 0)):.2%}` |",
            f"| Sharpe | `{float(m.get('sharpe_ratio', 0)):.3f}` | -- |",
            f"| Sortino | `{float(m.get('sortino_ratio', 0)):.3f}` | -- |",
            f"| Alpha | `{float(m.get('alpha', 0)):.4f}` | -- |",
            f"| Win Rate | `{float(m.get('win_rate', 0)):.1%}` | -- |",
            f"| Profit Factor | `{float(m.get('profit_factor', 0)):.2f}` | -- |",
            "",
        ])

        # Component breakdown
        components = artifacts.get("component_scores", {})
        if components:
            lines.append("### Scorer Component Breakdown")
            lines.append("")
            for label, comp in components.items():
                raw = comp.get("score", 0)
                bar = "=" * int(raw * 20) + "-" * (20 - int(raw * 20))
                lines.append(f"- `{comp.get('category', '?')}` **{label}**: `[{bar}]` `{raw:.2f}` (w={comp.get('weight', 0)})")
            lines.append("")

    # Core scoring diagnostics
    if not one_shot_mode:
        lines.extend([
        "### Scoring Breakdown",
        "",
        f"| Component | Value |",
        f"|-----------|-------|",
        f"| Sharpe | `{artifacts.get('sharpe', 0):.4f}` |",
        f"| Max Drawdown | `{artifacts.get('max_drawdown', 0):.4f}` |",
        f"| Risk Score | `{artifacts.get('risk_score', 0):.4f}` |",
        ])

    return "\n".join(lines)


async def _handle_evaluate_command(draft: PitchDraft, command_name: str = "/evaluate") -> None:
    if not draft.ready_for_evaluation():
        _track_paid_usage(draft, "evaluate_blocked", command=command_name, reason="not_ready")
        await cl.Message(
            content=(
                "Evaluation is blocked until required items are complete.\n\n"
                f"{_checklist_markdown(draft)}"
            )
        ).send()
        return

    started_at = asyncio.get_running_loop().time()
    draft.status = "running"
    _save_pitch_snapshot(draft)
    _append_chat_event(draft, "system", "Evaluation started.")
    _track_paid_usage(draft, "evaluate_started", command=command_name)

    # --- TaskList: show all pipeline stages ---
    task_list = cl.TaskList()
    task_list.status = "Running evaluation pipeline..."

    task_data = cl.Task(title="Data Loading & Hard Checks", status=cl.TaskStatus.RUNNING)
    task_backtest = cl.Task(title="Backtest Agent")
    task_fabrication = cl.Task(title="Fabrication Detector")
    task_auditor = cl.Task(title="Pipeline Auditor")
    task_one_shot = cl.Task(title="One-Shot Validator")
    task_scoring = cl.Task(title="Scoring Engine")

    await task_list.add_task(task_data)
    await task_list.add_task(task_backtest)
    await task_list.add_task(task_fabrication)
    await task_list.add_task(task_auditor)
    await task_list.add_task(task_one_shot)
    await task_list.add_task(task_scoring)
    await task_list.send()

    await cl.Message(
        content=f"Starting evaluation pipeline for `{draft.pitch_id}`...",
        author="Orchestrator",
    ).send()

    # Mandatory anti-scam gate: validate every uploaded data file via CUA.
    data_fetcher_output = await _auto_validate_all_data_files_with_cua(draft, reason=command_name)

    # --- Run the monolithic evaluate_pitch (all agents inside) ---
    try:
        result = await cl.make_async(evaluate_pitch)(
            draft, data_fetcher_output=data_fetcher_output
        )
    except Exception as exc:
        elapsed_seconds = round(asyncio.get_running_loop().time() - started_at, 3)
        draft.status = "failed"
        _save_pitch_snapshot(draft)
        _append_chat_event(draft, "system", f"{command_name} failed. Error={exc.__class__.__name__}")
        _track_paid_usage(
            draft,
            "evaluate_failed",
            command=command_name,
            error_type=exc.__class__.__name__,
            elapsed_seconds=elapsed_seconds,
        )
        # Mark all tasks failed
        for t in [task_data, task_backtest, task_fabrication, task_auditor, task_one_shot, task_scoring]:
            t.status = cl.TaskStatus.FAILED
        task_list.status = "Evaluation failed"
        await task_list.send()
        await cl.Message(
            content=(
                "Evaluation failed because an LLM-dependent agent call did not complete.\n"
                f"Error: `{exc.__class__.__name__}: {exc}`\n"
                "Please correct the issue and run `/evaluate`."
            ),
            author="Orchestrator",
        ).send()
        return

    # --- Pipeline completed — render each agent's results ---
    if result.validation_outcome == "blocked_fabrication":
        draft.status = "rejected"
    elif result.validation_outcome == "needs_clarification":
        draft.status = "needs_clarification"
    else:
        draft.status = "ready_for_final_review"
    _save_pitch_snapshot(draft)

    result_path = _pitch_dir(draft.pitch_id) / "result.json"
    result_dict = result.to_dict()
    _write_json(result_path, result_dict)
    completed_at = _now_iso() if draft.status in {"ready_for_final_review", "rejected"} else None
    save_pitch_result(draft.pitch_id, _now_iso(), completed_at, result_dict)
    _append_chat_event(draft, "system", f"{command_name} complete. Decision={result.decision}")
    _track_paid_usage(
        draft,
        "evaluate_completed",
        command=command_name,
        decision=result.decision,
        validation_outcome=result.validation_outcome,
        overall_score=result.overall_score,
        elapsed_seconds=round(asyncio.get_running_loop().time() - started_at, 3),
    )

    # Mark data loading done
    task_data.status = cl.TaskStatus.DONE
    await task_list.send()

    # Check if backtest ran
    scoring_artifacts = result.agent_outputs.get("scoring", {}).get("artifacts", {})
    if scoring_artifacts.get("source") == "one_shot":
        task_backtest.status = cl.TaskStatus.DONE
    elif scoring_artifacts.get("source") == "strategy_scorer":
        task_backtest.status = cl.TaskStatus.DONE
    elif scoring_artifacts.get("backtest_attempt_count", 0) > 0:
        task_backtest.status = cl.TaskStatus.FAILED
    else:
        task_backtest.status = cl.TaskStatus.DONE  # skipped (no strategy script)
    await task_list.send()

    # Render each agent as a Step + authored message
    agent_order = ["data_fetcher", "data_validator", "pipeline_auditor", "one_shot_validator", "scoring"]
    task_map = {
        "data_fetcher": task_data,  # already done
        "data_validator": task_fabrication,
        "pipeline_auditor": task_auditor,
        "one_shot_validator": task_one_shot,
        "scoring": task_scoring,
    }
    for agent_key in agent_order:
        agent_output = result.agent_outputs.get(agent_key, {})
        if not agent_output:
            continue
        task_ref = task_map.get(agent_key)
        if task_ref and task_ref.status != cl.TaskStatus.DONE:
            agent_status = agent_output.get("status", "ok")
            task_ref.status = cl.TaskStatus.DONE if agent_status != "fail" else cl.TaskStatus.FAILED
            await task_list.send()
        await _render_agent_step(agent_key, agent_output)

    # --- Final outcome ---
    task_list.status = f"Evaluation complete — {result.decision}"
    await task_list.send()

    if result.validation_outcome == "blocked_fabrication":
        _set_session_validation_context([])
        await cl.Message(content="Goodbye.", author="Orchestrator").send()
        return

    if result.validation_outcome == "needs_clarification":
        context_items = _session_validation_context()
        # Store summary + all pending questions so the Clarifier can address them conversationally.
        context_entry = result.validation_summary
        if result.validation_questions:
            qs = " | ".join(result.validation_questions)
            context_entry = f"{result.validation_summary} | Pending questions: {qs}"
        context_items.append(context_entry)
        _set_session_validation_context(context_items)
        await cl.Message(
            content=_validation_followup_markdown(result.validation_summary, result.validation_questions),
            author="Orchestrator",
        ).send()
        return

    # Ready for final review — show score card
    _set_session_validation_context([])
    await cl.Message(
        content=_render_score_card(result),
        author="Orchestrator",
    ).send()

    # Hard reject reasons
    if result.hard_reject_reasons:
        lines = ["### Hard Reject Reasons"]
        for reason in result.hard_reject_reasons:
            lines.append(f"- {reason}")
        await cl.Message(content="\n".join(lines), author="Orchestrator").send()


def _parse_validate_data_command(content: str) -> tuple[str | None, str | None, str | None]:
    try:
        parts = shlex.split(content)
    except ValueError as exc:
        return None, None, f"Could not parse command arguments: {exc}"

    if len(parts) < 2:
        return None, None, "Usage: /validate_data \"file_to_validate\" \"notes\""

    file_name = parts[1].strip()
    notes = " ".join(parts[2:]).strip()
    if not file_name:
        return None, None, "File name cannot be empty."
    return file_name, notes, None


async def _handle_backtest_command(draft: PitchDraft) -> None:
    """Run the Claude backtest agent on uploaded strategy .py/.ipynb files."""
    if not _BACKTEST_AVAILABLE:
        _track_paid_usage(draft, "backtest_unavailable", reason="missing_dependency_or_key")
        await cl.Message(
            content=(
                "Backtest agent is unavailable. "
                "Ensure `anthropic` is installed and `ANTHROPIC_API_KEY` is set."
            ),
            author="Backtest Agent",
        ).send()
        return

    roles = _detect_file_roles(draft.uploaded_files, tickers=draft.tickers, thesis=draft.thesis)
    if not roles["strategy_scripts"]:
        _track_paid_usage(draft, "backtest_blocked", reason="no_strategy_script")
        await cl.Message(
            content=(
                "No strategy script found. Upload a `.py` or `.ipynb` file containing your strategy, "
                "then run `/backtest` again."
            ),
            author="Backtest Agent",
        ).send()
        return

    script_names = ", ".join(f"`{f.name}`" for f in roles["strategy_scripts"])
    data_names = ", ".join(f"`{f.name}`" for f in roles["data_files"] + roles["benchmark_files"])
    ticker = draft.tickers[0] if draft.tickers else "UNKNOWN"

    # Prepare file contents
    _strat_files: list[tuple[str, str]] = []
    for _f in roles["strategy_scripts"]:
        try:
            _strat_files.append(_load_strategy_source_for_backtest(_f))
        except Exception as exc:
            _track_paid_usage(
                draft,
                "backtest_prepare_failed",
                file_name=_f.name,
                error_type=exc.__class__.__name__,
            )
            await cl.Message(
                content=f"Could not prepare strategy file `{_f.name}`: {exc}",
                author="Backtest Agent",
            ).send()
            return

    _data_files: list[tuple[str, str]] = []
    data_read_errors: list[str] = []
    for _f in roles["data_files"] + roles["benchmark_files"]:
        try:
            _data_files.append((_f.name, Path(_f.path).read_text(encoding="utf-8", errors="replace")))
        except Exception as exc:
            data_read_errors.append(f"{_f.name}: {type(exc).__name__}: {exc}")
    if data_read_errors:
        _track_paid_usage(
            draft,
            "backtest_prepare_failed",
            reason="data_file_read_error",
            error_count=len(data_read_errors),
        )
        details = "\n".join(f"- {item}" for item in data_read_errors)
        await cl.Message(
            content=(
                "Could not read one or more data files for backtest execution.\n"
                f"{details}"
            ),
            author="Backtest Agent",
        ).send()
        return

    await cl.Message(
        content=(
            f"Starting backtest for {script_names}\n"
            f"Data files: {data_names or '(none -- will use Alpaca API)'}"
        ),
        author="Backtest Agent",
    ).send()

    started_at = asyncio.get_running_loop().time()
    _track_paid_usage(
        draft,
        "backtest_started",
        script_count=len(roles["strategy_scripts"]),
        data_file_count=len(roles["data_files"]) + len(roles["benchmark_files"]),
    )

    # Run with nested steps for each phase
    async with cl.Step(name="Backtest Agent", type="run") as parent_step:
        parent_step.input = f"Strategy: {script_names} | Ticker: {ticker}"

        async with cl.Step(name="Phase 1: Generate Runner", type="llm") as s1:
            s1.input = "Claude generates a standardised backtest runner script."
            s1.output = "Generating..."

        async with cl.Step(name="Phase 2: Execute Backtest", type="tool") as s2:
            s2.input = f"Running backtest with {BACKTEST_TIMEOUT_SECONDS}s timeout."
            s2.output = "Executing..."

        _flavor_msg = await cl.Message(
            content="⏳ **Backtest running…** Initializing agent...",
            author="Backtest Agent",
        ).send()
        _flavor_task = asyncio.create_task(_backtest_flavor_task(_flavor_msg))
        try:
            result = await cl.make_async(_run_backtest_agent)(
                strategy_files=_strat_files,
                data_files=_data_files,
                pitch_context={"name": draft.pitch_id, "ticker": ticker},
            )
        finally:
            _flavor_task.cancel()
            try:
                await _flavor_task
            except asyncio.CancelledError:
                pass
            _flavor_msg.content = "✅ **Backtest execution complete.**"
            await _flavor_msg.update()

        async with cl.Step(name="Phase 3: Validate Output", type="tool") as s3:
            s3.input = "Claude reviews the JSON output for completeness and sanity."
            s3.output = f"Verdict: `{result.status}` after {result.attempt_count} attempt(s)."

        # Update parent step
        parent_step.output = f"Status: `{result.status}` | Attempts: {result.attempt_count}"

    # Save backtest agent output
    backtest_path = _pitch_dir(draft.pitch_id) / "agent_outputs" / "backtest_agent.json"
    _write_json(backtest_path, result.to_dict())
    _append_chat_event(
        draft,
        "system",
        f"/backtest status={result.status} attempts={result.attempt_count}",
    )
    _track_paid_usage(
        draft,
        "backtest_completed",
        status=result.status,
        attempts=result.attempt_count,
        has_metrics=bool(result.metrics),
        elapsed_seconds=round(asyncio.get_running_loop().time() - started_at, 3),
    )

    if result.status == "success" and result.metrics:
        m = result.metrics
        lines = [
            f"## Backtest Complete",
            f"Completed in **{result.attempt_count}** attempt(s).\n",
            "### Key Metrics",
            "",
            "| Metric | Strategy | Benchmark |",
            "|--------|----------|-----------|",
            f"| CAGR | `{float(m.get('cagr', 0)):.2%}` | `{float(m.get('benchmark_cagr', 0)):.2%}` |",
            f"| Total Return | `{float(m.get('total_return', 0)):.2%}` | `{float(m.get('benchmark_total_return', 0)):.2%}` |",
            f"| Max Drawdown | `{float(m.get('max_drawdown', 0)):.2%}` | `{float(m.get('benchmark_max_drawdown', 0)):.2%}` |",
            f"| Sharpe | `{float(m.get('sharpe_ratio', 0)):.3f}` | -- |",
            f"| Sortino | `{float(m.get('sortino_ratio', 0)):.3f}` | -- |",
            f"| Calmar | `{float(m.get('calmar_ratio', 0)):.3f}` | -- |",
            f"| Excess Return | `{float(m.get('excess_return', 0)):.2%}` | -- |",
            f"| Alpha | `{float(m.get('alpha', 0)):.4f}` | -- |",
            f"| Win Rate | `{float(m.get('win_rate', 0)):.1%}` | -- |",
            f"| Total Trades | `{m.get('total_trades', 0)}` | -- |",
            f"| Profit Factor | `{float(m.get('profit_factor', 0)):.2f}` | -- |",
            f"| Up Capture | `{float(m.get('up_capture', 0)):.2f}` | -- |",
            f"| Down Capture | `{float(m.get('down_capture', 0)):.2f}` | -- |",
            "",
            "Run `/evaluate` to include these results in the full evaluation pipeline.",
        ]
        await cl.Message(content="\n".join(lines), author="Backtest Agent").send()

        # --- Equity curve chart (optional — only present if backtest script emitted curves) ---
        equity_curve = m.get("equity_curve")
        benchmark_curve = m.get("benchmark_curve")
        if equity_curve and isinstance(equity_curve, list) and len(equity_curve) > 1:
            try:
                import plotly.graph_objects as go  # noqa: PLC0415
                from plotly.subplots import make_subplots  # noqa: PLC0415

                dates_s = [pt[0] for pt in equity_curve]
                vals_s = [pt[1] for pt in equity_curve]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates_s, y=vals_s,
                    name="Strategy",
                    line=dict(color="#00e676", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0,230,118,0.07)",
                ))

                if benchmark_curve and isinstance(benchmark_curve, list) and len(benchmark_curve) > 1:
                    dates_b = [pt[0] for pt in benchmark_curve]
                    vals_b = [pt[1] for pt in benchmark_curve]
                    fig.add_trace(go.Scatter(
                        x=dates_b, y=vals_b,
                        name="Benchmark (B&H)",
                        line=dict(color="#9e9e9e", width=1.5, dash="dot"),
                    ))

                fig.update_layout(
                    title="📈 Equity Curve (normalised, start = 1.0)",
                    xaxis_title="Date",
                    yaxis_title="Portfolio value",
                    template="plotly_dark",
                    height=380,
                    margin=dict(l=40, r=20, t=50, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                await cl.Plotly(figure=fig, display="inline", size="large").send()
            except Exception as _chart_exc:
                _LOGGER.debug("Equity curve chart failed: %s", _chart_exc)

    elif result.status == "user_action_required":
        await cl.Message(
            content=(
                f"## User Action Required\n\n"
                f"{result.message}\n\n"
                "Please fix the issue above and re-upload your files, then run `/backtest` again."
            ),
            author="Backtest Agent",
        ).send()

    else:  # agent_fault
        await cl.Message(
            content=(
                f"## Could Not Complete\n\n"
                f"{result.message}\n\n"
                "Evaluation requires strategy_scorer output. Please resolve the backtest issue, then run `/evaluate` again."
            ),
            author="Backtest Agent",
        ).send()


async def _handle_validate_data_command(draft: PitchDraft, content: str) -> None:
    file_name, notes, error = _parse_validate_data_command(content)
    if error:
        _track_paid_usage(draft, "validate_data_blocked", reason=error)
        await cl.Message(content=error).send()
        return

    source_urls = ", ".join(draft.source_urls) if draft.source_urls else "(none)"

    await cl.Message(
        content=f"Validating `{file_name}` against source URLs. This can take a few minutes.",
        author="CUA Data Fetcher",
    ).send()

    started_at = asyncio.get_running_loop().time()
    _track_paid_usage(
        draft,
        "validate_data_started",
        file_name=file_name,
        source_url_count=len(draft.source_urls),
        has_notes=bool(notes),
    )

    # --- Live log streaming callback for CUA ---
    _cua_loop = asyncio.get_running_loop()
    _cua_log_lines: list[str] = []
    _cua_log_msg: list[cl.Message | None] = [None]  # mutable container for nonlocal
    _cua_last_update: list[float] = [0.0]

    async def _flush_cua_log() -> None:
        content = "```\n" + "\n".join(_cua_log_lines[-25:]) + "\n```"
        if _cua_log_msg[0] is None:
            _cua_log_msg[0] = cl.Message(content=content, author="CUA Live Feed")
            await _cua_log_msg[0].send()
        else:
            _cua_log_msg[0].content = content
            await _cua_log_msg[0].update()

    def _cua_log_cb(line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return
        _cua_log_lines.append(stripped)
        now = time.time()
        if now - _cua_last_update[0] > 0.7:
            _cua_last_update[0] = now
            asyncio.run_coroutine_threadsafe(_flush_cua_log(), _cua_loop)

    async with cl.Step(name="CUA Data Fetcher", type="tool") as step:
        step.input = f"File: `{file_name}` | Sources: {source_urls} | Notes: {notes or '(none)'}"
        step.output = "Running browser automation via Docker..."

        output = await cl.make_async(validate_data_with_cua)(
            draft, file_name or "", notes or "", log_callback=_cua_log_cb
        )
        # Final flush
        if _cua_log_lines:
            await _flush_cua_log()

        _set_session_data_fetcher_output(output)

        flags = output.get("flags", []) if isinstance(output.get("flags"), list) else []
        step.output = (
            f"**Status:** `{output.get('status', 'unknown')}`\n"
            f"**Summary:** {output.get('summary', 'CUA run completed.')}\n\n"
            f"**Flags:**\n{_format_flags_md(flags)}"
        )

    output_path = _pitch_dir(draft.pitch_id) / "agent_outputs" / "data_fetcher.json"
    _write_json(output_path, output)
    _append_chat_event(
        draft,
        "system",
        f"/validate_data file={file_name} status={output.get('status', 'unknown')}",
    )
    _track_paid_usage(
        draft,
        "validate_data_completed",
        file_name=file_name,
        status=output.get("status", "unknown"),
        flag_count=len(flags),
        elapsed_seconds=round(asyncio.get_running_loop().time() - started_at, 3),
    )

    flags = output.get("flags", []) if isinstance(output.get("flags"), list) else []
    await cl.Message(
        content=(
            f"**Status:** `{output.get('status', 'unknown')}`\n"
            f"**Summary:** {output.get('summary', '') or 'CUA run completed.'}\n\n"
            f"{_format_flags_md(flags)}\n\n"
            "This result will be used automatically in the next evaluation run."
        ),
        author="CUA Data Fetcher",
    ).send()


async def _handle_oneshot_command(draft: PitchDraft, content: str) -> None:
    parts = content.split()
    action = parts[1].lower() if len(parts) > 1 else "status"
    if action not in {"on", "off", "status"}:
        await cl.Message(content="Usage: `/oneshot on`, `/oneshot off`, or `/oneshot status`").send()
        return

    if action == "on":
        draft.one_shot_mode = True
        _save_pitch_snapshot(draft)
        await cl.Message(
            content=(
                "One-shot mode enabled. Evaluation will return binary recommendation (`VALID` / `NOT_VALID`) "
                "with no USD allocation."
            )
        ).send()
        return
    if action == "off":
        draft.one_shot_mode = False
        _save_pitch_snapshot(draft)
        await cl.Message(content="One-shot mode disabled. Standard allocation scoring is active.").send()
        return

    await cl.Message(content=f"One-shot mode is currently `{draft.one_shot_mode}`.").send()


async def _execute_orchestrator_actions(draft: PitchDraft, actions: list[dict[str, str]]) -> bool:
    handled = False
    for entry in actions:
        action = entry.get("action", "")
        reason = entry.get("reason", "").strip() or "Clarifier requested this pipeline step."

        if action == "run_backtest":
            handled = True
            await cl.Message(content=f"Clarifier action: running backtest. ({reason})", author="Orchestrator").send()
            await _handle_backtest_command(draft)
            continue

        if action == "run_validate_data":
            handled = True
            file_name = (entry.get("file_name") or "").strip()
            if not file_name:
                await cl.Message(
                    content="Clarifier requested `/validate_data`, but no `file_name` was provided.",
                    author="Orchestrator",
                ).send()
                continue
            notes = (entry.get("notes") or "").strip()
            quoted_notes = f' "{notes}"' if notes else ""
            await cl.Message(
                content=f"Clarifier action: validating data file `{file_name}`. ({reason})",
                author="Orchestrator",
            ).send()
            await _handle_validate_data_command(draft, f'/validate_data "{file_name}"{quoted_notes}')
            continue

        if action == "run_evaluate":
            handled = True
            await cl.Message(content=f"Clarifier action: running `/evaluate`. ({reason})", author="Orchestrator").send()
            await _handle_evaluate_command(draft, "/evaluate")
            continue

    return handled


@cl.on_chat_start
async def on_chat_start() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    draft = _new_pitch()
    _set_session_draft(draft)
    _set_session_history([])
    _set_session_validation_context([])
    _set_session_data_fetcher_output(None)

    # --- Onboarding step 1: strategy type ---
    type_res = await cl.AskActionMessage(
        content="**Step 1 of 2 — Pitch type**\n\nIs this a recurring strategy or a single one-shot trade?",
        actions=[
            cl.Action(name="strategy", payload={"value": "strategy"}, label="📈 Recurring strategy"),
            cl.Action(name="one_shot", payload={"value": "one_shot"}, label="⚡ One-shot trade"),
        ],
        timeout=300,
    ).send()
    if type_res and type_res.get("payload", {}).get("value") == "one_shot":
        draft.one_shot_mode = True

    # --- Onboarding step 2: time horizon ---
    horizon_res = await cl.AskActionMessage(
        content="**Step 2 of 2 — Time horizon**\n\nWhat is the expected holding period?",
        actions=[
            cl.Action(name="days", payload={"value": "days"}, label="Days (intraday â€“ 1 week)"),
            cl.Action(name="weeks", payload={"value": "weeks"}, label="Weeks (1 â€“ 8 weeks)"),
            cl.Action(name="months", payload={"value": "months"}, label="Months (2 â€“ 12 months)"),
            cl.Action(name="years", payload={"value": "years"}, label="Years (1 year+)"),
        ],
        timeout=300,
    ).send()
    if horizon_res:
        draft.time_horizon = horizon_res.get("payload", {}).get("value") or None

    _save_pitch_snapshot(draft)
    _track_paid_usage(
        draft,
        "chat_started",
        one_shot_mode=bool(draft.one_shot_mode),
        time_horizon=draft.time_horizon or "",
    )

    greeting = (
        "# Quant Pitch Evaluator\n\n"
        "Submit your quantitative trading strategy for structured evaluation.\n"
        "I will guide you through required fields, then automatically run the full validation pipeline.\n\n"
        "---\n\n"
        "**Agents in the pipeline:**\n"
        "- **Clarifier Agent** -- extracts structured fields from your natural-language pitch\n"
        "- **Backtest Agent** -- generates and runs a standardised backtest on your `.py` or `.ipynb` strategy\n"
        "- **Fabrication Detector** -- checks for intentional data manipulation\n"
        "- **Pipeline Auditor** -- reviews methodology for leakage and coding errors\n"
        "- **CUA Data Fetcher** -- browser-based validation of data against source URLs\n"
        "- **Scoring Engine** -- computes composite score and capital allocation\n\n"
        "---\n\n"
        f"{COMMANDS_TEXT}\n\n"
        "You do not need to run commands for normal flow; evaluation starts automatically when intake is complete.\n"
        "For event-driven one-shot theses, include `one_shot_mode=true` in your message or use `/oneshot on`.\n"
        "Start by describing your thesis and stock ticker(s).\n"
        "Then upload your strategy `.py`/`.ipynb` file.\n"
        "If you upload additional supporting CSV/TSV datasets, include source URL(s) for provenance checks."
    )
    await cl.Message(content=greeting, author="Orchestrator").send()
    await cl.Message(content=_checklist_markdown(draft)).send()
    if _get_client() is None:
        raise RuntimeError("GEMINI_API_KEY is not set. Startup failed because clarifier requires Gemini.")


@cl.on_message
async def on_message(message: cl.Message) -> None:
    draft = _session_draft()
    content = (message.content or "").strip()
    added_files = _ingest_message_files(draft, message)
    if added_files:
        _append_chat_event(draft, "system", f"Uploaded files: {', '.join(added_files)}")
        _track_paid_usage(
            draft,
            "files_uploaded",
            file_count=len(added_files),
            file_names=added_files,
        )
        await cl.Message(content=f"Added {len(added_files)} file(s): {', '.join(added_files)}").send()

    if not content and not added_files:
        await cl.Message(content="Send a message or attach at least one file.").send()
        return

    if content.startswith("/"):
        command = content.split()[0].lower()
        if command == "/help":
            await cl.Message(content=COMMANDS_TEXT).send()
            return
        if command == "/status":
            await cl.Message(content=_status_markdown(draft)).send()
            return
        if command == "/checklist":
            await cl.Message(content=_checklist_markdown(draft)).send()
            return
        if command == "/oneshot":
            await _handle_oneshot_command(draft, content)
            return
        if command == "/evaluate":
            await _handle_evaluate_command(draft, "/evaluate")
            return
        if command == "/validate":
            await cl.Message(content="`/validate` was removed. Use `/evaluate`.").send()
            return
        if command == "/backtest":
            await _handle_backtest_command(draft)
            return
        if command == "/validate_data":
            await _handle_validate_data_command(draft, content)
            return
        if command == "/reset":
            new_draft = _new_pitch()
            _set_session_draft(new_draft)
            _set_session_history([])
            _set_session_validation_context([])
            _set_session_data_fetcher_output(None)
            _save_pitch_snapshot(new_draft)
            await cl.Message(content=f"Started a new pitch: `{new_draft.pitch_id}`").send()
            return
        await cl.Message(content="Unknown command. Use `/help`.").send()
        return

    if content:
        _append_chat_event(draft, "user", content)
        _track_paid_usage(
            draft,
            "user_message",
            content_length=len(content),
            attached_file_count=len(added_files),
        )
        heuristic_update_from_user_text(draft, content)
        _apply_clarification_update(draft, content)
        lowered = content.lower()
        if "one_shot_mode=true" in lowered or "one-shot mode on" in lowered:
            draft.one_shot_mode = True
        elif "one_shot_mode=false" in lowered or "one-shot mode off" in lowered:
            draft.one_shot_mode = False

    clarifier_actions: list[dict[str, str]] = []
    if content:
        try:
            assistant_reply, clarifier_actions = await _run_clarifier_turn(draft, content)
            _append_chat_event(draft, "assistant", assistant_reply)
        except Exception as exc:
            draft.status = "failed"
            _save_pitch_snapshot(draft)
            _append_chat_event(
                draft,
                "system",
                f"Clarifier failure: {exc.__class__.__name__}: {exc}",
            )
            await cl.Message(
                content=(
                    "Clarifier failed because the Gemini call did not complete.\n"
                    f"Error: `{exc.__class__.__name__}: {exc}`\n"
                    "Fix the issue and resend your message."
                ),
                author="Clarifier Agent",
            ).send()
            return
    else:
        assistant_reply = _local_clarifier_reply(draft)

    if draft.ready_for_evaluation() and draft.status == "draft":
        draft.status = "ready"

    _save_pitch_snapshot(draft)

    # Stream the clarifier reply word-by-word for a live typing effect, then
    # append the checklist once streaming is complete.
    _cl_stream_msg = cl.Message(content="", author="Clarifier Agent")
    await _cl_stream_msg.send()
    _reply_words = assistant_reply.split(" ")
    for _wi, _wt in enumerate(_reply_words):
        await _cl_stream_msg.stream_token(_wt + (" " if _wi < len(_reply_words) - 1 else ""))
        await asyncio.sleep(0.012)  # ~80 words/sec — fast but visibly animated
    _cl_stream_msg.content = f"{assistant_reply}\n\n{_checklist_markdown(draft)}"
    await _cl_stream_msg.update()

    actions_handled = await _execute_orchestrator_actions(draft, clarifier_actions) if clarifier_actions else False

    # ChatGPT-style orchestration: auto-advance into evaluation once intake is complete.
    if not actions_handled and draft.ready_for_evaluation() and draft.status in {"ready", "needs_clarification"}:
        await cl.Message(
            content="Intake is complete. Running full anti-scam validation and evaluation now.",
            author="Orchestrator",
        ).send()
        await _handle_evaluate_command(draft, "/auto_evaluate")
