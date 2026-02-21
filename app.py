from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chainlit as cl
from anthropic import Anthropic
from dotenv import load_dotenv

from pitch_engine import (
    PitchDraft,
    UploadedFile,
    evaluate_pitch,
    extract_tagged_json,
    file_sha256,
    heuristic_update_from_user_text,
    strip_tagged_json,
)

load_dotenv()

DATA_ROOT = Path("data/pitches")

SYSTEM_PROMPT = """
You are the Clarifier Agent for a quant pitch intake flow.
Your goals:
1) Help the user produce a clear investment thesis.
2) Confirm methodology, target stock tickers, and time horizon.
3) Request source URLs when missing.
4) Keep responses short and practical.

At the end of every response, include one XML block with JSON:
<pitch_state>{"thesis": "...", "time_horizon": "days|weeks|months|years|null", "tickers": [], "source_urls": [], "methodology_summary": "...", "ready_for_evaluation": false}</pitch_state>

Rules:
- If you are uncertain about a field, return the current best value or empty string.
- `tickers` must always be a JSON array of stock tickers (e.g., ["AAPL", "MSFT"]).
- `source_urls` must always be a JSON array.
- Keep conversational text before the XML block.
- Do not include extra XML blocks.
""".strip()

COMMANDS_TEXT = """
Commands:
- `/status` show current pitch completeness
- `/checklist` show onboarding checklist
- `/evaluate` run v0 scoring and allocation
- `/reset` start a new pitch
- `/help` show commands
""".strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _anthropic_model() -> str:
    import os

    return os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")


def _get_client() -> Anthropic | None:
    import os

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None
    return Anthropic(api_key=api_key)


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
    _write_json(draft_path, draft.to_dict())


def _append_chat_event(draft: PitchDraft, role: str, content: str) -> None:
    history_path = _pitch_dir(draft.pitch_id) / "clarifier_history.jsonl"
    _append_jsonl(
        history_path,
        {
            "timestamp_utc": _now_iso(),
            "role": role,
            "content": content,
        },
    )


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
        f"- Time horizon: `{draft.time_horizon or '(missing)'}`\n"
        f"- Tickers: {tickers}\n"
        f"- Methodology summary: `{draft.methodology_summary or '(missing)'}`\n"
        f"- Source URLs (required): {urls}\n"
        f"- Uploaded files: {files}\n"
        f"- Missing fields: `{', '.join(missing) if missing else 'none'}`"
    )


def _checklist_markdown(draft: PitchDraft) -> str:
    checks = [
        ("Thesis (1-3 sentences)", bool(draft.thesis.strip())),
        ("Time horizon (`days`, `weeks`, `months`, `years`)", bool(draft.time_horizon)),
        ("Stock tickers (e.g., `AAPL, MSFT`)", len(draft.tickers) > 0),
        ("Methodology summary", bool(draft.methodology_summary.strip())),
        ("Source URL(s) for submitted data", len(draft.source_urls) > 0),
    ]
    lines = ["## Onboarding Checklist"]
    for label, done in checks:
        lines.append(f"- {'✅' if done else '⬜'} {label}")
    missing = draft.missing_fields()
    lines.append("")
    lines.append(f"- Ready: `{draft.ready_for_evaluation()}`")
    lines.append(f"- Missing: `{', '.join(missing) if missing else 'none'}`")
    lines.append("- Note: Source URLs are required for data provenance checks.")
    return "\n".join(lines)


def _local_clarifier_reply(draft: PitchDraft) -> str:
    missing = draft.missing_fields()
    if not missing:
        return "Your pitch looks complete. Run `/evaluate` when ready."

    prompt_map = {
        "thesis": "State the thesis in one sentence: what is mispriced and why now?",
        "time_horizon": "Pick a horizon: days, weeks, months, or years.",
        "tickers": "Share the stock ticker(s), for example: AAPL or AAPL, MSFT.",
        "methodology_summary": "Describe your method briefly: data used, signal construction, and validation.",
        "source_urls": "Share source URL(s) for every submitted dataset so we can verify provenance.",
    }
    first_missing = missing[0]
    return f"I captured your latest details. Next: {prompt_map.get(first_missing, first_missing)}"


async def _run_clarifier_turn(draft: PitchDraft, user_text: str) -> str:
    client = _get_client()
    if client is None:
        return _local_clarifier_reply(draft)

    context_payload = json.dumps(draft.to_llm_context(), ensure_ascii=True)
    wrapped_input = f"PITCH_CONTEXT={context_payload}\nUSER_MESSAGE={user_text}"
    history = _session_history()
    history.append({"role": "user", "content": wrapped_input})

    try:
        response = await cl.make_async(client.messages.create)(
            model=_anthropic_model(),
            max_tokens=700,
            temperature=0.2,
            system=SYSTEM_PROMPT,
            messages=history,
        )
        blocks = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                blocks.append(getattr(block, "text", ""))
        raw_text = "\n".join(blocks).strip()
        extracted = extract_tagged_json(raw_text, "pitch_state")
        if extracted:
            draft.merge_structured_update(extracted)
        assistant_text = strip_tagged_json(raw_text, "pitch_state") or _local_clarifier_reply(draft)
        history.append({"role": "assistant", "content": assistant_text})
        _set_session_history(history)
        return assistant_text
    except Exception as exc:
        fallback = (
            "Anthropic call failed for this turn, so I used local intake mode. "
            f"Error: {exc.__class__.__name__}"
        )
        history.append({"role": "assistant", "content": fallback})
        _set_session_history(history)
        return f"{fallback}\n\n{_local_clarifier_reply(draft)}"


def _copy_to_pitch_uploads(draft: PitchDraft, src_path: Path, original_name: str, mime_type: str, size: int) -> UploadedFile:
    upload_dir = _pitch_dir(draft.pitch_id) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / original_name
    if destination.exists():
        destination = upload_dir / f"{uuid.uuid4().hex[:8]}_{original_name}"
    shutil.copy2(src_path, destination)
    return UploadedFile(
        file_id=f"fil_{uuid.uuid4().hex[:12]}",
        name=original_name,
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


async def _handle_evaluate_command(draft: PitchDraft) -> None:
    if not draft.ready_for_evaluation():
        await cl.Message(
            content=(
                "Evaluation is blocked until required items are complete.\n\n"
                f"{_checklist_markdown(draft)}"
            )
        ).send()
        return

    draft.status = "running"
    _save_pitch_snapshot(draft)
    _append_chat_event(draft, "system", "Evaluation started.")

    result = evaluate_pitch(draft)
    draft.status = "completed"
    _save_pitch_snapshot(draft)

    result_path = _pitch_dir(draft.pitch_id) / "result.json"
    _write_json(result_path, result.to_dict())
    _append_chat_event(draft, "system", f"Evaluation complete. Decision={result.decision}")

    await cl.Message(content=result.report_markdown).send()


@cl.on_chat_start
async def on_chat_start() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    draft = _new_pitch()
    _set_session_draft(draft)
    _set_session_history([])
    _save_pitch_snapshot(draft)

    greeting = (
        "# Quant Pitch Evaluator\n"
        "Describe your pitch naturally. I will guide you through a checklist before evaluation.\n\n"
        f"{COMMANDS_TEXT}\n\n"
        "Tip: start with thesis + stock ticker(s) + source URL(s)."
    )
    await cl.Message(content=greeting).send()
    await cl.Message(content=_checklist_markdown(draft)).send()
    if _get_client() is None:
        await cl.Message(
            content="`ANTHROPIC_API_KEY` is not detected. Running in local clarifier mode only."
        ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    draft = _session_draft()
    content = (message.content or "").strip()
    added_files = _ingest_message_files(draft, message)
    if added_files:
        _append_chat_event(draft, "system", f"Uploaded files: {', '.join(added_files)}")
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
        if command == "/evaluate":
            await _handle_evaluate_command(draft)
            return
        if command == "/reset":
            new_draft = _new_pitch()
            _set_session_draft(new_draft)
            _set_session_history([])
            _save_pitch_snapshot(new_draft)
            await cl.Message(content=f"Started a new pitch: `{new_draft.pitch_id}`").send()
            return
        await cl.Message(content="Unknown command. Use `/help`.").send()
        return

    if content:
        _append_chat_event(draft, "user", content)
        heuristic_update_from_user_text(draft, content)

    if content:
        assistant_reply = await _run_clarifier_turn(draft, content)
        _append_chat_event(draft, "assistant", assistant_reply)
    else:
        assistant_reply = _local_clarifier_reply(draft)

    if draft.ready_for_evaluation() and draft.status == "draft":
        draft.status = "ready"

    _save_pitch_snapshot(draft)
    await cl.Message(content=f"{assistant_reply}\n\n{_checklist_markdown(draft)}").send()
