from __future__ import annotations

import json
import shlex
import shutil
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
    evaluate_pitch,
    extract_tagged_json,
    file_sha256,
    heuristic_update_from_user_text,
    strip_tagged_json,
    validate_data_with_cua,
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
- `/evaluate` run validation and scoring
- `/validate` re-run the validation loop after clarifications
- `/validate_data "file_to_validate" "notes"` run CUA data-source validation for one uploaded file
- `/reset` start a new pitch
- `/help` show commands
""".strip()


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


def _session_validation_context() -> list[str]:
    return cl.user_session.get("validation_context", [])


def _set_session_validation_context(items: list[str]) -> None:
    cl.user_session.set("validation_context", items[-3:])


def _session_data_fetcher_output() -> dict[str, Any] | None:
    output = cl.user_session.get("data_fetcher_output")
    return output if isinstance(output, dict) else None


def _set_session_data_fetcher_output(output: dict[str, Any] | None) -> None:
    cl.user_session.set("data_fetcher_output", output)


def _apply_clarification_update(draft: PitchDraft, user_text: str) -> None:
    if draft.status != "needs_clarification":
        return
    text = user_text.strip()
    if not text:
        return
    # Preserve concise clarification evidence for validator reruns.
    if draft.methodology_summary:
        draft.methodology_summary = f"{draft.methodology_summary}\n\nClarification:\n{text}"
    else:
        draft.methodology_summary = text


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

    try:
        response = await cl.make_async(client.models.generate_content)(
            model=_gemini_model(),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=700,
                system_instruction=SYSTEM_PROMPT,
            ),
        )
        raw_text = (getattr(response, "text", "") or "").strip()
        extracted = extract_tagged_json(raw_text, "pitch_state")
        if extracted:
            draft.merge_structured_update(extracted)
        assistant_text = strip_tagged_json(raw_text, "pitch_state") or _local_clarifier_reply(draft)
        history.append({"role": "assistant", "content": assistant_text})
        _set_session_history(history)
        return assistant_text
    except Exception as exc:
        fallback = (
            "Gemini call failed for this turn, so I used local intake mode. "
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


def _validation_followup_markdown(summary: str, questions: list[str]) -> str:
    lines = [
        "## Validation Follow-up",
        f"- Summary: {summary}",
    ]
    if questions:
        lines.append("- Please clarify the following:")
        for idx, question in enumerate(questions, start=1):
            lines.append(f"  {idx}. {question}")
    lines.append("- After updates, run `/validate` to re-run this loop.")
    return "\n".join(lines)


async def _handle_evaluate_command(draft: PitchDraft, command_name: str = "/evaluate") -> None:
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

    try:
        result = evaluate_pitch(draft, data_fetcher_output=_session_data_fetcher_output())
    except Exception as exc:
        draft.status = "failed"
        _save_pitch_snapshot(draft)
        _append_chat_event(draft, "system", f"{command_name} failed. Error={exc.__class__.__name__}")
        await cl.Message(
            content=(
                "Evaluation failed because validator agents could not run.\n"
                "Please ensure `GEMINI_API_KEY` is configured and retry."
            )
        ).send()
        return
    if result.validation_outcome == "blocked_fabrication":
        draft.status = "rejected"
    elif result.validation_outcome == "needs_clarification":
        draft.status = "needs_clarification"
    else:
        draft.status = "ready_for_final_review"
    _save_pitch_snapshot(draft)

    result_path = _pitch_dir(draft.pitch_id) / "result.json"
    _write_json(result_path, result.to_dict())
    _append_chat_event(draft, "system", f"{command_name} complete. Decision={result.decision}")

    if result.validation_outcome == "blocked_fabrication":
        _set_session_validation_context([])
        await cl.Message(content="Goodbye.").send()
        return

    if result.validation_outcome == "needs_clarification":
        context_items = _session_validation_context()
        context_items.append(result.validation_summary)
        _set_session_validation_context(context_items)
        await cl.Message(
            content=_validation_followup_markdown(result.validation_summary, result.validation_questions)
        ).send()
        return

    _set_session_validation_context([])
    await cl.Message(content="Congrats! Ready for final review.").send()
    await cl.Message(content=result.report_markdown).send()


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


async def _handle_validate_data_command(draft: PitchDraft, content: str) -> None:
    file_name, notes, error = _parse_validate_data_command(content)
    if error:
        await cl.Message(content=error).send()
        return

    await cl.Message(content=f"Starting CUA validation for `{file_name}`. This can take a few minutes.").send()
    output = await cl.make_async(validate_data_with_cua)(draft, file_name or "", notes or "")
    _set_session_data_fetcher_output(output)

    output_path = _pitch_dir(draft.pitch_id) / "agent_outputs" / "data_fetcher.json"
    _write_json(output_path, output)
    _append_chat_event(
        draft,
        "system",
        f"/validate_data file={file_name} status={output.get('status', 'unknown')}",
    )

    flags = output.get("flags", []) if isinstance(output.get("flags"), list) else []
    if flags:
        flag_lines = "\n".join(
            f"- `{flag.get('severity', 'medium')}` `{flag.get('code', 'CUA_ISSUE')}`: {flag.get('message', '')}"
            for flag in flags[:6]
        )
    else:
        flag_lines = "- none"

    await cl.Message(
        content=(
            "## CUA Data Validation\n"
            f"- Status: `{output.get('status', 'unknown')}`\n"
            f"- Summary: {output.get('summary', '') or 'CUA run completed.'}\n"
            f"- Flags:\n{flag_lines}\n\n"
            "Run `/evaluate` to include this CUA result in the full decision pipeline."
        )
    ).send()


@cl.on_chat_start
async def on_chat_start() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    draft = _new_pitch()
    _set_session_draft(draft)
    _set_session_history([])
    _set_session_validation_context([])
    _set_session_data_fetcher_output(None)
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
            content="`GEMINI_API_KEY` is not detected. Running in local clarifier mode only."
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
            await _handle_evaluate_command(draft, "/evaluate")
            return
        if command == "/validate":
            await _handle_evaluate_command(draft, "/validate")
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
        heuristic_update_from_user_text(draft, content)
        _apply_clarification_update(draft, content)

    if content:
        assistant_reply = await _run_clarifier_turn(draft, content)
        _append_chat_event(draft, "assistant", assistant_reply)
    else:
        assistant_reply = _local_clarifier_reply(draft)

    if draft.ready_for_evaluation() and draft.status == "draft":
        draft.status = "ready"

    _save_pitch_snapshot(draft)
    await cl.Message(content=f"{assistant_reply}\n\n{_checklist_markdown(draft)}").send()
