from __future__ import annotations

import json
import re
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
    _detect_file_roles,
    _load_strategy_source_for_backtest,
    evaluate_pitch,
    extract_tagged_json,
    file_sha256,
    heuristic_update_from_user_text,
    strip_tagged_json,
    validate_data_with_cua,
)

try:
    from backtest_agent import run_backtest_agent as _run_backtest_agent
    _BACKTEST_AVAILABLE = True
except ImportError:
    _BACKTEST_AVAILABLE = False

load_dotenv()

import os as _os
BACKTEST_TIMEOUT_SECONDS = int(_os.getenv("BACKTEST_TIMEOUT_SECONDS", "120"))

DATA_ROOT = Path("data/pitches")

SYSTEM_PROMPT = """
You are the Clarifier Agent for a quant pitch intake flow.
Your goals:
1) Help the user produce a clear investment thesis.
2) Confirm target stock tickers and time horizon.
3) Request source URLs when missing.
4) Remind the user to upload their strategy files (.py or .ipynb) and/or price data (.csv) if not yet attached.
5) Keep responses short and practical.

At the end of every response, include one XML block with JSON:
<pitch_state>{"thesis": "...", "time_horizon": "days|weeks|months|years|null", "tickers": [], "source_urls": [], "methodology_summary": "...", "one_shot_mode": false, "ready_for_evaluation": false}</pitch_state>

Rules:
- If you are uncertain about a field, return the current best value or empty string.
- `tickers` must always be a JSON array of stock tickers (e.g., ["AAPL", "MSFT"]).
- `source_urls` must always be a JSON array.
- Keep conversational text before the XML block.
- Do not include extra XML blocks.
- Do NOT require methodology_summary — it is optional. Focus on thesis, tickers, time_horizon, source_urls, and file uploads.
""".strip()

COMMANDS_TEXT = """
Optional commands:
- `/status` show current pitch completeness
- `/checklist` show onboarding checklist
- `/oneshot on|off|status` explicitly toggle one-shot validation mode
- `/evaluate` run validation and scoring
- `/validate` re-run the validation loop after clarifications
- `/backtest` run Claude backtest agent on uploaded strategy script (.py or .ipynb)
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
        f"- One-shot mode: `{draft.one_shot_mode}`\n"
        f"- Time horizon: `{draft.time_horizon or '(missing)'}`\n"
        f"- Tickers: {tickers}\n"
        f"- Source URLs (required): {urls}\n"
        f"- Uploaded files: {files}\n"
        f"- Missing fields: `{', '.join(missing) if missing else 'none'}`"
    )


def _checklist_markdown(draft: PitchDraft) -> str:
    checks = [
        ("Strategy mode explicitly set (`/oneshot on` for one-shot, otherwise standard)", True),
        ("Thesis (1-3 sentences)", bool(draft.thesis.strip())),
        ("Time horizon (`days`, `weeks`, `months`, `years`)", bool(draft.time_horizon)),
        ("Stock tickers (e.g., `AAPL, MSFT`)", len(draft.tickers) > 0),
        ("Source URL(s) for submitted data", len(draft.source_urls) > 0),
        ("Strategy files (`.py`/`.ipynb` script or `.csv` data)", len(draft.uploaded_files) > 0),
    ]
    lines = ["## Onboarding Checklist"]
    for label, done in checks:
        lines.append(f"- {'✅' if done else '⬜'} {label}")
    missing = draft.missing_fields()
    lines.append("")
    lines.append(f"- Ready: `{draft.ready_for_evaluation()}`")
    lines.append(f"- Missing: `{', '.join(missing) if missing else 'none'}`")
    lines.append("- Upload your strategy `.py` or `.ipynb` file and/or price data `.csv` using the attachment button.")
    return "\n".join(lines)


def _local_clarifier_reply(draft: PitchDraft) -> str:
    missing = draft.missing_fields()
    if not missing:
        return "Your pitch looks complete. I will automatically run validation now."

    prompt_map = {
        "thesis": "State the thesis in one sentence: what is mispriced and why now?",
        "time_horizon": "Pick a horizon: days, weeks, months, or years.",
        "tickers": "Share the stock ticker(s), for example: AAPL or AAPL, MSFT.",
        "source_urls": "Share source URL(s) for every submitted dataset so we can verify provenance.",
        "uploaded_files": "Upload your strategy file (`.py` or `.ipynb`) and/or price data (`.csv`) using the attachment button below.",
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
        async with cl.Step(name="Clarifier Agent", type="llm") as step:
            step.input = f"Analyzing user input and extracting pitch fields..."
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
                step.output = f"Extracted fields: {json.dumps(extracted, indent=2)}"
            else:
                step.output = "No structured fields extracted this turn."

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


def _files_requiring_cua(draft: PitchDraft) -> list[UploadedFile]:
    roles = _detect_file_roles(draft.uploaded_files)
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
    hints = _parse_source_fetch_hints(draft.methodology_summary or "")
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
        f"expected_file_format={expected_format}",
        "strict requirement: downloaded source artifact must match reference schema/entity/date range",
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
            "status": "fail",
            "confidence": 0.0,
            "summary": "No data files were available for mandatory CUA validation.",
            "flags": [
                {
                    "code": "CUA_DATA_FILES_MISSING",
                    "message": "Upload at least one CSV/TSV data file so CUA can validate source provenance.",
                }
            ],
            "artifacts": {"validated_file_names": [], "match_rate": 0.0, "per_file": {}},
            "latency_ms": 0,
        }

    merged_flags: list[dict[str, str]] = []
    total_latency = 0
    confidence_sum = 0.0
    match_rates: list[float] = []
    ok_count = 0

    for file_name, output in per_file_outputs.items():
        total_latency += int(output.get("latency_ms", 0) or 0)
        confidence_sum += float(output.get("confidence", 0.0) or 0.0)
        status = str(output.get("status", "warn")).lower()
        if status == "ok":
            ok_count += 1

        artifacts = output.get("artifacts", {})
        if isinstance(artifacts, dict):
            match = artifacts.get("match_rate")
            if isinstance(match, (int, float)):
                match_rates.append(float(match))
            expected_format = str(artifacts.get("expected_file_format", "")).strip().lower()
            expected_urls = artifacts.get("expected_source_urls", [])
            checked_urls = artifacts.get("source_urls_checked", [])
            candidate_downloads = artifacts.get("candidate_downloaded_files", [])

            if expected_format:
                valid_ext = {expected_format}
                if expected_format in {"csv", "tsv"}:
                    valid_ext = {"csv", "tsv"}
                has_matching_format = False
                if isinstance(candidate_downloads, list):
                    for candidate in candidate_downloads:
                        ext = Path(str(candidate)).suffix.lower().lstrip(".")
                        if ext in valid_ext:
                            has_matching_format = True
                            break
                if not has_matching_format:
                    merged_flags.append(
                        {
                            "code": "CUA_EXPECTED_FORMAT_MISMATCH",
                            "message": f"[{file_name}] No downloaded artifact matched expected format `{expected_format}`.",
                        }
                    )
                    status = "fail"

            if isinstance(expected_urls, list) and expected_urls:
                expected_set = {str(url).strip() for url in expected_urls if str(url).strip()}
                checked_set = set()
                if isinstance(checked_urls, list):
                    checked_set = {str(url).strip() for url in checked_urls if str(url).strip()}
                if not expected_set.issubset(checked_set):
                    merged_flags.append(
                        {
                            "code": "CUA_EXPECTED_SOURCE_MISSING",
                            "message": f"[{file_name}] CUA result did not include expected source link(s).",
                        }
                    )
                    status = "fail"

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

        if status != "ok":
            merged_flags.append(
                {
                    "code": "CUA_FILE_NOT_CLEAN",
                    "message": f"[{file_name}] CUA validation did not return status=ok.",
                }
            )

    total = len(per_file_outputs)
    match_rate = sum(match_rates) / len(match_rates) if match_rates else 0.0
    confidence = confidence_sum / total if total else 0.0
    all_ok = ok_count == total
    status = "ok" if all_ok else "fail"
    summary = (
        f"CUA validated all {total} data file(s) successfully."
        if all_ok
        else f"CUA found unresolved issues in {total - ok_count} of {total} data file(s)."
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

    outputs: dict[str, dict[str, Any]] = {}
    async with cl.Step(name="CUA Data Fetcher", type="tool") as step:
        step.input = f"Auto-validating {len(files)} file(s) against submitted source URLs."
        for index, file_entry in enumerate(files, start=1):
            step.output = f"Validating `{file_entry.name}` ({index}/{len(files)})..."
            notes, source_urls_override, expected_format = _build_cua_context_for_file(draft, file_entry, reason)
            try:
                output = await cl.make_async(validate_data_with_cua)(
                    draft,
                    file_entry.name,
                    notes,
                    source_urls_override,
                )
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
            outputs[file_entry.name] = output
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
    lines.append("- After updates, run `/validate` to re-run this loop.")
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
        f"**Status:** {_status_icon(status)} `{status}`  |  **Confidence:** `{confidence:.0%}`  |  **Latency:** `{latency}ms`",
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
    source = artifacts.get("source", "csv_approximation")
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

    # CSV-based scoring breakdown
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
        draft.status = "failed"
        _save_pitch_snapshot(draft)
        _append_chat_event(draft, "system", f"{command_name} failed. Error={exc.__class__.__name__}")
        # Mark all tasks failed
        for t in [task_data, task_backtest, task_fabrication, task_auditor, task_one_shot, task_scoring]:
            t.status = cl.TaskStatus.FAILED
        task_list.status = "Evaluation failed"
        await task_list.send()
        await cl.Message(
            content=(
                "Evaluation failed because validator agents could not run.\n"
                "Please ensure `GEMINI_API_KEY` is configured and retry."
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
    _write_json(result_path, result.to_dict())
    _append_chat_event(draft, "system", f"{command_name} complete. Decision={result.decision}")

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
        context_items.append(result.validation_summary)
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
        await cl.Message(
            content=(
                "Backtest agent is unavailable. "
                "Ensure `anthropic` is installed and `ANTHROPIC_API_KEY` is set."
            ),
            author="Backtest Agent",
        ).send()
        return

    roles = _detect_file_roles(draft.uploaded_files)
    if not roles["strategy_scripts"]:
        await cl.Message(
            content=(
                "No strategy script found. Upload a `.py` or `.ipynb` file containing your strategy "
                "and optionally a price data CSV, then run `/backtest` again."
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
            await cl.Message(
                content=f"Could not prepare strategy file `{_f.name}`: {exc}",
                author="Backtest Agent",
            ).send()
            return

    _data_files: list[tuple[str, str]] = []
    for _f in roles["data_files"] + roles["benchmark_files"]:
        try:
            _data_files.append((_f.name, Path(_f.path).read_text(encoding="utf-8", errors="replace")))
        except Exception:
            pass

    await cl.Message(
        content=(
            f"Starting backtest for {script_names}\n"
            f"Data files: {data_names or '(none -- will use Alpaca API)'}"
        ),
        author="Backtest Agent",
    ).send()

    # Run with nested steps for each phase
    async with cl.Step(name="Backtest Agent", type="run") as parent_step:
        parent_step.input = f"Strategy: {script_names} | Ticker: {ticker}"

        async with cl.Step(name="Phase 1: Generate Runner", type="llm") as s1:
            s1.input = "Claude generates a standardised backtest runner script."
            s1.output = "Generating..."

        async with cl.Step(name="Phase 2: Execute Backtest", type="tool") as s2:
            s2.input = f"Running backtest with {BACKTEST_TIMEOUT_SECONDS}s timeout."
            s2.output = "Executing..."

        result = await cl.make_async(_run_backtest_agent)(
            strategy_files=_strat_files,
            data_files=_data_files,
            pitch_context={"name": draft.pitch_id, "ticker": ticker},
        )

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
                "The evaluation will fall back to CSV-based scoring when you run `/evaluate`."
            ),
            author="Backtest Agent",
        ).send()


async def _handle_validate_data_command(draft: PitchDraft, content: str) -> None:
    file_name, notes, error = _parse_validate_data_command(content)
    if error:
        await cl.Message(content=error).send()
        return

    source_urls = ", ".join(draft.source_urls) if draft.source_urls else "(none)"

    await cl.Message(
        content=f"Validating `{file_name}` against source URLs. This can take a few minutes.",
        author="CUA Data Fetcher",
    ).send()

    async with cl.Step(name="CUA Data Fetcher", type="tool") as step:
        step.input = f"File: `{file_name}` | Sources: {source_urls} | Notes: {notes or '(none)'}"
        step.output = "Running browser automation via Docker..."

        output = await cl.make_async(validate_data_with_cua)(draft, file_name or "", notes or "")

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
        "Start by describing your thesis, stock ticker(s), and source URL(s).\n"
        "Then upload your strategy `.py`/`.ipynb` file and/or price data `.csv`."
    )
    await cl.Message(content=greeting, author="Orchestrator").send()
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
        if command == "/oneshot":
            await _handle_oneshot_command(draft, content)
            return
        if command == "/evaluate":
            await _handle_evaluate_command(draft, "/evaluate")
            return
        if command == "/validate":
            await _handle_evaluate_command(draft, "/validate")
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
        heuristic_update_from_user_text(draft, content)
        _apply_clarification_update(draft, content)
        lowered = content.lower()
        if "one_shot_mode=true" in lowered or "one-shot mode on" in lowered:
            draft.one_shot_mode = True
        elif "one_shot_mode=false" in lowered or "one-shot mode off" in lowered:
            draft.one_shot_mode = False

    if content:
        assistant_reply = await _run_clarifier_turn(draft, content)
        _append_chat_event(draft, "assistant", assistant_reply)
    else:
        assistant_reply = _local_clarifier_reply(draft)

    if draft.ready_for_evaluation() and draft.status == "draft":
        draft.status = "ready"

    _save_pitch_snapshot(draft)
    await cl.Message(content=f"{assistant_reply}\n\n{_checklist_markdown(draft)}", author="Clarifier Agent").send()

    # ChatGPT-style orchestration: auto-advance into evaluation once intake is complete.
    if draft.ready_for_evaluation() and draft.status in {"ready", "needs_clarification"}:
        await cl.Message(
            content="Intake is complete. Running full anti-scam validation and evaluation now.",
            author="Orchestrator",
        ).send()
        await _handle_evaluate_command(draft, "/auto_evaluate")
