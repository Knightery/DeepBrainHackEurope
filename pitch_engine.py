from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

REQUIRED_FIELDS = ("thesis", "time_horizon", "methodology_summary")
TIME_HORIZON_VALUES = {"days", "weeks", "months", "years"}
SEVERITY_LEVELS = {"low", "medium", "high", "critical"}


def _gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-3.1-pro")
VALIDATION_OUTCOME_BLOCKED = "blocked_fabrication"
VALIDATION_OUTCOME_CLARIFY = "needs_clarification"
VALIDATION_OUTCOME_READY = "ready_for_final_review"

FABRICATION_VALIDATOR_PROMPT = """
You are the Fabrication Detector for quant pitch evaluation.

Objective:
- Find signs of intentional cheating or fabricated/manipulated market data.
- Be conservative: only flag when there is concrete evidence or strong statistical suspicion.

Checklist:
1) Impossible values (negative volume, impossible prices, repeated constants where variation is expected)
2) Overly smooth or synthetic-looking returns inconsistent with market behavior
3) Timestamp anomalies that suggest post-editing (non-monotonic or suspiciously regular patterns)
4) Data/source mismatch clues between submitted data summary and declared source URLs
5) Evidence of suspicious manual tampering patterns

Output rules:
- Return exactly one XML block with strict JSON:
<validator_output>{"summary":"...","confidence":0.0,"flags":[{"code":"...","severity":"low|medium|high|critical","message":"..."}],"artifacts":{"verdict":"clean|fabrication|unclear","questions":[],"checklist_findings":[]}}</validator_output>
- Use short, specific flag messages with direct evidence from input.
- If no material concerns, return empty flags.
""".strip()

CODING_ERRORS_VALIDATOR_PROMPT = """
You are the Coding Errors Detector for quant pitch evaluation.

Objective:
- Find unintentional ML/quant methodology mistakes that inflate backtest quality.
- Focus on methodological risk, not fraud intent.

Checklist:
1) Look-ahead bias / future leakage
2) Target leakage from features
3) Survivorship bias in ticker universe framing
4) Overfitting or data snooping risk
5) Weak validation protocol (no clear out-of-sample / walk-forward split)
6) Missing transaction-cost/slippage realism for active strategies

Output rules:
- Return exactly one XML block with strict JSON:
<validator_output>{"summary":"...","confidence":0.0,"flags":[{"code":"...","severity":"low|medium|high|critical","message":"..."}],"artifacts":{"verdict":"clean|needs_clarification|unclear","questions":[],"checklist_findings":[]}}</validator_output>
- Use actionable, concise flag messages.
- If no material concerns, return empty flags.
""".strip()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def round_to_100(value: float) -> int:
    return int(round(value / 100.0) * 100)


def normalize_time_horizon(raw_value: str | None) -> str | None:
    if not raw_value:
        return None
    text = raw_value.strip().lower()
    if text in TIME_HORIZON_VALUES:
        return text
    if "day" in text:
        return "days"
    if "week" in text:
        return "weeks"
    if "month" in text:
        return "months"
    if "year" in text or "annual" in text or "long" in text:
        return "years"
    return None


def extract_urls(text: str) -> list[str]:
    pattern = r"https?://[^\s)]+"
    urls = re.findall(pattern, text)
    return sorted(set(urls))


def normalize_ticker(raw: str) -> str | None:
    token = raw.strip().upper().lstrip("$")
    if not token:
        return None
    if re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z]{1,2})?", token):
        return token
    return None


def parse_tickers(value: Any) -> list[str]:
    if value is None:
        return []

    candidates: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                candidates.extend(re.split(r"[\s,;|]+", item.strip()))
    elif isinstance(value, str):
        candidates.extend(re.split(r"[\s,;|]+", value.strip()))
    else:
        return []

    normalized = [normalize_ticker(candidate) for candidate in candidates]
    return sorted({item for item in normalized if item})


def extract_tickers_from_text(text: str) -> list[str]:
    extracted: list[str] = []

    tagged_line_pattern = r"(?:ticker|tickers|stock|stocks|symbol|symbols)\s*[:\-]\s*([^\n]+)"
    for line in re.findall(tagged_line_pattern, text, flags=re.IGNORECASE):
        extracted.extend(parse_tickers(line))

    dollar_prefixed = re.findall(r"\$[A-Za-z]{1,5}(?:\.[A-Za-z]{1,2})?", text)
    extracted.extend([item for item in (normalize_ticker(token) for token in dollar_prefixed) if item])

    short_text = text.strip()
    if len(short_text) <= 60 and "," in short_text:
        extracted.extend(parse_tickers(short_text))

    return sorted(set(extracted))


def extract_tagged_json(text: str, tag: str = "pitch_state") -> dict[str, Any] | None:
    pattern = rf"<{tag}>\s*(\{{.*?\}})\s*</{tag}>"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def strip_tagged_json(text: str, tag: str = "pitch_state") -> str:
    pattern = rf"\s*<{tag}>.*?</{tag}>"
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_source_urls(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        merged: list[str] = []
        for item in value:
            if isinstance(item, str):
                merged.extend(extract_urls(item) or [item.strip()])
        return [item for item in sorted(set(merged)) if item]
    if isinstance(value, str):
        urls = extract_urls(value)
        if urls:
            return urls
        split_candidates = [part.strip() for part in value.split(",")]
        return [part for part in split_candidates if part]
    return []


@dataclass
class UploadedFile:
    file_id: str
    name: str
    path: str
    mime_type: str = ""
    size_bytes: int = 0
    sha256: str = ""


@dataclass
class PitchDraft:
    pitch_id: str
    created_at: str
    status: str = "draft"
    thesis: str = ""
    time_horizon: str | None = None
    tickers: list[str] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)
    methodology_summary: str = ""
    submitter: dict[str, Any] = field(default_factory=dict)
    uploaded_files: list[UploadedFile] = field(default_factory=list)

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        for field_name in REQUIRED_FIELDS:
            value = getattr(self, field_name)
            if not value:
                missing.append(field_name)
        if not self.tickers:
            missing.append("tickers")
        if not self.source_urls:
            missing.append("source_urls")
        return missing

    def ready_for_evaluation(self) -> bool:
        return len(self.missing_fields()) == 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["uploaded_files"] = [asdict(file) for file in self.uploaded_files]
        return payload

    def to_llm_context(self) -> dict[str, Any]:
        return {
            "pitch_id": self.pitch_id,
            "status": self.status,
            "thesis": self.thesis,
            "time_horizon": self.time_horizon,
            "tickers": self.tickers,
            "source_urls": self.source_urls,
            "methodology_summary": self.methodology_summary,
            "uploaded_file_names": [entry.name for entry in self.uploaded_files],
            "missing_fields": self.missing_fields(),
            "ready_for_evaluation": self.ready_for_evaluation(),
        }

    def merge_structured_update(self, data: dict[str, Any]) -> None:
        thesis = data.get("thesis")
        if isinstance(thesis, str) and thesis.strip():
            self.thesis = thesis.strip()

        horizon = normalize_time_horizon(data.get("time_horizon"))
        if horizon:
            self.time_horizon = horizon

        incoming_tickers = parse_tickers(data.get("tickers"))
        if incoming_tickers:
            self.tickers = sorted(set(self.tickers + incoming_tickers))

        methodology = data.get("methodology_summary")
        if isinstance(methodology, str) and methodology.strip():
            self.methodology_summary = methodology.strip()

        incoming_urls = parse_source_urls(data.get("source_urls"))
        if incoming_urls:
            self.source_urls = sorted(set(self.source_urls + incoming_urls))


def heuristic_update_from_user_text(draft: PitchDraft, text: str) -> None:
    urls = extract_urls(text)
    if urls:
        draft.source_urls = sorted(set(draft.source_urls + urls))

    tickers = extract_tickers_from_text(text)
    if tickers:
        draft.tickers = sorted(set(draft.tickers + tickers))

    horizon = normalize_time_horizon(text)
    if horizon and not draft.time_horizon:
        draft.time_horizon = horizon

    lowered = text.lower()
    if not draft.thesis and len(text.strip()) > 30:
        if any(token in lowered for token in ("thesis", "idea", "mispriced", "expect", "edge")):
            draft.thesis = text.strip()

    if not draft.methodology_summary and len(text.strip()) > 40:
        if any(token in lowered for token in ("method", "backtest", "data", "model", "signal", "features")):
            draft.methodology_summary = text.strip()


def _load_first_table(uploaded_files: list[UploadedFile]) -> tuple[pd.DataFrame | None, list[str]]:
    errors: list[str] = []
    for file_entry in uploaded_files:
        path = Path(file_entry.path)
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".tsv"}:
            continue
        try:
            sep = "\t" if suffix == ".tsv" else ","
            frame = pd.read_csv(path, sep=sep)
            if frame.empty:
                errors.append(f"{file_entry.name}: file is empty")
                continue
            return frame, errors
        except Exception as exc:  # pragma: no cover - depends on user files
            errors.append(f"{file_entry.name}: {exc}")
    return None, errors


def _pick_close_column(frame: pd.DataFrame) -> str | None:
    lowered_map = {column.lower(): column for column in frame.columns}
    for candidate in ("close", "adj_close", "adj close", "price", "last"):
        if candidate in lowered_map:
            return lowered_map[candidate]
    numeric_columns = frame.select_dtypes(include=["number"]).columns.tolist()
    if numeric_columns:
        return numeric_columns[0]
    return None


def _parse_timestamps(frame: pd.DataFrame) -> pd.Series | None:
    lowered_map = {column.lower(): column for column in frame.columns}
    for candidate in ("timestamp_utc", "timestamp", "date", "datetime"):
        column = lowered_map.get(candidate)
        if not column:
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
        if parsed.notna().sum() > 0:
            return parsed
    return None


def _methodology_score(text: str) -> tuple[float, list[str]]:
    if not text.strip():
        return 0.0, ["Methodology is missing."]

    warnings: list[str] = []
    words = len(text.split())
    score = 0.2

    if words >= 30:
        score += 0.2
    else:
        warnings.append("Methodology is short; add details on validation and assumptions.")

    keyword_weights = {
        "out-of-sample": 0.15,
        "walk-forward": 0.15,
        "validation": 0.1,
        "risk": 0.1,
        "drawdown": 0.1,
        "leakage": 0.1,
        "assumption": 0.1,
    }
    lowered = text.lower()
    for keyword, weight in keyword_weights.items():
        if keyword in lowered:
            score += weight

    if "out-of-sample" not in lowered and "walk-forward" not in lowered:
        warnings.append("No explicit out-of-sample or walk-forward validation noted.")

    return clamp(score), warnings


def _match_rate(draft: PitchDraft) -> float:
    if draft.source_urls and draft.uploaded_files:
        return 0.8
    if draft.source_urls or draft.uploaded_files:
        return 0.5
    return 0.0


def _compute_allocation(score: float, horizon: str | None) -> int:
    if score < 55:
        base = 0
    elif score < 65:
        base = 1000
    elif score < 75:
        base = 2500
    elif score < 85:
        base = 5000
    elif score < 93:
        base = 10000
    else:
        base = 15000

    multiplier_map = {"days": 0.8, "weeks": 1.0, "months": 1.2, "years": 1.4}
    multiplier = multiplier_map.get(horizon or "", 1.0)

    allocation = round_to_100(base * multiplier)
    return min(20000, allocation)


def _safe_head_records(frame: pd.DataFrame | None, limit: int = 40) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    try:
        json_text = frame.head(limit).to_json(orient="records", date_format="iso")
        return json.loads(json_text)
    except Exception:
        return []


def _series_stats(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {}
    return {
        "count": float(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _build_validator_payload(
    draft: PitchDraft,
    frame: pd.DataFrame | None,
    close_series: pd.Series,
    load_errors: list[str],
    match_rate: float,
    sharpe: float,
    max_drawdown: float,
) -> dict[str, Any]:
    return {
        "pitch": {
            "pitch_id": draft.pitch_id,
            "thesis": draft.thesis,
            "time_horizon": draft.time_horizon,
            "tickers": draft.tickers,
            "source_urls": draft.source_urls,
            "methodology_summary": draft.methodology_summary,
        },
        "data_summary": {
            "row_count": int(frame.shape[0]) if frame is not None else 0,
            "column_count": int(frame.shape[1]) if frame is not None else 0,
            "columns": list(frame.columns) if frame is not None else [],
            "head_records": _safe_head_records(frame),
            "close_stats": _series_stats(close_series),
            "load_errors": load_errors,
        },
        "scoring_context": {
            "match_rate": round(match_rate, 4),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_drawdown, 4),
        },
    }


def _extract_text_blocks(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    blocks: list[str] = []
    for block in content:
        text_value = getattr(block, "text", None)
        if isinstance(text_value, str):
            blocks.append(text_value)
        elif isinstance(block, dict) and block.get("type") == "text":
            blocks.append(str(block.get("text", "")))
    return "\n".join(blocks).strip()


def _normalize_flags(flags: Any) -> list[dict[str, str]]:
    if not isinstance(flags, list):
        return []
    normalized: list[dict[str, str]] = []
    for raw_flag in flags:
        if not isinstance(raw_flag, dict):
            continue
        code = str(raw_flag.get("code", "")).strip().upper() or "UNSPECIFIED"
        severity = str(raw_flag.get("severity", "medium")).strip().lower()
        message = str(raw_flag.get("message", "")).strip()
        if severity not in SEVERITY_LEVELS:
            severity = "medium"
        if not message:
            continue
        normalized.append({"code": code, "severity": severity, "message": message})
    return normalized


def _status_from_flags(flags: list[dict[str, str]]) -> str:
    severities = {flag["severity"] for flag in flags}
    if "critical" in severities:
        return "fail"
    if "high" in severities or "medium" in severities:
        return "warn"
    return "ok"


def _summarize_flags(flag_groups: list[list[dict[str, str]]], limit: int = 4) -> str:
    merged: list[str] = []
    for flags in flag_groups:
        for flag in flags:
            message = flag.get("message", "").strip()
            if message:
                merged.append(message)
    if not merged:
        return "No material validation concerns."
    return " | ".join(merged[:limit])


def _question_from_flag(flag: dict[str, str]) -> str | None:
    code = flag.get("code", "").upper()
    message = flag.get("message", "").lower()
    if "SOURCE" in code or "source" in message or "url" in message:
        return "Please clarify the exact public source URL and how the submitted file was derived from it."
    if "ASSUMPTION" in code or "assumption" in message:
        return "Please list your key modeling assumptions and why they are realistic."
    if "LOOKAHEAD" in code or "leak" in message:
        return "Please explain how your train/test split prevents look-ahead leakage."
    if "SURVIV" in code:
        return "Please explain how you avoided survivorship bias in the ticker universe."
    if "OVERFIT" in code or "snoop" in message:
        return "Please describe your out-of-sample or walk-forward validation setup."
    if "COST" in code or "slippage" in message:
        return "Please include assumptions for transaction costs and slippage."
    return None


def _collect_validation_questions(
    fetcher_flags: list[dict[str, str]],
    validator_flags: list[dict[str, str]],
    auditor_flags: list[dict[str, str]],
    coding_artifacts: dict[str, Any],
) -> list[str]:
    questions: list[str] = []
    maybe_questions = coding_artifacts.get("questions", [])
    if isinstance(maybe_questions, list):
        for item in maybe_questions:
            text = str(item).strip()
            if text:
                questions.append(text)
    for flag in fetcher_flags + validator_flags + auditor_flags:
        generated = _question_from_flag(flag)
        if generated:
            questions.append(generated)
    deduped = list(dict.fromkeys(questions))
    return deduped[:4]


def _run_llm_validator(agent_name: str, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for validator agents.")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=_gemini_model(),
            system_instruction=system_prompt,
        )
        response = model.generate_content(
            json.dumps(payload, ensure_ascii=True),
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 900,
            },
        )
        raw_text = (getattr(response, "text", "") or "").strip()
        parsed = extract_tagged_json(raw_text, "validator_output")
        if parsed is None:
            raise ValueError("Missing or invalid validator_output JSON block")

        flags = _normalize_flags(parsed.get("flags"))
        confidence = clamp(parsed.get("confidence", 0.5))
        summary = str(parsed.get("summary", "")).strip() or f"{agent_name} completed."
        artifacts = parsed.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}

        return {
            "status": _status_from_flags(flags),
            "confidence": confidence,
            "summary": summary,
            "flags": flags,
            "artifacts": artifacts,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
    except Exception as exc:
        return {
            "status": "warn",
            "confidence": 0.2,
            "summary": f"{agent_name} failed; falling back to non-LLM checks.",
            "flags": [
                {
                    "code": "LLM_VALIDATOR_ERROR",
                    "severity": "high",
                    "message": f"{exc.__class__.__name__} while running LLM validator.",
                }
            ],
            "artifacts": {"verdict": "unclear", "questions": [], "checklist_findings": []},
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }


def _run_llm_validators(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        fabrication_future = executor.submit(
            _run_llm_validator,
            "fabrication_detector",
            FABRICATION_VALIDATOR_PROMPT,
            payload,
        )
        coding_future = executor.submit(
            _run_llm_validator,
            "coding_errors_detector",
            CODING_ERRORS_VALIDATOR_PROMPT,
            payload,
        )
        return {
            "fabrication_detector": fabrication_future.result(),
            "coding_errors_detector": coding_future.result(),
        }


@dataclass
class EvaluationResult:
    pitch_id: str
    status: str
    overall_score: float
    allocation_usd: int
    decision: str
    validation_outcome: str
    user_facing_message: str
    validation_summary: str
    validation_questions: list[str]
    hard_reject_reasons: list[str]
    agent_outputs: dict[str, Any]
    report_markdown: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_pitch(draft: PitchDraft) -> EvaluationResult:
    frame, load_errors = _load_first_table(draft.uploaded_files)
    method_score, methodology_warnings = _methodology_score(draft.methodology_summary)

    hard_reject_reasons: list[str] = []
    validator_flags: list[dict[str, str]] = []
    auditor_flags: list[dict[str, str]] = []
    fetcher_flags: list[dict[str, str]] = []

    if not draft.source_urls:
        hard_reject_reasons.append("No source URL was provided for submitted data.")
        fetcher_flags.append(
            {
                "code": "MISSING_SOURCE_URLS",
                "severity": "critical",
                "message": "Provide source URL(s) for all submitted datasets.",
            }
        )

    if not draft.tickers:
        hard_reject_reasons.append("No stock ticker(s) were provided.")
        validator_flags.append(
            {
                "code": "MISSING_TICKERS",
                "severity": "critical",
                "message": "Provide the stock ticker(s) for this pitch (for example: AAPL, MSFT).",
            }
        )

    close_series = pd.Series(dtype=float)
    row_count = 0
    symbol_count = 1
    sharpe = 0.0
    max_drawdown = -1.0
    risk_score = 1.0
    data_quality_score = 0.0
    time_to_return_days: int | None = None

    if frame is None:
        hard_reject_reasons.append("No parseable CSV/TSV data was found for scoring.")
        validator_flags.append(
            {
                "code": "INSUFFICIENT_DATA",
                "severity": "critical",
                "message": "Upload a non-empty CSV or TSV with a close price column.",
            }
        )
    else:
        row_count = int(frame.shape[0])
        if "symbol" in frame.columns:
            symbol_count = int(frame["symbol"].nunique(dropna=True)) or 1

        close_column = _pick_close_column(frame)
        if close_column is None:
            hard_reject_reasons.append("No numeric price column found for scoring.")
            validator_flags.append(
                {
                    "code": "MISSING_PRICE_COLUMN",
                    "severity": "critical",
                    "message": "Expected one of close/adj_close/price columns.",
                }
            )
        else:
            close_series = pd.to_numeric(frame[close_column], errors="coerce").dropna()

        if len(close_series) < 30:
            hard_reject_reasons.append("Fewer than 30 valid price rows.")
            validator_flags.append(
                {
                    "code": "INSUFFICIENT_DATA",
                    "severity": "critical",
                    "message": "At least 30 valid rows are required.",
                }
            )

        if len(close_series) > 1:
            returns = close_series.pct_change().replace([math.inf, -math.inf], pd.NA).dropna()
            if len(returns) > 0:
                mean_return = float(returns.mean())
                std_return = float(returns.std(ddof=1))
                sharpe = (math.sqrt(252.0) * mean_return / std_return) if std_return > 1e-12 else 0.0

                equity_curve = (1.0 + returns).cumprod()
                running_max = equity_curve.cummax()
                drawdown = equity_curve / running_max - 1.0
                max_drawdown = float(drawdown.min())

                annualized_vol = std_return * math.sqrt(252.0)
                risk_score = clamp(annualized_vol / 0.60)

                timestamps = _parse_timestamps(frame)
                if timestamps is not None and len(timestamps) >= len(equity_curve):
                    aligned = timestamps.iloc[-len(equity_curve):].reset_index(drop=True)
                    positive_idx = (equity_curve - 1.0 > 0).to_numpy().nonzero()[0]
                    if len(positive_idx) > 0:
                        start_ts = aligned.iloc[0]
                        end_ts = aligned.iloc[int(positive_idx[0])]
                        if pd.notna(start_ts) and pd.notna(end_ts):
                            delta_days = int((end_ts - start_ts).total_seconds() // 86400)
                            time_to_return_days = max(delta_days, 0)

        total_cells = float(max(frame.shape[0] * max(frame.shape[1], 1), 1))
        missing_fraction = float(frame.isna().sum().sum()) / total_cells
        duplicate_fraction = float(frame.duplicated().mean()) if frame.shape[0] > 0 else 0.0
        data_quality_score = clamp(1.0 - missing_fraction - (0.5 * duplicate_fraction))

    fetcher_flags.append(
        {
            "code": "CUA_NOT_CONNECTED",
            "severity": "low",
            "message": "CUA fetch run is not yet wired from Chainlit; using placeholder match score.",
        }
    )

    for warning in methodology_warnings:
        auditor_flags.append({"code": "METHOD_WARN", "severity": "medium", "message": warning})

    sharpe_n = clamp((sharpe + 1.0) / 3.0)
    drawdown_n = 1.0 - clamp(abs(max_drawdown) / 0.60)
    risk_n = 1.0 - risk_score
    match_rate = _match_rate(draft)

    validator_payload = _build_validator_payload(
        draft=draft,
        frame=frame,
        close_series=close_series,
        load_errors=load_errors,
        match_rate=match_rate,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
    )
    llm_validators = _run_llm_validators(validator_payload)
    fabrication_result = llm_validators["fabrication_detector"]
    coding_errors_result = llm_validators["coding_errors_detector"]

    validator_flags.extend(fabrication_result["flags"])
    auditor_flags.extend(coding_errors_result["flags"])

    score = 100.0 * (
        (0.30 * sharpe_n)
        + (0.20 * drawdown_n)
        + (0.15 * risk_n)
        + (0.15 * data_quality_score)
        + (0.10 * method_score)
        + (0.10 * match_rate)
    )
    overall_score = round(score, 1)

    has_critical = any(flag["severity"] == "critical" for flag in (validator_flags + auditor_flags + fetcher_flags))
    if has_critical and "Critical validation issues detected." not in hard_reject_reasons:
        hard_reject_reasons.append("Critical validation issues detected.")

    fabrication_verdict = str(fabrication_result.get("artifacts", {}).get("verdict", "unclear")).strip().lower()
    fabrication_has_critical = any(flag.get("severity") == "critical" for flag in fabrication_result["flags"])
    is_fabrication_blocked = fabrication_has_critical or fabrication_verdict == "fabrication"

    validation_summary = _summarize_flags([fabrication_result["flags"], coding_errors_result["flags"], fetcher_flags])
    validation_questions = _collect_validation_questions(
        fetcher_flags=fetcher_flags,
        validator_flags=fabrication_result["flags"],
        auditor_flags=coding_errors_result["flags"],
        coding_artifacts=coding_errors_result.get("artifacts", {}),
    )

    has_actionable_issue = any(
        flag.get("severity") in {"medium", "high", "critical"} for flag in (fetcher_flags + validator_flags + auditor_flags)
    )
    if is_fabrication_blocked:
        validation_outcome = VALIDATION_OUTCOME_BLOCKED
        user_facing_message = "Goodbye."
        allocation_usd = 0
        decision = "reject_fabrication"
    elif has_actionable_issue or hard_reject_reasons:
        validation_outcome = VALIDATION_OUTCOME_CLARIFY
        user_facing_message = "I found issues that need clarification before final review."
        allocation_usd = 0
        decision = "needs_clarification"
    else:
        validation_outcome = VALIDATION_OUTCOME_READY
        user_facing_message = "Congrats! Your pitch is ready for final review."
        allocation_usd = _compute_allocation(overall_score, draft.time_horizon)
        decision = "ready_for_final_review"

    fetcher_status = "fail" if any(flag["severity"] == "critical" for flag in fetcher_flags) else "warn"
    validator_status = "fail" if any(flag["severity"] == "critical" for flag in validator_flags) else "ok"
    auditor_status = "fail" if any(flag["severity"] == "critical" for flag in auditor_flags) else "warn"

    agent_outputs = {
        "data_fetcher": {
            "agent": "data_fetcher",
            "status": fetcher_status,
            "confidence": 0.45,
            "summary": "CUA integration placeholder from Chainlit intake.",
            "flags": fetcher_flags,
            "artifacts": {"match_rate": match_rate, "load_errors": load_errors},
            "latency_ms": 0,
        },
        "data_validator": {
            "agent": "data_validator",
            "status": validator_status if validator_status == "fail" else fabrication_result["status"],
            "confidence": float(fabrication_result["confidence"]),
            "summary": fabrication_result["summary"],
            "flags": validator_flags,
            "artifacts": {
                "data_quality_score": round(data_quality_score, 4),
                "row_count": row_count,
                "symbol_count": symbol_count,
                "llm_artifacts": fabrication_result["artifacts"],
            },
            "latency_ms": int(fabrication_result["latency_ms"]),
        },
        "pipeline_auditor": {
            "agent": "pipeline_auditor",
            "status": auditor_status if auditor_status == "fail" else coding_errors_result["status"],
            "confidence": float(coding_errors_result["confidence"]),
            "summary": coding_errors_result["summary"],
            "flags": auditor_flags,
            "artifacts": {
                "methodology_score": round(method_score, 4),
                "llm_artifacts": coding_errors_result["artifacts"],
            },
            "latency_ms": int(coding_errors_result["latency_ms"]),
        },
        "scoring": {
            "agent": "scoring",
            "status": "ok",
            "confidence": 0.8,
            "summary": "Deterministic v0 scoring and allocation.",
            "flags": [],
            "artifacts": {
                "sharpe": round(sharpe, 4),
                "max_drawdown": round(max_drawdown, 4),
                "risk_score": round(risk_score, 4),
                "time_to_return_days": time_to_return_days,
            },
            "latency_ms": 0,
        },
    }

    report_lines = [
        f"# Pitch Result: `{draft.pitch_id}`",
        "",
        f"- Decision: **{decision}**",
        f"- Validation outcome: **{validation_outcome}**",
        f"- User message: **{user_facing_message}**",
        f"- Overall score: **{overall_score}** / 100",
        f"- Allocation (USD): **{allocation_usd}**",
        "",
        "## Metrics",
        f"- Sharpe: `{round(sharpe, 4)}`",
        f"- Max drawdown: `{round(max_drawdown, 4)}`",
        f"- Risk score: `{round(risk_score, 4)}`",
        f"- Data quality score: `{round(data_quality_score, 4)}`",
        f"- Methodology score: `{round(method_score, 4)}`",
        f"- Match rate: `{round(match_rate, 4)}`",
    ]

    if hard_reject_reasons:
        report_lines.extend(["", "## Hard Reject Reasons"])
        for reason in hard_reject_reasons:
            report_lines.append(f"- {reason}")

    if validation_questions:
        report_lines.extend(["", "## Clarification Questions"])
        for question in validation_questions:
            report_lines.append(f"- {question}")

    report_lines.extend(["", "## Agent Flags"])
    for agent_name in ("data_fetcher", "data_validator", "pipeline_auditor"):
        flags = agent_outputs[agent_name]["flags"]
        if flags:
            for flag in flags:
                report_lines.append(
                    f"- `{agent_name}` `{flag['severity']}` `{flag['code']}`: {flag['message']}"
                )
        else:
            report_lines.append(f"- `{agent_name}`: no flags")

    return EvaluationResult(
        pitch_id=draft.pitch_id,
        status="completed",
        overall_score=overall_score,
        allocation_usd=allocation_usd,
        decision=decision,
        validation_outcome=validation_outcome,
        user_facing_message=user_facing_message,
        validation_summary=validation_summary,
        validation_questions=validation_questions,
        hard_reject_reasons=hard_reject_reasons,
        agent_outputs=agent_outputs,
        report_markdown="\n".join(report_lines),
    )
