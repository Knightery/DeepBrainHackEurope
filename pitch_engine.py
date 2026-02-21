from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

# Optional backtest agent and strategy scorer (requires anthropic package)
try:
    from backtest_agent import BacktestTermination, run_backtest_agent  # noqa: F401
    from strategy_scorer import score_strategy, validate_and_load
    _BACKTEST_AGENT_AVAILABLE = True
except ImportError:
    _BACKTEST_AGENT_AVAILABLE = False

try:
    from one_shot_validator import evaluate_one_shot_strategy
    _ONE_SHOT_VALIDATOR_AVAILABLE = True
except ImportError:
    _ONE_SHOT_VALIDATOR_AVAILABLE = False

REQUIRED_FIELDS = ("thesis", "time_horizon")
TIME_HORIZON_VALUES = {"days", "weeks", "months", "years"}
BACKTEST_NOTEBOOK_MAX_SCRIPT_CHARS = int(os.getenv("BACKTEST_NOTEBOOK_MAX_SCRIPT_CHARS", "180000"))
BACKTEST_NOTEBOOK_MAX_CODE_CELLS = int(os.getenv("BACKTEST_NOTEBOOK_MAX_CODE_CELLS", "400"))


def _gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
VALIDATION_OUTCOME_BLOCKED = "blocked_fabrication"
VALIDATION_OUTCOME_CLARIFY = "needs_clarification"
VALIDATION_OUTCOME_READY = "ready_for_final_review"

FABRICATION_VALIDATOR_PROMPT = """
You are the Fabrication Detector for quant pitch evaluation.
Your PRIMARY job: detect intentional data manipulation or fraud. You may note coding errors as secondary
observations, but do not call them fabrication.

Verdict rules (strict):
- verdict=fabrication: only when there is strong, concrete evidence of intentional manipulation
  (e.g. prices are mathematically generated, volume is identically repeated, impossible market statistics
  that cannot occur even in a buggy backtest). Requires confidence >= 0.8.
- verdict=unclear: suspicious but insufficient evidence to confirm intent.
- verdict=clean: no material fabrication concerns.

Do NOT return verdict=fabrication solely because metrics are high, a small dataset was used, or a
methodology error inflated results. Those are coding errors, not fabrication.

Fabrication checklist:
1) Prices or volumes that are mathematically generated (perfect increments, constant values, impossible precision)
2) Returns that are physically impossible regardless of strategy (Sharpe > 50, zero drawdown over 30+ days)
3) Non-monotonic timestamps or signs of post-hoc data editing
4) Data that provably does not match the declared source (e.g. ticker mismatch, wrong price range for period)

Questions rule:
- When verdict=unclear or verdict=fabrication, populate `questions` with 1-3 specific, direct questions
  that would allow the user to resolve your uncertainty if answered honestly.
  Examples: "Can you share the raw source file before any preprocessing?"
           "Your prices show zero variance from 2023-01-05 to 2023-01-12 — what caused this?"
           "The declared ticker is AAPL but the price range matches TSLA for this period — please clarify."
- When verdict=clean, leave `questions` empty.

Output — strict JSON only:
{"summary":"...","confidence":0.0,"flags":[{"code":"...","message":"..."}],"artifacts":{"verdict":"clean|fabrication|unclear","questions":[]}}
""".strip()

CODING_ERRORS_VALIDATOR_PROMPT = """
You are the Coding Errors Detector for quant pitch evaluation.
Your PRIMARY job: detect unintentional ML/quant methodology mistakes that inflate backtest quality.
You may note suspicious data patterns as secondary observations, but your verdict must reflect
methodology quality only — not fraud.

Verdict rules:
- verdict=errors_found: one or more methodology errors that would materially inflate reported results.
- verdict=clean: methodology is sound for the scope described.

Methodology checklist:
1) Look-ahead bias — features or targets derived using future data (negative shift, rolling windows not anchored to t-1)
2) Feature leakage — target variable used as or directly derivable from input features
3) Survivorship bias — universe constructed using post-period knowledge
4) Overfitting / data snooping — hyperparameter search on the test set, too few observations for statistical significance
5) Weak validation — no out-of-sample or walk-forward split documented
6) Unrealistic assumptions — no transaction costs / slippage for high-frequency or daily rebalancing strategies

Questions rule:
- When verdict=errors_found, populate `questions` with 1-4 specific, actionable questions that would let
  the user clarify or correct each detected issue.
  Each question should point at the specific problem found, e.g.:
  "Your rolling mean uses a window of 20 bars with no shift — is the window anchored to the current bar
   or the previous bar at prediction time?"
  "No transaction cost assumption is documented for a daily rebalancing strategy — what cost did you assume?"
- When verdict=clean, leave `questions` empty.

Output — strict JSON only:
{"summary":"...","confidence":0.0,"flags":[{"code":"...","message":"..."}],"artifacts":{"verdict":"clean|errors_found","questions":[]}}
""".strip()

class ValidatorFlagSchema(BaseModel):
    code: str
    message: str


class ValidatorArtifactsSchema(BaseModel):
    verdict: str = "unclear"
    questions: list[str] = Field(default_factory=list)


class ValidatorResponseSchema(BaseModel):
    summary: str
    confidence: float
    flags: list[ValidatorFlagSchema] = Field(default_factory=list)
    artifacts: ValidatorArtifactsSchema = Field(default_factory=ValidatorArtifactsSchema)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def round_to_100(value: float) -> int:
    return int(round(value / 100.0) * 100)


def normalize_time_horizon(raw_value: str | None) -> str | None:
    """Accept only exact canonical values. Time horizon is set via UI selection, not free text."""
    if not raw_value:
        return None
    text = raw_value.strip().lower()
    if text in TIME_HORIZON_VALUES:
        return text
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
    one_shot_mode: bool = False
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
        if not self.uploaded_files:
            missing.append("uploaded_files")
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
            "one_shot_mode": self.one_shot_mode,
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

    # Time horizon is set via UI selection on chat start — do not infer from free text.

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


_FILE_ROLE_CLASSIFIER_PROMPT = """
You are classifying uploaded files for a quant strategy pitch evaluation pipeline.

You will receive a JSON payload with:
- "tickers": the strategy's target tickers
- "thesis": the investment thesis
- "files": a list of file profiles, each with "name", "columns", and "sample_rows"

Classify each file into exactly one role:
- "strategy_data": main price/returns/signal data for the strategy being pitched
- "benchmark": reference or comparison data (market index, SPY, risk-free, macro series)
- "other": unrecognised or irrelevant files

Classification rules:
- Files whose columns or sample values relate to the pitched tickers or thesis are strategy_data.
- Files with purely index/market/macro data, or whose name/columns suggest a reference series
  (e.g. spy, spx, index, benchmark, rf, macro, vix, treasury) are benchmark.
- When ambiguous, prefer strategy_data over benchmark.
- strategy_data files typically contain: close, open, high, low, volume, price, return, signal
  for the assets described in the thesis.

Output strict JSON only:
{"files": [{"name": "...", "role": "strategy_data|benchmark|other", "reason": "..."}]}
""".strip()


def _profile_csv_file(file_entry: UploadedFile, max_rows: int = 5) -> dict[str, Any]:
    """Read column names and a few sample rows from a CSV/TSV for LLM classification."""
    try:
        path = Path(file_entry.path)
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        frame = pd.read_csv(path, sep=sep, nrows=max_rows)
        return {
            "name": file_entry.name,
            "columns": list(frame.columns),
            "sample_rows": frame.head(max_rows).fillna("").to_dict(orient="records"),
        }
    except Exception as exc:
        return {
            "name": file_entry.name,
            "columns": [],
            "sample_rows": [],
            "profile_error": f"{exc.__class__.__name__}: {exc}",
        }


def _detect_file_roles(
    uploaded_files: list[UploadedFile],
    tickers: list[str] | None = None,
    thesis: str | None = None,
) -> dict[str, list[UploadedFile]]:
    """
    LLM-powered file role detection. Reads actual file content (column headers +
    sample rows) and uses Gemini with pitch context to classify each CSV/TSV.
    """
    if not uploaded_files:
        return {"strategy_scripts": [], "data_files": [], "benchmark_files": []}

    # Scripts are unambiguous — classify them immediately without an LLM call.
    strategy_scripts = [f for f in uploaded_files if Path(f.path).suffix.lower() in {".py", ".ipynb"}]
    csv_files = [f for f in uploaded_files if Path(f.path).suffix.lower() in {".csv", ".tsv"}]

    if not csv_files:
        return {"strategy_scripts": strategy_scripts, "data_files": [], "benchmark_files": []}

    # If only one CSV/TSV, no LLM needed — it must be strategy data.
    if len(csv_files) == 1:
        return {"strategy_scripts": strategy_scripts, "data_files": csv_files, "benchmark_files": []}

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for file-role classification.")

    profiles = [_profile_csv_file(f) for f in csv_files]
    payload = {
        "tickers": tickers or [],
        "thesis": thesis or "",
        "files": profiles,
    }

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=_gemini_model(),
        contents=json.dumps(payload, ensure_ascii=True),
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=10000,
            system_instruction=_FILE_ROLE_CLASSIFIER_PROMPT,
        ),
    )
    raw = (getattr(response, "text", "") or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    parsed = json.loads(raw)
    classifications: dict[str, str] = {
        item["name"]: item["role"]
        for item in parsed.get("files", [])
        if isinstance(item, dict) and "name" in item and "role" in item
    }

    data_files: list[UploadedFile] = []
    benchmark_files: list[UploadedFile] = []
    for f in csv_files:
        role = classifications.get(f.name, "strategy_data")
        if role == "benchmark":
            benchmark_files.append(f)
        else:
            data_files.append(f)

    # Safety: if the LLM classified everything as benchmark, promote all to data.
    if not data_files:
        data_files = benchmark_files
        benchmark_files = []

    return {
        "strategy_scripts": strategy_scripts,
        "data_files": data_files,
        "benchmark_files": benchmark_files,
    }


def _sanitize_notebook_code_line(line: str) -> str:
    stripped = line.lstrip()
    # Jupyter magics and shell escapes are not valid plain-Python syntax.
    if stripped.startswith("%") or stripped.startswith("!"):
        return f"# {line}"
    if stripped.startswith("get_ipython("):
        return f"# {line}"
    return line


def _compile_notebook_to_script(notebook_text: str, notebook_name: str) -> str:
    try:
        notebook_obj = json.loads(notebook_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Notebook JSON is invalid: {exc}") from exc

    cells = notebook_obj.get("cells")
    if not isinstance(cells, list):
        raise ValueError("Notebook does not contain a valid 'cells' list.")

    compiled_chunks = [
        f"# Compiled from notebook: {notebook_name}",
        "# Executed as a linear Python script for MVP backtesting.",
    ]
    code_cells = 0

    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        source = str(source)
        if not source.strip():
            continue

        sanitized = "\n".join(_sanitize_notebook_code_line(line) for line in source.splitlines())
        code_cells += 1
        compiled_chunks.append(f"\n# --- notebook cell {idx} ---\n{sanitized}\n")

        if code_cells > BACKTEST_NOTEBOOK_MAX_CODE_CELLS:
            raise ValueError(
                f"Notebook has more than {BACKTEST_NOTEBOOK_MAX_CODE_CELLS} code cells; "
                "please trim it before upload."
            )

    if code_cells == 0:
        raise ValueError("Notebook has no executable code cells.")

    compiled_script = "\n".join(compiled_chunks)
    if len(compiled_script) > BACKTEST_NOTEBOOK_MAX_SCRIPT_CHARS:
        raise ValueError(
            f"Compiled notebook script exceeds {BACKTEST_NOTEBOOK_MAX_SCRIPT_CHARS} characters; "
            "please upload a smaller notebook or convert it to a focused strategy .py."
        )

    return compiled_script


def _load_strategy_source_for_backtest(file_entry: UploadedFile) -> tuple[str, str]:
    path = Path(file_entry.path)
    suffix = path.suffix.lower()
    raw_text = path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".py":
        return file_entry.name, raw_text
    if suffix == ".ipynb":
        compiled_name = f"{Path(file_entry.name).stem}_compiled.py"
        compiled_script = _compile_notebook_to_script(raw_text, file_entry.name)
        return compiled_name, compiled_script

    raise ValueError(f"Unsupported strategy file type: '{suffix}'")


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


def _safe_head_records(frame: pd.DataFrame | None, limit: int = 25) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    try:
        json_text = frame.head(limit).to_json(orient="records", date_format="iso")
        return json.loads(json_text)
    except Exception as exc:
        return [{"_error": f"{exc.__class__.__name__}: {exc}"}]


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


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _cua_dir() -> Path:
    return _project_root() / "cua"


def _cua_downloads_dir() -> Path:
    return _cua_dir() / "downloads"


def _find_uploaded_file(draft: PitchDraft, file_to_validate: str) -> UploadedFile | None:
    requested = file_to_validate.strip().lower()
    if not requested:
        return None
    for uploaded in draft.uploaded_files:
        if uploaded.name.lower() == requested:
            return uploaded
    return None


def _reference_file_brief(path: Path, mime_type: str, max_chars: int = 2400) -> str:
    suffix = path.suffix.lower()
    size_bytes = path.stat().st_size if path.exists() else 0
    parts = [
        f"name={path.name}",
        f"mime_type={mime_type or 'unknown'}",
        f"size_bytes={size_bytes}",
    ]

    try:
        if suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            frame = pd.read_csv(path, sep=sep, nrows=8)
            columns = ", ".join(str(column) for column in frame.columns[:30])
            preview_rows = frame.head(5).to_dict(orient="records")
            parts.append(f"columns={columns}")
            parts.append(f"sample_rows={json.dumps(preview_rows, ensure_ascii=True)}")
        elif suffix == ".json":
            text = path.read_text(encoding="utf-8", errors="replace")
            clipped = text[:max_chars]
            parts.append(f"json_preview={clipped}")
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            clipped = text[:max_chars]
            if clipped.strip():
                parts.append(f"text_preview={clipped}")
    except Exception as exc:
        parts.append(f"preview_error={exc.__class__.__name__}")

    return "\n".join(parts)


def _extract_json_after_separator(raw_output: str) -> dict[str, Any] | None:
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
        pass

    decoder = json.JSONDecoder()
    for idx in range(len(candidate) - 1, -1, -1):
        if candidate[idx] != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(candidate[idx:])
            if isinstance(parsed, dict) and end <= len(candidate[idx:]):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _resolve_downloaded_host_paths(downloaded_files: list[str], excluded_filenames: set[str] | None = None) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    downloads_dir = _cua_downloads_dir()
    excluded = {name.lower() for name in (excluded_filenames or set())}
    for raw_path in downloaded_files:
        filename = Path(str(raw_path)).name
        if not filename:
            continue
        if filename.lower() in excluded:
            continue
        host_path = downloads_dir / filename
        if host_path.exists() and host_path not in seen:
            resolved.append(host_path)
            seen.add(host_path)
    return resolved


def _review_download_match_with_llm(
    reference_path: Path,
    candidate_paths: list[Path],
    notes: str,
) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for download match review.")

    reference_profile = _reference_file_brief(reference_path, "")
    candidate_profiles: list[dict[str, str]] = []
    for candidate in candidate_paths[:6]:
        candidate_profiles.append(
            {
                "name": candidate.name,
                "profile": _reference_file_brief(candidate, ""),
            }
        )

    prompt = (
        "You are reviewing downloaded source files against a submitted reference file.\n"
        "Decide if any candidate is a close enough match to proceed.\n"
        "Compare schema, entities/tickers, date ranges, granularity, and obvious semantic alignment.\n"
        "If mismatch, provide concrete retry guidance for a browser automation agent.\n\n"
        f"Reference profile:\n{reference_profile}\n\n"
        f"Candidate profiles JSON:\n{json.dumps(candidate_profiles, ensure_ascii=True)}\n\n"
    )
    if notes.strip():
        prompt += f"Additional context:\n{notes.strip()}\n\n"
    prompt += (
        "Return exactly one XML block with strict JSON:\n"
        "<download_match>{\"verdict\":\"match|mismatch|unclear\",\"confidence\":0.0,"
        "\"best_candidate\":\"filename-or-empty\",\"reason\":\"...\","
        "\"retry_guidance\":\"...\"}</download_match>"
    )

    attempts_raw = os.getenv("CUA_MATCH_REVIEW_MAX_ATTEMPTS", "3")
    delay_raw = os.getenv("CUA_MATCH_REVIEW_RETRY_DELAY_SECONDS", "1.0")
    try:
        max_attempts = max(1, min(5, int(attempts_raw)))
    except ValueError:
        max_attempts = 3
    try:
        base_delay = max(0.0, min(20.0, float(delay_raw)))
    except ValueError:
        base_delay = 1.0

    client = genai.Client(api_key=api_key)
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=_gemini_model(),
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=10000,
                ),
            )
            raw_text = (getattr(response, "text", "") or "").strip()
            parsed = extract_tagged_json(raw_text, "download_match")
            if not parsed:
                raise ValueError("Download match review model returned an invalid response.")

            verdict = str(parsed.get("verdict", "unclear")).strip().lower()
            if verdict not in {"match", "mismatch", "unclear"}:
                verdict = "unclear"
            confidence = clamp(parsed.get("confidence", 0.5))
            best_candidate = str(parsed.get("best_candidate", "")).strip()
            reason = str(parsed.get("reason", "")).strip() or "LLM did not provide a reason."
            retry_guidance = str(parsed.get("retry_guidance", "")).strip() or (
                "Find a different source download that better matches schema/entities/date range."
            )
            return {
                "verdict": verdict,
                "confidence": confidence,
                "best_candidate": best_candidate,
                "reason": reason,
                "retry_guidance": retry_guidance,
            }
        except Exception as exc:  # pragma: no cover - runtime/network dependent
            last_exc = exc
            if attempt >= max_attempts:
                break
            time.sleep(base_delay * (2 ** (attempt - 1)))

    reason = "LLM match review failed after retries."
    if last_exc is not None:
        reason = (
            f"LLM match review failed after {max_attempts} attempt(s): "
            f"{last_exc.__class__.__name__}: {last_exc}"
        )
    return {
        "verdict": "mismatch",
        "confidence": 0.0,
        "best_candidate": "",
        "reason": reason,
        "retry_guidance": (
            "Fix Gemini match-review availability/configuration and rerun CUA. "
            "Heuristic fallback matching is disabled."
        ),
    }


def validate_data_with_cua(
    draft: PitchDraft,
    file_to_validate: str,
    notes: str = "",
    source_urls_override: list[str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    file_entry = _find_uploaded_file(draft, file_to_validate)
    if file_entry is None:
        available = [entry.name for entry in draft.uploaded_files]
        return {
            "agent": "data_fetcher",
            "status": "fail",
            "confidence": 0.0,
            "summary": "Requested file was not found among uploaded files.",
            "flags": [
                {
                    "code": "REFERENCE_FILE_NOT_FOUND",
                    "message": f"Upload exists: {', '.join(available) if available else '(none)'}",
                }
            ],
            "artifacts": {"requested_file": file_to_validate, "available_files": available},
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    selected_source_urls = [url.strip() for url in (source_urls_override or draft.source_urls) if str(url).strip()]

    if not selected_source_urls:
        return {
            "agent": "data_fetcher",
            "status": "fail",
            "confidence": 0.0,
            "summary": "No source URLs were provided for CUA validation.",
            "flags": [
                {
                    "code": "MISSING_SOURCE_URLS",
                    "message": "Add at least one source URL before running /validate_data.",
                }
            ],
            "artifacts": {"requested_file": file_to_validate},
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    source_path = Path(file_entry.path)
    if not source_path.exists():
        return {
            "agent": "data_fetcher",
            "status": "fail",
            "confidence": 0.0,
            "summary": "Uploaded reference file path is missing on disk.",
            "flags": [
                {
                    "code": "REFERENCE_FILE_MISSING_ON_DISK",
                    "message": f"Missing file: {source_path}",
                }
            ],
            "artifacts": {"requested_file": file_entry.name, "path": str(source_path)},
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    cua_downloads = _cua_downloads_dir()
    cua_downloads.mkdir(parents=True, exist_ok=True)

    staged_reference_name = f"reference_{draft.pitch_id}_{file_entry.name}"
    staged_reference_path = cua_downloads / staged_reference_name
    shutil.copy2(source_path, staged_reference_path)

    max_attempts_raw = os.getenv("CUA_MAX_ATTEMPTS", "3")
    try:
        max_attempts = max(1, min(5, int(max_attempts_raw)))
    except ValueError:
        max_attempts = 3

    timeout_raw = os.getenv("CUA_RUN_TIMEOUT_SECONDS", "360")
    try:
        run_timeout = max(60, int(timeout_raw))
    except ValueError:
        run_timeout = 360

    source_list = "\n".join(f"- {url}" for url in selected_source_urls)
    reference_brief = _reference_file_brief(source_path, file_entry.mime_type)

    attempt_history: list[dict[str, Any]] = []
    retry_feedback = ""
    last_failure_output: dict[str, Any] | None = None

    for attempt in range(1, max_attempts + 1):
        start_url = selected_source_urls[(attempt - 1) % len(selected_source_urls)]
        description = (
            "You are validating whether the submitted reference file matches publicly available source data.\n"
            "First, inspect the local reference file in ~/Downloads using GUI actions.\n"
            "Then navigate source pages and download candidate files, preferring browser GUI actions first.\n"
            "If GUI is blocked or clearly inferior, you may use a script/terminal fallback.\n"
            "After each download, compare against the reference and continue searching if mismatch.\n"
            "Do not stop at the first file unless it is a strong match.\n\n"
            f"Attempt: {attempt}/{max_attempts}\n"
            f"Pitch ID: {draft.pitch_id}\n"
            f"Reference filename: {staged_reference_name}\n"
            f"Source URLs to check:\n{source_list}\n\n"
            f"Reference file profile:\n{reference_brief}\n"
        )
        if notes.strip():
            description += f"\nUser notes:\n{notes.strip()}\n"
        if retry_feedback:
            description += f"\nPrevious mismatch feedback:\n{retry_feedback}\n"

        command = [
            "docker",
            "compose",
            "run",
            "--rm",
            "--remove-orphans",
            "data-fetcher",
            start_url,
            description,
            staged_reference_name,
        ]

        try:
            completed = subprocess.run(
                command,
                cwd=_cua_dir(),
                capture_output=True,
                text=True,
                check=False,
                timeout=run_timeout,
            )
        except subprocess.TimeoutExpired:
            last_failure_output = {
                "agent": "data_fetcher",
                "status": "fail",
                "confidence": 0.0,
                "summary": f"CUA run timed out after {run_timeout}s.",
                "flags": [
                    {
                        "code": "CUA_RUN_TIMEOUT",
                        "message": f"Timeout after {run_timeout}s while running docker compose.",
                    }
                ],
                "artifacts": {"attempt": attempt, "command": " ".join(command)},
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
            break
        except Exception as exc:
            last_failure_output = {
                "agent": "data_fetcher",
                "status": "fail",
                "confidence": 0.0,
                "summary": "Failed to start CUA container process.",
                "flags": [
                    {
                        "code": "CUA_RUN_ERROR",
                        "message": f"{exc.__class__.__name__} while invoking docker compose.",
                    }
                ],
                "artifacts": {"attempt": attempt, "command": " ".join(command)},
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
            break

        parsed = _extract_json_after_separator((completed.stdout or "") + "\n" + (completed.stderr or ""))
        if not parsed:
            last_failure_output = {
                "agent": "data_fetcher",
                "status": "fail",
                "confidence": 0.2,
                "summary": "CUA run completed but returned unparsable output.",
                "flags": [
                    {
                        "code": "CUA_OUTPUT_PARSE_FAILED",
                        "message": "Could not parse JSON result from CUA output.",
                    }
                ],
                "artifacts": {
                    "attempt": attempt,
                    "return_code": completed.returncode,
                    "stdout_tail": (completed.stdout or "")[-3000:],
                    "stderr_tail": (completed.stderr or "")[-3000:],
                },
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
            continue

        validation = parsed.get("validation", {}) if isinstance(parsed.get("validation"), dict) else {}
        issues = validation.get("issues", []) if isinstance(validation.get("issues"), list) else []
        flags: list[dict[str, str]] = []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            code = str(issue.get("code", "CUA_ISSUE")).upper()
            message = str(issue.get("message", "CUA reported an issue.")).strip()
            if message:
                flags.append({"code": code, "message": message})

        advisories = validation.get("advisories", []) if isinstance(validation.get("advisories"), list) else []
        for advisory in advisories:
            if not isinstance(advisory, dict):
                continue
            code = str(advisory.get("code", "CUA_ADVISORY")).strip().upper() or "CUA_ADVISORY"
            message = str(advisory.get("message", "")).strip()
            if message:
                flags.append({"code": code, "message": message})

        downloaded = parsed.get("downloaded_files", []) if isinstance(parsed.get("downloaded_files"), list) else []
        excluded_names = {staged_reference_name, source_path.name}
        filtered_downloaded = [
            file_path
            for file_path in downloaded
            if Path(str(file_path)).name.lower() not in {name.lower() for name in excluded_names}
        ]
        candidate_paths = _resolve_downloaded_host_paths(filtered_downloaded, excluded_filenames=excluded_names)
        match_review = _review_download_match_with_llm(source_path, candidate_paths, notes)
        match_verdict = str(match_review.get("verdict", "unclear")).lower()

        if match_review.get("best_candidate", "").strip().lower() in {name.lower() for name in excluded_names}:
            match_verdict = "mismatch"
            match_review["verdict"] = "mismatch"
            match_review["reason"] = "Staged reference file was selected as best candidate; this is not a valid source download."
            match_review["retry_guidance"] = "Ignore the staged reference file and fetch a real source artifact from the target page."

        if match_verdict == "mismatch":
            flags.append(
                {
                    "code": "SOURCE_MISMATCH_SEVERE",
                    "message": str(match_review.get("reason", "Downloaded files did not match reference data.")).strip(),
                }
            )

        status_text = str(parsed.get("status", "")).strip().lower()
        status = "ok" if status_text == "success" and not flags and match_verdict == "match" else "warn"
        if status_text == "fail":
            status = "fail"
        if match_verdict == "mismatch" and attempt >= max_attempts:
            status = "fail"

        confidence = clamp(match_review.get("confidence", 0.5))
        if status == "warn":
            confidence = min(confidence, 0.55)
        elif status == "fail":
            confidence = min(confidence, 0.25)

        summary = str(parsed.get("summary", "")).strip() or "CUA run completed."
        attempt_record = {
            "attempt": attempt,
            "start_url": start_url,
            "status": status,
            "match_review": match_review,
            "downloaded_files": downloaded,
            "candidate_downloaded_files": filtered_downloaded,
        }
        attempt_history.append(attempt_record)

        result_payload = {
            "agent": "data_fetcher",
            "status": status,
            "confidence": confidence,
            "summary": summary,
            "flags": flags,
            "artifacts": {
                "requested_file": file_entry.name,
                "staged_reference": staged_reference_name,
                "source_urls_checked": selected_source_urls,
                "downloaded_files": downloaded,
                "candidate_downloaded_files": filtered_downloaded,
                "validation": validation,
                "return_code": completed.returncode,
                "match_rate": confidence if match_verdict == "match" else 0.35,
                "match_review": match_review,
                "attempt_history": attempt_history,
            },
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

        if status == "ok":
            return result_payload

        retry_feedback = str(match_review.get("retry_guidance", "")).strip()
        last_failure_output = result_payload

    if last_failure_output:
        artifacts = last_failure_output.get("artifacts", {})
        if isinstance(artifacts, dict):
            artifacts["attempt_history"] = attempt_history
        return last_failure_output

    return {
        "agent": "data_fetcher",
        "status": "fail",
        "confidence": 0.0,
        "summary": "CUA validation failed before any attempt could complete.",
        "flags": [
            {
                "code": "CUA_UNKNOWN_ERROR",
                "message": "Unknown CUA failure before attempt completion.",
            }
        ],
        "artifacts": {
            "requested_file": file_entry.name,
            "staged_reference": staged_reference_name,
            "attempt_history": attempt_history,
        },
        "latency_ms": int((time.perf_counter() - started) * 1000),
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
        message = str(raw_flag.get("message", "")).strip()
        if not message:
            continue
        normalized.append({"code": code, "message": message})
    return normalized


def _status_from_flags(flags: list[dict[str, str]]) -> str:
    return "warn" if flags else "ok"


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
    fabrication_artifacts: dict[str, Any] | None = None,
) -> list[str]:
    questions: list[str] = []
    # Questions directly emitted by the LLM detectors take priority.
    for artifacts in (fabrication_artifacts or {}, coding_artifacts):
        maybe_questions = artifacts.get("questions", [])
        if isinstance(maybe_questions, list):
            for item in maybe_questions:
                text = str(item).strip()
                if text:
                    questions.append(text)
    # Fallback: derive questions from flag codes for any remaining uncovered flags.
    for flag in fetcher_flags + validator_flags + auditor_flags:
        generated = _question_from_flag(flag)
        if generated:
            questions.append(generated)
    return list(dict.fromkeys(questions))


def _validator_log(agent_name: str, message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[validator:{agent_name}] {ts} | {message}")


def _parse_validator_json(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _validator_retry_attempts() -> int:
    raw = os.getenv("VALIDATOR_LLM_MAX_ATTEMPTS", "3")
    try:
        value = int(raw)
    except ValueError:
        value = 3
    return max(1, min(6, value))


def _validator_retry_delay_seconds() -> float:
    raw = os.getenv("VALIDATOR_LLM_RETRY_DELAY_SECONDS", "1.5")
    try:
        value = float(raw)
    except ValueError:
        value = 1.5
    return max(0.0, min(30.0, value))


def _validator_temperature() -> float:
    raw = os.getenv("VALIDATOR_LLM_TEMPERATURE", "0.2")
    try:
        value = float(raw)
    except ValueError:
        value = 0.2
    return max(0.0, min(1.0, value))


def _run_llm_validator(agent_name: str, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for validator agents.")

    model_name = _gemini_model()
    _validator_log(agent_name, f"start model={model_name}")
    _validator_log(agent_name, f"payload_bytes={len(json.dumps(payload, ensure_ascii=True))}")
    client = genai.Client(api_key=api_key)
    max_attempts = _validator_retry_attempts()
    base_delay = _validator_retry_delay_seconds()
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            _validator_log(agent_name, f"retry attempt={attempt}/{max_attempts}")
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=json.dumps(payload, ensure_ascii=True),
                config=types.GenerateContentConfig(
                    temperature=_validator_temperature(),
                    max_output_tokens=10000,
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                ),
            )
            raw_text = (getattr(response, "text", "") or "").strip()
            _validator_log(agent_name, "raw_response_begin")
            print(raw_text)
            _validator_log(agent_name, "raw_response_end")

            parsed_any = getattr(response, "parsed", None)
            if parsed_any is None:
                parsed_dict = _parse_validator_json(raw_text)
                if parsed_dict is None:
                    _validator_log(agent_name, "parse_failed: invalid JSON response")
                    raise ValueError("Invalid JSON response from validator model")
                parsed_model = ValidatorResponseSchema.model_validate(parsed_dict)
            elif isinstance(parsed_any, ValidatorResponseSchema):
                parsed_model = parsed_any
            elif isinstance(parsed_any, dict):
                parsed_model = ValidatorResponseSchema.model_validate(parsed_any)
            else:
                parsed_model = ValidatorResponseSchema.model_validate(
                    getattr(parsed_any, "model_dump", lambda: {})()
                )
            parsed = parsed_model.model_dump()

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
        except Exception as exc:  # pragma: no cover - runtime/network dependent
            last_exc = exc
            _validator_log(agent_name, f"attempt={attempt} failed error={exc.__class__.__name__}: {exc}")
            if attempt >= max_attempts:
                break
            sleep_seconds = base_delay * (2 ** (attempt - 1))
            time.sleep(sleep_seconds)

    assert last_exc is not None
    raise RuntimeError(
        f"{agent_name} failed after {max_attempts} LLM attempt(s): "
        f"{last_exc.__class__.__name__}: {last_exc}"
    ) from last_exc


def _run_llm_validators(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    max_workers = 1 if os.getenv("VALIDATOR_SERIAL_MODE", "1") == "1" else 2
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
        try:
            fabrication_result = fabrication_future.result()
        except Exception as exc:  # pragma: no cover - defensive
            _validator_log("fabrication_detector", f"future_error={exc.__class__.__name__}: {exc}")
            fabrication_result = _validator_error_result("fabrication_detector", time.perf_counter(), exc, 1)
        try:
            coding_result = coding_future.result()
        except Exception as exc:  # pragma: no cover - defensive
            _validator_log("coding_errors_detector", f"future_error={exc.__class__.__name__}: {exc}")
            coding_result = _validator_error_result("coding_errors_detector", time.perf_counter(), exc, 1)
        return {
            "fabrication_detector": fabrication_result,
            "coding_errors_detector": coding_result,
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


def evaluate_pitch(draft: PitchDraft, data_fetcher_output: dict[str, Any] | None = None) -> EvaluationResult:
    # Classify uploaded files first so CSV scoring uses the right data file.
    roles = _detect_file_roles(draft.uploaded_files, tickers=draft.tickers, thesis=draft.thesis)
    scoring_files = roles["data_files"] if roles["data_files"] else draft.uploaded_files
    frame, load_errors = _load_first_table(scoring_files)
    method_score, methodology_warnings = _methodology_score(draft.methodology_summary)

    hard_reject_reasons: list[str] = []
    validator_flags: list[dict[str, str]] = []
    auditor_flags: list[dict[str, str]] = []
    fetcher_flags: list[dict[str, str]] = []
    one_shot_result: Any = None

    # --- Backtest agent ---
    # When the user uploads a strategy .py/.ipynb, Claude generates and runs a standardised
    # backtest runner that computes all strategy_scorer.Strategy metrics.
    backtest_result: Any = None
    backtest_scored: Any = None
    if roles["strategy_scripts"] and _BACKTEST_AGENT_AVAILABLE:
        _strat_files: list[tuple[str, str]] = []
        for _f in roles["strategy_scripts"]:
            try:
                _strat_files.append(_load_strategy_source_for_backtest(_f))
            except ValueError as _compile_err:
                auditor_flags.append({
                    "code": "BACKTEST_STRATEGY_PREP_ERROR",
                    "message": f"Could not prepare strategy file '{_f.name}': {_compile_err}",
                })
            except OSError as _read_err:
                auditor_flags.append({
                    "code": "BACKTEST_FILE_READ_ERROR",
                    "message": f"Could not read strategy file '{_f.name}': {_read_err}",
                })
        _data_files: list[tuple[str, str]] = []
        for _f in roles["data_files"] + roles["benchmark_files"]:
            try:
                _data_files.append((_f.name, Path(_f.path).read_text(encoding="utf-8", errors="replace")))
            except OSError as _read_err:
                auditor_flags.append({
                    "code": "BACKTEST_FILE_READ_ERROR",
                    "message": f"Could not read data file '{_f.name}': {_read_err}",
                })
        _ticker = draft.tickers[0] if draft.tickers else "UNKNOWN"
        if _strat_files:
            try:
                backtest_result = run_backtest_agent(
                    strategy_files=_strat_files,
                    data_files=_data_files,
                    pitch_context={"name": draft.pitch_id, "ticker": _ticker},
                )
            except ValueError as _cfg_err:
                raise RuntimeError(f"Backtest agent misconfigured: {_cfg_err}") from _cfg_err

            if backtest_result is not None:
                if backtest_result.status == "success" and backtest_result.metrics:
                    _strategy_obj, _v_errors = validate_and_load(backtest_result.metrics)
                    if _v_errors:
                        auditor_flags.append({
                            "code": "BACKTEST_SCHEMA_INVALID",
                            "message": f"Backtest metrics failed validation: {_v_errors}",
                        })
                    if _strategy_obj:
                        backtest_scored = score_strategy(_strategy_obj)
                        if backtest_scored.disqualified:
                            for _reason in backtest_scored.disqualification_reasons:
                                if _reason not in hard_reject_reasons:
                                    hard_reject_reasons.append(_reason)
                elif backtest_result.status == "user_action_required":
                    auditor_flags.append({
                        "code": "BACKTEST_USER_ACTION_REQUIRED",
                        "message": backtest_result.message,
                    })
                elif backtest_result.status == "agent_fault":
                    raise RuntimeError(
                        f"Backtest agent failed after {backtest_result.attempt_count} attempt(s). "
                        f"Message: {backtest_result.message}"
                    )

    if not draft.source_urls:
        hard_reject_reasons.append("No source URL was provided for submitted data.")
        fetcher_flags.append(
            {
                "code": "MISSING_SOURCE_URLS",
                "message": "Provide source URL(s) for all submitted datasets.",
            }
        )

    if not draft.tickers:
        hard_reject_reasons.append("No stock ticker(s) were provided.")
        validator_flags.append(
            {
                "code": "MISSING_TICKERS",
                "message": "Provide the stock ticker(s) for this pitch (for example: AAPL, MSFT).",
            }
        )

    if draft.one_shot_mode:
        if not _ONE_SHOT_VALIDATOR_AVAILABLE:
            auditor_flags.append(
                {
                    "code": "ONE_SHOT_VALIDATOR_UNAVAILABLE",
                    "message": "One-shot validator module is unavailable.",
                }
            )
        else:
            one_shot_result = evaluate_one_shot_strategy(draft=draft)
            auditor_flags.extend(one_shot_result.flags)

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

    for warning in methodology_warnings:
        auditor_flags.append({"code": "METHOD_WARN", "message": warning})

    sharpe_n = clamp((sharpe + 1.0) / 3.0)
    drawdown_n = 1.0 - clamp(abs(max_drawdown) / 0.60)
    risk_n = 1.0 - risk_score
    if data_fetcher_output:
        fetcher_flags.extend(_normalize_flags(data_fetcher_output.get("flags")))
        artifacts = data_fetcher_output.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
        match_rate = clamp(artifacts.get("match_rate", _match_rate(draft)))
    else:
        fetcher_flags.append(
            {
                "code": "CUA_NOT_CONNECTED",
                "message": "CUA validation is mandatory and was not run for this evaluation.",
            }
        )
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

    strategy_scorer_ready = backtest_scored is not None and not backtest_scored.disqualified
    if not draft.one_shot_mode and not strategy_scorer_ready:
        auditor_flags.append(
            {
                "code": "STRATEGY_SCORER_REQUIRED",
                "message": "Backtest strategy scoring is required. CSV approximation scoring is disabled.",
            }
        )

    if draft.one_shot_mode:
        score = 100.0 if (one_shot_result and one_shot_result.recommendation == "VALID") else 0.0
    elif strategy_scorer_ready:
        # Richer formula: 65% strategy-scorer composite + 35% data/methodology/provenance
        score = (
            0.65 * backtest_scored.composite_score
            + 15.0 * data_quality_score
            + 10.0 * method_score
            + 10.0 * match_rate
        )
    else:
        score = 0.0
    overall_score = round(score, 1)

    if hard_reject_reasons:
        hard_reject_reasons = list(dict.fromkeys(hard_reject_reasons))

    fabrication_verdict = str(fabrication_result.get("artifacts", {}).get("verdict", "unclear")).strip().lower()
    fabrication_confidence = float(fabrication_result.get("confidence", 0.0))
    is_fabrication_blocked = fabrication_verdict == "fabrication" and fabrication_confidence >= 0.8

    validation_summary = _summarize_flags([fabrication_result["flags"], coding_errors_result["flags"], fetcher_flags])
    validation_questions = _collect_validation_questions(
        fetcher_flags=fetcher_flags,
        validator_flags=fabrication_result["flags"],
        auditor_flags=coding_errors_result["flags"],
        coding_artifacts=coding_errors_result.get("artifacts", {}),
        fabrication_artifacts=fabrication_result.get("artifacts", {}),
    )

    coding_verdict = str(coding_errors_result.get("artifacts", {}).get("verdict", "clean")).strip().lower()
    has_actionable_issue = (
        coding_verdict == "errors_found"
        or bool(fetcher_flags + auditor_flags)
        or (fabrication_verdict == "unclear" and fabrication_confidence >= 0.7)
    )
    if is_fabrication_blocked:
        validation_outcome = VALIDATION_OUTCOME_BLOCKED
        user_facing_message = "Goodbye."
        allocation_usd = 0
        decision = "reject_fabrication"
    elif draft.one_shot_mode and one_shot_result is not None:
        allocation_usd = 0
        if one_shot_result.recommendation == "VALID":
            validation_outcome = VALIDATION_OUTCOME_READY
            user_facing_message = "One-shot strategy is statistically valid and ready for final review."
            decision = "one_shot_valid"
        else:
            validation_outcome = VALIDATION_OUTCOME_CLARIFY
            user_facing_message = "One-shot strategy is not yet statistically valid. Please address the listed gaps."
            decision = "one_shot_not_valid"
            validation_questions = sorted(set(validation_questions + one_shot_result.validation_questions))
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

    fetcher_status = "warn" if fetcher_flags else "ok"
    validator_status = "warn" if validator_flags else "ok"
    auditor_status = "warn" if auditor_flags else "ok"

    fetcher_summary = "CUA integration placeholder from Chainlit intake."
    fetcher_confidence = 0.45
    fetcher_latency = 0
    fetcher_artifacts: dict[str, Any] = {"match_rate": match_rate, "load_errors": load_errors}

    if data_fetcher_output:
        fetcher_status = str(data_fetcher_output.get("status", fetcher_status))
        fetcher_summary = str(data_fetcher_output.get("summary", fetcher_summary))
        fetcher_confidence = clamp(float(data_fetcher_output.get("confidence", fetcher_confidence)))
        fetcher_latency = int(data_fetcher_output.get("latency_ms", 0) or 0)
        maybe_artifacts = data_fetcher_output.get("artifacts")
        if isinstance(maybe_artifacts, dict):
            fetcher_artifacts = maybe_artifacts

    agent_outputs = {
        "data_fetcher": {
            "agent": "data_fetcher",
            "status": fetcher_status,
            "confidence": fetcher_confidence,
            "summary": fetcher_summary,
            "flags": fetcher_flags,
            "artifacts": fetcher_artifacts,
            "latency_ms": fetcher_latency,
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
            "confidence": (
                float(one_shot_result.confidence)
                if (draft.one_shot_mode and one_shot_result is not None)
                else (0.9 if strategy_scorer_ready else 0.3)
            ),
            "summary": (
                "One-shot strategy binary recommendation."
                if draft.one_shot_mode
                else (
                    "Strategy-scorer composite from backtest agent."
                    if strategy_scorer_ready
                    else "Strategy-scorer output unavailable."
                )
            ),
            "flags": [] if not draft.one_shot_mode else (one_shot_result.flags if one_shot_result else []),
            "artifacts": {
                "source": (
                    "one_shot"
                    if draft.one_shot_mode
                    else ("strategy_scorer" if strategy_scorer_ready else "strategy_scorer_unavailable")
                ),
                "sharpe": round(sharpe, 4),
                "max_drawdown": round(max_drawdown, 4),
                "risk_score": round(risk_score, 4),
                "time_to_return_days": time_to_return_days,
                "one_shot_mode": draft.one_shot_mode,
                "one_shot_recommendation": (
                    one_shot_result.recommendation if (draft.one_shot_mode and one_shot_result is not None) else None
                ),
                "one_shot_artifacts": (
                    one_shot_result.artifacts if (draft.one_shot_mode and one_shot_result is not None) else None
                ),
                "backtest_attempt_count": backtest_result.attempt_count if backtest_result else 0,
                "backtest_metrics": backtest_result.metrics if backtest_result else None,
                **(
                    {
                        "composite_score": backtest_scored.composite_score,
                        "disqualified": backtest_scored.disqualified,
                        "disqualification_reasons": backtest_scored.disqualification_reasons,
                        "component_scores": {
                            c.label: {
                                "score": round(c.raw_score, 3),
                                "weight": c.weight,
                                "weighted": round(c.weighted_score, 3),
                                "category": c.category,
                            }
                            for c in backtest_scored.component_scores
                        },
                    }
                    if backtest_scored is not None
                    else {}
                ),
            },
            "latency_ms": 0,
        },
    }
    if draft.one_shot_mode and one_shot_result is not None:
        agent_outputs["one_shot_validator"] = {
            "agent": "one_shot_validator",
            "status": one_shot_result.status,
            "confidence": one_shot_result.confidence,
            "summary": one_shot_result.summary,
            "flags": one_shot_result.flags,
            "artifacts": one_shot_result.artifacts,
            "latency_ms": one_shot_result.latency_ms,
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
        f"- Scoring source: `{'one_shot' if draft.one_shot_mode else ('strategy_scorer' if strategy_scorer_ready else 'strategy_scorer_unavailable')}`",
    ]

    if draft.one_shot_mode and one_shot_result is not None:
        report_lines.extend(
            [
                "",
                "## One-Shot Validation",
                f"- Recommendation: **{one_shot_result.recommendation}**",
                f"- Summary: {one_shot_result.summary}",
            ]
        )
        for criterion in one_shot_result.artifacts.get("criteria", []):
            criterion_name = str(criterion.get("node", "criterion"))
            verdict = "PASS" if criterion.get("pass") else "FAIL"
            report_lines.append(f"- `{criterion_name}`: **{verdict}**")

    if (
        not draft.one_shot_mode
        and backtest_scored is not None
        and not backtest_scored.disqualified
        and backtest_result
        and backtest_result.metrics
    ):
        m = backtest_result.metrics
        report_lines.extend([
            "",
            "## Strategy Backtest Metrics",
            f"- Period: `{m.get('backtest_start', '?')}` → `{m.get('backtest_end', '?')}`",
            f"- CAGR: `{float(m.get('cagr', 0)):.2%}`  (benchmark: `{float(m.get('benchmark_cagr', 0)):.2%}`)",
            f"- Excess return: `{float(m.get('excess_return', 0)):.2%}`  |  Alpha: `{float(m.get('alpha', 0)):.4f}`",
            f"- Sharpe: `{float(m.get('sharpe_ratio', 0)):.3f}`  |  Sortino: `{float(m.get('sortino_ratio', 0)):.3f}`  |  Calmar: `{float(m.get('calmar_ratio', 0)):.3f}`",
            f"- Max drawdown: `{float(m.get('max_drawdown', 0)):.2%}`  (benchmark: `{float(m.get('benchmark_max_drawdown', 0)):.2%}`)",
            f"- Win rate: `{float(m.get('win_rate', 0)):.1%}`  |  Trades: `{m.get('total_trades', 0)}`  |  Profit factor: `{float(m.get('profit_factor', 0)):.2f}`",
            f"- Up capture: `{float(m.get('up_capture', 0)):.2f}`  |  Down capture: `{float(m.get('down_capture', 0)):.2f}`",
            "",
            "## Strategy Scorer Component Breakdown",
        ])
        for _c in backtest_scored.component_scores:
            _bar = "█" * int(_c.raw_score * 20) + "░" * (20 - int(_c.raw_score * 20))
            report_lines.append(
                f"- `{_c.category}` **{_c.label}**: {_bar} `{_c.raw_score:.2f}` (w={_c.weight})"
            )

    if hard_reject_reasons:
        report_lines.extend(["", "## Hard Reject Reasons"])
        for reason in hard_reject_reasons:
            report_lines.append(f"- {reason}")

    if validation_questions:
        report_lines.extend(["", "## Clarification Questions"])
        for question in validation_questions:
            report_lines.append(f"- {question}")

    report_lines.extend(["", "## Agent Flags"])
    for agent_name in ("data_fetcher", "data_validator", "pipeline_auditor", "one_shot_validator"):
        if agent_name not in agent_outputs:
            continue
        flags = agent_outputs[agent_name]["flags"]
        if flags:
            for flag in flags:
                report_lines.append(
                    f"- `{agent_name}` `{flag['code']}`: {flag['message']}"
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
