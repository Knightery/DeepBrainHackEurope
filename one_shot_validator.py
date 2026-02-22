from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _read_tables(uploaded_files: list[Any]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    tables: dict[str, pd.DataFrame] = {}
    read_errors: list[str] = []
    for file_entry in uploaded_files:
        path = Path(getattr(file_entry, "path", ""))
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".tsv"}:
            continue
        if not path.exists():
            continue
        try:
            sep = "\t" if suffix == ".tsv" else ","
            frame = pd.read_csv(path, sep=sep)
        except Exception as exc:
            read_errors.append(f"{path.name}: {exc.__class__.__name__}: {exc}")
            continue
        if frame.empty:
            continue
        tables[path.name.lower()] = frame
    return tables, read_errors


def _pick_series(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    lower_map = {col.lower(): col for col in frame.columns}
    for candidate in candidates:
        match = lower_map.get(candidate)
        if match:
            values = pd.to_numeric(frame[match], errors="coerce").dropna()
            if len(values) > 0:
                return values.reset_index(drop=True)
    return None


def _find_table_with_columns(
    tables: dict[str, pd.DataFrame],
    left_candidates: tuple[str, ...],
    right_candidates: tuple[str, ...],
) -> tuple[pd.Series | None, pd.Series | None]:
    for frame in tables.values():
        left = _pick_series(frame, left_candidates)
        right = _pick_series(frame, right_candidates)
        if left is None or right is None:
            continue
        n = min(len(left), len(right))
        if n > 0:
            return left.iloc[:n].reset_index(drop=True), right.iloc[:n].reset_index(drop=True)
    return None, None


def _permutation_p_value(x: np.ndarray, y: np.ndarray, n_perm: int = 2000, seed: int = 7) -> float:
    if len(x) != len(y) or len(x) < 3:
        return 1.0
    rng = np.random.default_rng(seed)
    observed = abs(pd.Series(x).rank().corr(pd.Series(y).rank(), method="pearson"))
    if not math.isfinite(observed):
        return 1.0
    exceed = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        corr = abs(pd.Series(x).rank().corr(pd.Series(y_perm).rank(), method="pearson"))
        if corr >= observed:
            exceed += 1
    return (exceed + 1.0) / (n_perm + 1.0)


def _parse_numeric_from_text(text: str, key: str) -> float | None:
    pattern = rf"{re.escape(key)}\s*[:=]\s*(-?\d+(?:\.\d+)?)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return _as_float(match.group(1))


@dataclass
class OneShotResult:
    status: str
    recommendation: str
    summary: str
    confidence: float
    flags: list[dict[str, str]]
    artifacts: dict[str, Any]
    latency_ms: int
    missing_inputs: list[str]
    validation_questions: list[str]


def _parse_token_from_text(text: str, key: str) -> str | None:
    pattern = rf"{re.escape(key)}\s*[:=]\s*([A-Za-z_]+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().lower()


# ---------------------------------------------------------------------------
# LLM Extraction Agent
# ---------------------------------------------------------------------------

def _gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-pro-preview-03-25")


ONE_SHOT_EXTRACTOR_PROMPT = """
You are the One-Shot Strategy Extraction Agent for a quant pitch evaluation pipeline.

Your sole job: extract structured inputs needed for statistical validation of a one-shot
event-driven strategy from the user's free-form thesis, methodology description, and the
actual column names and sample rows of every uploaded CSV/TSV file.

## 1. Event type â€” infer from content, never require a magic keyword

Choose ONE of:
- "causal_chain": a causal mechanism connects a driver variable to an asset return
  (e.g. drought -> wheat price rises -> McDonald's input costs rise -> stock underperforms).
  Needs Nodes 1, 2, 3, 4.
- "binary_event": a discrete catalyst (earnings surprise, FDA decision, regulatory ruling,
  product launch) where the user models a probability vs. market-implied probability.
  Needs Nodes 2, 4.
- "deal_spread": merger/acquisition arbitrage â€” user models deal close probability vs.
  market-implied break-even price. Needs Node 2 and deal pricing inputs.

## 2. Column mapping â€” use actual file content

Map uploaded CSV/TSV columns to semantic roles. Use both column names AND sample values:
- "driver":        independent/causal variable that precedes the return (crop yield,
                   rainfall index, macro indicator, sentiment score, any leading factor)
- "asset_return":  realized returns or price changes of the target asset
- "severity":      magnitude/intensity of historical driver episodes (for Node 3 OLS)
- "magnitude":     price change corresponding to each historical episode (for Node 3 OLS)
- "forecast_prob": probability forecasts â€” values should be in [0, 1]
- "outcome":       binary realized outcome â€” values should be 0 or 1

Return null for a role that is not needed for the inferred event type, or that is absent
from all files. Return null rather than guessing â€” false positives cause wrong statistics.

## 3. Numeric parameter extraction â€” handle free-form text

Extract from supporting_notes using semantic understanding, not just regex:
- "I believe there is a 65% chance" -> p_true = 0.65
- "market implies around 50%" -> p_market = 0.50
- "upside ~120%" or "payoff if correct is 1.2x" -> payoff_up = 1.2
- "downside risk is -30%" -> payoff_down = -0.30
- "I'll assume 1% transaction costs" -> transaction_cost = 0.01
- For deal_spread: extract p_close, current_price, price_if_close, price_if_break

Set a param to null if you cannot determine it with reasonable confidence.

## 4. Questions â€” specific, actionable, plain English

For each role or param you cannot determine, produce one focused question.
- Bad:  "Please provide probability inputs."
- Good: "What probability do you assign to the event occurring, and what does the market
  currently imply? (e.g. 'I think there is a 65% chance, market implies around 50%')"

## Output â€” strict JSON only

{
  "event_type": "causal_chain|binary_event|deal_spread",
  "event_type_reasoning": "one short sentence",
  "column_mappings": {
    "driver":        {"file": "name.csv", "column": "col_name", "confidence": 0.9},
    "asset_return":  {"file": "name.csv", "column": "col_name", "confidence": 0.9},
    "severity":      null,
    "magnitude":     null,
    "forecast_prob": {"file": "name.csv", "column": "col_name", "confidence": 0.9},
    "outcome":       {"file": "name.csv", "column": "col_name", "confidence": 0.9}
  },
  "numeric_params": {
    "p_true": null, "p_market": null,
    "payoff_up": null, "payoff_down": null, "transaction_cost": 0.0,
    "p_close": null, "current_price": null,
    "price_if_close": null, "price_if_break": null
  },
  "extraction_questions": [],
  "extraction_confidence": 0.0
}
""".strip()


@dataclass
class OneShotExtractionResult:
    """Structured output of the LLM extraction agent."""
    event_type: str = "causal_chain"
    event_type_reasoning: str = ""
    column_mappings: dict = field(default_factory=dict)
    numeric_params: dict = field(default_factory=dict)
    extraction_questions: list = field(default_factory=list)
    extraction_confidence: float = 0.0
    extraction_latency_ms: int = 0


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _profile_uploaded_csvs(uploaded_files: list[Any], max_rows: int = 5) -> list[dict[str, Any]]:
    """Build {name, row_count, columns, sample_rows} profiles for the LLM extractor payload."""
    profiles: list[dict[str, Any]] = []
    for file_entry in uploaded_files:
        path = Path(getattr(file_entry, "path", ""))
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".tsv"}:
            continue
        if not path.exists():
            continue
        try:
            sep = "\t" if suffix == ".tsv" else ","
            frame = __import__("pandas").read_csv(path, sep=sep)
            if frame.empty:
                continue
            profiles.append({
                "name": path.name,
                "row_count": len(frame),
                "columns": list(frame.columns),
                "sample_rows": frame.head(max_rows).fillna("").to_dict(orient="records"),
            })
        except Exception as exc:
            profiles.append(
                {
                    "name": path.name,
                    "row_count": 0,
                    "columns": [],
                    "sample_rows": [],
                    "profile_error": f"{exc.__class__.__name__}: {exc}",
                }
            )
    return profiles


def _extract_one_shot_params(draft: Any) -> OneShotExtractionResult:
    """Run the Gemini extraction agent to parse event type, column roles, and numeric params.

    Raises RuntimeError if GEMINI_API_KEY is not set or google-genai is not installed.
    Re-raises any API exception so callers are aware of failures.
    """
    started = time.monotonic()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Gemini extraction is required for one-shot validation."
        )

    try:
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "google-genai package is not installed. "
            "Install it with: pip install google-genai"
        ) from exc

    file_profiles = _profile_uploaded_csvs(getattr(draft, "uploaded_files", []))
    payload = {
        "thesis": getattr(draft, "thesis", "") or "",
        "tickers": getattr(draft, "tickers", []) or [],
        "source_urls": getattr(draft, "source_urls", []) or [],
        "supporting_notes": getattr(draft, "supporting_notes", "") or "",
        "files": file_profiles,
    }

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=_gemini_model(),
        contents=json.dumps(payload, ensure_ascii=True),
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=10000,
            system_instruction=ONE_SHOT_EXTRACTOR_PROMPT,
            response_mime_type="application/json",
        ),
    )
    raw = (getattr(response, "text", "") or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    parsed = json.loads(raw)

    latency_ms = int((time.monotonic() - started) * 1000)

    event_type = str(parsed.get("event_type", "causal_chain")).strip().lower()
    if event_type not in {"causal_chain", "binary_event", "deal_spread"}:
        event_type = "causal_chain"

    col_mappings = parsed.get("column_mappings", {})
    if not isinstance(col_mappings, dict):
        col_mappings = {}

    raw_params = parsed.get("numeric_params", {})
    if not isinstance(raw_params, dict):
        raw_params = {}
    cleaned_params: dict[str, Any] = {}
    for key in (
        "p_true", "p_market", "payoff_up", "payoff_down", "transaction_cost",
        "p_close", "current_price", "price_if_close", "price_if_break",
    ):
        cleaned_params[key] = _as_float(raw_params.get(key))

    questions = parsed.get("extraction_questions", [])
    if not isinstance(questions, list):
        questions = []
    questions = [str(q) for q in questions if q]

    return OneShotExtractionResult(
        event_type=event_type,
        event_type_reasoning=str(parsed.get("event_type_reasoning", "")),
        column_mappings=col_mappings,
        numeric_params=cleaned_params,
        extraction_questions=questions,
        extraction_confidence=_clamp(float(parsed.get("extraction_confidence", 0.5))),
        extraction_latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# LLM-aware series resolution helpers
# ---------------------------------------------------------------------------

def _resolve_mapped_series(
    tables: dict[str, Any],
    mapping: dict[str, Any] | None,
) -> "pd.Series | None":
    """Resolve one LLM column mapping to a numeric Series. Returns None if mapping absent or column not found."""
    if not mapping or not isinstance(mapping, dict):
        return None
    file_name = str(mapping.get("file", "")).strip().lower()
    col_name = str(mapping.get("column", "")).strip()
    if not file_name or not col_name:
        return None
    frame = tables.get(file_name)
    if frame is None:
        return None
    lower_map = {c.lower(): c for c in frame.columns}
    actual_col = lower_map.get(col_name.lower())
    if actual_col is None:
        return None
    values = pd.to_numeric(frame[actual_col], errors="coerce").dropna()
    return values.reset_index(drop=True) if len(values) > 0 else None


def _find_pair_with_fallback(
    tables: dict[str, Any],
    left_mapping: dict[str, Any] | None,
    right_mapping: dict[str, Any] | None,
) -> "tuple[pd.Series | None, pd.Series | None]":
    """Resolve a paired series from LLM column mappings. Returns (None, None) if either can't be resolved."""
    left = _resolve_mapped_series(tables, left_mapping)
    right = _resolve_mapped_series(tables, right_mapping)
    if left is None or right is None:
        return None, None
    n = min(len(left), len(right))
    return left.iloc[:n].reset_index(drop=True), right.iloc[:n].reset_index(drop=True)


def _resolve_numeric(
    extraction: OneShotExtractionResult,
    key: str,
    confidence_threshold: float = 0.6,
) -> float | None:
    """Return LLM-extracted numeric value, or None if below confidence threshold or key absent."""
    if extraction.extraction_confidence < confidence_threshold:
        return None
    return extraction.numeric_params.get(key)


def _run_node2_forecast_calibration(
    tables: dict[str, pd.DataFrame],
    extraction: OneShotExtractionResult,
    flags: list[dict[str, str]],
    questions: list[str],
    missing_inputs: list[str],
) -> dict[str, Any]:
    col_maps = extraction.column_mappings
    forecast, outcome = _find_pair_with_fallback(
        tables,
        col_maps.get("forecast_prob"),
        col_maps.get("outcome"),
    )
    node2: dict[str, Any] = {"node": "forecast_calibration", "pass": False, "details": {}}
    if forecast is None or outcome is None or min(len(forecast), len(outcome)) < 20:
        missing_inputs.append("node2_forecast_history")
        questions.append(
            "Please upload a forecast calibration history CSV with at least 20 rows. "
            "The file should have one column of probability estimates (values between 0 and 1) "
            "and one column of binary realized outcomes (0 or 1). "
            "Column names can be anything â€” the system will identify them automatically."
        )
        flags.append({
            "code": "ONE_SHOT_NODE2_INPUT_MISSING",
            "message": "Missing or insufficient forecast calibration history for Node 2.",
        })
        return node2

    n = min(len(forecast), len(outcome))
    f = np.clip(forecast.iloc[:n].to_numpy(dtype=float), 0.0, 1.0)
    o = np.clip(outcome.iloc[:n].to_numpy(dtype=float), 0.0, 1.0)
    bs = float(brier_score_loss(o, f))
    base_rate = float(np.mean(o))
    bs_clim = float(np.mean((base_rate - o) ** 2))
    bss = 1.0 - (bs / bs_clim) if bs_clim > 1e-12 else 0.0

    bins = np.linspace(0.0, 1.0, 11)
    idx = np.digitize(f, bins, right=True)
    weighted_gap = 0.0
    for b in range(1, len(bins)):
        mask = idx == b
        if not np.any(mask):
            continue
        weighted_gap += float(np.mean(mask)) * abs(float(np.mean(f[mask]) - np.mean(o[mask])))

    passed = bool(bss > 0.0 and weighted_gap <= 0.10)
    node2["pass"] = passed
    node2["details"] = {
        "n": n,
        "brier_score": round(bs, 6),
        "brier_skill_score": round(bss, 6),
        "weighted_calibration_gap": round(weighted_gap, 6),
    }
    if not passed:
        flags.append(
            {
                "code": "ONE_SHOT_NODE2_POOR_CALIBRATION",
                "message": "Node 2 failed calibration criteria (requires BSS > 0 and low calibration gap).",
            }
        )
    return node2


def evaluate_one_shot_strategy(*, draft: Any, min_positive_edge_prob: float = 0.75) -> OneShotResult:
    started = time.monotonic()
    flags: list[dict[str, str]] = []
    questions: list[str] = []
    missing_inputs: list[str] = []
    fallback_inputs_used: list[str] = []
    criteria: list[dict[str, Any]] = []

    # â”€â”€ Phase 0: LLM Extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Gemini reads thesis, methodology, tickers, and actual CSV column profiles to
    # infer event type, map columns to semantic roles, and extract numeric params
    # from free-form text. Raises RuntimeError if GEMINI_API_KEY is not set.
    extraction = _extract_one_shot_params(draft)
    # Surface LLM-generated questions that aren't already covered by node checks below
    questions.extend(extraction.extraction_questions)

    tables, table_read_errors = _read_tables(getattr(draft, "uploaded_files", []))
    if table_read_errors:
        flags.append(
            {
                "code": "ONE_SHOT_TABLE_READ_ERROR",
                "message": "Some uploaded CSV/TSV files could not be parsed: " + "; ".join(table_read_errors),
            }
        )
    col_maps = extraction.column_mappings

    variant = extraction.event_type
    if variant not in {"causal_chain", "binary_event", "deal_spread"}:
        flags.append({
            "code": "ONE_SHOT_EVENT_TYPE_UNKNOWN",
            "message": f"Unknown one-shot event type '{variant}'. Falling back to 'causal_chain'.",
        })
        variant = "causal_chain"

    beta = 0.0
    beta_std = 0.15
    node1_n = 0

    # â”€â”€ Node 2: Forecast Calibration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    node2 = _run_node2_forecast_calibration(tables, extraction, flags, questions, missing_inputs)
    criteria.append(node2)

    if variant == "causal_chain":
        # â”€â”€ Node 1: Causal Relationship â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        x, y = _find_pair_with_fallback(
            tables,
            col_maps.get("driver"),
            col_maps.get("asset_return"),
        )
        node1: dict[str, Any] = {"node": "causal_relationship", "pass": False, "details": {}}
        if x is None or y is None or min(len(x), len(y)) < 30:
            missing_inputs.append("node1_relationship_series")
            questions.append(
                "Please upload a CSV with at least 30 rows containing your causal driver series "
                "and the target asset return series side-by-side. "
                "Column names can be anything â€” the system will identify them automatically."
            )
            flags.append({
                "code": "ONE_SHOT_NODE1_INPUT_MISSING",
                "message": "Missing or insufficient relationship time series for Node 1 (minimum 30 rows).",
            })
            node1_n = int(min(len(x), len(y))) if (x is not None and y is not None) else 0
            node1["details"] = {"n": node1_n}
        else:
            node1_n = min(len(x), len(y))
            x_arr = x.iloc[:node1_n].to_numpy(dtype=float)
            y_arr = y.iloc[:node1_n].to_numpy(dtype=float)
            split = int(max(1, round(node1_n * 0.7)))
            in_corr = float(pd.Series(x_arr[:split]).corr(pd.Series(y_arr[:split]), method="spearman"))
            oos_corr = (
                float(pd.Series(x_arr[split:]).corr(pd.Series(y_arr[split:]), method="spearman"))
                if node1_n - split >= 3
                else 0.0
            )
            p_value = _permutation_p_value(x_arr, y_arr)
            sign_stable = (in_corr == 0.0 and oos_corr == 0.0) or (np.sign(in_corr) == np.sign(oos_corr))
            passed = bool(p_value < 0.05 and oos_corr > 0.0 and sign_stable)
            node1["pass"] = passed
            node1["details"] = {
                "n": node1_n,
                "spearman_in_sample": round(in_corr, 4),
                "spearman_out_of_sample": round(oos_corr, 4),
                "p_value": round(p_value, 6),
                "sign_stable": bool(sign_stable),
            }
            if not passed:
                flags.append({
                    "code": "ONE_SHOT_NODE1_WEAK_RELATIONSHIP",
                    "message": "Node 1 failed significance/stability checks (requires p<0.05 and stable positive OOS relationship).",
                })
        criteria.append(node1)

        # â”€â”€ Node 3: Magnitude Estimate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        severity, change = _find_pair_with_fallback(
            tables,
            col_maps.get("severity"),
            col_maps.get("magnitude"),
        )
        node3: dict[str, Any] = {"node": "magnitude_estimate", "pass": False, "details": {}}
        if severity is None or change is None or min(len(severity), len(change)) < 8:
            missing_inputs.append("node3_magnitude_history")
            questions.append(
                "Please upload a CSV with at least 8 historical episodes showing the intensity "
                "of the driver event alongside the resulting price change. "
                "Column names can be anything â€” the system will identify them automatically."
            )
            flags.append({
                "code": "ONE_SHOT_NODE3_INPUT_MISSING",
                "message": "Missing or insufficient magnitude history for Node 3.",
            })
        else:
            n3 = min(len(severity), len(change))
            x_arr = severity.iloc[:n3].to_numpy(dtype=float)
            y_arr = change.iloc[:n3].to_numpy(dtype=float)
            x_mean = float(np.mean(x_arr))
            y_mean = float(np.mean(y_arr))
            ssx = float(np.sum((x_arr - x_mean) ** 2))
            if ssx <= 1e-12:
                ci_low, ci_high = -math.inf, math.inf
                flags.append({
                    "code": "ONE_SHOT_NODE3_DEGENERATE_INPUT",
                    "message": "Node 3 severity input has near-zero variance; cannot estimate slope reliably.",
                })
            else:
                beta = float(np.sum((x_arr - x_mean) * (y_arr - y_mean)) / ssx)
                alpha = y_mean - beta * x_mean
                residuals = y_arr - (alpha + beta * x_arr)
                sigma2 = float(np.sum(residuals**2) / max(n3 - 2, 1))
                beta_std = math.sqrt(max(sigma2 / ssx, 1e-12))
                ci_low = beta - 1.96 * beta_std
                ci_high = beta + 1.96 * beta_std
            passed = bool(ci_low > 0.0 and math.isfinite(ci_low))
            node3["pass"] = passed
            node3["details"] = {
                "n": n3,
                "beta": round(beta, 6),
                "beta_std_err": round(beta_std, 6),
                "ci95_low": round(ci_low, 6),
                "ci95_high": round(ci_high, 6),
            }
            if not passed:
                flags.append({
                    "code": "ONE_SHOT_NODE3_CI_INCLUDES_ZERO",
                    "message": "Node 3 failed magnitude significance (95% CI lower bound must be > 0).",
                })
        criteria.append(node3)

    # â”€â”€ Node 4: Market Mispricing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p_true = _resolve_numeric(extraction, "p_true")
    p_market = _resolve_numeric(extraction, "p_market")
    payoff_up = _resolve_numeric(extraction, "payoff_up")
    payoff_down = _resolve_numeric(extraction, "payoff_down")
    transaction_cost = _resolve_numeric(extraction, "transaction_cost")
    if transaction_cost is None:
        fallback_inputs_used.append("transaction_cost")
        transaction_cost = 0.0

    if variant == "deal_spread":
        p_close = _resolve_numeric(extraction, "p_close")
        current_price = _resolve_numeric(extraction, "current_price")
        price_if_close = _resolve_numeric(extraction, "price_if_close")
        price_if_break = _resolve_numeric(extraction, "price_if_break")
        node4: dict[str, Any] = {"node": "deal_spread_pricing", "pass": False, "details": {}}
        if p_close is None or current_price is None or price_if_close is None or price_if_break is None:
            missing_inputs.append("deal_pricing_inputs")
            questions.append(
                "For a deal-spread strategy, please provide: your probability of the deal "
                "closing, the current market price, the expected price if it closes, and the "
                "expected price if it breaks. You can write these conversationally â€” for example: "
                "'I think there is a 75% chance the deal closes. Stock is at $45, acquisition "
                "price is $55, break price is around $35.'"
            )
            flags.append({
                "code": "ONE_SHOT_DEAL_INPUT_MISSING",
                "message": "Missing deal-spread inputs required for pricing edge.",
            })
            if p_close is None:
                fallback_inputs_used.append("p_close")
                p_close = 0.5
            if current_price is None:
                fallback_inputs_used.append("current_price")
                current_price = 1.0
            if price_if_close is None:
                fallback_inputs_used.append("price_if_close")
                price_if_close = current_price
            if price_if_break is None:
                fallback_inputs_used.append("price_if_break")
                price_if_break = current_price
        p_true = _clamp(float(p_close))
        denom = float(price_if_close - price_if_break)
        if abs(denom) > 1e-12:
            p_market = _clamp((float(current_price) - float(price_if_break)) / denom)
        else:
            fallback_inputs_used.append("deal_market_implied_probability")
            p_market = 0.5
        delta_p = float(p_true - p_market)
        node4["pass"] = bool(delta_p >= 0.05)
        node4["details"] = {
            "p_close_model": round(p_true, 4),
            "p_close_market_implied": round(p_market, 4),
            "delta_p": round(delta_p, 4),
            "current_price": round(float(current_price), 6),
            "price_if_close": round(float(price_if_close), 6),
            "price_if_break": round(float(price_if_break), 6),
            "transaction_cost": round(float(transaction_cost or 0.0), 6),
        }
        if not node4["pass"]:
            flags.append({
                "code": "ONE_SHOT_DEAL_WEAK_EDGE",
                "message": "Deal-spread mispricing edge is weak (requires model probability gap >= 5pp).",
            })
    else:
        node4 = {"node": "market_mispricing", "pass": False, "details": {}}
        if p_true is None or p_market is None or payoff_up is None or payoff_down is None:
            missing_inputs.append("node4_market_inputs")
            questions.append(
                "What probability do you assign to the event occurring, and what does the market "
                "currently imply? Also, what is the upside return if correct and the downside if wrong? "
                "For example: 'I think there is a 65% chance, market implies 50%, "
                "upside is +120%, downside is -30%.' You do not need to use any special format."
            )
            flags.append({
                "code": "ONE_SHOT_NODE4_INPUT_MISSING",
                "message": "Missing market-implied probability and payoff assumptions for Node 4.",
            })
            if p_true is None:
                fallback_inputs_used.append("p_true")
                p_true = 0.5
            if p_market is None:
                fallback_inputs_used.append("p_market")
                p_market = 0.5
            if payoff_up is None:
                fallback_inputs_used.append("payoff_up")
                payoff_up = 1.0
            if payoff_down is None:
                fallback_inputs_used.append("payoff_down")
                payoff_down = -1.0
        p_true = _clamp(float(p_true))
        p_market = _clamp(float(p_market))
        delta_p = float(p_true - p_market)
        node4["pass"] = bool(delta_p >= 0.05)
        node4["details"] = {
            "p_true": round(p_true, 4),
            "p_market": round(p_market, 4),
            "delta_p": round(delta_p, 4),
            "payoff_up": round(float(payoff_up), 6),
            "payoff_down": round(float(payoff_down), 6),
            "transaction_cost": round(float(transaction_cost or 0.0), 6),
        }
        if not node4["pass"]:
            flags.append({
                "code": "ONE_SHOT_NODE4_WEAK_MISPRICING",
                "message": "Node 4 failed mispricing criterion (requires p_true - p_market >= 5pp).",
            })
    criteria.append(node4)

    # Monte Carlo uncertainty aggregation
    rng = np.random.default_rng(11)
    samples = 10_000
    n2 = int(node2["details"].get("n", 0) or 0)
    p_true_std = math.sqrt(max(float(p_true) * (1.0 - float(p_true)) / max(n2, 1), 0.05**2))
    if "node2_forecast_history" in missing_inputs:
        p_true_std = max(p_true_std, 0.12)

    if variant == "causal_chain":
        rho_mean = 0.0
        for item in criteria:
            if item.get("node") == "causal_relationship":
                rho_mean = float(item.get("details", {}).get("spearman_in_sample", 0.0))
                node1_n = int(item.get("details", {}).get("n", 0) or 0)
                break
        rho_std = math.sqrt(max((1.0 - rho_mean * rho_mean) / max(node1_n - 2, 1), 1e-4))
        if "node1_relationship_series" in missing_inputs:
            rho_std = max(rho_std, 0.30)
        if "node3_magnitude_history" in missing_inputs:
            beta_std = max(beta_std, 0.20)

        rho_draws = np.clip(rng.normal(rho_mean, rho_std, size=samples), -1.0, 1.0)
        beta_draws = rng.normal(beta, beta_std, size=samples)
        p_true_draws = np.clip(rng.normal(float(p_true), p_true_std, size=samples), 0.0, 1.0)
        base_ev_draws = p_true_draws * float(payoff_up) + (1.0 - p_true_draws) * float(payoff_down)
        relationship_multiplier = np.clip(rho_draws, 0.0, 1.0)
        magnitude_multiplier = np.clip(beta_draws / max(abs(beta) if abs(beta) > 1e-6 else 1.0, 1e-6), 0.0, 2.0)
        ev_draws = base_ev_draws * relationship_multiplier * magnitude_multiplier
    elif variant == "deal_spread":
        current_price = float(node4["details"].get("current_price", 1.0))
        price_if_close = float(node4["details"].get("price_if_close", current_price))
        price_if_break = float(node4["details"].get("price_if_break", current_price))
        p_draws = np.clip(rng.normal(float(p_true), p_true_std, size=samples), 0.0, 1.0)
        close_std = abs(price_if_close) * 0.02
        break_std = abs(price_if_break) * 0.02
        close_draws = rng.normal(price_if_close, close_std, size=samples)
        break_draws = rng.normal(price_if_break, break_std, size=samples)
        expected_prices = p_draws * close_draws + (1.0 - p_draws) * break_draws
        ev_draws = (expected_prices / max(current_price, 1e-6)) - 1.0 - float(transaction_cost or 0.0)
    else:
        p_true_draws = np.clip(rng.normal(float(p_true), p_true_std, size=samples), 0.0, 1.0)
        ev_draws = p_true_draws * float(payoff_up) + (1.0 - p_true_draws) * float(payoff_down)

    ev_mean = float(np.mean(ev_draws))
    ev_p5 = float(np.percentile(ev_draws, 5))
    ev_p95 = float(np.percentile(ev_draws, 95))
    prob_positive_edge = float(np.mean(ev_draws > 0.0))

    monte_carlo_pass = bool(ev_p5 > float(transaction_cost or 0.0) and prob_positive_edge >= min_positive_edge_prob)
    if not monte_carlo_pass:
        flags.append({
            "code": "ONE_SHOT_MONTE_CARLO_FAIL",
            "message": (
                "Monte Carlo failed one-shot thresholds "
                f"(requires EV p5 > costs and P(EV>0) >= {min_positive_edge_prob:.0%})."
            ),
        })

    criteria.append({
        "node": "monte_carlo_edge",
        "pass": monte_carlo_pass,
        "details": {
            "simulations": samples,
            "mean_ev": round(ev_mean, 6),
            "ev_p5": round(ev_p5, 6),
            "ev_p95": round(ev_p95, 6),
            "prob_positive_edge": round(prob_positive_edge, 6),
            "threshold_prob_positive_edge": min_positive_edge_prob,
            "transaction_cost": round(float(transaction_cost or 0.0), 6),
        },
    })

    pass_count = sum(1 for item in criteria if bool(item.get("pass")))
    total = len(criteria)
    recommendation = "VALID" if pass_count == total and not missing_inputs else "NOT_VALID"

    if recommendation == "VALID":
        summary = "One-shot validation passed all node criteria and Monte Carlo edge checks."
    else:
        summary = f"One-shot validation failed {total - pass_count} of {total} criteria."

    if missing_inputs:
        flags.append({
            "code": "ONE_SHOT_INPUTS_INCOMPLETE",
            "message": "One-shot inputs are incomplete; uncertainty was widened and recommendation is provisional.",
        })
    fallback_inputs_used = list(dict.fromkeys(fallback_inputs_used))
    if fallback_inputs_used:
        flags.append(
            {
                "code": "ONE_SHOT_FALLBACK_VALUES_USED",
                "message": (
                    "One-shot computation used hardcoded fallback values for: "
                    + ", ".join(fallback_inputs_used)
                ),
            }
        )

    status = "ok" if recommendation == "VALID" else "warn"
    confidence = _clamp((pass_count / max(total, 1)) * 0.9)

    return OneShotResult(
        status=status,
        recommendation=recommendation,
        summary=summary,
        confidence=confidence,
        flags=flags,
        artifacts={
            "mode": "one_shot",
            "event_type": variant,
            "criteria": criteria,
            "pass_count": pass_count,
            "total_criteria": total,
            "recommendation": recommendation,
            "monte_carlo": criteria[-1]["details"],
            "missing_inputs": missing_inputs,
            "fallback_inputs_used": fallback_inputs_used,
            "extraction": {
                "confidence": extraction.extraction_confidence,
                "latency_ms": extraction.extraction_latency_ms,
                "event_type_reasoning": extraction.event_type_reasoning,
                "column_mappings": extraction.column_mappings,
                "numeric_params": extraction.numeric_params,
            },
        },
        latency_ms=int((time.monotonic() - started) * 1000),
        missing_inputs=missing_inputs,
        validation_questions=questions,
    )

