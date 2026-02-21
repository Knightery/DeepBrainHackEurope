from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
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


def _read_tables(uploaded_files: list[Any]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
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
        except Exception:
            continue
        if frame.empty:
            continue
        tables[path.name.lower()] = frame
    return tables


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


def _run_node2_forecast_calibration(
    tables: dict[str, pd.DataFrame],
    flags: list[dict[str, str]],
    questions: list[str],
    missing_inputs: list[str],
) -> dict[str, Any]:
    forecast, outcome = _find_table_with_columns(
        tables,
        ("forecast_prob", "probability", "p_forecast", "forecast"),
        ("outcome", "event", "observed", "label"),
    )
    node2 = {"node": "forecast_calibration", "pass": False, "details": {}}
    if forecast is None or outcome is None or min(len(forecast), len(outcome)) < 20:
        missing_inputs.append("node2_forecast_history")
        questions.append(
            "Upload forecast calibration history with `forecast_prob` and binary `outcome` columns "
            "(minimum 20 rows; 30+ preferred)."
        )
        flags.append(
            {
                "code": "ONE_SHOT_NODE2_INPUT_MISSING",
                "message": "Missing or insufficient forecast calibration history for Node 2.",
            }
        )
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
    criteria: list[dict[str, Any]] = []

    tables = _read_tables(draft.uploaded_files)
    method_text = draft.methodology_summary or ""
    variant = _parse_token_from_text(method_text, "one_shot_event_type") or "causal_chain"
    if variant not in {"causal_chain", "binary_event", "deal_spread"}:
        flags.append(
            {
                "code": "ONE_SHOT_EVENT_TYPE_UNKNOWN",
                "message": f"Unknown one-shot event type '{variant}'. Falling back to 'causal_chain'.",
            }
        )
        variant = "causal_chain"

    # Node 1 + Node 3 are specific to causal-chain setups.
    beta = 0.0
    beta_std = 0.15
    node1_n = 0
    node2 = _run_node2_forecast_calibration(tables, flags, questions, missing_inputs)
    criteria.append(node2)

    if variant == "causal_chain":
        x, y = _find_table_with_columns(
            tables,
            ("wheat_price", "driver_value", "variable_a", "x", "factor"),
            ("mcd_return", "asset_return", "target_return", "y", "response"),
        )
        node1 = {"node": "causal_relationship", "pass": False, "details": {}}
        if x is None or y is None or min(len(x), len(y)) < 30:
            missing_inputs.append("node1_relationship_series")
            questions.append(
                "Upload one CSV with at least 30 rows containing both driver and asset-return columns "
                "(for example `wheat_price` and `mcd_return`)."
            )
            flags.append(
                {
                    "code": "ONE_SHOT_NODE1_INPUT_MISSING",
                    "message": "Missing or insufficient relationship time series for Node 1 (minimum 30 rows).",
                }
            )
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
                flags.append(
                    {
                        "code": "ONE_SHOT_NODE1_WEAK_RELATIONSHIP",
                        "message": "Node 1 failed significance/stability checks (requires p<0.05 and stable positive OOS relationship).",
                    }
                )
        criteria.append(node1)

        severity, change = _find_table_with_columns(
            tables,
            ("drought_severity", "severity", "x_magnitude", "driver_magnitude"),
            ("wheat_change", "price_change", "magnitude_change", "delta_price"),
        )
        node3 = {"node": "magnitude_estimate", "pass": False, "details": {}}
        if severity is None or change is None or min(len(severity), len(change)) < 8:
            missing_inputs.append("node3_magnitude_history")
            questions.append(
                "Upload historical magnitude data with severity and resulting price-change columns "
                "(minimum 8 episodes)."
            )
            flags.append(
                {
                    "code": "ONE_SHOT_NODE3_INPUT_MISSING",
                    "message": "Missing or insufficient magnitude history for Node 3.",
                }
            )
        else:
            n3 = min(len(severity), len(change))
            x_arr = severity.iloc[:n3].to_numpy(dtype=float)
            y_arr = change.iloc[:n3].to_numpy(dtype=float)
            x_mean = float(np.mean(x_arr))
            y_mean = float(np.mean(y_arr))
            ssx = float(np.sum((x_arr - x_mean) ** 2))
            if ssx <= 1e-12:
                ci_low, ci_high = -math.inf, math.inf
                flags.append(
                    {
                        "code": "ONE_SHOT_NODE3_DEGENERATE_INPUT",
                        "message": "Node 3 severity input has near-zero variance; cannot estimate slope reliably.",
                    }
                )
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
                flags.append(
                    {
                        "code": "ONE_SHOT_NODE3_CI_INCLUDES_ZERO",
                        "message": "Node 3 failed magnitude significance (95% CI lower bound must be > 0).",
                    }
                )
        criteria.append(node3)

    # Node 4: market mispricing inputs
    p_true = _parse_numeric_from_text(method_text, "p_true")
    p_market = _parse_numeric_from_text(method_text, "p_market")
    payoff_up = _parse_numeric_from_text(method_text, "payoff_up")
    payoff_down = _parse_numeric_from_text(method_text, "payoff_down")
    transaction_cost = _parse_numeric_from_text(method_text, "transaction_cost")
    if transaction_cost is None:
        transaction_cost = 0.0

    if variant == "deal_spread":
        p_close = _parse_numeric_from_text(method_text, "p_close")
        current_price = _parse_numeric_from_text(method_text, "current_price")
        price_if_close = _parse_numeric_from_text(method_text, "price_if_close")
        price_if_break = _parse_numeric_from_text(method_text, "price_if_break")
        node4 = {"node": "deal_spread_pricing", "pass": False, "details": {}}
        if p_close is None or current_price is None or price_if_close is None or price_if_break is None:
            missing_inputs.append("deal_pricing_inputs")
            questions.append(
                "For `one_shot_event_type=deal_spread`, include: "
                "`p_close=..., current_price=..., price_if_close=..., price_if_break=..., transaction_cost=...`."
            )
            flags.append(
                {
                    "code": "ONE_SHOT_DEAL_INPUT_MISSING",
                    "message": "Missing deal-spread inputs required for pricing edge.",
                }
            )
            p_close = p_close if p_close is not None else 0.5
            current_price = current_price if current_price is not None else 1.0
            price_if_close = price_if_close if price_if_close is not None else current_price
            price_if_break = price_if_break if price_if_break is not None else current_price
        p_true = _clamp(float(p_close))
        denom = float(price_if_close - price_if_break)
        if abs(denom) > 1e-12:
            p_market = _clamp((float(current_price) - float(price_if_break)) / denom)
        else:
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
            flags.append(
                {
                    "code": "ONE_SHOT_DEAL_WEAK_EDGE",
                    "message": "Deal-spread mispricing edge is weak (requires model probability gap >= 5pp).",
                }
            )
    else:
        node4 = {"node": "market_mispricing", "pass": False, "details": {}}
        if p_true is None or p_market is None or payoff_up is None or payoff_down is None:
            missing_inputs.append("node4_market_inputs")
            questions.append(
                "Add one-shot market inputs in methodology text: "
                "`p_true=0.60, p_market=0.50, payoff_up=1.0, payoff_down=-1.0, transaction_cost=0.02`."
            )
            flags.append(
                {
                    "code": "ONE_SHOT_NODE4_INPUT_MISSING",
                    "message": "Missing market-implied probability and payoff assumptions for Node 4.",
                }
            )
            p_true = p_true if p_true is not None else 0.5
            p_market = p_market if p_market is not None else 0.5
            payoff_up = payoff_up if payoff_up is not None else 1.0
            payoff_down = payoff_down if payoff_down is not None else -1.0
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
            flags.append(
                {
                    "code": "ONE_SHOT_NODE4_WEAK_MISPRICING",
                    "message": "Node 4 failed mispricing criterion (requires p_true - p_market >= 5pp).",
                }
            )
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
        },
        latency_ms=int((time.monotonic() - started) * 1000),
        missing_inputs=missing_inputs,
        validation_questions=questions,
    )
