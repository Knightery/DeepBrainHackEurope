"""
Live one-shot validation smoke test.

Scenario: Consumer confidence as a causal driver of XRT (SPDR S&P Retail ETF) returns.

Hypothesis (natural language, no magic keywords):
  "When US consumer confidence readings drop more than 5 points in a month, the
   XRT retail ETF tends to sell off over the following two weeks. I think the market
   is currently underpricing the probability that this drawdown materialises,
   given that the most recent Michigan survey came in at a 3-year low. My rough
   estimate is there is about a 64% chance of a meaningful decline, while options
   pricing implies something closer to 48%. If I'm right I'd expect roughly 12%
   downside from here; if wrong, a 6% rally as sentiment recovers faster than expected."

Data is intentionally messy:
  - consumer_sentiment_weekly.csv  (driver + asset return in same file, extra cols, ~10% NaN)
  - analyst_track_record.csv       (calibration history with opaque column names)
  - historical_drawdown_episodes.csv (magnitude history, mixed date formats, extra labels)
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from one_shot_validator import evaluate_one_shot_strategy  # noqa: E402

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Build messy CSV files
# ---------------------------------------------------------------------------

def _make_sentiment_returns(folder: Path) -> SimpleNamespace:
    """
    consumer_sentiment_weekly.csv
    Columns (intentionally odd names):
      week_ending, conf_index, prev_conf, survey_n, region_code,
      xrt_wkly_pct, notes
    ~10 % of xrt_wkly_pct values are NaN (data gaps).
    Strong positive correlation: high confidence → positive retail returns.
    """
    n = 90
    trend = np.linspace(95, 105, n)
    conf_index = trend + RNG.normal(0, 4, n)

    # Positive relationship: returns track confidence deviations from mean
    base_return = (conf_index - conf_index.mean()) * 0.004
    xrt_return = base_return + RNG.normal(0, 0.012, n)

    # Inject ~10 % NaN in xrt column
    nan_mask = RNG.random(n) < 0.10
    xrt_return_messy = xrt_return.astype(object)
    xrt_return_messy[nan_mask] = None

    df = pd.DataFrame({
        "week_ending": pd.date_range("2022-01-07", periods=n, freq="W-FRI").strftime("%d/%m/%Y"),
        "conf_index": np.round(conf_index, 1),
        "prev_conf": np.round(np.roll(conf_index, 1), 1),
        "survey_n": RNG.integers(400, 600, n),
        "region_code": RNG.choice(["N", "S", "E", "W", "MW"], n),
        "xrt_wkly_pct": [None if v is None else round(float(v), 4) for v in xrt_return_messy],
        "notes": RNG.choice(["", "revised", "preliminary", "n/a"], n),
    })

    path = folder / "consumer_sentiment_weekly.csv"
    df.to_csv(path, index=False)
    return SimpleNamespace(path=str(path))


def _make_analyst_track_record(folder: Path) -> SimpleNamespace:
    """
    analyst_track_record.csv
    Intentionally opaque column names for forecast_prob / outcome:
      qtr_id, analyst_id, prob_call, was_right, call_direction, rationale_tag
    BSS > 0, calibration gap ≤ 0.10.
    """
    n = 80
    # Simulate reasonably calibrated analysts
    true_probs = RNG.uniform(0.2, 0.85, n)
    was_right = (RNG.random(n) < true_probs).astype(int)

    # Add small miscalibration: inflate probs slightly
    reported_probs = np.clip(true_probs + RNG.normal(0.03, 0.04, n), 0.05, 0.95)

    df = pd.DataFrame({
        "qtr_id": [f"Q{i // 10 + 1}-{2020 + i // 40}" for i in range(n)],
        "analyst_id": RNG.choice(["JT", "MR", "PK", "SL"], n),
        "prob_call": np.round(reported_probs, 3),
        "was_right": was_right,
        "call_direction": RNG.choice(["long", "short"], n),
        "rationale_tag": RNG.choice(["macro", "technical", "fundamental"], n),
    })

    path = folder / "analyst_track_record.csv"
    df.to_csv(path, index=False)
    return SimpleNamespace(path=str(path))


def _make_historical_episodes(folder: Path) -> SimpleNamespace:
    """
    historical_drawdown_episodes.csv
    Intentionally messy: mixed date formats, extra label cols, column names
    that don't scream 'severity' or 'magnitude'.
      ep_id, event_label, trigger_yyyymm, conf_drop_pts, xrt_fwd3w_ret, sector_tag
    Positive beta: larger confidence drops → larger XRT drawdowns.
    """
    n = 22
    conf_drop = RNG.uniform(3, 20, n)          # severity
    xrt_drawdown = -conf_drop * 0.009 + RNG.normal(0, 0.015, n)  # magnitude (negative)

    # For Node 3 to pass we need CI lower > 0, so let's use absolute values
    # and frame it as "confidence drop → magnitude of drawdown (positive number)"
    xrt_mag = np.abs(xrt_drawdown)  # positive: larger drop = bigger mag

    trigger_months = pd.date_range("2010-01", periods=n, freq="6ME").strftime("%Y-%m")

    df = pd.DataFrame({
        "ep_id": [f"EP{i+1:02d}" for i in range(n)],
        "event_label": RNG.choice(
            ["debt_ceiling", "fed_surprise", "cpi_shock", "geopolit", "bank_stress"], n
        ),
        "trigger_yyyymm": trigger_months,
        "conf_drop_pts": np.round(conf_drop, 2),
        "xrt_fwd3w_ret": np.round(xrt_mag, 4),
        "sector_tag": "retail",
    })

    path = folder / "historical_drawdown_episodes.csv"
    df.to_csv(path, index=False)
    return SimpleNamespace(path=str(path))


# ---------------------------------------------------------------------------
# Run the test
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 70)
    print("ONE-SHOT LIVE VALIDATION TEST")
    print("Scenario: Consumer Sentiment → XRT Retail ETF (causal chain)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        f_sentiment = _make_sentiment_returns(base)
        f_track = _make_analyst_track_record(base)
        f_episodes = _make_historical_episodes(base)

        print("\n[data] Files written:")
        for f in (f_sentiment, f_track, f_episodes):
            p = Path(f.path)
            df = pd.read_csv(p)
            print(f"  {p.name}: {len(df)} rows × {len(df.columns)} cols  "
                  f"| columns: {list(df.columns)}")

        draft = SimpleNamespace(
            uploaded_files=[f_sentiment, f_track, f_episodes],
            thesis=(
                "US consumer confidence is a leading indicator for US retail equity performance. "
                "When the Michigan Consumer Sentiment Index drops more than 5 points in a month, "
                "XRT (S&P Retail ETF) tends to underperform the market over the following 2 weeks. "
                "The most recent reading came in at a 3-year low, suggesting a meaningful drawdown "
                "is likely over the short horizon."
            ),
            methodology_summary=(
                "This is a causal-chain pitch. I believe there's roughly a 64% chance we see a "
                "significant retail sector drawdown over the next two weeks, while options skew "
                "and analyst consensus seems to imply something closer to 48%. "
                "If the drawdown materialises, I'd expect XRT to fall around 12% from current levels. "
                "If sentiment stabilises or bounces back faster, the upside is about 6%. "
                "Transaction costs including spread are around 0.3%."
            ),
            tickers=["XRT"],
            source_urls=["https://data.sca.isr.umich.edu/", "https://finance.yahoo.com/quote/XRT"],
        )

        print("\n[extraction] Calling Gemini extraction agent (live)...")

        try:
            result = evaluate_one_shot_strategy(draft=draft, min_positive_edge_prob=0.60)
        except Exception as exc:
            print(f"\n[ERROR] evaluate_one_shot_strategy raised: {exc.__class__.__name__}: {exc}")
            raise

        ext = result.artifacts.get("extraction", {})
        print("\n" + "-" * 70)
        print("EXTRACTION RESULTS")
        print("-" * 70)
        print(f"  event_type_reasoning : {ext.get('event_type_reasoning', '')}")
        print(f"  extraction_confidence: {ext.get('confidence', 'n/a')}")
        print(f"  latency_ms           : {ext.get('latency_ms', 'n/a')}")
        print("\n  column_mappings:")
        for role, mapping in (ext.get("column_mappings") or {}).items():
            print(f"    {role:16s} → {mapping}")
        print("\n  numeric_params:")
        for k, v in (ext.get("numeric_params") or {}).items():
            if v is not None:
                print(f"    {k:20s} = {v}")
        if result.validation_questions:
            print("\n  extraction_questions:")
            for q in result.validation_questions:
                print(f"    • {q}")

        print("\n" + "-" * 70)
        print("NODE RESULTS")
        print("-" * 70)
        for node in result.artifacts.get("criteria", []):
            status = "✓ PASS" if node.get("pass") else "✗ FAIL"
            print(f"  [{status}] {node['node']}")
            for k, v in node.get("details", {}).items():
                print(f"           {k}: {v}")

        print("\n" + "-" * 70)
        print("FINAL RESULT")
        print("-" * 70)
        print(f"  recommendation : {result.recommendation}")
        print(f"  status         : {result.status}")
        print(f"  confidence     : {result.confidence:.3f}")
        print(f"  latency_ms     : {result.latency_ms}")
        print(f"  missing_inputs : {result.missing_inputs}")
        if result.flags:
            print("\n  flags:")
            for flag in result.flags:
                print(f"    [{flag['code']}] {flag['message']}")
        print("\n  summary:")
        print(f"    {result.summary}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
