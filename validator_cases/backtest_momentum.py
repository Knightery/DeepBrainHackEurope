"""
Time-series momentum backtest on SPY (2010-2024).

Strategy (Moskowitz, Ooi & Pedersen 2012):
  - Signal: 12-1 month return (252-day lookback, skip most recent 21 days)
  - Long (+1) when signal > 0, short (-1) when signal < 0
  - Daily rebalancing; 1bp cost per round-trip side
  - Pure rule-based: no ML, no look-ahead, parameters fixed before test period

Run: python validator_cases/backtest_momentum.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKER = "SPY"
START = "2005-01-01"   # extra history for signal warm-up
END = "2024-12-31"
TEST_START = "2010-01-01"   # OOS evaluation window
COST_BPS = 1.0

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)


def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = raw[["Close", "Volume"]].copy()
    df.columns = ["close", "volume"]
    df.index.name = "date"
    return df


def build_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df["close"].pct_change()

    # 12-1 month momentum: return from 252 days ago to 21 days ago, known at t-1
    past_ret = df["close"].pct_change(252 - 21).shift(21)
    df["signal"] = past_ret.apply(lambda x: 1 if x > 0 else -1)

    df["fwd_ret"] = df["ret"].shift(-1)
    return df.dropna(subset=["signal", "fwd_ret"])


def run_backtest(df: pd.DataFrame, test_start: str) -> tuple[dict, pd.DataFrame]:
    test = df[df.index >= test_start].copy()

    cost = COST_BPS / 10_000
    # cost charged when signal flips (long-to-short or vice versa = 2 sides)
    test["cost"] = test["signal"].diff().abs().fillna(0) * cost
    test["strat_ret"] = test["signal"] * test["fwd_ret"] - test["cost"]
    test["buy_hold_ret"] = test["fwd_ret"]

    test["equity_strat"] = (1 + test["strat_ret"]).cumprod()
    test["equity_bh"] = (1 + test["buy_hold_ret"]).cumprod()

    n_years = len(test) / 252
    strat_ann = test["equity_strat"].iloc[-1] ** (1 / n_years) - 1
    bh_ann = test["equity_bh"].iloc[-1] ** (1 / n_years) - 1
    strat_vol = test["strat_ret"].std() * math.sqrt(252)
    bh_vol = test["buy_hold_ret"].std() * math.sqrt(252)

    def max_dd(eq: pd.Series) -> float:
        return float((eq / eq.cummax() - 1).min())

    n_flips = int(test["signal"].diff().abs().fillna(0).sum() // 2)

    metrics = {
        "ticker": TICKER,
        "signal_warmup_start": str(df.index[0].date()),
        "test_start": str(test.index[0].date()),
        "test_end": str(test.index[-1].date()),
        "n_test_days": len(test),
        "n_signal_flips": n_flips,
        "pct_days_long": round(float((test["signal"] == 1).mean()) * 100, 1),
        "strategy": {
            "cagr_pct": round(strat_ann * 100, 2),
            "ann_vol_pct": round(strat_vol * 100, 2),
            "sharpe": round(strat_ann / strat_vol if strat_vol else 0.0, 3),
            "max_drawdown_pct": round(max_dd(test["equity_strat"]) * 100, 2),
            "total_return_pct": round((test["equity_strat"].iloc[-1] - 1) * 100, 2),
        },
        "buy_and_hold": {
            "cagr_pct": round(bh_ann * 100, 2),
            "ann_vol_pct": round(bh_vol * 100, 2),
            "sharpe": round(bh_ann / bh_vol if bh_vol else 0.0, 3),
            "max_drawdown_pct": round(max_dd(test["equity_bh"]) * 100, 2),
            "total_return_pct": round((test["equity_bh"].iloc[-1] - 1) * 100, 2),
        },
        "cost_bps_per_side": COST_BPS,
    }

    return metrics, test


def main() -> None:
    print(f"Downloading {TICKER} {START} to {END}...")
    raw = fetch(TICKER, START, END)
    print(f"  {len(raw)} trading days fetched")

    df = build_signal(raw)
    print(f"  {len(df)} rows after signal construction")

    metrics, test_df = run_backtest(df, TEST_START)

    print("\n-- Backtest Results (OOS: 2010-2024) -------------------------")
    print(f"  Test  : {metrics['test_start']} to {metrics['test_end']}  ({metrics['n_test_days']} days)")
    print(f"  Signal flips: {metrics['n_signal_flips']}  |  % days long: {metrics['pct_days_long']}%")
    print()
    w = 20
    print(f"  {'Metric':{w}}  {'Strategy':>12}  {'Buy & Hold':>12}")
    print(f"  {'-'*w}  {'----------':>12}  {'----------':>12}")
    for label, k in [
        ("CAGR (%)",       "cagr_pct"),
        ("Ann. Vol (%)",   "ann_vol_pct"),
        ("Sharpe",         "sharpe"),
        ("Max DD (%)",     "max_drawdown_pct"),
        ("Total Ret (%)",  "total_return_pct"),
    ]:
        sv = metrics["strategy"][k]
        bv = metrics["buy_and_hold"][k]
        print(f"  {label:{w}}  {sv:>12.2f}  {bv:>12.2f}")
    print("--------------------------------------------------------------")

    metrics_path = OUT_DIR / "backtest_results.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics  -> {metrics_path}")

    equity_path = OUT_DIR / "backtest_equity.csv"
    test_df[["equity_strat", "equity_bh", "signal"]].to_csv(equity_path)
    print(f"Saved equity   -> {equity_path}")


if __name__ == "__main__":
    main()
