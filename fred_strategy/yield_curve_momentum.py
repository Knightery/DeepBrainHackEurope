"""
Yield Curve Regime Momentum Strategy
=====================================
Combines two signals:
  1. Price momentum  - equity is above its 200-day SMA (trend filter)
  2. Macro regime    - 10Y-2Y Treasury yield spread (from FRED) is above -0.10%
                       (avoid holding equities during deep yield-curve inversions)

External data source: FRED (Federal Reserve Bank of St. Louis)
  URL: https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y
  No API key required. Returns daily observations of the 10Y-2Y spread.

Thesis:
  The yield curve inversion (10Y < 2Y) historically precedes recessions by
  6–18 months. Combining an inversion filter with a 200-day trend filter should
  reduce equity exposure during the worst drawdown windows while retaining
  most of the upside during expansion phases.

Usage:
  python yield_curve_momentum.py
  or upload as a strategy file to the Quant Pitch Evaluator.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ── Parameters ────────────────────────────────────────────────────────────────
TICKER = "SPY"
BENCHMARK_TICKER = "SPY"          # buy-and-hold comparison
START_DATE = "2005-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

TREND_WINDOW = 200                # days for price moving average
YIELD_SPREAD_THRESHOLD = -0.10    # percent; below this = inversion caution zone
YIELD_SPREAD_SMOOTHING = 20       # business days to smooth noisy daily spread
TRANSACTION_COST = 0.0005         # 0.05% per trade (one-way)

FRED_SERIES = "T10Y2Y"            # 10-Year minus 2-Year Treasury Constant Maturity Rate
FRED_URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_SERIES}"


# ── 1. Fetch price data ───────────────────────────────────────────────────────
def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No price data returned for {ticker}")
    df = raw[["Close"]].copy()
    df.columns = ["close"]
    df.index = pd.to_datetime(df.index)
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values(0)
    df.sort_index(inplace=True)
    return df


# ── 2. Fetch FRED yield curve data ────────────────────────────────────────────
def fetch_yield_curve(start: str, end: str) -> pd.Series:
    """Download 10Y-2Y spread from FRED as a pandas Series (percent)."""
    try:
        resp = requests.get(FRED_URL, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), parse_dates=["observation_date"], index_col="observation_date")
        df.columns = ["spread"]
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
        df.sort_index(inplace=True)
        df = df.loc[start:end]
        spread = df["spread"].dropna()
        return spread
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch FRED yield curve data ({FRED_SERIES}): {exc}\n"
            "Check network access or try again later."
        ) from exc


# ── 3. Build signal ───────────────────────────────────────────────────────────
def build_signals(prices: pd.DataFrame, spread: pd.Series) -> pd.DataFrame:
    df = prices.copy()

    # Price momentum: above 200-day SMA
    df["sma200"] = df["close"].rolling(TREND_WINDOW).mean()
    df["above_trend"] = (df["close"] > df["sma200"]).astype(int)

    # Yield curve regime: forward-fill FRED data (published on business days),
    # then smooth to reduce whipsaw around the threshold
    spread_ff = spread.reindex(df.index, method="ffill")
    df["spread"] = spread_ff
    df["spread_smooth"] = spread_ff.rolling(YIELD_SPREAD_SMOOTHING).mean()
    df["curve_ok"] = (df["spread_smooth"] > YIELD_SPREAD_THRESHOLD).astype(int)

    # Combined signal: long only when both filters pass
    df["signal"] = df["above_trend"] * df["curve_ok"]

    return df


# ── 4. Backtest ───────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["daily_ret"] = df["close"].pct_change()

    # Signal is known at close of day t → trade opens at close of day t
    # (assumes liquid ETF, end-of-day execution)
    df["position"] = df["signal"].shift(1).fillna(0)

    # Transaction costs on position changes
    df["trades"] = df["position"].diff().abs().fillna(0)
    df["strat_ret"] = df["position"] * df["daily_ret"] - df["trades"] * TRANSACTION_COST

    # Benchmark: buy and hold
    df["bm_ret"] = df["daily_ret"].fillna(0)

    # Drop warm-up period (need 200 days to form SMA + smoothing buffer)
    warmup = TREND_WINDOW + YIELD_SPREAD_SMOOTHING + 5
    df = df.iloc[warmup:].copy()

    # Equity curves
    df["equity"] = (1 + df["strat_ret"]).cumprod()
    df["bm_equity"] = (1 + df["bm_ret"]).cumprod()

    n_years = len(df) / 252.0

    # ── Core metrics ──────────────────────────────────────────────────────────
    total_return = float(df["equity"].iloc[-1] - 1)
    bm_total_return = float(df["bm_equity"].iloc[-1] - 1)

    cagr = float((df["equity"].iloc[-1]) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    bm_cagr = float((df["bm_equity"].iloc[-1]) ** (1 / n_years) - 1) if n_years > 0 else 0.0

    vol = float(df["strat_ret"].std() * np.sqrt(252))
    bm_vol = float(df["bm_ret"].std() * np.sqrt(252))

    rf = 0.04  # approximate risk-free rate
    sharpe = float((df["strat_ret"].mean() * 252 - rf) / (vol + 1e-10))
    sortino_neg = df["strat_ret"][df["strat_ret"] < 0].std() * np.sqrt(252)
    sortino = float((df["strat_ret"].mean() * 252 - rf) / (sortino_neg + 1e-10))

    # Max drawdown
    rolling_max = df["equity"].cummax()
    drawdown = df["equity"] / rolling_max - 1
    max_dd = float(drawdown.min())

    bm_rolling_max = df["bm_equity"].cummax()
    bm_drawdown = df["bm_equity"] / bm_rolling_max - 1
    bm_max_dd = float(bm_drawdown.min())

    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0

    # Trade stats
    trade_mask = df["trades"] > 0
    n_trades = int(trade_mask.sum())

    trade_returns: list[float] = []
    in_trade = False
    entry_eq = 1.0
    for _, row in df.iterrows():
        if row["position"] == 1 and not in_trade:
            in_trade = True
            entry_eq = row["equity"]
        elif row["position"] == 0 and in_trade:
            in_trade = False
            trade_returns.append(float(row["equity"] / entry_eq - 1))

    win_rate = float(np.mean([r > 0 for r in trade_returns])) if trade_returns else 0.0
    wins = sum(r for r in trade_returns if r > 0)
    losses = abs(sum(r for r in trade_returns if r < 0))
    profit_factor = float(wins / losses) if losses > 0 else float("inf")

    # Alpha / beta vs benchmark
    cov_matrix = np.cov(df["strat_ret"].fillna(0), df["bm_ret"].fillna(0))
    beta = float(cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-10))
    alpha = float(cagr - (rf + beta * (bm_cagr - rf)))

    # Up/down capture
    up_days = df[df["bm_ret"] > 0]
    down_days = df[df["bm_ret"] < 0]
    up_capture = (
        float(up_days["strat_ret"].mean() / up_days["bm_ret"].mean())
        if len(up_days) > 0 and up_days["bm_ret"].mean() != 0 else 0.0
    )
    down_capture = (
        float(down_days["strat_ret"].mean() / down_days["bm_ret"].mean())
        if len(down_days) > 0 and down_days["bm_ret"].mean() != 0 else 0.0
    )

    # Time in market
    time_in_market = float(df["position"].mean())

    metrics = {
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6),
        "benchmark_total_return": round(bm_total_return, 6),
        "benchmark_cagr": round(bm_cagr, 6),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown": round(max_dd, 6),
        "benchmark_max_drawdown": round(bm_max_dd, 6),
        "calmar_ratio": round(calmar, 4),
        "alpha": round(alpha, 6),
        "beta": round(beta, 4),
        "volatility": round(vol, 6),
        "benchmark_volatility": round(bm_vol, 6),
        "win_rate": round(win_rate, 4),
        "total_trades": n_trades,
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 9999.0,
        "up_capture": round(up_capture, 4),
        "down_capture": round(down_capture, 4),
        "time_in_market": round(time_in_market, 4),
    }
    return metrics, df


# ── 5. Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Fetching price data for {TICKER} ({START_DATE} → {END_DATE})...")
    prices = fetch_prices(TICKER, START_DATE, END_DATE)
    print(f"  {len(prices)} trading days loaded.")

    print(f"Fetching FRED yield curve data (series: {FRED_SERIES})...")
    spread = fetch_yield_curve(START_DATE, END_DATE)
    print(f"  {len(spread)} FRED observations loaded.")
    print(f"  Latest spread: {spread.iloc[-1]:.2f}% (as of {spread.index[-1].date()})")

    print("Building signals...")
    df = build_signals(prices, spread)

    # Summary of signal composition
    pct_trend_long = df["above_trend"].mean() * 100
    pct_curve_ok = df["curve_ok"].mean() * 100
    pct_in_market = df["signal"].mean() * 100
    print(f"  Days above 200-SMA:          {pct_trend_long:.1f}%")
    print(f"  Days curve NOT inverted:     {pct_curve_ok:.1f}%")
    print(f"  Days in market (both pass):  {pct_in_market:.1f}%")

    print("Running backtest...")
    metrics, result_df = run_backtest(df)

    print("\n" + "=" * 55)
    print("  YIELD CURVE REGIME MOMENTUM — BACKTEST RESULTS")
    print("=" * 55)
    print(f"  Ticker:           {TICKER}")
    print(f"  Period:           {result_df.index[0].date()} → {result_df.index[-1].date()}")
    print(f"  External source:  FRED {FRED_SERIES} (10Y-2Y spread)")
    print("-" * 55)
    print(f"  CAGR (strategy):  {metrics['cagr']:.2%}")
    print(f"  CAGR (B&H):       {metrics['benchmark_cagr']:.2%}")
    print(f"  Total Return:     {metrics['total_return']:.2%}")
    print(f"  Sharpe:           {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino:          {metrics['sortino_ratio']:.3f}")
    print(f"  Max Drawdown:     {metrics['max_drawdown']:.2%}")
    print(f"  B&H Max DD:       {metrics['benchmark_max_drawdown']:.2%}")
    print(f"  Calmar:           {metrics['calmar_ratio']:.3f}")
    print(f"  Alpha:            {metrics['alpha']:.4f}")
    print(f"  Beta:             {metrics['beta']:.4f}")
    print(f"  Win Rate:         {metrics['win_rate']:.1%}")
    print(f"  Trades:           {metrics['total_trades']}")
    print(f"  Profit Factor:    {metrics['profit_factor']:.2f}")
    print(f"  Up Capture:       {metrics['up_capture']:.2f}")
    print(f"  Down Capture:     {metrics['down_capture']:.2f}")
    print(f"  Time in Market:   {metrics['time_in_market']:.1%}")
    print("=" * 55)

    return metrics


if __name__ == "__main__":
    main()
