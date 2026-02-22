"""
backtest_agent.py
-----------------
Claude-powered backtest creator agent.

Given a user's strategy script(s) and data file(s), Claude generates a
standardized runner that computes all strategy_scorer.Strategy metrics,
executes it in a temp directory, and reviews the results.

3-attempt loop per call:
  Phase 1  CREATE/FIX  Claude writes (or rewrites) the runner script
  Phase 2  RUN         subprocess.run the generated script
  Phase 3  REVIEW      Claude validates output & maps to Strategy fields

Termination statuses:
  "success"               All required fields extracted, Strategy validated.
  "agent_fault"           3 attempts exhausted; bugs appear self-introduced.
  "user_action_required"  Claude determined the user must fix or provide something.

Usage (standalone test):
  python backtest_agent.py strategy.py prices.csv AAPL
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKTEST_TIMEOUT_SECONDS = int(os.getenv("BACKTEST_TIMEOUT_SECONDS", "1200"))
BACKTEST_MAX_ATTEMPTS = int(os.getenv("BACKTEST_MAX_ATTEMPTS", "3"))
_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5-20251101")

# All fields that the generated backtest script MUST emit in its JSON output.
# "name" and "ticker" are injected by us from pitch context, not computed.
REQUIRED_OUTPUT_FIELDS = {
    "backtest_start",
    "backtest_end",
    "cagr",
    "total_return",
    "volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "max_drawdown_duration",
    "total_trades",
    "win_rate",
    "avg_win",
    "avg_loss",
    "profit_factor",
    "expectancy",
    "benchmark_cagr",
    "benchmark_max_drawdown",
    "benchmark_total_return",
    "alpha",
    "information_ratio",
    "excess_return",
    "up_capture",
    "down_capture",
}

# JSON schema description injected into the Claude system prompt.
REQUIRED_OUTPUT_SCHEMA = """
The generated script MUST print a single JSON object to stdout as its LAST output.
All other debug/progress output must go to stderr (print(..., file=sys.stderr)).

Required JSON fields:
{
  "backtest_start":          "YYYY-MM-DD",   // first date of backtest period
  "backtest_end":            "YYYY-MM-DD",   // last date of backtest period
  "cagr":                    float,          // annualised compound return, e.g. 0.15 = 15%
  "total_return":            float,          // total return over period, e.g. 0.45
  "volatility":              float,          // annualised std dev of daily returns
  "sharpe_ratio":            float,
  "sortino_ratio":           float,
  "calmar_ratio":            float,          // cagr / abs(max_drawdown)
  "max_drawdown":            float,          // NEGATIVE, e.g. -0.22
  "max_drawdown_duration":   int,            // longest days below prior equity peak
  "total_trades":            int,
  "win_rate":                float,          // fraction [0,1]
  "avg_win":                 float,          // POSITIVE average return of winning trades
  "avg_loss":                float,          // NEGATIVE average return of losing trades
  "profit_factor":           float,          // abs(sum wins) / abs(sum losses), >= 0
  "expectancy":              float,          // win_rate*avg_win + (1-win_rate)*avg_loss
  "benchmark_cagr":          float,          // buy-and-hold CAGR on same ticker same period
  "benchmark_max_drawdown":  float,          // buy-and-hold max drawdown, NEGATIVE
  "benchmark_total_return":  float,          // buy-and-hold total return
  "alpha":                   float,          // Jensen alpha (regression intercept * 252)
  "information_ratio":       float,          // mean(excess_daily) / std(excess_daily) * sqrt(252)
  "excess_return":           float,          // cagr - benchmark_cagr  (must match within 0.5%)
  "up_capture":              float,          // strategy_up_mean / benchmark_up_mean
  "down_capture":            float          // strategy_down_mean / benchmark_down_mean
}
""".strip()

# ---------------------------------------------------------------------------
# Alpaca benchmark fetch snippet â€” injected verbatim into every generated script.
# ---------------------------------------------------------------------------
ALPACA_BENCHMARK_SNIPPET = '''
def _fetch_benchmark_alpaca(
    ticker: str,
    start: str,
    end: str,
    timeframe: str = "1Day",
    adjustment: str = "all",
    feed: str = "sip",
    limit: int = 10000,
    sort: str = "asc",
) -> "pd.DataFrame":
    """
    Fetch OHLCV prices from Alpaca Markets.
    Supports daily or intraday bars depending on `timeframe`.
    Returns a DataFrame sorted by timestamp with lowercase OHLCV columns and
    common strategy aliases (Open/High/Low/Close/Volume/Adj Close).
    Raises RuntimeError on any failure; no silent fallbacks.
    """
    import os, json, urllib.request, urllib.parse, urllib.error, re
    api_key = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_API_SECRET", "")
    if not api_key or not api_secret:
        raise RuntimeError("ALPACA_API_KEY or ALPACA_API_SECRET environment variable is not set.")

    timeframe = str(timeframe or "1Day").strip()
    if not re.match(r"^[1-9][0-9]*(Min|Hour|Day|Week|Month)$", timeframe):
        raise RuntimeError(
            f"Invalid Alpaca timeframe '{timeframe}'. "
            "Expected values like 1Min, 5Min, 15Min, 1Hour, 1Day."
        )
    adjustment = str(adjustment or "all").strip().lower()
    if adjustment not in {"raw", "split", "dividend", "all"}:
        raise RuntimeError(f"Invalid Alpaca adjustment '{adjustment}'.")
    feed = str(feed or "sip").strip().lower()
    if feed not in {"sip", "iex", "otc"}:
        raise RuntimeError(f"Invalid Alpaca feed '{feed}'.")
    sort = str(sort or "asc").strip().lower()
    if sort not in {"asc", "desc"}:
        raise RuntimeError(f"Invalid Alpaca sort '{sort}'.")
    try:
        limit = max(1, min(10000, int(limit)))
    except Exception as exc:
        raise RuntimeError(f"Invalid Alpaca limit '{limit}'.") from exc

    base_url = "https://data.alpaca.markets/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
        "Accept": "application/json",
    }
    bars = []
    feed_candidates = [feed] if feed != "sip" else ["sip", "iex"]
    last_http_error = None
    chosen_feed = None
    for feed_candidate in feed_candidates:
        bars = []
        page_token = None
        request_failed = False
        while True:
            params = {
                "symbols": ticker,
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "adjustment": adjustment,
                "feed": feed_candidate,
                "limit": str(limit),
                "sort": sort,
            }
            if page_token:
                params["page_token"] = page_token
            url = base_url + "?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                request_failed = True
                last_http_error = (exc.code, exc.reason, feed_candidate)
                break
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Alpaca API network error: {exc.reason}") from exc

            symbol_bars = data.get("bars", {}).get(ticker.upper(), [])
            bars.extend(symbol_bars)
            page_token = data.get("next_page_token")
            if not page_token:
                break

        if not request_failed and bars:
            chosen_feed = feed_candidate
            break

    if not bars:
        if last_http_error is not None:
            code, reason, failing_feed = last_http_error
            raise RuntimeError(
                f"Alpaca API HTTP error {code}: {reason} (feed={failing_feed}, tried={feed_candidates})"
            )
        raise RuntimeError(
            f"Alpaca returned 0 bars for {ticker} between {start} and {end}. "
            "Check that the ticker is valid and the date range covers trading days."
        )

    import pandas as pd
    df = pd.DataFrame(bars)
    # Alpaca fields: t(timestamp), o(open), h(high), l(low), c(close), v(volume), vw(vwap), n(trade_count)
    df["date"] = pd.to_datetime(df.get("t"), errors="coerce", utc=True).dt.tz_localize(None)
    df["open"] = pd.to_numeric(df.get("o"), errors="coerce")
    df["high"] = pd.to_numeric(df.get("h"), errors="coerce")
    df["low"] = pd.to_numeric(df.get("l"), errors="coerce")
    df["close"] = pd.to_numeric(df.get("c"), errors="coerce")
    df["volume"] = pd.to_numeric(df.get("v"), errors="coerce")
    df["vwap"] = pd.to_numeric(df.get("vw"), errors="coerce")
    df["trade_count"] = pd.to_numeric(df.get("n"), errors="coerce")
    df = (
        df[["date", "open", "high", "low", "close", "volume", "vwap", "trade_count"]]
        .dropna(subset=["date", "close"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    if df.empty:
        raise RuntimeError(f"Alpaca bars for {ticker} contained no parseable close prices.")

    # Common aliases used by many notebooks/scripts.
    df["Open"] = df["open"]
    df["High"] = df["high"]
    df["Low"] = df["low"]
    df["Close"] = df["close"]
    df["Volume"] = df["volume"]
    df["Adj Close"] = df["close"]
    df["alpaca_feed"] = chosen_feed or feed
    return df
'''.strip()

SOLANA_BENCHMARK_SNIPPET = '''
def _fetch_benchmark_solana_xstocks(
    ticker: str,
    start: str,
    end: str,
    timeframe: str = "1D",
    currency: str = "usd",
) -> "pd.DataFrame":
    """
    Fetch OHLCV bars for Solana xStocks from Birdeye.
    `ticker` can be an xStock symbol (e.g., TSLAx) or a Solana mint address.
    Raises RuntimeError on failure.
    """
    import os, json, urllib.request, urllib.parse, urllib.error, datetime, re
    import pandas as pd

    api_key = os.getenv("BIRDEYE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("BIRDEYE_API_KEY environment variable is not set.")

    mint_map = {
        "AAPLX": "XsbEhLAtcf6HdfpFZ5xEMdqW8nfAvcsP5bdudRLJzJp",
        "ABBVX": "XswbinNKyPmzTa5CskMbCPvMW6G5CMnZXZEeQSSQoie",
        "ABTX": "XsHtf5RpxsQ7jeJ9ivNewouZKJHbPxhPoEy6yYvULr7",
        "ACNX": "Xs5UJzmCRQ8DWZjskExdSQDnbE6iLkRu2jjrRAB1JSU",
        "AMBRX": "XsaQTCgebC2KPbf27KUhdv5JFvHhQ4GDAPURwrEhAzb",
        "AMDX": "XsXcJ6GZ9kVnjqGsjBnktRcuwMBmvKWh8S93RefZ1rF",
        "AMZNX": "Xs3eBt7uRfJX8QUs4suhyU8p2M6DoUDrJyWBa8LLZsg",
        "APPX": "XsPdAVBi8Zc1xvv53k4JcMrQaEDTgkGqKYeh7AYgPHV",
        "AVGOX": "XsgSaSvNSqLTtFuyWPBhK9196Xb9Bbdyjj4fH3cPJGo",
        "AZNX": "Xs3ZFkPYT2BN7qBMqf1j1bfTeTm1rFzEFSsQ1z3wAKU",
        "BACX": "XswsQk4duEQmCbGzfqUUWYmi7pV7xpJ9eEmLHXCaEQP",
        "BMNRX": "XsrBCwaH8c46xiqXBChzobgufRKxQxAWUWbndgBNzFn",
        "BRK.BX": "Xs6B6zawENwAbWVi7w92rjazLuAr5Az59qgWKcNb45x",
        "BTBTX": "XsPLBFy59Q3hY59KLAJur8QyvziMF4xUxGTxXqXE7cT",
        "BTGOX": "XsvHMmbDcd14DHHW16PkxPGW7ks77ehxUv1E9Zmxgj4",
        "CMCSAX": "XsvKCaNsxg2GN8jjUmq71qukMJr7Q1c5R2Mk9P8kcS8",
        "COINX": "Xs7ZdzSHLU9ftNJsii5fCeJhoRWSC32SQGzGQtePxNu",
        "COPXX": "XsybfiKkD4UmjkAGT2uR8X2sq9AWFtvGJM2KTffoALZ",
        "CRCLX": "XsueG8BtpquVJX9LVLLEGuViXUungE6WmK5YZ3p3bd1",
        "CRMX": "XsczbcQ3zfcgAEt9qHQES8pxKAVG5rujPSHQEXi4kaN",
        "CRWDX": "Xs7xXqkcK7K8urEqGg52SECi79dRp2cEKKuYjUePYDw",
        "CSCOX": "Xsr3pdLQyXvDJBFgpR5nexCEZwXvigb8wbPYp4YoNFf",
        "CVXX": "XsNNMt7WTNA2sV3jrb1NNfNgapxRF5i4i6GcnTRRHts",
        "DFDVX": "Xs2yquAgsHByNzx68WJC55WHjHBvG9JsMB7CWjTLyPy",
        "DHRX": "Xseo8tgCZfkHxWS9xbFYeKFyMSbWEvZGFV1Gh53GtCV",
        "GLDX": "Xsv9hRk1z5ystj9MhnA7Lq4vjSsLwzL2nxrwmwtD3re",
        "GMEX": "Xsf9mBktVB9BSU5kf4nHxPq5hCBJ2j2ui3ecFGxPRGc",
        "GOOGLX": "XsCPL9dNWBMvFtTmwcCA5v3xWPSMEBCszbQdiLLq6aN",
        "GSX": "XsgaUyp4jd1fNBCxgtTKkW64xnnhQcvgaxzsbAq5ZD1",
        "HDX": "XszjVtyhowGjSC5odCqBpW1CtXXwXjYokymrk7fGKD3",
        "HONX": "XsRbLZthfABAPAfumWNEJhPyiKDW6TvDVeAeW7oKqA2",
        "HOODX": "XsvNBAYkrDRNhA7wPHQfX3ZUXZyZLdnCQDfHZ56bzpg",
        "IBMX": "XspwhyYPdWVM8XBHZnpS9hgyag9MKjLRyE3tVfmCbSr",
        "IEMGX": "XsFnZawJdLdXfBSEt5Vw29K5vdBiHotdPLjUPafpfHs",
        "IJRX": "XsyZcb97BzETAqi9BoP2C9D196MiMNBisGMVNje2Thz",
        "INTCX": "XshPgPdXFRWB8tP1j82rebb2Q9rPgGX37RuqzohmArM",
        "IWMX": "XsbELVbLGBkn7xfMfyYuUipKGt1iRUc2B7pYRvFTFu3",
        "JNJX": "XsGVi5eo1Dh2zUpic4qACcjuWGjNv8GCt3dm5XcX6Dn",
        "JPMX": "XsMAqkcKsUewDrzVkait4e5u4y8REgtyS7jWgCpLV2C",
        "KOX": "XsaBXg8dU5cPM6ehmVctMkVqoiRG2ZjMo1cyBJ3AykQ",
        "KRAQX": "XsAiRejKuvLAdq9KtedrMSrabz7SWdzKoVK6Qgac1Ki",
        "LINX": "XsSr8anD1hkvNMu8XQiVcmiaTP7XGvYu7Q58LdmtE8Z",
        "LLYX": "Xsnuv4omNoHozR6EEW5mXkw8Nrny5rB3jVfLqi6gKMH",
        "MAX": "XsApJFV9MAktqnAc6jqzsHVujxkGm9xcSUffaBoYLKC",
        "MCDX": "XsqE9cRRpzxcGKDXj1BJ7Xmg4GRhZoyY1KpmGSxAWT2",
        "MDTX": "XsDgw22qRLTv5Uwuzn6T63cW69exG41T6gwQhEK22u2",
        "METAX": "Xsa62P5mvPszXL1krVUnU5ar38bBSVcWAB6fmPCo5Zu",
        "MRKX": "XsnQnU7AdbRZYe2akqqpibDdXjkieGFfSkbkjX1Sd1X",
        "MRVLX": "XsuxRGDzbLjnJ72v74b7p9VY6N66uYgTCyfwwRjVCJA",
        "MSFTX": "XspzcW1PRtgf6Wj92HCiZdjzKCyFekVD8P5Ueh3dRMX",
        "MSTRX": "XsP7xzNPvEHS1m6qfanPUGjNmdnmsLKEoNAnHjdxxyZ",
        "NFLXX": "XsEH7wWfJJu2ZT3UCFeVfALnVA6CP5ur7Ee11KmzVpL",
        "NVDAX": "Xsc9qvGR1efVDFGLrVsmkzv3qi45LTBjeUKSPmx9qEh",
        "NVOX": "XsfAzPzYrYjd4Dpa9BU3cusBsvWfVB9gBcyGC87S57n",
        "OPENX": "XsGtpmjhmC8kyjVSWL4VicGu36ceq9u55PTgF8bhGv6",
        "ORCLX": "XsjFwUPiLofddX5cWFHW35GCbXcSu1BCUGfxoQAQjeL",
        "PALLX": "XsTTtPA5V19YwHKDv4xeVXNM6kdsQNJvg3MyWkRUckt",
        "PEPX": "Xsv99frTRUeornyvCfvhnDesQDWuvns1M852Pez91vF",
        "PFEX": "XsAtbqkAP1HJxy7hFDeq7ok6yM43DQ9mQ1Rh861X8rw",
        "PGX": "XsYdjDjNUygZ7yGKfQaB6TxLh2gC6RRjzLtLAGJrhzV",
        "PLTRX": "XsoBhf2ufR8fTyNSjqfU71DYGaE6Z3SUGAidpzriAA4",
        "PMX": "Xsba6tUnSjDae2VcopDB6FGGDaxRrewFCDa5hKn5vT3",
        "PPLTX": "Xst6eFD4YT6sz9RLMysN9SyvaZWtraSdVJQGu5ZkAme",
        "QQQX": "Xs8S1uUs1zvS2p7iwtsG3b6fkhpvmwz4GYU3gWAmWHZ",
        "SCHFX": "XsWAnFM77x6YvpdaZoos79R12o4Yj4r7EVkaTWddzhU",
        "SLVX": "XsxAd6okt8y1RRK6gNg7iJaqiWNiq5Md5EDf3ZrF2dm",
        "SPYX": "XsoCS1TfEyfFhfvj8EtZ528L3CaKBDBRqRapnBbDF2W",
        "STRCX": "Xs78JED6PFZxWc2wCEPspZW9kL3Se5J7L5TChKgsidH",
        "TBLLX": "XsqBC5tcVQLYt8wqGCHRnAUUecbRYXoJCReD6w7QEKp",
        "TMOX": "Xs8drBWy3Sd5QY3aifG9kt9KFs2K3PGZmx7jWrsrk57",
        "TONXX": "XscE4GUcsYhcyZu5ATiGUMmhxYa1D5fwbpJw4K6K4dp",
        "TQQQX": "XsjQP3iMAaQ3kQScQKthQpx9ALRbjKAjQtHg6TFomoc",
        "TSLAX": "XsDoVfqeBukxuZHWhdvWHBhgEHjGNst4MLodqsJHzoB",
        "UNHX": "XszvaiXGPwvk2nwb3o9C1CX4K6zH8sez11E6uyup6fe",
        "VTIX": "XsssYEQjzxBCFgvYFFNuhJFBeHNdLWYeUSP8F45cDr9",
        "VTX": "XsEdDDTcVGJU6nvdRdVnj53eKTrsCkvtrVfXGmUK68V",
        "VX": "XsqgsbXwWogGJsNcVZ3TyVouy2MbTkfCFhCGGGcQZ2p",
        "WMTX": "Xs151QeqTCiuKtinzfRATnUESM2xTU6V9Wy8Vy538ci",
        "XOMX": "XsaHND8sHyfMfsWPj6kSdd5VwvCayZvjYgKmmcNL5qh",
    }
    symbol = str(ticker or "").strip()
    if not symbol:
        raise RuntimeError("Ticker is required for Solana benchmark fetch.")

    mint = mint_map.get(symbol.upper())
    if mint is None:
        if re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", symbol):
            mint = symbol
        else:
            raise RuntimeError(
                f"Unknown xStock ticker '{symbol}'. Provide a Solana mint address or supported xStock symbol."
            )

    timeframe = str(timeframe or "1D").strip()
    valid_timeframes = {"1m", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "6H", "8H", "12H", "1D", "3D", "1W", "1M"}
    if timeframe not in valid_timeframes:
        raise RuntimeError(f"Invalid Birdeye timeframe '{timeframe}'.")
    currency = str(currency or "usd").strip().lower()
    if currency not in {"usd", "native"}:
        raise RuntimeError(f"Invalid Birdeye currency '{currency}'.")

    def _to_unix(value: str) -> int:
        text = str(value).strip()
        if re.fullmatch(r"\\d+", text):
            return int(text)
        dt = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return int(dt.timestamp())

    time_from = _to_unix(start)
    time_to = _to_unix(end)
    if time_to <= time_from:
        raise RuntimeError("Invalid time range for Solana fetch (end <= start).")

    query = urllib.parse.urlencode({
        "address": mint,
        "type": timeframe,
        "currency": currency,
        "time_from": str(time_from),
        "time_to": str(time_to),
    })
    url = f"https://public-api.birdeye.so/defi/v3/ohlcv?{query}"
    headers = {"X-API-KEY": api_key, "x-chain": "solana", "Accept": "application/json"}
    req = urllib.request.Request(url, headers=headers, method="GET")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Birdeye HTTP error {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Birdeye network error: {exc.reason}") from exc

    if not isinstance(payload, dict) or not payload.get("success"):
        raise RuntimeError(f"Birdeye returned failure payload: {payload}")
    data = payload.get("data", {}) if isinstance(payload.get("data"), dict) else {}
    items = data.get("items", []) if isinstance(data.get("items"), list) else []
    if not items:
        raise RuntimeError(f"Birdeye returned 0 bars for {symbol} ({mint}).")

    is_scaled = bool(data.get("is_scaled_ui_token", False))
    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        o = item.get("scaled_o") if is_scaled and item.get("scaled_o") is not None else item.get("o")
        h = item.get("scaled_h") if is_scaled and item.get("scaled_h") is not None else item.get("h")
        l = item.get("scaled_l") if is_scaled and item.get("scaled_l") is not None else item.get("l")
        c = item.get("scaled_c") if is_scaled and item.get("scaled_c") is not None else item.get("c")
        v = item.get("scaled_v") if is_scaled and item.get("scaled_v") is not None else item.get("v")
        ts = item.get("unix_time")
        rows.append({"unix_time": ts, "open": o, "high": h, "low": l, "close": c, "volume": v, "v_usd": item.get("v_usd")})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(pd.to_numeric(df.get("unix_time"), errors="coerce"), unit="s", utc=True).dt.tz_localize(None)
    for col in ("open", "high", "low", "close", "volume", "v_usd"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df = (
        df[["date", "open", "high", "low", "close", "volume", "v_usd"]]
        .dropna(subset=["date", "close"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    if df.empty:
        raise RuntimeError("Birdeye bars contained no parseable close prices.")
    df["Open"] = df["open"]
    df["High"] = df["high"]
    df["Low"] = df["low"]
    df["Close"] = df["close"]
    df["Volume"] = df["volume"]
    df["Adj Close"] = df["close"]
    df["solana_mint"] = mint
    return df
'''.strip()

# System prompt for the CREATE/FIX phase (Phase 1).
CREATE_SYSTEM_PROMPT = ("""
You are an expert quant Python developer. Your job is to produce a SINGLE,
self-contained Python script that:

1. Loads the user's strategy files and data files from the CURRENT WORKING DIRECTORY.
2. Runs the strategy and any machine-learning / signal logic it contains.
3. Fetches benchmark buy-and-hold data for the same ticker and period using the
   provider-appropriate API function(s) provided below (copy verbatim).
4. Computes ALL required output metrics (see schema below).
5. Prints the final JSON object to stdout as the very last thing.
   All other prints MUST use stderr.

Constraints:
- You may import ANY Python package the strategy requires. If a package is not available,
  install it at the top of the script using subprocess: e.g.
      import subprocess, sys
      subprocess.check_call([sys.executable, "-m", "pip", "install", "<package>"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  Do this before any other imports that may be missing.
- If the strategy file defines a function, call it. If it is a script, adapt it.
- NEVER require the user to upload price CSV files.
- If a strategy uses nsepy, yfinance, pandas_datareader, AlphaVantage, or other
  external market data clients, replace that logic with provider function bar fetching.
- If local price files referenced by the strategy are missing, do not fail fast or
  request user uploads by default; reconstruct equivalent price inputs from fetched bars.
- ALWAYS call the selected benchmark function at least once to build canonical strategy price data.
- Use fetched bars as the canonical internal dataset for strategy calculations.
- Persist that canonical dataset to "__internal_price_data.csv" in the working directory
  with at least date/open/high/low/close/volume columns.
- After a successful fetch, print one exact marker to stderr:
    ALPACA_FETCH_OK timeframe=<timeframe> rows=<row_count>
    OR
    SOLANA_FETCH_OK timeframe=<timeframe> rows=<row_count>
- You may call the selected benchmark function for both strategy inputs and benchmark inputs.
- The equity curve must be derived from the strategy's actual signals/positions.
- Trades must be extracted or inferred from the position changes.
- Provider selection rules:
    - Use Solana function when ticker looks like xStock (suffix "x"/"X") or is a Solana mint.
    - Otherwise use Alpaca function.
- For Alpaca benchmark call:
      bench_df = _fetch_benchmark_alpaca(
          ticker=ticker,
          start=backtest_start,
          end=backtest_end,
          timeframe=benchmark_timeframe,
          adjustment=benchmark_adjustment,
          feed=benchmark_feed,
          limit=benchmark_limit,
          sort="asc",
      )
- For Solana benchmark call:
      bench_df = _fetch_benchmark_solana_xstocks(
          ticker=ticker,
          start=backtest_start,
          end=backtest_end,
          timeframe=benchmark_timeframe,
          currency="usd",
      )
- Both functions raise RuntimeError on any failure (missing credentials, network
  error, no data). Do NOT catch or suppress these errors â€” let the script exit with
  a non-zero code so the issue is surfaced clearly.
- Infer benchmark API parameters from the strategy/data:
    1) timeframe: derive from bar cadence if possible
    2) for Alpaca use values like 1Min/5Min/15Min/1Hour/1Day
    3) for Solana use values like 1m/5m/15m/1H/4H/1D
    4) prefer daily bars when cadence is unclear
- If using intraday bars, compute annualization using an inferred bars-per-year factor
  consistent with the chosen timeframe (do not assume 252 daily bars in that case).
- If cadence cannot be inferred confidently, default to timeframe="1Day" and print that
  fallback decision to stderr.
- Column name aliases for price data (accept any of these):
    date column:   date, Date, timestamp, timestamp_utc, datetime, Datetime
    close column:  close, Close, adj_close, Adj Close, adj close, price, Price, last
    return column: daily_return, returns, return, pct_return, ret, Return
    portfolio value: portfolio_value, equity, value, nav, wealth, NAV

ALPACA FUNCTION TO COPY VERBATIM:
{alpaca_snippet}

SOLANA FUNCTION TO COPY VERBATIM:
{solana_snippet}

{schema}

OPTIONAL EXTRAS — strongly recommended for visualisation (wrap ENTIRELY in try/except; NEVER let them crash the script or affect the required JSON fields above):
After computing all required metrics, also emit two normalised equity curve series in the same JSON object:
  "equity_curve":    [["YYYY-MM-DD", float], ...],  // strategy portfolio value, normalised so first bar = 1.0
  "benchmark_curve": [["YYYY-MM-DD", float], ...],  // benchmark portfolio value, normalised so first bar = 1.0
One entry per trading bar. If the series would exceed 1000 points, downsample evenly so output stays ≤1000 entries.
Both series MUST start at exactly 1.0. Example emission code (adapt as needed):
  try:
      _eq = portfolio_value_series / portfolio_value_series.iloc[0]
      _bm = benchmark_value_series / benchmark_value_series.iloc[0]
      _step = max(1, len(_eq) // 1000)
      metrics["equity_curve"] = [[str(d)[:10], round(float(v), 6)] for d, v in zip(_eq.index[::_step], _eq.iloc[::_step])]
      metrics["benchmark_curve"] = [[str(d)[:10], round(float(v), 6)] for d, v in zip(_bm.index[::_step], _bm.iloc[::_step])]
  except Exception:
      pass  # curves are optional — never fail the script over them

CRITICAL: Output ONLY the Python script. No explanation, no markdown fences, no prose.
Just raw Python code starting with imports.
"""
    .strip()
    .format(
        alpaca_snippet=ALPACA_BENCHMARK_SNIPPET,
        solana_snippet=SOLANA_BENCHMARK_SNIPPET,
        schema=REQUIRED_OUTPUT_SCHEMA,
    )
)

# System prompt for the REVIEW phase (Phase 3).
REVIEW_SYSTEM_PROMPT = """
You are a quant backtest result validator. Your job is to review the stdout JSON
output of a generated backtest script and decide one of three outcomes.

Rules:
1. If all required fields are present and values are sensible (no Inf, no NaN,
   max_drawdown is negative, avg_loss is negative, win_rate in [0,1],
   total_trades >= 1, excess_return within 0.5% of cagr - benchmark_cagr):
   -> verdict = "success"

2. If fields are missing or values are corrupt/nonsensical due to a BUG in the
   generated script (not the user's strategy):
   -> verdict = "agent_fault"
   -> feedback = concise description of the bug to fix in the next attempt

3. If the user's strategy script itself is the problem (import error for a custom
   library, unsupported ticker/no Alpaca data, fundamentally broken logic that cannot
   be fixed by rewriting the runner even after Alpaca fallback):
   -> verdict = "user_action_required"
   -> message = clear, user-facing explanation of what they need to fix or provide

Respond ONLY with a JSON object:
{
  "verdict": "success" | "agent_fault" | "user_action_required",
  "feedback": "...",   // only for agent_fault â€” what to fix
  "message": "..."     // only for user_action_required â€” user-facing message
}
""".strip()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AttemptRecord:
    attempt: int
    phase_reached: str        # "create" | "run" | "review"
    script: str               # generated Python code
    stdout: str
    stderr: str
    returncode: int | None
    review_verdict: str       # "success" | "agent_fault" | "user_action_required" | "not_reached"
    review_feedback: str
    review_message: str
    latency_ms: int


@dataclass
class BacktestTermination:
    status: str               # "success" | "agent_fault" | "user_action_required"
    metrics: dict | None      # fully-populated dict matching REQUIRED_OUTPUT_FIELDS, or None
    message: str              # human-readable summary
    attempt_count: int
    attempt_history: list[AttemptRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Helper â€” get Anthropic client
# ---------------------------------------------------------------------------

def _get_anthropic_client():
    """Return an anthropic.Anthropic client, or raise ImportError / ValueError."""
    try:
        import anthropic  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for the backtest agent. "
            "Run: pip install anthropic>=0.50.0"
        ) from exc

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set. Backtest agent cannot run.")

    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Helper â€” file schema summary
# ---------------------------------------------------------------------------

def _file_schema_summary(filename: str, content: str, max_rows: int = 8) -> str:
    """Return a brief schema description of a CSV/TSV data file for Claude."""
    lines = content.splitlines()
    if not lines:
        return f"# {filename}: (empty)"

    preview_lines = lines[: max_rows + 1]
    preview = "\n".join(preview_lines)
    total_rows = len(lines) - 1  # exclude header
    return (
        f"# {filename} â€” {total_rows} data rows\n"
        f"{preview}\n"
        f"{'...(truncated)' if len(lines) > max_rows + 1 else ''}"
    ).strip()


def _sanitize_staged_filename(raw_name: str, fallback_prefix: str) -> str:
    """
    Keep only a filesystem-safe leaf filename for temp staging.
    Prevents path traversal and absolute-path writes.
    """
    text = str(raw_name or "").strip().replace("\x00", "")
    leaf = text.replace("\\", "/").split("/")[-1]
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", leaf).strip("._")
    if not safe:
        safe = fallback_prefix
    return safe[:180]


def _normalize_staged_files(files: list[tuple[str, str]], fallback_prefix: str) -> list[tuple[str, str]]:
    """
    Sanitize filenames and de-duplicate after normalization.
    Keeps filenames stable for both CREATE and RUN phases.
    """
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, (fname, content) in enumerate(files, start=1):
        base = _sanitize_staged_filename(fname, f"{fallback_prefix}_{index}")
        stem = Path(base).stem or f"{fallback_prefix}_{index}"
        suffix = Path(base).suffix
        candidate = base
        counter = 1
        while candidate.lower() in seen:
            candidate = f"{stem}_{counter}{suffix}"
            counter += 1
        seen.add(candidate.lower())
        normalized.append((candidate, content if isinstance(content, str) else str(content)))
    return normalized


def _is_solana_xstock_ticker(ticker: str) -> bool:
    symbol = str(ticker or "").strip()
    if not symbol:
        return False
    if symbol.upper().endswith("X"):
        return True
    return bool(re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", symbol))


# ---------------------------------------------------------------------------
# Helper â€” extract JSON from Claude response text
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """Extract the last JSON object from a text response."""
    # Try direct parse first
    stripped = text.strip()
    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Find last {...} block
    decoder = json.JSONDecoder()
    best: dict | None = None
    for i in range(len(stripped)):
        if stripped[i] == "{":
            try:
                obj, _ = decoder.raw_decode(stripped, i)
                if isinstance(obj, dict):
                    best = obj
            except json.JSONDecodeError:
                pass
    return best


# ---------------------------------------------------------------------------
# Helper â€” extract Python script from Claude response
# ---------------------------------------------------------------------------

def _extract_script(text: str) -> str:
    """
    Claude should return raw Python with no fences. But in case it adds
    markdown fences, strip them.
    Raises ValueError if the result is empty.
    """
    # Strip ```python ... ``` or ``` ... ```
    fenced = re.sub(r"^```(?:python)?\s*\n", "", text.strip(), flags=re.MULTILINE)
    fenced = re.sub(r"\n```\s*$", "", fenced.strip(), flags=re.MULTILINE)
    result = fenced.strip()
    if not result:
        raise ValueError("Claude returned an empty script response.")
    return result


# ---------------------------------------------------------------------------
# Helper â€” extract last JSON from stdout
# ---------------------------------------------------------------------------

def _extract_stdout_json(stdout: str) -> dict | None:
    """
    The generated script should print the JSON object as its last output line(s).
    Find and parse it.
    """
    # Try full stdout first
    result = _extract_json(stdout.strip())
    if result:
        return result

    # Try last non-empty line(s) progressively
    lines = [line for line in stdout.splitlines() if line.strip()]
    for start in range(len(lines) - 1, -1, -1):
        fragment = "\n".join(lines[start:])
        result = _extract_json(fragment)
        if result:
            return result
    return None


# ---------------------------------------------------------------------------
# Phase 1: CREATE / FIX
# ---------------------------------------------------------------------------

def _phase_create(
    client,
    strategy_files: list[tuple[str, str]],
    data_files: list[tuple[str, str]],
    pitch_context: dict,
    prior_feedback: str | None,
    prior_script: str | None,
    prior_stdout: str | None,
    prior_stderr: str | None,
) -> str:
    """Ask Claude to generate (or fix) the runner script. Returns Python source."""

    strategy_block = "\n\n".join(
        f"# === STRATEGY FILE: {fname} ===\n{content}"
        for fname, content in strategy_files
    )

    data_block = "\n\n".join(
        _file_schema_summary(fname, content)
        for fname, content in data_files
    )

    ticker = pitch_context.get("ticker", "UNKNOWN")
    name = pitch_context.get("name", "strategy")
    data_source_hint = "solana_xstocks_birdeye" if _is_solana_xstock_ticker(ticker) else "alpaca_stocks"

    context_block = textwrap.dedent(f"""\
        Strategy name: {name}
        Ticker: {ticker}
        Data source hint: {data_source_hint}
        Available files in working directory: {[f for f, _ in strategy_files + data_files]}
    """)

    if prior_feedback:
        fix_block = textwrap.dedent(f"""\
            PREVIOUS ATTEMPT FAILED. Fix the following issue and rewrite the complete script:
            FEEDBACK: {prior_feedback}

            PREVIOUS SCRIPT (for reference):
            {prior_script or "(none)"}

            PREVIOUS STDOUT:
            {(prior_stdout or "(empty)")[:2000]}

            PREVIOUS STDERR:
            {(prior_stderr or "(empty)")[:2000]}
        """)
    else:
        fix_block = ""

    user_message = "\n\n".join(filter(None, [
        "## Context",
        context_block,
        "## Strategy Files",
        strategy_block,
        "## Data File Schemas",
        data_block,
        fix_block if fix_block else None,
        "Generate the complete backtest runner script now.",
    ]))

    response = client.messages.create(
        model=_ANTHROPIC_MODEL,
        max_tokens=8192,
        system=CREATE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return _extract_script(response.content[0].text)


# ---------------------------------------------------------------------------
# Phase 2: RUN
# ---------------------------------------------------------------------------

def _phase_run(
    script: str,
    strategy_files: list[tuple[str, str]],
    data_files: list[tuple[str, str]],
    tmp_dir: Path,
) -> tuple[str, str, int]:
    """
    Write script + all files to tmp_dir, execute it.
    Returns (stdout, stderr, returncode).
    Timeouts are returned as ("TIMEOUT", ...) so the review phase can be specific.
    """
    tmp_root = tmp_dir.resolve()

    def _write_staged_file(filename: str, content: str) -> None:
        target = (tmp_root / filename).resolve()
        try:
            target.relative_to(tmp_root)
        except ValueError as exc:
            raise ValueError(f"Unsafe staged filename: {filename!r}") from exc
        target.write_text(content, encoding="utf-8")

    # Write generated runner
    runner_path = tmp_dir / "_backtest_runner.py"
    runner_path.write_text(script, encoding="utf-8")

    # Copy strategy files
    for fname, content in strategy_files:
        _write_staged_file(fname, content)

    # Copy data files
    for fname, content in data_files:
        _write_staged_file(fname, content)

    try:
        result = subprocess.run(
            [sys.executable, str(runner_path)],
            cwd=str(tmp_dir),
            capture_output=True,
            text=True,
            timeout=BACKTEST_TIMEOUT_SECONDS,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        def _decode(v: Any) -> str:
            if isinstance(v, bytes):
                return v.decode(errors="replace")
            return str(v) if v is not None else ""
        partial_stdout = _decode(exc.stdout)
        partial_stderr = _decode(exc.stderr)
        return partial_stdout, f"TIMEOUT: script exceeded {BACKTEST_TIMEOUT_SECONDS}s.\n{partial_stderr}", 1


# ---------------------------------------------------------------------------
# Phase 3: REVIEW
# ---------------------------------------------------------------------------

def _phase_review(
    client,
    stdout: str,
    stderr: str,
    returncode: int,
    script: str,
    ticker: str = "UNKNOWN",
) -> tuple[str, str, str]:
    """
    Ask Claude to validate the output.
    Returns (verdict, feedback, message).
    verdict: "success" | "agent_fault" | "user_action_required"
    """
    # Quick pre-check: if script crashed with returncode != 0
    if returncode != 0:
        # Let Claude decide if it's fixable or user issue
        failure_context = (
            f"The script returned exit code {returncode}.\n"
            f"STDERR:\n{stderr[:3000]}\n"
            f"STDOUT:\n{stdout[:1000]}"
        )
    else:
        failure_context = None

    expected_marker = "SOLANA_FETCH_OK" if _is_solana_xstock_ticker(ticker) else "ALPACA_FETCH_OK"
    if expected_marker not in (stderr or ""):
        return (
            "agent_fault",
            (
                "Runner did not confirm benchmark data fetch. "
                "Ensure it calls the provider-appropriate benchmark fetch function, "
                "writes __internal_price_data.csv, and prints "
                f"{expected_marker} timeframe=<...> rows=<...> to stderr."
            ),
            "",
        )

    parsed_output = _extract_stdout_json(stdout)
    missing_fields = REQUIRED_OUTPUT_FIELDS - set(parsed_output.keys()) if parsed_output else REQUIRED_OUTPUT_FIELDS

    review_input = json.dumps({
        "exit_code": returncode,
        "stdout_json": parsed_output,
        "missing_fields": sorted(missing_fields),
        "stderr_tail": stderr[-2000:] if stderr else "",
        "script_tail": script[-2000:],
        "failure_context": failure_context,
    }, indent=2, default=str)

    response = client.messages.create(
        model=_ANTHROPIC_MODEL,
        max_tokens=512,
        system=REVIEW_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": review_input}],
    )

    raw_text = response.content[0].text
    result = _extract_json(raw_text)
    if not result:
        # Claude responded with unparseable text â€” treat as agent fault
        return "agent_fault", f"Review LLM returned unparseable response: {raw_text[:300]}", ""

    verdict = str(result.get("verdict", "agent_fault"))
    feedback = str(result.get("feedback", ""))
    message = str(result.get("message", ""))
    return verdict, feedback, message


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_backtest_agent(
    strategy_files: list[tuple[str, str]],
    data_files: list[tuple[str, str]],
    pitch_context: dict,
    max_attempts: int = BACKTEST_MAX_ATTEMPTS,
) -> BacktestTermination:
    """
    Run the Claude backtest agent loop.

    Args:
        strategy_files: List of (filename, file_content) for .py strategy files.
        data_files:     List of (filename, file_content) for .csv/.tsv data files.
        pitch_context:  Dict with at minimum "name" and "ticker".
        max_attempts:   Number of CREATEâ†’RUNâ†’REVIEW cycles (default 3).

    Returns:
        BacktestTermination with status, metrics, message, attempt_history.
    """
    try:
        client = _get_anthropic_client()
    except ImportError as exc:
        return BacktestTermination(
            status="agent_fault",
            metrics=None,
            message=str(exc),
            attempt_count=0,
        )
    except ValueError:
        # Misconfiguration (missing API key) â€” re-raise so the caller knows
        # this is an infrastructure problem, not a strategy problem.
        raise

    attempt_history: list[AttemptRecord] = []
    prior_feedback: str | None = None
    prior_script: str | None = None
    prior_stdout: str | None = None
    prior_stderr: str | None = None

    strategy_files = _normalize_staged_files(strategy_files, "strategy")
    data_files = _normalize_staged_files(data_files, "data")

    tmp_dir = Path(tempfile.mkdtemp(prefix="backtest_agent_"))

    try:
        for attempt_num in range(1, max_attempts + 1):
            t_start = time.monotonic()
            script = ""
            stdout = ""
            stderr = ""
            returncode: int | None = None
            review_verdict = "not_reached"
            review_feedback = ""
            review_message = ""
            phase_reached = "create"

            # ---- Phase 1: CREATE / FIX ----
            try:
                script = _phase_create(
                    client=client,
                    strategy_files=strategy_files,
                    data_files=data_files,
                    pitch_context=pitch_context,
                    prior_feedback=prior_feedback,
                    prior_script=prior_script,
                    prior_stdout=prior_stdout,
                    prior_stderr=prior_stderr,
                )
            except Exception as exc:
                review_feedback = (
                    f"Phase-1 (create) exception: {type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                attempt_history.append(AttemptRecord(
                    attempt=attempt_num,
                    phase_reached=phase_reached,
                    script=script,
                    stdout=stdout,
                    stderr=stderr,
                    returncode=returncode,
                    review_verdict="agent_fault",
                    review_feedback=review_feedback,
                    review_message="",
                    latency_ms=int((time.monotonic() - t_start) * 1000),
                ))
                prior_feedback = review_feedback
                continue

            # ---- Phase 2: RUN ----
            phase_reached = "run"
            # Clean tmp_dir for this attempt (keep it isolated per attempt)
            attempt_dir = tmp_dir / f"attempt_{attempt_num}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            stdout, stderr, returncode = _phase_run(
                script=script,
                strategy_files=strategy_files,
                data_files=data_files,
                tmp_dir=attempt_dir,
            )

            # ---- Phase 3: REVIEW ----
            phase_reached = "review"
            try:
                review_verdict, review_feedback, review_message = _phase_review(
                    client=client,
                    stdout=stdout,
                    stderr=stderr,
                    returncode=returncode,
                    script=script,
                    ticker=str(pitch_context.get("ticker", "UNKNOWN")),
                )
            except Exception as exc:
                review_verdict = "agent_fault"
                review_feedback = (
                    f"Phase-3 (review) exception: {type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                review_message = ""

            latency_ms = int((time.monotonic() - t_start) * 1000)
            attempt_history.append(AttemptRecord(
                attempt=attempt_num,
                phase_reached=phase_reached,
                script=script,
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                review_verdict=review_verdict,
                review_feedback=review_feedback,
                review_message=review_message,
                latency_ms=latency_ms,
            ))

            if review_verdict == "success":
                # Extract and enrich metrics with name + ticker from context
                raw_metrics = _extract_stdout_json(stdout) or {}
                raw_metrics["name"] = pitch_context.get("name", "strategy")
                raw_metrics["ticker"] = pitch_context.get("ticker", "UNKNOWN")

                # Coerce and validate numeric types â€” treat NaN/Inf as agent fault
                try:
                    metrics = _coerce_metric_types(raw_metrics)
                except ValueError as exc:
                    review_verdict = "agent_fault"
                    review_feedback = f"Metric type coercion failed: {exc}"
                    prior_feedback = review_feedback
                    prior_script = script
                    prior_stdout = stdout
                    prior_stderr = stderr
                    attempt_history[-1].review_verdict = review_verdict
                    attempt_history[-1].review_feedback = review_feedback
                    continue

                return BacktestTermination(
                    status="success",
                    metrics=metrics,
                    message=f"Backtest completed successfully on attempt {attempt_num}.",
                    attempt_count=attempt_num,
                    attempt_history=attempt_history,
                )

            elif review_verdict == "user_action_required":
                return BacktestTermination(
                    status="user_action_required",
                    metrics=None,
                    message=review_message or "The strategy script requires user intervention.",
                    attempt_count=attempt_num,
                    attempt_history=attempt_history,
                )

            else:
                # agent_fault â€” set up feedback for next attempt
                prior_feedback = review_feedback
                prior_script = script
                prior_stdout = stdout
                prior_stderr = stderr

        # All attempts exhausted
        return BacktestTermination(
            status="agent_fault",
            metrics=None,
            message=(
                f"Backtest agent exhausted {max_attempts} attempt(s) without success. "
                f"Last feedback: {prior_feedback or 'unknown error'}"
            ),
            attempt_count=max_attempts,
            attempt_history=attempt_history,
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Post-processing: coerce types on extracted metrics
# ---------------------------------------------------------------------------

def _coerce_metric_types(metrics: dict) -> dict:
    """
    Cast numeric fields to correct Python types.
    Raises ValueError if any required numeric field contains NaN or Inf,
    which indicates a corrupt backtest result that should be caught by review.
    """
    int_fields = {"max_drawdown_duration", "total_trades"}
    float_fields = REQUIRED_OUTPUT_FIELDS - int_fields - {"backtest_start", "backtest_end"}
    str_fields = {"backtest_start", "backtest_end", "name", "ticker"}

    result = dict(metrics)
    for key in float_fields:
        if key in result:
            try:
                v = float(result[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Metric '{key}' is not numeric: {result[key]!r}") from exc
            if math.isnan(v) or math.isinf(v):
                raise ValueError(
                    f"Metric '{key}' is {v!r} â€” backtest produced non-finite result. "
                    "This is a bug in the generated script."
                )
            result[key] = v
    for key in int_fields:
        if key in result:
            try:
                result[key] = int(result[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Metric '{key}' is not an integer: {result[key]!r}") from exc
    for key in str_fields:
        if key in result:
            result[key] = str(result[key])
    return result


# ---------------------------------------------------------------------------
# Standalone CLI for testing
# ---------------------------------------------------------------------------

def _infer_ticker_from_filename(filename: str) -> str:
    stem = Path(filename).stem.upper()
    # Try common patterns: prices_AAPL -> AAPL, AAPL_prices -> AAPL
    parts = re.split(r"[_\-\.]", stem)
    for part in parts:
        if re.fullmatch(r"[A-Z]{1,5}", part) and part not in {"CSV", "DATA", "PRICES", "PRICE"}:
            return part
    return "SPY"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the backtest agent on a strategy file.")
    parser.add_argument("strategy", help="Path to strategy .py file")
    parser.add_argument("data", nargs="*", help="Path(s) to data CSV/TSV file(s)")
    parser.add_argument("--ticker", default=None, help="Ticker symbol (inferred from data filename if omitted)")
    parser.add_argument("--name", default="test_strategy", help="Strategy name")
    parser.add_argument("--attempts", type=int, default=BACKTEST_MAX_ATTEMPTS)
    args = parser.parse_args()

    strategy_path = Path(args.strategy)
    strategy_files = [(strategy_path.name, strategy_path.read_text(encoding="utf-8"))]

    data_file_list: list[tuple[str, str]] = []
    for dp in args.data:
        p = Path(dp)
        data_file_list.append((p.name, p.read_text(encoding="utf-8")))

    ticker = args.ticker or (
        _infer_ticker_from_filename(args.data[0]) if args.data else "SPY"
    )

    print(f"Running backtest agent: strategy={strategy_path.name}, ticker={ticker}, max_attempts={args.attempts}")

    result = run_backtest_agent(
        strategy_files=strategy_files,
        data_files=data_file_list,
        pitch_context={"name": args.name, "ticker": ticker},
        max_attempts=args.attempts,
    )

    print(f"\nStatus:  {result.status}")
    print(f"Message: {result.message}")
    print(f"Attempts: {result.attempt_count}")
    if result.metrics:
        print("\nMetrics:")
        for k, v in sorted(result.metrics.items()):
            print(f"  {k}: {v}")

