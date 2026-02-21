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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKTEST_TIMEOUT_SECONDS = int(os.getenv("BACKTEST_TIMEOUT_SECONDS", "120"))
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

# System prompt for the CREATE/FIX phase (Phase 1).
CREATE_SYSTEM_PROMPT = """
You are an expert quant Python developer. Your job is to produce a SINGLE,
self-contained Python script that:

1. Loads the user's strategy files and data files from the CURRENT WORKING DIRECTORY.
2. Runs the strategy and any machine-learning / signal logic it contains.
3. Fetches benchmark buy-and-hold data for the same ticker and period using yfinance.
4. Computes ALL required output metrics (see schema below).
5. Prints the final JSON object to stdout as the very last thing.
   All other prints MUST use stderr.

Constraints:
- Only use: json, math, os, sys, pathlib, datetime, numpy, pandas, scipy, sklearn, yfinance,
  statsmodels. No other third-party imports.
- If the strategy file defines a function, call it. If it is a script, adapt it.
- The equity curve must be derived from the strategy's actual signals/positions.
- Trades must be extracted or inferred from the position changes.
- For the benchmark, call:
    import yfinance as yf
    bench = yf.download(ticker, start=backtest_start, end=backtest_end, progress=False)
  Use the Adj Close column for benchmark returns. If yfinance fails, compute
  buy-and-hold from the same price data the user uploaded.
- Column name aliases for price data (accept any of these):
    date column:   date, Date, timestamp, timestamp_utc, datetime, Datetime
    close column:  close, Close, adj_close, Adj Close, adj close, price, Price, last
    return column: daily_return, returns, return, pct_return, ret, Return
    portfolio value: portfolio_value, equity, value, nav, wealth, NAV

{schema}

CRITICAL: Output ONLY the Python script. No explanation, no markdown fences, no prose.
Just raw Python code starting with imports.
""".strip().format(schema=REQUIRED_OUTPUT_SCHEMA)

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
   library, references a file we don't have, fundamentally broken logic that cannot
   be fixed by rewriting the runner):
   -> verdict = "user_action_required"
   -> message = clear, user-facing explanation of what they need to fix or provide

Respond ONLY with a JSON object:
{
  "verdict": "success" | "agent_fault" | "user_action_required",
  "feedback": "...",   // only for agent_fault — what to fix
  "message": "..."     // only for user_action_required — user-facing message
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
# Helper — get Anthropic client
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
# Helper — file schema summary
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
        f"# {filename} — {total_rows} data rows\n"
        f"{preview}\n"
        f"{'...(truncated)' if len(lines) > max_rows + 1 else ''}"
    ).strip()


# ---------------------------------------------------------------------------
# Helper — extract JSON from Claude response text
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
# Helper — extract Python script from Claude response
# ---------------------------------------------------------------------------

def _extract_script(text: str) -> str:
    """
    Claude should return raw Python with no fences. But in case it adds
    markdown fences, strip them.
    """
    # Strip ```python ... ``` or ``` ... ```
    fenced = re.sub(r"^```(?:python)?\s*\n", "", text.strip(), flags=re.MULTILINE)
    fenced = re.sub(r"\n```\s*$", "", fenced.strip(), flags=re.MULTILINE)
    return fenced.strip()


# ---------------------------------------------------------------------------
# Helper — extract last JSON from stdout
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

    context_block = textwrap.dedent(f"""\
        Strategy name: {name}
        Ticker: {ticker}
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
        max_tokens=4096,
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
    """
    # Write generated runner
    runner_path = tmp_dir / "_backtest_runner.py"
    runner_path.write_text(script, encoding="utf-8")

    # Copy strategy files
    for fname, content in strategy_files:
        (tmp_dir / fname).write_text(content, encoding="utf-8")

    # Copy data files
    for fname, content in data_files:
        (tmp_dir / fname).write_text(content, encoding="utf-8")

    try:
        result = subprocess.run(
            [sys.executable, str(runner_path)],
            cwd=str(tmp_dir),
            capture_output=True,
            text=True,
            timeout=BACKTEST_TIMEOUT_SECONDS,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Script timed out after {BACKTEST_TIMEOUT_SECONDS}s", 1
    except Exception as exc:
        return "", str(exc), 1


# ---------------------------------------------------------------------------
# Phase 3: REVIEW
# ---------------------------------------------------------------------------

def _phase_review(
    client,
    stdout: str,
    stderr: str,
    returncode: int,
    script: str,
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
        # Claude responded with unparseable text — treat as agent fault
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
        max_attempts:   Number of CREATE→RUN→REVIEW cycles (default 3).

    Returns:
        BacktestTermination with status, metrics, message, attempt_history.
    """
    try:
        client = _get_anthropic_client()
    except (ImportError, ValueError) as exc:
        return BacktestTermination(
            status="agent_fault",
            metrics=None,
            message=str(exc),
            attempt_count=0,
        )

    attempt_history: list[AttemptRecord] = []
    prior_feedback: str | None = None
    prior_script: str | None = None
    prior_stdout: str | None = None
    prior_stderr: str | None = None

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
                review_feedback = f"Phase-1 (create) exception: {exc}"
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
                )
            except Exception as exc:
                review_verdict = "agent_fault"
                review_feedback = f"Phase-3 (review) exception: {exc}"
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
                metrics = _extract_stdout_json(stdout) or {}
                metrics["name"] = pitch_context.get("name", "strategy")
                metrics["ticker"] = pitch_context.get("ticker", "UNKNOWN")

                # Ensure numeric types
                metrics = _coerce_metric_types(metrics)

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
                # agent_fault — set up feedback for next attempt
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
    """Cast numeric fields to correct Python types."""
    int_fields = {"max_drawdown_duration", "total_trades"}
    float_fields = REQUIRED_OUTPUT_FIELDS - int_fields - {"backtest_start", "backtest_end"}
    str_fields = {"backtest_start", "backtest_end", "name", "ticker"}

    result = dict(metrics)
    for key in float_fields:
        if key in result:
            try:
                v = float(result[key])
                result[key] = 0.0 if (math.isnan(v) or math.isinf(v)) else v
            except (TypeError, ValueError):
                result[key] = 0.0
    for key in int_fields:
        if key in result:
            try:
                result[key] = int(result[key])
            except (TypeError, ValueError):
                result[key] = 0
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
