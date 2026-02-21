"""
strategy_scorer.py
------------------
Scores and ranks quantitative trading strategies from a JSON input.

Each strategy JSON must include:
  - name (str)
  - ticker (str)
  - backtest_start (str, YYYY-MM-DD)
  - backtest_end (str, YYYY-MM-DD)
  - benchmark_cagr (float)          -- buy-and-hold CAGR for the ticker over same period
  - benchmark_max_drawdown (float)  -- buy-and-hold max drawdown (negative float, e.g. -0.25)
  - benchmark_total_return (float)  -- buy-and-hold total return over same period

  # Strategy performance
  - cagr (float)
  - total_return (float)
  - volatility (float)              -- annualised std dev of returns
  - sharpe_ratio (float)
  - sortino_ratio (float)
  - calmar_ratio (float)
  - max_drawdown (float)            -- negative float
  - max_drawdown_duration (int)     -- days

  # Trades
  - total_trades (int)
  - win_rate (float)
  - avg_win (float)
  - avg_loss (float)                -- negative float
  - profit_factor (float)
  - expectancy (float)

  # Benchmark-relative (computed by the agent ideally, validated here)
  - alpha (float)
  - information_ratio (float)
  - excess_return (float)           -- cagr - benchmark_cagr
  - up_capture (float)              -- fraction of benchmark upside captured (e.g. 0.9 = 90%)
  - down_capture (float)            -- fraction of benchmark downside captured (lower is better)

Usage:
  python strategy_scorer.py strategies.json
  python strategy_scorer.py strategies.json --ticker AAPL
  python strategy_scorer.py strategies.json --top 5
"""

import json
import sys
import argparse
import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Schema & validation
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "name", "ticker", "backtest_start", "backtest_end",
    "benchmark_cagr", "benchmark_max_drawdown", "benchmark_total_return",
    "cagr", "total_return", "volatility",
    "sharpe_ratio", "sortino_ratio", "calmar_ratio",
    "max_drawdown", "max_drawdown_duration",
    "total_trades", "win_rate", "avg_win", "avg_loss",
    "profit_factor", "expectancy",
    "alpha", "information_ratio", "excess_return",
    "up_capture", "down_capture",
}

OPTIONAL_FIELDS = {
    "description": None,
    "exposure_time": None,
}


@dataclass
class Strategy:
    name: str
    ticker: str
    backtest_start: str
    backtest_end: str

    # Benchmark
    benchmark_cagr: float
    benchmark_max_drawdown: float
    benchmark_total_return: float

    # Returns
    cagr: float
    total_return: float
    volatility: float

    # Risk ratios
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int

    # Trades
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float

    # Benchmark-relative
    alpha: float
    information_ratio: float
    excess_return: float
    up_capture: float
    down_capture: float

    # Optional
    description: Optional[str] = None
    exposure_time: Optional[float] = None

    # Computed post-init
    win_loss_ratio: float = field(init=False)
    drawdown_vs_benchmark: float = field(init=False)  # dd - benchmark_dd (negative = better)
    capture_ratio: float = field(init=False)           # up / down (higher is better)

    def __post_init__(self):
        # Win/loss ratio — more meaningful than win rate alone
        self.win_loss_ratio = (
            abs(self.avg_win / self.avg_loss)
            if self.avg_loss and self.avg_loss != 0
            else 0.0
        )
        # How much better/worse is drawdown vs benchmark?
        # Positive means strategy drew down MORE than benchmark (bad)
        self.drawdown_vs_benchmark = self.max_drawdown - self.benchmark_max_drawdown

        # Up-capture / down-capture ratio. Inf-guard: if down_capture is 0, cap at 10
        if self.down_capture and self.down_capture > 0:
            self.capture_ratio = self.up_capture / self.down_capture
        else:
            self.capture_ratio = self.up_capture * 10  # very low downside capture = excellent


def validate_and_load(raw: dict) -> tuple[Optional[Strategy], list[str]]:
    """Return (Strategy, []) on success, (None, [errors]) on failure."""
    errors = []

    missing = REQUIRED_FIELDS - set(raw.keys())
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")
        return None, errors

    # Sanity checks
    if raw["win_rate"] < 0 or raw["win_rate"] > 1:
        errors.append(f"win_rate must be in [0, 1], got {raw['win_rate']}")
    if raw["max_drawdown"] > 0:
        errors.append(f"max_drawdown should be negative, got {raw['max_drawdown']}")
    if raw["benchmark_max_drawdown"] > 0:
        errors.append(f"benchmark_max_drawdown should be negative, got {raw['benchmark_max_drawdown']}")
    if raw["avg_loss"] > 0:
        errors.append(f"avg_loss should be negative, got {raw['avg_loss']}")
    if raw["total_trades"] < 1:
        errors.append(f"total_trades must be >= 1, got {raw['total_trades']}")

    # Cross-validate excess_return
    computed_excess = raw["cagr"] - raw["benchmark_cagr"]
    if abs(computed_excess - raw["excess_return"]) > 0.005:
        errors.append(
            f"excess_return {raw['excess_return']:.4f} doesn't match "
            f"cagr - benchmark_cagr = {computed_excess:.4f} (tolerance 0.5%)"
        )

    if errors:
        return None, errors

    kwargs = {k: raw[k] for k in REQUIRED_FIELDS}
    for k, default in OPTIONAL_FIELDS.items():
        kwargs[k] = raw.get(k, default)

    return Strategy(**kwargs), []


# ---------------------------------------------------------------------------
# Hard gates — strategies that fail these are disqualified
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    passed: bool
    reason: str = ""


def apply_hard_gates(s: Strategy) -> list[GateResult]:
    """
    A strategy must pass ALL gates to be eligible for scoring.
    Failing any gate = disqualified (score = 0, flagged in output).
    """
    gates = []

    # Gate 1: Must beat buy-and-hold on a raw return basis
    gates.append(GateResult(
        passed=s.excess_return > 0,
        reason=f"Excess return {s.excess_return:.2%} <= 0 (underperforms buy-and-hold)"
    ))

    # Gate 2: Positive alpha (risk-adjusted outperformance)
    gates.append(GateResult(
        passed=s.alpha > 0,
        reason=f"Alpha {s.alpha:.4f} <= 0"
    ))

    # Gate 3: Sharpe must be positive
    gates.append(GateResult(
        passed=s.sharpe_ratio > 0,
        reason=f"Sharpe ratio {s.sharpe_ratio:.2f} <= 0"
    ))

    # Gate 4: Profit factor must be > 1 (more won than lost)
    gates.append(GateResult(
        passed=s.profit_factor > 1.0,
        reason=f"Profit factor {s.profit_factor:.2f} <= 1.0"
    ))

    # Gate 5: Must not have catastrophic drawdown (> 50%)
    gates.append(GateResult(
        passed=s.max_drawdown > -0.50,
        reason=f"Max drawdown {s.max_drawdown:.2%} exceeds -50%"
    ))

    return gates


# ---------------------------------------------------------------------------
# Metric scoring — each metric returns a score in [0, 1]
# ---------------------------------------------------------------------------

def sigmoid(x: float, center: float = 0, scale: float = 1) -> float:
    """Smooth bounded mapping. Returns ~0.5 at center."""
    return 1 / (1 + math.exp(-scale * (x - center)))


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def score_excess_return(s: Strategy) -> float:
    """Excess return over benchmark CAGR. Sigmoid centred at 5% outperformance."""
    return clamp(sigmoid(s.excess_return, center=0.05, scale=20))


def score_alpha(s: Strategy) -> float:
    """Jensen's alpha. Good alpha is 2–5%+."""
    return clamp(sigmoid(s.alpha, center=0.02, scale=30))


def score_information_ratio(s: Strategy) -> float:
    """IR > 0.5 is decent, > 1.0 is excellent."""
    return clamp(sigmoid(s.information_ratio, center=0.5, scale=3))


def score_sharpe(s: Strategy) -> float:
    """Sharpe > 1 is good, > 2 is excellent."""
    return clamp(sigmoid(s.sharpe_ratio, center=1.0, scale=2))


def score_sortino(s: Strategy) -> float:
    """Sortino > 1.5 is good."""
    return clamp(sigmoid(s.sortino_ratio, center=1.5, scale=1.8))


def score_calmar(s: Strategy) -> float:
    """Calmar > 1 is good (return/max_dd)."""
    return clamp(sigmoid(s.calmar_ratio, center=1.0, scale=2))


def score_drawdown_vs_benchmark(s: Strategy) -> float:
    """
    Did the strategy draw down less than buy-and-hold?
    drawdown_vs_benchmark < 0 means the strategy had a SMALLER drawdown (good).
    Score = 1 if much better, 0.5 if equal, 0 if much worse.
    """
    # Positive = worse than benchmark, negative = better
    return clamp(sigmoid(-s.drawdown_vs_benchmark, center=0, scale=15))


def score_capture_ratio(s: Strategy) -> float:
    """Up-capture / down-capture. >1 is good (captures more up than down)."""
    return clamp(sigmoid(s.capture_ratio, center=1.0, scale=3))


def score_profit_factor(s: Strategy) -> float:
    """Profit factor > 1.5 is solid, > 2 is great."""
    return clamp(sigmoid(s.profit_factor, center=1.5, scale=2))


def score_win_loss_ratio(s: Strategy) -> float:
    """Avg win / avg loss. > 1.5 is decent, > 2.5 is good."""
    return clamp(sigmoid(s.win_loss_ratio, center=1.5, scale=1.5))


def score_expectancy(s: Strategy) -> float:
    """
    Expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    Positive expectancy is required; higher is better.
    Score sigmoid centred at 0.5% per trade.
    """
    return clamp(sigmoid(s.expectancy, center=0.005, scale=200))


def score_consistency(s: Strategy) -> float:
    """
    Proxy for consistency: number of trades. More trades = more evidence.
    Caps out around 200 trades (diminishing returns beyond that).
    """
    return clamp(sigmoid(s.total_trades, center=50, scale=0.04))


# ---------------------------------------------------------------------------
# Weighted composite scorer
# ---------------------------------------------------------------------------

# (scoring_function, weight, label, category)
SCORING_COMPONENTS = [
    # --- Benchmark-relative (highest weight) ---
    (score_excess_return,          3.0,  "Excess Return",             "benchmark"),
    (score_alpha,                  3.0,  "Alpha",                     "benchmark"),
    (score_information_ratio,      2.5,  "Information Ratio",         "benchmark"),
    (score_capture_ratio,          2.0,  "Up/Down Capture Ratio",     "benchmark"),
    (score_drawdown_vs_benchmark,  2.0,  "Drawdown vs Benchmark",     "benchmark"),

    # --- Risk-adjusted returns ---
    (score_sharpe,                 2.5,  "Sharpe Ratio",              "risk"),
    (score_sortino,                2.0,  "Sortino Ratio",             "risk"),
    (score_calmar,                 1.5,  "Calmar Ratio",              "risk"),

    # --- Trade quality ---
    (score_profit_factor,          1.5,  "Profit Factor",             "trades"),
    (score_win_loss_ratio,         1.0,  "Win/Loss Ratio",            "trades"),
    (score_expectancy,             1.5,  "Expectancy",                "trades"),
    (score_consistency,            0.5,  "Trade Consistency",         "trades"),
]

TOTAL_WEIGHT = sum(w for _, w, _, _ in SCORING_COMPONENTS)


@dataclass
class ComponentScore:
    label: str
    category: str
    weight: float
    raw_score: float       # [0, 1]
    weighted_score: float  # raw_score * weight


@dataclass
class ScoredStrategy:
    strategy: Strategy
    disqualified: bool
    disqualification_reasons: list[str]
    component_scores: list[ComponentScore]
    composite_score: float   # [0, 100]
    rank: int = 0            # filled in after ranking


def score_strategy(s: Strategy) -> ScoredStrategy:
    gate_results = apply_hard_gates(s)
    failed_gates = [g for g in gate_results if not g.passed]

    if failed_gates:
        return ScoredStrategy(
            strategy=s,
            disqualified=True,
            disqualification_reasons=[g.reason for g in failed_gates],
            component_scores=[],
            composite_score=0.0,
        )

    components = []
    for fn, weight, label, category in SCORING_COMPONENTS:
        raw = fn(s)
        components.append(ComponentScore(
            label=label,
            category=category,
            weight=weight,
            raw_score=raw,
            weighted_score=raw * weight,
        ))

    composite = sum(c.weighted_score for c in components) / TOTAL_WEIGHT * 100

    return ScoredStrategy(
        strategy=s,
        disqualified=False,
        disqualification_reasons=[],
        component_scores=components,
        composite_score=round(composite, 2),
    )


# ---------------------------------------------------------------------------
# Population-based scoring — z-score normalization against pool
# ---------------------------------------------------------------------------

def _zscore_normalize(value: float, population: list[float]) -> float:
    """
    Z-score normalize a value against a population, then map to [0, 1] via sigmoid.
    Average performer scores 0.5. +2 std above mean scores ~0.95.
    """
    if len(population) < 2:
        return 0.5
    mean = sum(population) / len(population)
    variance = sum((x - mean) ** 2 for x in population) / (len(population) - 1)
    std = variance ** 0.5
    if std < 1e-9:
        return 0.5
    z = (value - mean) / std
    return clamp(1 / (1 + math.exp(-z)))


def _minmax_normalize(value: float, population: list[float]) -> float:
    """Min-max normalize to [0, 1]. Falls back to 0.5 if all values are equal."""
    lo, hi = min(population), max(population)
    if hi - lo < 1e-9:
        return 0.5
    return clamp((value - lo) / (hi - lo))


def score_against_pool(target: Strategy, pool: list[Strategy]) -> ScoredStrategy:
    """
    Score target strategy using z-score normalization against same-ticker strategies
    in the pool. Falls back to sigmoid-based score_strategy if pool is too small.

    Same-ticker comparison is intentional — comparing AAPL strategies against
    TSLA strategies is not meaningful since benchmark difficulty differs.
    """
    same_ticker = [s for s in pool if s.ticker.upper() == target.ticker.upper()]

    # Need at least 3 same-ticker strategies for population stats to be meaningful
    if len(same_ticker) < 3:
        return score_strategy(target)

    # Still apply hard gates regardless of pool size
    gate_results = apply_hard_gates(target)
    failed_gates = [g for g in gate_results if not g.passed]
    if failed_gates:
        return ScoredStrategy(
            strategy=target,
            disqualified=True,
            disqualification_reasons=[g.reason for g in failed_gates],
            component_scores=[],
            composite_score=0.0,
        )

    components = []
    for fn, weight, label, category in SCORING_COMPONENTS:
        # Compute raw sigmoid score for every strategy in the same-ticker pool
        pool_raw_scores = [fn(s) for s in same_ticker]
        target_raw = fn(target)

        # Z-score normalize target's raw score against the pool distribution
        normalized = _zscore_normalize(target_raw, pool_raw_scores)

        components.append(ComponentScore(
            label=label,
            category=category,
            weight=weight,
            raw_score=target_raw,           # raw sigmoid score for display
            weighted_score=normalized * weight,  # z-score normalized for ranking
        ))

    total_weight = sum(c.weight for c in components)
    composite = sum(c.weighted_score for c in components) / total_weight * 100

    return ScoredStrategy(
        strategy=target,
        disqualified=False,
        disqualification_reasons=[],
        component_scores=components,
        composite_score=round(clamp(composite, 0.0, 100.0), 2),
    )


def rescore_pool(pool: list[Strategy]) -> list[ScoredStrategy]:
    """
    Rescore all strategies in the pool against each other.
    Called when a new submission shifts the population distribution.
    Returns ranked list.
    """
    scored = [score_against_pool(s, pool) for s in pool]
    return rank_strategies(scored)


# ---------------------------------------------------------------------------
# Ranking — within ticker groups first, then overall
# ---------------------------------------------------------------------------

def rank_strategies(scored: list[ScoredStrategy]) -> list[ScoredStrategy]:
    """
    Sort by composite score descending.
    Disqualified strategies go to the bottom.
    """
    qualified = sorted(
        [s for s in scored if not s.disqualified],
        key=lambda s: s.composite_score,
        reverse=True,
    )
    disqualified = [s for s in scored if s.disqualified]

    ranked = qualified + disqualified
    for i, s in enumerate(ranked):
        s.rank = i + 1

    return ranked


def group_by_ticker(scored: list[ScoredStrategy]) -> dict[str, list[ScoredStrategy]]:
    groups: dict[str, list[ScoredStrategy]] = {}
    for s in scored:
        ticker = s.strategy.ticker.upper()
        groups.setdefault(ticker, []).append(s)
    for ticker in groups:
        groups[ticker] = rank_strategies(groups[ticker])
    return groups


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_report(
    ranked: list[ScoredStrategy],
    ticker_groups: dict[str, list[ScoredStrategy]],
    verbose: bool = True,
) -> str:
    lines = []
    sep = "─" * 72

    lines.append("\n" + sep)
    lines.append("  QUANT STRATEGY RANKING REPORT")
    lines.append(sep)

    # --- Overall ranking ---
    lines.append("\n  OVERALL RANKING\n")
    lines.append(f"  {'Rank':<6} {'Strategy':<22} {'Ticker':<8} {'Score':>7}  {'Period'}")
    lines.append("  " + "─" * 65)

    for s in ranked:
        if s.disqualified:
            lines.append(
                f"  {'DQ':<6} {s.strategy.name:<22} {s.strategy.ticker:<8} "
                f"{'—':>7}  DISQUALIFIED"
            )
        else:
            period = f"{s.strategy.backtest_start} → {s.strategy.backtest_end}"
            lines.append(
                f"  {s.rank:<6} {s.strategy.name:<22} {s.strategy.ticker:<8} "
                f"{s.composite_score:>7.1f}  {period}"
            )

    # --- Per-ticker ranking ---
    if len(ticker_groups) > 1:
        lines.append(f"\n\n  WITHIN-TICKER RANKINGS\n")
        for ticker, group in sorted(ticker_groups.items()):
            lines.append(f"  {ticker}")
            lines.append("  " + "─" * 50)
            for s in group:
                if s.disqualified:
                    lines.append(f"    DQ   {s.strategy.name:<22} DISQUALIFIED")
                else:
                    lines.append(f"    {s.rank:<4} {s.strategy.name:<22} {s.composite_score:.1f}")
            lines.append("")

    # --- Detailed breakdown per strategy ---
    if verbose:
        lines.append(f"\n  DETAILED BREAKDOWNS\n")
        for s in ranked:
            lines.append(sep)
            status = "DISQUALIFIED" if s.disqualified else f"SCORE: {s.composite_score:.1f} / 100"
            lines.append(f"  {s.strategy.name}  [{s.strategy.ticker}]  —  {status}")
            lines.append(
                f"  Period: {s.strategy.backtest_start} → {s.strategy.backtest_end}"
            )
            lines.append(
                f"  Benchmark (buy & hold): CAGR {s.strategy.benchmark_cagr:.2%}  "
                f"MaxDD {s.strategy.benchmark_max_drawdown:.2%}"
            )
            lines.append(
                f"  Strategy:               CAGR {s.strategy.cagr:.2%}  "
                f"MaxDD {s.strategy.max_drawdown:.2%}  "
                f"Excess {s.strategy.excess_return:.2%}"
            )

            if s.disqualified:
                lines.append("\n  ✗ Failed gates:")
                for reason in s.disqualification_reasons:
                    lines.append(f"      • {reason}")
            else:
                # Group components by category
                cats = {}
                for c in s.component_scores:
                    cats.setdefault(c.category, []).append(c)

                lines.append("")
                for cat, comps in cats.items():
                    lines.append(f"  [{cat.upper()}]")
                    for c in comps:
                        bar = "█" * int(c.raw_score * 20) + "░" * (20 - int(c.raw_score * 20))
                        lines.append(
                            f"    {c.label:<28} {bar}  {c.raw_score:.2f}  (w={c.weight})"
                        )
                    lines.append("")

            lines.append("")

    return "\n".join(lines)


def to_json_output(ranked: list[ScoredStrategy]) -> dict:
    results = []
    for s in ranked:
        entry = {
            "rank": s.rank,
            "name": s.strategy.name,
            "ticker": s.strategy.ticker,
            "score": s.composite_score,
            "disqualified": s.disqualified,
            "disqualification_reasons": s.disqualification_reasons,
            "period": {
                "start": s.strategy.backtest_start,
                "end": s.strategy.backtest_end,
            },
            "key_metrics": {
                "excess_return": s.strategy.excess_return,
                "alpha": s.strategy.alpha,
                "information_ratio": s.strategy.information_ratio,
                "sharpe_ratio": s.strategy.sharpe_ratio,
                "max_drawdown": s.strategy.max_drawdown,
                "benchmark_max_drawdown": s.strategy.benchmark_max_drawdown,
                "capture_ratio": s.strategy.capture_ratio,
            },
        }
        if not s.disqualified:
            entry["component_scores"] = {
                c.label: {
                    "score": round(c.raw_score, 3),
                    "weight": c.weight,
                    "weighted": round(c.weighted_score, 3),
                }
                for c in s.component_scores
            }
        results.append(entry)
    return {"rankings": results, "total_strategies": len(ranked)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Score and rank quant strategies.")
    parser.add_argument("input", help="Path to JSON file (list of strategy objects)")
    parser.add_argument("--ticker", help="Filter to a specific ticker only", default=None)
    parser.add_argument("--top", type=int, help="Show only top N strategies", default=None)
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of text report")
    parser.add_argument("--quiet", action="store_true", help="Skip detailed per-strategy breakdown")
    args = parser.parse_args()

    with open(args.input) as f:
        raw_list = json.load(f)

    if not isinstance(raw_list, list):
        raw_list = [raw_list]

    strategies: list[Strategy] = []
    for i, raw in enumerate(raw_list):
        if args.ticker and raw.get("ticker", "").upper() != args.ticker.upper():
            continue
        s, errors = validate_and_load(raw)
        if errors:
            print(f"[ERROR] Strategy #{i} ({raw.get('name', '?')}): {errors}", file=sys.stderr)
        else:
            strategies.append(s)

    if not strategies:
        print("No valid strategies to score.", file=sys.stderr)
        sys.exit(1)

    scored = [score_strategy(s) for s in strategies]
    ranked = rank_strategies(scored)
    ticker_groups = group_by_ticker(scored)

    if args.top:
        ranked = ranked[: args.top]

    if args.json:
        print(json.dumps(to_json_output(ranked), indent=2))
    else:
        print(format_report(ranked, ticker_groups, verbose=not args.quiet))


if __name__ == "__main__":
    main()