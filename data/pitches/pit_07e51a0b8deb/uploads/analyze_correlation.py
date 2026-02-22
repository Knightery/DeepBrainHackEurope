from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lib import (
    Paths,
    add_forward_returns,
    build_feature_frame,
    compute_metrics,
    parse_eia_history,
    parse_eqt_prices,
    parse_noaa_csv,
    pick_latest_row,
    write_metrics_json,
)


def _write_report(
    report_path: Path,
    metrics: dict,
    latest_row: pd.Series,
    hold_days: int,
    threshold: float,
) -> None:
    lines = [
        "# EQT One-Shot Correlation Proof",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Data Sources",
        "- EIA weekly natural gas storage history: `data/raw/eia_ngshistory.xls`",
        "- NOAA CPC weekly heating degree day archive: `data/raw/noaa_weekly_hdd_us.csv`",
        "- EQT daily prices: `data/raw/eqt_prices_stooq_daily.csv`",
        "",
        "## Feature Construction",
        "- `storage_surprise_bcf`: current EIA weekly net change minus prior expanding mean for same week-of-year.",
        "- `warmth_surprise`: negative NOAA HDD deviation from normal minus prior expanding mean for same week-of-year.",
        "- `signal_score = z(storage_surprise_bcf) + z(warmth_surprise)` using a 104-week rolling z-score (shifted by 1 week).",
        "",
        "## Historical Results",
        f"- Sample rows: {int(metrics['sample_rows'])} ({metrics['start_date']} to {metrics['end_date']})",
        f"- Hold period: {hold_days} trading days",
        f"- Entry lag: 6 calendar days after EIA week-ending date (release timing proxy)",
        f"- `corr(signal_score, EQT fwd return)`: {metrics['corr_signal_vs_return']:.4f}",
        f"- `corr(storage_surprise_z, EQT fwd return)`: {metrics['corr_storage_vs_return']:.4f}",
        f"- `corr(warmth_surprise_z, EQT fwd return)`: {metrics['corr_warmth_vs_return']:.4f}",
        "",
        "### Quintile Spread Check",
        f"- Q0 mean fwd return: {metrics['quintile_0_mean']:.4%}",
        f"- Q4 mean fwd return: {metrics['quintile_4_mean']:.4%}",
        f"- Q4 - Q0 spread: {metrics['quintile_4_minus_0']:.4%}",
        f"- Welch t-stat (Q4 - Q0): {metrics['quintile_spread_t_stat']:.3f}",
        "",
        "### Tradable Threshold Rule",
        f"- Rule: if `signal_score > {threshold}`, short EQT; if `< -{threshold}`, long EQT; else no trade.",
        f"- Trades triggered: {int(metrics['trade_count'])}",
        f"- Average strategy return per trade: {metrics['strategy_avg_return']:.4%}",
        f"- Hit rate: {metrics['strategy_hit_rate']:.2%}",
        f"- Strategy t-stat: {metrics['strategy_t_stat']:.3f}",
        "",
        "## Latest Read",
        f"- Week ending: {latest_row['week_ending'].date().isoformat()}",
        f"- EIA net change (Bcf): {latest_row['net_change_bcf']:.1f}",
        f"- Storage surprise (Bcf): {latest_row['storage_surprise_bcf']:.2f}",
        f"- NOAA HDD dev from normal: {latest_row['hdd_week_dev_norm']:.1f}",
        f"- Warmth surprise: {latest_row['warmth_surprise']:.2f}",
        f"- Signal score: {latest_row['signal_score']:.3f}",
        "",
        "## Caveats",
        "- This is a tactical statistical relationship, not a structural certainty.",
        "- NOAA archive data are preliminary by source labeling.",
        "- Price history source is Stooq adjusted daily data.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hold-days", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1.0)
    args = parser.parse_args()

    paths = Paths.from_script(Path(__file__))
    eia_path = paths.raw_dir / "eia_ngshistory.xls"
    noaa_path = paths.raw_dir / "noaa_weekly_hdd_us.csv"
    eqt_path = paths.raw_dir / "eqt_prices_stooq_daily.csv"

    if not (eia_path.exists() and noaa_path.exists() and eqt_path.exists()):
        raise FileNotFoundError(
            "Missing required raw data files. Run scripts/download_data.py first."
        )

    eia = parse_eia_history(eia_path)
    noaa = parse_noaa_csv(noaa_path)
    prices = parse_eqt_prices(eqt_path)

    features = build_feature_frame(eia, noaa)
    frame = add_forward_returns(features, prices, hold_days=args.hold_days, release_lag_days=6)
    metrics = compute_metrics(frame, hold_days=args.hold_days, threshold=args.threshold)
    latest_row = pick_latest_row(frame)

    processed_csv = paths.processed_dir / "feature_dataset.csv"
    frame.to_csv(processed_csv, index=False)

    metrics_json = paths.reports_dir / "correlation_metrics.json"
    write_metrics_json(metrics_json, metrics)

    report_md = paths.reports_dir / "correlation_proof.md"
    _write_report(
        report_path=report_md,
        metrics=metrics,
        latest_row=latest_row,
        hold_days=args.hold_days,
        threshold=args.threshold,
    )

    print("Analysis complete.")
    print(f"Processed dataset: {processed_csv}")
    print(f"Metrics JSON: {metrics_json}")
    print(f"Report: {report_md}")


if __name__ == "__main__":
    main()

