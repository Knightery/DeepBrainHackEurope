from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lib import Paths, current_signal, parse_eqt_prices, pick_latest_row


def _risk_bands(entry_price: float, realized_vol: float, hold_days: int, side: str) -> tuple:
    move = max(0.04, min(0.12, realized_vol * np.sqrt(max(1, hold_days))))
    if side == "SHORT":
        stop = entry_price * (1.0 + move)
        target = entry_price * (1.0 - (1.25 * move))
    elif side == "LONG":
        stop = entry_price * (1.0 - move)
        target = entry_price * (1.0 + (1.25 * move))
    else:
        stop = np.nan
        target = np.nan
    return float(stop), float(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--hold-days", type=int, default=10)
    args = parser.parse_args()

    paths = Paths.from_script(Path(__file__))
    feature_path = paths.processed_dir / "feature_dataset.csv"
    price_path = paths.raw_dir / "eqt_prices_stooq_daily.csv"

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing {feature_path}. Run scripts/analyze_correlation.py first."
        )
    if not price_path.exists():
        raise FileNotFoundError(
            f"Missing {price_path}. Run scripts/download_data.py first."
        )

    frame = pd.read_csv(feature_path)
    frame["week_ending"] = pd.to_datetime(frame["week_ending"], errors="coerce")
    frame["signal_score"] = pd.to_numeric(frame["signal_score"], errors="coerce")
    latest = pick_latest_row(frame)

    prices = parse_eqt_prices(price_path)
    prices["daily_return"] = prices["Close"].pct_change()
    realized_vol = float(prices["daily_return"].tail(20).std(ddof=1))
    last_price = float(prices.iloc[-1]["Close"])
    last_date = pd.Timestamp(prices.iloc[-1]["Date"]).date().isoformat()

    side = current_signal(latest, threshold=args.threshold)
    stop, target = _risk_bands(last_price, realized_vol, args.hold_days, side)
    confidence = min(1.0, abs(float(latest["signal_score"])) / 3.0)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": "EQT",
        "signal": side,
        "signal_score": round(float(latest["signal_score"]), 4),
        "threshold": args.threshold,
        "entry_reference_price": round(last_price, 4),
        "entry_reference_date": last_date,
        "recommended_holding_days": args.hold_days,
        "stop_price": None if np.isnan(stop) else round(stop, 4),
        "target_price": None if np.isnan(target) else round(target, 4),
        "confidence_0_to_1": round(confidence, 4),
        "drivers": {
            "week_ending": latest["week_ending"].date().isoformat(),
            "eia_net_change_bcf": round(float(latest["net_change_bcf"]), 2),
            "eia_storage_surprise_bcf": round(
                float(latest["storage_surprise_bcf"]), 2
            ),
            "noaa_hdd_dev_from_normal": round(float(latest["hdd_week_dev_norm"]), 2),
            "warmth_surprise": round(float(latest["warmth_surprise"]), 2),
        },
    }

    out_path = paths.reports_dir / "latest_trade_signal.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

