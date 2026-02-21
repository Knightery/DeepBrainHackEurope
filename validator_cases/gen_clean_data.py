"""One-off script to generate clean_momentum_prices.csv."""
from __future__ import annotations

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

# Use a clearly fictional ticker so the LLM cannot compare against real market data.
TICKER = "BACKTST"

# 2024 NYSE market holidays
MARKET_HOLIDAYS_2024 = {
    date(2024, 1, 1),   # New Year's Day
    date(2024, 1, 15),  # MLK Day
    date(2024, 2, 19),  # Presidents Day
    date(2024, 3, 29),  # Good Friday
    date(2024, 5, 27),  # Memorial Day
    date(2024, 6, 19),  # Juneteenth
    date(2024, 7, 4),   # Independence Day
    date(2024, 9, 2),   # Labor Day
    date(2024, 11, 28), # Thanksgiving
    date(2024, 12, 25), # Christmas
}


price = 104.37  # fictional ETF, not round
daily_vol = 0.010
drift = 0.00035

MARKET_HOLIDAYS_2025 = {
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 20),  # MLK Day
    date(2025, 2, 17),  # Presidents Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Christmas
}

ALL_HOLIDAYS = MARKET_HOLIDAYS_2024 | MARKET_HOLIDAYS_2025


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in ALL_HOLIDAYS


prices: list[tuple[str, str, float, int]] = []
d = date(2024, 1, 2)
while len(prices) < 504:  # ~2 years
    if is_trading_day(d):
        ret = random.gauss(drift, daily_vol)
        if random.random() < 0.04:
            ret = random.gauss(drift, daily_vol * 2.5)
        price *= 1 + ret
        vol_raw = max(8_000_000, int(random.gauss(18_500_000, 3_200_000)))
        prices.append((d.isoformat(), TICKER, round(price, 2), vol_raw))
    d += timedelta(days=1)

rows = []
for i, (dt, sym, close, volume) in enumerate(prices):
    mom_21 = (prices[i][2] / prices[max(0, i - 21)][2] - 1) if i >= 21 else 0.0
    mom_63 = (prices[i][2] / prices[max(0, i - 63)][2] - 1) if i >= 63 else 0.0
    if i >= 20:
        rets = [prices[j][2] / prices[j - 1][2] - 1 for j in range(i - 19, i + 1)]
        rvol = math.sqrt(sum(r ** 2 for r in rets) / len(rets)) * math.sqrt(252)
    else:
        rvol = 0.0
    rows.append({
        "date": dt, "symbol": sym, "close": close, "volume": volume,
        "mom_21": round(mom_21, 4), "mom_63": round(mom_63, 4), "rvol_21": round(rvol, 4),
    })

# Drop the 63-row warmup period so all feature values in the file are valid.
rows = [r for r in rows if r["mom_63"] != 0.0]

out = Path(__file__).parent / "data" / "clean_momentum_prices.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["date", "symbol", "close", "volume", "mom_21", "mom_63", "rvol_21"])
    w.writeheader()
    w.writerows(rows)

closes = [r["close"] for r in rows]
vols = [r["volume"] for r in rows]
print(f"Generated {len(rows)} rows -> {out}")
print(f"Close range : {min(closes):.2f} - {max(closes):.2f}")
print(f"Volume range: {min(vols):,} - {max(vols):,}")
