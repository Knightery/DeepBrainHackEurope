from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_skills import run_data_skill


def _iso_utc(dt: datetime) -> str:
    return dt.replace(microsecond=0).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    load_dotenv()

    symbol = os.getenv("ALPACA_SMOKE_SYMBOL", "AAPL").strip().upper()
    days = int(os.getenv("ALPACA_SMOKE_DAYS", "30"))
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(2, days))

    result = run_data_skill(
        "alpaca_historical_bars",
        {
            "symbol": symbol,
            "start": _iso_utc(start),
            "end": _iso_utc(end),
            "timeframe": os.getenv("ALPACA_SMOKE_TIMEFRAME", "1Day"),
            "feed": os.getenv("ALPACA_SMOKE_FEED", "iex"),
            "adjustment": os.getenv("ALPACA_SMOKE_ADJUSTMENT", "raw"),
            "limit": int(os.getenv("ALPACA_SMOKE_LIMIT", "1000")),
        },
    )

    summary = {
        "skill": result.get("skill"),
        "status": result.get("status"),
        "summary": result.get("summary"),
        "bar_count": len(result.get("bars", [])) if isinstance(result.get("bars"), list) else 0,
        "artifacts": result.get("artifacts", {}),
    }
    print(json.dumps(summary, indent=2))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
