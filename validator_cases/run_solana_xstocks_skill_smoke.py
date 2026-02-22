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


def _to_unix(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp())


def main() -> int:
    load_dotenv()
    symbol = os.getenv("SOLANA_XSTOCK_SMOKE_SYMBOL", "TSLAx").strip()
    interval = os.getenv("SOLANA_XSTOCK_SMOKE_INTERVAL", "1D").strip()
    days = int(os.getenv("SOLANA_XSTOCK_SMOKE_DAYS", "30"))
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(2, days))

    result = run_data_skill(
        "solana_xstocks_bars",
        {
            "symbol": symbol,
            "time_from": _to_unix(start),
            "time_to": _to_unix(end),
            "interval": interval,
            "currency": "usd",
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
