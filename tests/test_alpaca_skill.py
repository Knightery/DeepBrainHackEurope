from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agent_skills import list_data_skills, run_data_skill
from agent_skills.alpaca_historical import fetch_alpaca_historical_bars


class AlpacaSkillTests(unittest.TestCase):
    def test_registry_lists_skill(self) -> None:
        self.assertIn("alpaca_historical_bars", list_data_skills())

    def test_registry_rejects_unknown_skill(self) -> None:
        result = run_data_skill("does_not_exist", {})
        self.assertEqual("fail", result.get("status"))
        self.assertIn("available_skills", result.get("artifacts", {}))

    @patch.dict(os.environ, {"ALPACA_API_KEY": "test-key", "ALPACA_API_SECRET": "test-secret"}, clear=False)
    @patch("agent_skills.alpaca_historical._http_get_json")
    def test_fetch_paginates_and_normalizes(self, mock_get: object) -> None:
        mock_get.side_effect = [
            (
                {
                    "bars": [
                        {"t": "2026-01-02T00:00:00Z", "o": 100, "h": 101, "l": 99.5, "c": 100.5, "v": 2500, "n": 50, "vw": 100.2}
                    ],
                    "next_page_token": "next-1",
                },
                "req-1",
            ),
            (
                {
                    "bars": [
                        {"t": "2026-01-03T00:00:00Z", "o": 101, "h": 102, "l": 100.1, "c": 101.7, "v": 2100, "n": 44, "vw": 101.4}
                    ]
                },
                "req-2",
            ),
        ]

        result = fetch_alpaca_historical_bars(
            {
                "symbol": "AAPL",
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-01-05T00:00:00Z",
                "timeframe": "1Day",
                "limit": 1,
            }
        )

        self.assertEqual("ok", result.get("status"))
        bars = result.get("bars", [])
        self.assertEqual(2, len(bars))
        self.assertEqual("AAPL", bars[0]["symbol"])
        self.assertEqual(100.5, bars[0]["close"])
        self.assertEqual(2500, bars[0]["volume"])
        self.assertEqual(["req-1", "req-2"], result.get("artifacts", {}).get("request_ids"))

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_credentials_fails(self) -> None:
        result = run_data_skill(
            "alpaca_historical_bars",
            {
                "symbol": "AAPL",
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-01-03T00:00:00Z",
            },
        )
        self.assertEqual("fail", result.get("status"))
        self.assertIn("ALPACA_API_KEY", str(result.get("summary", "")))


if __name__ == "__main__":
    unittest.main()
