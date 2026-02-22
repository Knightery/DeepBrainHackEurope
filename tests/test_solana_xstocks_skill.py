from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agent_skills import list_data_skills, run_data_skill
from agent_skills.solana_xstocks import fetch_solana_xstocks_bars


class SolanaXStocksSkillTests(unittest.TestCase):
    def test_registry_lists_skill(self) -> None:
        self.assertIn("solana_xstocks_bars", list_data_skills())

    @patch.dict(os.environ, {"BIRDEYE_API_KEY": "test-key"}, clear=False)
    @patch("agent_skills.solana_xstocks._http_get_json")
    def test_fetch_normalizes_scaled_bars(self, mock_get: object) -> None:
        mock_get.return_value = {
            "success": True,
            "data": {
                "is_scaled_ui_token": True,
                "items": [
                    {
                        "unix_time": 1760000000,
                        "scaled_o": 151.1,
                        "scaled_h": 152.2,
                        "scaled_l": 150.7,
                        "scaled_c": 151.8,
                        "scaled_v": 12.5,
                        "v_usd": 1890.0,
                    }
                ],
            },
        }
        result = fetch_solana_xstocks_bars(
            {
                "symbol": "TSLAx",
                "time_from": 1759990000,
                "time_to": 1760010000,
                "interval": "1D",
            }
        )
        self.assertEqual("ok", result.get("status"))
        bars = result.get("bars", [])
        self.assertEqual(1, len(bars))
        self.assertEqual("TSLAX", bars[0]["symbol"])
        self.assertEqual(151.8, bars[0]["close"])
        self.assertEqual(12.5, bars[0]["volume"])

    @patch.dict(os.environ, {"BIRDEYE_API_KEY": "test-key"}, clear=False)
    @patch("agent_skills.solana_xstocks._http_get_json")
    def test_symbol_with_dot_is_supported(self, mock_get: object) -> None:
        mock_get.return_value = {
            "success": True,
            "data": {"is_scaled_ui_token": False, "items": [{"unix_time": 1760000000, "o": 500, "h": 505, "l": 498, "c": 503, "v": 2.0}]},
        }
        result = fetch_solana_xstocks_bars(
            {"symbol": "BRK.Bx", "time_from": 1759990000, "time_to": 1760010000, "interval": "1D"}
        )
        self.assertEqual("ok", result.get("status"))
        bars = result.get("bars", [])
        self.assertEqual("BRK.BX", bars[0]["symbol"])
        self.assertEqual("Xs6B6zawENwAbWVi7w92rjazLuAr5Az59qgWKcNb45x", bars[0]["mint"])

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_credentials_fails(self) -> None:
        result = run_data_skill(
            "solana_xstocks_bars",
            {
                "symbol": "TSLAx",
                "time_from": 1759990000,
                "time_to": 1760010000,
                "interval": "1D",
            },
        )
        self.assertEqual("fail", result.get("status"))
        self.assertIn("BIRDEYE_API_KEY", str(result.get("summary", "")))


if __name__ == "__main__":
    unittest.main()
