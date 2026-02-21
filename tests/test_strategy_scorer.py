from __future__ import annotations

import unittest

from strategy_scorer import validate_and_load


def _base_payload() -> dict:
    return {
        "name": "test_strategy",
        "ticker": "AAPL",
        "backtest_start": "2024-01-01",
        "backtest_end": "2024-12-31",
        "benchmark_cagr": 0.1,
        "benchmark_max_drawdown": -0.2,
        "benchmark_total_return": 0.12,
        "cagr": 0.2,
        "total_return": 0.25,
        "volatility": 0.3,
        "sharpe_ratio": 1.2,
        "sortino_ratio": 1.5,
        "calmar_ratio": 1.0,
        "max_drawdown": -0.15,
        "max_drawdown_duration": 30,
        "total_trades": 10,
        "win_rate": 0.6,
        "avg_win": 0.02,
        "avg_loss": -0.01,
        "profit_factor": 1.5,
        "expectancy": 0.005,
        "alpha": 0.03,
        "information_ratio": 0.8,
        "excess_return": 0.1,
        "up_capture": 1.1,
        "down_capture": 0.8,
    }


class StrategyScorerValidationTests(unittest.TestCase):
    def test_numeric_strings_are_coerced(self) -> None:
        raw = _base_payload()
        raw["win_rate"] = "0.6"
        raw["cagr"] = "0.2"
        raw["benchmark_cagr"] = "0.1"
        raw["excess_return"] = "0.1"
        raw["total_trades"] = "10"

        strategy, errors = validate_and_load(raw)

        self.assertEqual(errors, [])
        self.assertIsNotNone(strategy)
        assert strategy is not None
        self.assertAlmostEqual(strategy.win_rate, 0.6)
        self.assertEqual(strategy.total_trades, 10)

    def test_invalid_numeric_string_returns_validation_error(self) -> None:
        raw = _base_payload()
        raw["win_rate"] = "abc"

        strategy, errors = validate_and_load(raw)

        self.assertIsNone(strategy)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("win_rate must be a float" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
