from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from one_shot_validator import OneShotExtractionResult, evaluate_one_shot_strategy


def _make_extraction(**kwargs) -> OneShotExtractionResult:
    """Build an OneShotExtractionResult with sensible defaults for tests."""
    defaults: dict = {
        "event_type": "causal_chain",
        "event_type_reasoning": "test",
        "column_mappings": {},
        "numeric_params": {},
        "extraction_questions": [],
        "extraction_confidence": 0.9,
        "extraction_latency_ms": 0,
    }
    defaults.update(kwargs)
    return OneShotExtractionResult(**defaults)


class OneShotValidatorTests(unittest.TestCase):
    def _write_csv(self, folder: Path, name: str, frame: pd.DataFrame) -> SimpleNamespace:
        path = folder / name
        frame.to_csv(path, index=False)
        return SimpleNamespace(path=str(path))

    def test_one_shot_valid_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            n1 = 120
            driver = pd.Series(range(n1), dtype=float)
            target = driver * 0.02 + 0.1
            relationship = pd.DataFrame({"wheat_price": driver, "mcd_return": target})

            n2 = 80
            probs = pd.Series([0.7] * 50 + [0.2] * 30, dtype=float)
            outcomes = pd.Series([1] * 35 + [0] * 15 + [1] * 6 + [0] * 24, dtype=float)
            calibration = pd.DataFrame({"forecast_prob": probs, "outcome": outcomes})

            n3 = 30
            severity = pd.Series(range(1, n3 + 1), dtype=float)
            change = severity * 0.05 + 0.2
            magnitude = pd.DataFrame({"drought_severity": severity, "wheat_change": change})

            uploaded_files = [
                self._write_csv(base, "node1.csv", relationship),
                self._write_csv(base, "node2.csv", calibration),
                self._write_csv(base, "node3.csv", magnitude),
            ]

            draft = SimpleNamespace(
                uploaded_files=uploaded_files,
                supporting_notes="causal chain wheat â†’ MCD",
            )

            mock_extraction = _make_extraction(
                event_type="causal_chain",
                column_mappings={
                    "driver": {"file": "node1.csv", "column": "wheat_price"},
                    "asset_return": {"file": "node1.csv", "column": "mcd_return"},
                    "severity": {"file": "node3.csv", "column": "drought_severity"},
                    "magnitude": {"file": "node3.csv", "column": "wheat_change"},
                    "forecast_prob": {"file": "node2.csv", "column": "forecast_prob"},
                    "outcome": {"file": "node2.csv", "column": "outcome"},
                },
                numeric_params={
                    "p_true": 0.65, "p_market": 0.50,
                    "payoff_up": 1.2, "payoff_down": -0.3, "transaction_cost": 0.0,
                },
            )

            with patch("one_shot_validator._extract_one_shot_params", return_value=mock_extraction):
                result = evaluate_one_shot_strategy(draft=draft, min_positive_edge_prob=0.6)
            self.assertEqual(result.recommendation, "VALID")
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.missing_inputs, [])

    def test_one_shot_missing_inputs_not_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            relationship = pd.DataFrame({"wheat_price": [1, 2, 3], "mcd_return": [0.1, 0.2, 0.1]})
            uploaded_files = [self._write_csv(base, "small.csv", relationship)]

            draft = SimpleNamespace(uploaded_files=uploaded_files, supporting_notes="causal pitch")

            # Extraction maps columns but data is too small; params mostly absent
            mock_extraction = _make_extraction(
                event_type="causal_chain",
                column_mappings={
                    "driver": {"file": "small.csv", "column": "wheat_price"},
                    "asset_return": {"file": "small.csv", "column": "mcd_return"},
                },
                numeric_params={"p_true": 0.55},
            )

            with patch("one_shot_validator._extract_one_shot_params", return_value=mock_extraction):
                result = evaluate_one_shot_strategy(draft=draft)
            self.assertEqual(result.recommendation, "NOT_VALID")
            self.assertGreater(len(result.missing_inputs), 0)
            self.assertGreater(len(result.validation_questions), 0)

    def test_binary_event_valid_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            probs = pd.Series([0.75] * 40 + [0.25] * 40, dtype=float)
            outcomes = pd.Series([1] * 31 + [0] * 9 + [1] * 8 + [0] * 32, dtype=float)
            calibration = pd.DataFrame({"forecast_prob": probs, "outcome": outcomes})
            uploaded_files = [self._write_csv(base, "node2.csv", calibration)]
            draft = SimpleNamespace(
                uploaded_files=uploaded_files,
                supporting_notes="binary event FDA approval",
            )

            mock_extraction = _make_extraction(
                event_type="binary_event",
                column_mappings={
                    "forecast_prob": {"file": "node2.csv", "column": "forecast_prob"},
                    "outcome": {"file": "node2.csv", "column": "outcome"},
                },
                numeric_params={
                    "p_true": 0.62, "p_market": 0.50,
                    "payoff_up": 1.0, "payoff_down": -0.2, "transaction_cost": 0.0,
                },
            )

            with patch("one_shot_validator._extract_one_shot_params", return_value=mock_extraction):
                result = evaluate_one_shot_strategy(draft=draft, min_positive_edge_prob=0.6)
            self.assertEqual(result.recommendation, "VALID")
            self.assertEqual(result.artifacts.get("event_type"), "binary_event")

    def test_deal_spread_valid_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            probs = pd.Series([0.7] * 50 + [0.3] * 30, dtype=float)
            outcomes = pd.Series([1] * 37 + [0] * 13 + [1] * 10 + [0] * 20, dtype=float)
            calibration = pd.DataFrame({"forecast_prob": probs, "outcome": outcomes})
            uploaded_files = [self._write_csv(base, "node2.csv", calibration)]
            draft = SimpleNamespace(
                uploaded_files=uploaded_files,
                supporting_notes="deal spread acquisition",
            )

            mock_extraction = _make_extraction(
                event_type="deal_spread",
                column_mappings={
                    "forecast_prob": {"file": "node2.csv", "column": "forecast_prob"},
                    "outcome": {"file": "node2.csv", "column": "outcome"},
                },
                numeric_params={
                    "p_close": 0.78, "current_price": 45.0,
                    "price_if_close": 55.0, "price_if_break": 35.0, "transaction_cost": 0.0,
                },
            )

            with patch("one_shot_validator._extract_one_shot_params", return_value=mock_extraction):
                result = evaluate_one_shot_strategy(draft=draft, min_positive_edge_prob=0.6)
            self.assertEqual(result.recommendation, "VALID")
            self.assertEqual(result.artifacts.get("event_type"), "deal_spread")


if __name__ == "__main__":
    unittest.main()

