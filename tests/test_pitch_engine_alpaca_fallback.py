from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import pitch_engine
from pitch_engine import (
    PitchDraft,
    UploadedFile,
    _alpaca_fallback_data_summary,
    _build_validator_payload,
    _candidate_alpaca_timeframes,
    evaluate_pitch,
)


class AlpacaFallbackTests(unittest.TestCase):
    def test_candidate_timeframes_prioritize_intraday_for_short_windows(self) -> None:
        candidates = _candidate_alpaca_timeframes("2026-02-01T00:00:00Z", "2026-02-02T23:59:59Z")
        self.assertGreaterEqual(len(candidates), 3)
        self.assertEqual("1Min", candidates[0])
        self.assertIn("1Day", candidates)

    @patch.object(pitch_engine, "_ALPACA_SKILL_AVAILABLE", True)
    @patch("pitch_engine.run_data_skill")
    def test_fallback_tries_multiple_timeframes_until_success(self, mock_run_data_skill: object) -> None:
        mock_run_data_skill.side_effect = [
            {"status": "fail", "summary": "no minute bars", "bars": []},
            {"status": "fail", "summary": "no minute bars on iex", "bars": []},
            {"status": "fail", "summary": "no 5min bars on sip", "bars": []},
            {
                "status": "ok",
                "summary": "ok",
                "bars": [
                    {"timestamp": "2026-02-01T14:30:00Z", "close": 100.0, "open": 99.0},
                    {"timestamp": "2026-02-01T15:30:00Z", "close": 101.0, "open": 100.0},
                ],
                "artifacts": {"request_ids": ["abc"]},
            },
        ]

        summary, error = _alpaca_fallback_data_summary(
            ticker="AAPL",
            backtest_start="2026-02-01",
            backtest_end="2026-02-02",
        )

        self.assertIsNone(error)
        assert summary is not None
        self.assertEqual("alpaca_fallback", summary.get("source"))
        self.assertEqual("5Min", summary.get("timeframe"))
        self.assertEqual("iex", summary.get("feed"))
        self.assertEqual(2, summary.get("row_count"))
        self.assertIn("attempted_timeframes", summary)
        self.assertEqual(4, mock_run_data_skill.call_count)

    def test_validator_payload_uses_fallback_data_when_frame_missing(self) -> None:
        draft = PitchDraft(
            pitch_id="pit_test",
            created_at="2026-02-21T00:00:00Z",
            thesis="Mean reversion",
            time_horizon="weeks",
            tickers=["AAPL"],
        )
        fallback = {
            "source": "alpaca_fallback",
            "ticker": "AAPL",
            "timeframe": "1Hour",
            "window": {"start": "2026-01-01T00:00:00Z", "end": "2026-02-01T00:00:00Z"},
            "row_count": 120,
            "column_count": 8,
            "columns": ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"],
            "head_records": [{"timestamp": "2026-01-01T10:00:00Z", "close": 100.1}],
            "close_stats": {"count": 120.0, "mean": 110.0, "std": 2.0, "min": 100.0, "max": 115.0},
        }

        payload = _build_validator_payload(
            draft=draft,
            frame=None,
            close_series=pd.Series(dtype=float),
            load_errors=["No CSV/TSV uploaded; fallback used."],
            match_rate=1.0,
            sharpe=1.2,
            max_drawdown=-0.25,
            fallback_data_summary=fallback,
            strategy_file_names=["algo-backend.ipynb"],
            backtest_context={"status": "success", "attempt_count": 1},
        )

        self.assertEqual(120, payload["data_summary"]["row_count"])
        self.assertEqual("alpaca_fallback", payload["data_summary"]["source"])
        self.assertEqual(["algo-backend.ipynb"], payload["pitch"]["strategy_file_names"])
        self.assertEqual("success", payload["execution_context"]["status"])
        self.assertIn("No CSV/TSV uploaded; fallback used.", payload["data_summary"]["load_errors"])

    @patch("pitch_engine._run_llm_validators")
    @patch("pitch_engine._alpaca_fallback_data_summary")
    @patch("pitch_engine.run_backtest_agent")
    def test_evaluate_pitch_script_only_includes_fallback_data_in_validator_payload(
        self,
        mock_backtest_agent: object,
        mock_fallback_summary: object,
        mock_run_llm_validators: object,
    ) -> None:
        captured: dict[str, object] = {}

        def _fake_validators(payload: dict) -> dict:
            captured["payload"] = payload
            return {
                "fabrication_detector": {
                    "status": "ok",
                    "confidence": 0.2,
                    "summary": "clean",
                    "flags": [],
                    "artifacts": {"verdict": "clean", "questions": []},
                    "latency_ms": 1,
                },
                "coding_errors_detector": {
                    "status": "ok",
                    "confidence": 0.2,
                    "summary": "clean",
                    "flags": [],
                    "artifacts": {"verdict": "clean", "questions": []},
                    "latency_ms": 1,
                },
            }

        mock_run_llm_validators.side_effect = _fake_validators
        mock_backtest_agent.return_value = SimpleNamespace(
            status="success",
            metrics={
                "ticker": "AAPL",
                "backtest_start": "2026-01-01",
                "backtest_end": "2026-02-01",
            },
            attempt_count=1,
            message="ok",
        )
        mock_fallback_summary.return_value = (
            {
                "source": "alpaca_fallback",
                "ticker": "AAPL",
                "row_count": 100,
                "column_count": 3,
                "columns": ["timestamp", "close", "volume"],
                "head_records": [{"timestamp": "2026-01-01T00:00:00Z", "close": 100.0, "volume": 10}],
                "close_stats": {"count": 100.0, "mean": 100.0, "std": 1.0, "min": 98.0, "max": 102.0},
            },
            None,
        )

        with TemporaryDirectory() as tmp:
            strategy_path = f"{tmp}/algo.py"
            with open(strategy_path, "w", encoding="utf-8") as handle:
                handle.write("print('strategy')")

            draft = PitchDraft(
                pitch_id="pit_eval",
                created_at="2026-02-21T00:00:00Z",
                thesis="Momentum",
                time_horizon="weeks",
                tickers=["AAPL"],
                uploaded_files=[
                    UploadedFile(
                        file_id="f1",
                        name="algo.py",
                        path=strategy_path,
                        mime_type="text/x-python",
                    )
                ],
            )

            evaluate_pitch(draft=draft, data_fetcher_output={"status": "ok", "summary": "", "confidence": 1, "flags": [], "artifacts": {}, "latency_ms": 0})

        assert "payload" in captured
        payload = captured["payload"]
        assert isinstance(payload, dict)
        self.assertEqual(100, payload["data_summary"]["row_count"])
        self.assertEqual("alpaca_fallback", payload["data_summary"]["source"])
        self.assertEqual(["algo.py"], payload["pitch"]["strategy_file_names"])

    @patch("pitch_engine._run_llm_validators")
    @patch("pitch_engine._alpaca_fallback_data_summary")
    @patch.object(pitch_engine, "_BACKTEST_AGENT_AVAILABLE", False)
    def test_evaluate_pitch_uses_horizon_window_when_backtest_unavailable(
        self,
        mock_fallback_summary: object,
        mock_run_llm_validators: object,
    ) -> None:
        captured: dict[str, object] = {}

        def _fake_validators(payload: dict) -> dict:
            captured["payload"] = payload
            return {
                "fabrication_detector": {
                    "status": "ok",
                    "confidence": 0.2,
                    "summary": "clean",
                    "flags": [],
                    "artifacts": {"verdict": "clean", "questions": []},
                    "latency_ms": 1,
                },
                "coding_errors_detector": {
                    "status": "ok",
                    "confidence": 0.2,
                    "summary": "clean",
                    "flags": [],
                    "artifacts": {"verdict": "clean", "questions": []},
                    "latency_ms": 1,
                },
            }

        mock_run_llm_validators.side_effect = _fake_validators
        mock_fallback_summary.return_value = (
            {
                "source": "alpaca_fallback",
                "ticker": "AAPL",
                "timeframe": "1Hour",
                "window": {"start": "2025-01-01T00:00:00Z", "end": "2026-01-01T00:00:00Z"},
                "row_count": 60,
                "column_count": 3,
                "columns": ["timestamp", "close", "volume"],
                "head_records": [{"timestamp": "2025-01-01T00:00:00Z", "close": 100.0, "volume": 10}],
                "close_stats": {"count": 60.0, "mean": 100.0, "std": 1.0, "min": 98.0, "max": 102.0},
            },
            None,
        )

        with TemporaryDirectory() as tmp:
            strategy_path = f"{tmp}/algo.py"
            with open(strategy_path, "w", encoding="utf-8") as handle:
                handle.write("print('strategy')")

            draft = PitchDraft(
                pitch_id="pit_eval_horizon",
                created_at="2026-02-21T00:00:00Z",
                thesis="Momentum",
                time_horizon="months",
                tickers=["AAPL"],
                uploaded_files=[
                    UploadedFile(
                        file_id="f1",
                        name="algo.py",
                        path=strategy_path,
                        mime_type="text/x-python",
                    )
                ],
            )

            evaluate_pitch(draft=draft, data_fetcher_output={"status": "ok", "summary": "", "confidence": 1, "flags": [], "artifacts": {}, "latency_ms": 0})

        self.assertEqual(1, mock_fallback_summary.call_count)
        kwargs = mock_fallback_summary.call_args.kwargs
        self.assertEqual("AAPL", kwargs["ticker"])
        self.assertTrue(kwargs["backtest_start"])
        self.assertTrue(kwargs["backtest_end"])

        assert "payload" in captured
        payload = captured["payload"]
        assert isinstance(payload, dict)
        self.assertEqual(60, payload["data_summary"]["row_count"])


if __name__ == "__main__":
    unittest.main()
