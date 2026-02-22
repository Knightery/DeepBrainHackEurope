from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pitch_engine import _review_download_match_with_llm, _run_llm_validator, _run_llm_validators


def _fake_response(text: str, parsed=None):
    return SimpleNamespace(text=text, parsed=parsed)


class ValidatorRetryTests(unittest.TestCase):
    @patch("pitch_engine.ThreadPoolExecutor")
    def test_run_llm_validators_forces_parallel_pool(self, mock_pool_cls: Mock) -> None:
        fake_pool = Mock()
        mock_pool_cls.return_value.__enter__.return_value = fake_pool

        fabrication_future = Mock()
        coding_future = Mock()
        fabrication_future.result.return_value = {"status": "ok", "flags": []}
        coding_future.result.return_value = {"status": "ok", "flags": []}
        fake_pool.submit.side_effect = [fabrication_future, coding_future]

        result = _run_llm_validators({"payload": 1})

        mock_pool_cls.assert_called_once_with(max_workers=2)
        self.assertIn("fabrication_detector", result)
        self.assertIn("coding_errors_detector", result)

    @patch.dict(
        "os.environ",
        {
            "GEMINI_API_KEY": "test-key",
            "VALIDATOR_LLM_MAX_ATTEMPTS": "2",
            "VALIDATOR_LLM_RETRY_DELAY_SECONDS": "0",
        },
        clear=False,
    )
    @patch("pitch_engine.genai.Client")
    def test_run_llm_validator_retries_then_succeeds(self, mock_client_cls: Mock) -> None:
        valid_payload = {
            "summary": "ok",
            "confidence": 0.9,
            "flags": [],
            "artifacts": {"verdict": "clean", "questions": []},
        }
        generate_mock = Mock(
            side_effect=[
                _fake_response("not-json", None),
                _fake_response(json.dumps(valid_payload), None),
            ]
        )
        mock_client_cls.return_value = SimpleNamespace(models=SimpleNamespace(generate_content=generate_mock))

        result = _run_llm_validator("coding_errors_detector", "prompt", {"x": 1})
        self.assertEqual("ok", result.get("status"))
        self.assertEqual("ok", result.get("summary"))
        self.assertEqual(2, generate_mock.call_count)

    @patch.dict(
        "os.environ",
        {
            "GEMINI_API_KEY": "test-key",
            "VALIDATOR_LLM_MAX_ATTEMPTS": "2",
            "VALIDATOR_LLM_RETRY_DELAY_SECONDS": "0",
        },
        clear=False,
    )
    @patch("pitch_engine.genai.Client")
    def test_run_llm_validator_returns_error_after_exhausted_retries(self, mock_client_cls: Mock) -> None:
        generate_mock = Mock(side_effect=[_fake_response("bad", None), _fake_response("still bad", None)])
        mock_client_cls.return_value = SimpleNamespace(models=SimpleNamespace(generate_content=generate_mock))

        with self.assertRaises(RuntimeError) as ctx:
            _run_llm_validator("fabrication_detector", "prompt", {"x": 1})
        self.assertIn("after 2 LLM attempt(s)", str(ctx.exception))
        self.assertEqual(2, generate_mock.call_count)

    @patch.dict(
        "os.environ",
        {
            "GEMINI_API_KEY": "test-key",
            "CUA_MATCH_REVIEW_MAX_ATTEMPTS": "2",
            "CUA_MATCH_REVIEW_RETRY_DELAY_SECONDS": "0",
        },
        clear=False,
    )
    @patch("pitch_engine.genai.Client")
    def test_review_download_match_retries_then_parses(self, mock_client_cls: Mock) -> None:
        parsed_block = (
            "<download_match>"
            "{\"verdict\":\"match\",\"confidence\":0.95,\"best_candidate\":\"candidate.csv\","
            "\"reason\":\"good\",\"retry_guidance\":\"none\"}"
            "</download_match>"
        )
        generate_mock = Mock(side_effect=[_fake_response("invalid"), _fake_response(parsed_block)])
        mock_client_cls.return_value = SimpleNamespace(models=SimpleNamespace(generate_content=generate_mock))

        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / "ref.csv"
            candidate = Path(tmp) / "candidate.csv"
            reference.write_text("date,close\n2026-01-01,1\n", encoding="utf-8")
            candidate.write_text("date,close\n2026-01-01,1\n", encoding="utf-8")

            result = _review_download_match_with_llm(reference, [candidate], "note")

        self.assertEqual("match", result.get("verdict"))
        self.assertEqual("candidate.csv", result.get("best_candidate"))
        self.assertEqual(2, generate_mock.call_count)


if __name__ == "__main__":
    unittest.main()
