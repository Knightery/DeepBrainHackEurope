from __future__ import annotations

import json
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pitch_engine import (
    PitchDraft,
    UploadedFile,
    _review_download_match_with_llm,
    _run_llm_validator,
    _run_llm_validators,
    validate_data_with_cua,
)


def _fake_response(text: str, parsed=None):
    return SimpleNamespace(text=text, parsed=parsed)


class ValidatorRetryTests(unittest.TestCase):
    def test_validate_data_with_cua_returns_on_matched_warn_attempt(self) -> None:
        class _FakeProc:
            def __init__(self) -> None:
                self.stdout = io.StringIO("cua-log-line\n")
                self.returncode = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            reference = tmp_path / "reference.csv"
            reference.write_text("date,spread_pct\n2026-01-01,1.23\n", encoding="utf-8")
            downloads_dir = tmp_path / "downloads"
            downloads_dir.mkdir(parents=True, exist_ok=True)

            draft = PitchDraft(
                pitch_id="pit_test_cua_warn_match",
                created_at="2026-02-22T00:00:00Z",
                thesis="test",
                time_horizon="years",
                tickers=["TLT"],
                source_urls=["https://fred.stlouisfed.org/series/T10Y2Y"],
                uploaded_files=[
                    UploadedFile(
                        file_id="fil_reference",
                        name=reference.name,
                        path=str(reference),
                        mime_type="text/csv",
                        size_bytes=reference.stat().st_size,
                    )
                ],
            )

            parsed_output = {
                "status": "success",
                "summary": "CUA run completed.",
                "downloaded_files": [],
                "validation": {
                    "issues": [],
                    "advisories": [{"code": "NO_GET_PAGE_TEXT", "message": "advisory only"}],
                },
            }
            match_review = {
                "verdict": "match",
                "confidence": 0.95,
                "best_candidate": "candidate.csv",
                "reason": "Matched reference schema and values.",
                "retry_guidance": "",
            }

            with (
                patch.dict("os.environ", {"CUA_MAX_ATTEMPTS": "3"}, clear=False),
                patch("pitch_engine._cua_downloads_dir", return_value=downloads_dir),
                patch("pitch_engine.subprocess.Popen", return_value=_FakeProc()) as mock_popen,
                patch("pitch_engine._extract_json_after_separator", return_value=parsed_output),
                patch("pitch_engine._resolve_downloaded_host_paths", return_value=[]),
                patch("pitch_engine._review_download_match_with_llm", return_value=match_review),
            ):
                result = validate_data_with_cua(draft, reference.name, notes="test")

        self.assertEqual("warn", result.get("status"))
        self.assertEqual("match", (result.get("artifacts") or {}).get("match_review", {}).get("verdict"))
        self.assertEqual(1, mock_popen.call_count)

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
