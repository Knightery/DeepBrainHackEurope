from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from app import _copy_to_pitch_uploads
from backtest_agent import _normalize_staged_files, _phase_run
from pitch_engine import PitchDraft


class SecurityHardeningTests(unittest.TestCase):
    def test_copy_to_pitch_uploads_sanitizes_and_contains_path(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                source_path = Path(tmp) / "payload.csv"
                source_path.write_text("x\n1\n", encoding="utf-8")

                draft = PitchDraft(pitch_id="pit_secure", created_at="2026-01-01T00:00:00Z")
                uploaded = _copy_to_pitch_uploads(
                    draft=draft,
                    src_path=source_path,
                    original_name="..\\escape.csv",
                    mime_type="text/csv",
                    size=0,
                )

                uploads_dir = (Path("data/pitches") / "pit_secure" / "uploads").resolve()
                stored_path = Path(uploaded.path).resolve()
                self.assertTrue(stored_path.is_file())
                self.assertTrue(str(stored_path).startswith(str(uploads_dir)))
                self.assertEqual("escape.csv", uploaded.name)
            finally:
                os.chdir(original_cwd)

    def test_phase_run_rejects_unsafe_staged_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _phase_run(
                    script="print('ok')",
                    strategy_files=[("..\\outside.py", "print('x')")],
                    data_files=[],
                    tmp_dir=Path(tmp),
                )

    def test_normalize_staged_files_sanitizes_and_deduplicates(self) -> None:
        files = [
            ("..\\alpha.py", "print(1)"),
            ("../alpha.py", "print(2)"),
            ("", "print(3)"),
        ]
        normalized = _normalize_staged_files(files, "strategy")
        names = [name for name, _ in normalized]
        self.assertEqual("alpha.py", names[0])
        self.assertEqual("alpha_1.py", names[1])
        self.assertTrue(names[2].startswith("strategy_3"))


if __name__ == "__main__":
    unittest.main()
