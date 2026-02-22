from __future__ import annotations

import unittest
from unittest.mock import Mock

from backtest_agent import _phase_review


class BacktestAgentAlpacaMarkerTests(unittest.TestCase):
    def test_phase_review_requires_alpaca_fetch_marker(self) -> None:
        client = Mock()
        verdict, feedback, message = _phase_review(
            client=client,
            stdout='{"cagr": 0.1}',
            stderr="runner completed",
            returncode=0,
            script="print('hello')",
        )

        self.assertEqual("agent_fault", verdict)
        self.assertIn("ALPACA_FETCH_OK", feedback)
        self.assertEqual("", message)
        client.messages.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
