from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from paid_usage import PaidUsageTracker


class PaidUsageTrackerTests(unittest.TestCase):
    @patch.dict("os.environ", {}, clear=True)
    def test_from_env_disabled_without_api_key(self) -> None:
        tracker = PaidUsageTracker.from_env()
        self.assertFalse(tracker.enabled)

    @patch("paid_usage._PaidSignal")
    @patch("paid_usage._ProductByExternalId")
    @patch("paid_usage._CustomerByExternalId")
    @patch("paid_usage._PaidClient")
    def test_send_usage_record_uses_paid_sdk(
        self,
        mock_paid_client,
        mock_customer,
        mock_product,
        mock_signal,
    ) -> None:
        client = MagicMock()
        mock_paid_client.return_value = client
        tracker = PaidUsageTracker(
            api_key="paid-test-key",
            event_name="eva_by_anyquant",
            external_product_id="product_456",
            timeout_seconds=5,
            enabled=True,
        )

        ok = tracker.send_usage_record(
            external_customer_id="customer_123",
            data={"usage_action": "evaluate_completed", "pitch_id": "pit_abc"},
            idempotency_key="idem-123",
        )

        self.assertTrue(ok)
        mock_paid_client.assert_called_once_with(token="paid-test-key", timeout=5.0)
        mock_customer.assert_called_once_with(externalCustomerId="customer_123")
        mock_product.assert_called_once_with(externalProductId="product_456")
        mock_signal.assert_called_once()
        client.signals.create_signals.assert_called_once()

    @patch("paid_usage._PaidSignal")
    @patch("paid_usage._CustomerByExternalId")
    @patch("paid_usage._PaidClient")
    def test_send_usage_record_returns_false_on_sdk_error(
        self,
        mock_paid_client,
        _mock_customer,
        _mock_signal,
    ) -> None:
        client = MagicMock()
        client.signals.create_signals.side_effect = RuntimeError("boom")
        mock_paid_client.return_value = client
        tracker = PaidUsageTracker(api_key="bad-key", max_retries=0, enabled=True)

        ok = tracker.send_usage_record(external_customer_id="customer_123")

        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
