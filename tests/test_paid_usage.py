from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from paid_usage import PaidUsageTracker


class _FakeResponse:
    def __init__(self, status: int = 202) -> None:
        self.status = status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class PaidUsageTrackerTests(unittest.TestCase):
    @patch.dict("os.environ", {}, clear=True)
    def test_from_env_disabled_without_api_key(self) -> None:
        tracker = PaidUsageTracker.from_env()
        self.assertFalse(tracker.enabled)

    @patch("paid_usage.urllib.request.urlopen")
    def test_send_usage_record_posts_expected_bulk_payload(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeResponse(status=202)
        tracker = PaidUsageTracker(
            api_key="paid-test-key",
            event_name="eva_by_anyquant",
            external_product_id="product_456",
            enabled=True,
        )

        ok = tracker.send_usage_record(
            external_customer_id="customer_123",
            data={"usage_action": "evaluate_completed", "pitch_id": "pit_abc"},
            idempotency_key="idem-123",
        )

        self.assertTrue(ok)
        req = mock_urlopen.call_args.args[0]
        body = json.loads(req.data.decode("utf-8"))

        self.assertEqual("https://api.paid.ai/v2/usage/bulk", req.full_url)
        self.assertEqual("Bearer paid-test-key", req.headers.get("Authorization"))
        self.assertEqual("application/json", req.headers.get("Content-type"))
        self.assertEqual("eva_by_anyquant", body["usageRecords"][0]["event_name"])
        self.assertEqual("customer_123", body["usageRecords"][0]["external_customer_id"])
        self.assertEqual("product_456", body["usageRecords"][0]["external_product_id"])
        self.assertEqual("idem-123", body["usageRecords"][0]["idempotency_key"])

    @patch("paid_usage.urllib.request.urlopen")
    def test_send_usage_record_handles_http_error(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.paid.ai/v2/usage/bulk",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"bad key"}'),
        )
        tracker = PaidUsageTracker(api_key="bad-key", enabled=True)

        ok = tracker.send_usage_record(external_customer_id="customer_123")

        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
