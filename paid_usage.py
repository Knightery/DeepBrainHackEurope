from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

PAID_USAGE_BULK_ENDPOINT = "https://api.paid.ai/v2/usage/bulk"
_LOGGER = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class PaidUsageTracker:
    def __init__(
        self,
        *,
        api_key: str,
        event_name: str = "eva_by_anyquant",
        external_product_id: str | None = None,
        endpoint: str = PAID_USAGE_BULK_ENDPOINT,
        timeout_seconds: float = 5.0,
        enabled: bool | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.event_name = (event_name or "eva_by_anyquant").strip() or "eva_by_anyquant"
        self.external_product_id = (external_product_id or "").strip() or None
        self.endpoint = (endpoint or PAID_USAGE_BULK_ENDPOINT).strip() or PAID_USAGE_BULK_ENDPOINT
        self.timeout_seconds = timeout_seconds
        self.enabled = bool(self.api_key) if enabled is None else bool(enabled and self.api_key)

    @classmethod
    def from_env(cls, default_event_name: str = "eva_by_anyquant") -> "PaidUsageTracker":
        api_key = os.getenv("PAID_API_KEY", "").strip()
        event_name = os.getenv("PAID_EVENT_NAME", default_event_name).strip() or default_event_name
        product_id = os.getenv("PAID_EXTERNAL_PRODUCT_ID", "").strip() or None
        endpoint = os.getenv("PAID_USAGE_ENDPOINT", PAID_USAGE_BULK_ENDPOINT).strip() or PAID_USAGE_BULK_ENDPOINT
        timeout_raw = os.getenv("PAID_USAGE_TIMEOUT_SECONDS", "5").strip()
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError:
            timeout_seconds = 5.0
        enabled = _env_bool("PAID_USAGE_ENABLED", default=bool(api_key))
        return cls(
            api_key=api_key,
            event_name=event_name,
            external_product_id=product_id,
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            enabled=enabled,
        )

    def send_usage_record(
        self,
        *,
        external_customer_id: str | None = None,
        external_product_id: str | None = None,
        data: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> bool:
        if not self.enabled:
            return False

        record: dict[str, Any] = {"event_name": self.event_name}
        if external_customer_id:
            record["external_customer_id"] = external_customer_id

        product_id = (external_product_id or self.external_product_id or "").strip()
        if product_id:
            record["external_product_id"] = product_id

        if data:
            record["data"] = data
        if idempotency_key:
            record["idempotency_key"] = idempotency_key

        body = json.dumps({"usageRecords": [record]}).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                status = int(getattr(resp, "status", 200))
                if 200 <= status < 300:
                    return True
                _LOGGER.warning("Paid usage call returned non-success status: %s", status)
                return False
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:250]
            except Exception:
                detail = ""
            _LOGGER.warning("Paid usage request failed with HTTP %s: %s", exc.code, detail)
            return False
        except urllib.error.URLError as exc:
            _LOGGER.warning("Paid usage request failed: %s", exc)
            return False

    async def send_usage_record_async(
        self,
        *,
        external_customer_id: str | None = None,
        external_product_id: str | None = None,
        data: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> bool:
        return await asyncio.to_thread(
            self.send_usage_record,
            external_customer_id=external_customer_id,
            external_product_id=external_product_id,
            data=data,
            idempotency_key=idempotency_key,
        )
