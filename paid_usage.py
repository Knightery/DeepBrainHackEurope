from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any

try:
    from paid import Paid as _PaidClient
    from paid.types import CustomerByExternalId as _CustomerByExternalId
    from paid.types import ProductByExternalId as _ProductByExternalId
    from paid.types import Signal as _PaidSignal
except Exception:
    _PaidClient = None
    _CustomerByExternalId = None
    _ProductByExternalId = None
    _PaidSignal = None

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
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.35,
        dns_error_cooldown_seconds: float = 45.0,
        use_sdk: bool = True,
        enabled: bool | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.event_name = (event_name or "eva_by_anyquant").strip() or "eva_by_anyquant"
        self.external_product_id = (external_product_id or "").strip() or None
        self.endpoint = (endpoint or PAID_USAGE_BULK_ENDPOINT).strip() or PAID_USAGE_BULK_ENDPOINT
        self.timeout_seconds = max(float(timeout_seconds), 0.1)
        self.max_retries = max(int(max_retries), 0)
        self.retry_backoff_seconds = max(float(retry_backoff_seconds), 0.0)
        self.dns_error_cooldown_seconds = max(float(dns_error_cooldown_seconds), 0.0)
        self.use_sdk = bool(use_sdk)
        self.enabled = bool(self.api_key) if enabled is None else bool(enabled and self.api_key)
        self._dns_cooldown_until_monotonic = 0.0
        self._paid_client = _PaidClient(token=self.api_key) if (_PaidClient and self.api_key and self.use_sdk) else None

    @classmethod
    def from_env(cls, default_event_name: str = "eva_by_anyquant") -> "PaidUsageTracker":
        api_key = os.getenv("PAID_API_KEY", "").strip()
        event_name = os.getenv("PAID_EVENT_NAME", default_event_name).strip() or default_event_name
        product_id = (
            os.getenv("PAID_EXTERNAL_PRODUCT_ID", "").strip()
            or os.getenv("PAID_PRODUCT_ID", "").strip()
            or None
        )
        endpoint = os.getenv("PAID_USAGE_ENDPOINT", PAID_USAGE_BULK_ENDPOINT).strip() or PAID_USAGE_BULK_ENDPOINT
        timeout_raw = os.getenv("PAID_USAGE_TIMEOUT_SECONDS", "5").strip()
        retries_raw = os.getenv("PAID_USAGE_MAX_RETRIES", "1").strip()
        backoff_raw = os.getenv("PAID_USAGE_RETRY_BACKOFF_SECONDS", "0.35").strip()
        dns_cooldown_raw = os.getenv("PAID_USAGE_DNS_ERROR_COOLDOWN_SECONDS", "45").strip()
        use_sdk = _env_bool("PAID_USE_SDK", default=True)
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError:
            timeout_seconds = 5.0
        try:
            max_retries = int(retries_raw)
        except ValueError:
            max_retries = 1
        try:
            retry_backoff_seconds = float(backoff_raw)
        except ValueError:
            retry_backoff_seconds = 0.35
        try:
            dns_error_cooldown_seconds = float(dns_cooldown_raw)
        except ValueError:
            dns_error_cooldown_seconds = 45.0
        enabled = _env_bool("PAID_USAGE_ENABLED", default=bool(api_key))
        return cls(
            api_key=api_key,
            event_name=event_name,
            external_product_id=product_id,
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            dns_error_cooldown_seconds=dns_error_cooldown_seconds,
            use_sdk=use_sdk,
            enabled=enabled,
        )

    @staticmethod
    def _is_dns_resolution_error(exc: urllib.error.URLError) -> bool:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, socket.gaierror):
            return True
        reason_text = str(reason or exc).lower()
        return "getaddrinfo failed" in reason_text or "name or service not known" in reason_text

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
        if time.monotonic() < self._dns_cooldown_until_monotonic:
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

        if self._paid_client is not None:
            try:
                if _PaidSignal is None or _CustomerByExternalId is None:
                    raise RuntimeError("Paid SDK types are unavailable.")
                customer_external_id = record.get("external_customer_id")
                if not customer_external_id:
                    raise ValueError("Paid signal requires external_customer_id.")
                attribution = None
                product_external_id = record.get("external_product_id")
                if product_external_id and _ProductByExternalId is not None:
                    attribution = _ProductByExternalId(externalProductId=product_external_id)
                signal_kwargs: dict[str, Any] = {
                    "eventName": record["event_name"],
                    "customer": _CustomerByExternalId(externalCustomerId=customer_external_id),
                    "attribution": attribution,
                    "data": record.get("data"),
                }
                idempotency_value = record.get("idempotency_key")
                if idempotency_value:
                    signal_kwargs["idempotencyKey"] = idempotency_value
                signal = _PaidSignal(
                    **signal_kwargs,
                )
                self._paid_client.signals.create_signals(signals=[signal])
                return True
            except Exception as exc:
                # Keep legacy HTTP path as resilient fallback for SDK/network quirks.
                _LOGGER.warning(
                    "Paid SDK signal send failed (%s: %s); falling back to HTTP endpoint.",
                    exc.__class__.__name__,
                    exc,
                )

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
        for attempt in range(self.max_retries + 1):
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
                if self._is_dns_resolution_error(exc):
                    self._dns_cooldown_until_monotonic = (
                        time.monotonic() + self.dns_error_cooldown_seconds
                    )
                    _LOGGER.warning(
                        "Paid usage DNS resolution failed: %s. "
                        "Suppressing paid-usage sends for %.1fs.",
                        exc,
                        self.dns_error_cooldown_seconds,
                    )
                    return False
                is_last_attempt = attempt >= self.max_retries
                if is_last_attempt:
                    _LOGGER.warning("Paid usage request failed: %s", exc)
                    return False
                sleep_seconds = self.retry_backoff_seconds * (attempt + 1)
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
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
