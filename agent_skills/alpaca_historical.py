from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

DEFAULT_DATA_BASE_URL = "https://data.alpaca.markets"
DEFAULT_TIMEOUT_SECONDS = 20
MAX_LIMIT = 10000
DEFAULT_LIMIT = 1000
DEFAULT_MAX_PAGES = 50
DEFAULT_TIMEFRAME = "1Day"
VALID_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.\-]{0,14}$")


@dataclass
class AlpacaHistoricalBarsRequest:
    symbol: str
    start: str
    end: str | None = None
    timeframe: str = DEFAULT_TIMEFRAME
    feed: str | None = None
    adjustment: str | None = "raw"
    sort: str = "asc"
    limit: int = DEFAULT_LIMIT
    currency: str | None = None
    max_pages: int = DEFAULT_MAX_PAGES


def _normalize_symbol(raw: Any) -> str:
    symbol = str(raw or "").strip().upper()
    if not symbol:
        raise ValueError("`symbol` is required.")
    if not VALID_SYMBOL.fullmatch(symbol):
        raise ValueError(f"Invalid symbol `{symbol}`.")
    return symbol


def _normalize_datetime(value: Any, field_name: str) -> str:
    if value is None:
        if field_name == "end":
            return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        raise ValueError("`start` is required.")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    text = str(value).strip()
    if not text:
        raise ValueError(f"`{field_name}` cannot be empty.")
    return text


def _normalize_limit(value: Any) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = DEFAULT_LIMIT
    return max(1, min(MAX_LIMIT, limit))


def _normalize_max_pages(value: Any) -> int:
    try:
        pages = int(value)
    except (TypeError, ValueError):
        pages = DEFAULT_MAX_PAGES
    return max(1, min(200, pages))


def _auth_headers() -> dict[str, str]:
    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    api_secret = os.getenv("ALPACA_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET are required.")
    return {
        "Accept": "application/json",
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
        "User-Agent": "quant-pitch-evaluator/alpaca-skill",
    }


def _http_get_json(url: str, headers: dict[str, str], timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> tuple[dict[str, Any], str | None]:
    request = Request(url=url, headers=headers, method="GET")
    with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310 - fixed host from configuration
        request_id = response.headers.get("x-request-id")
        body = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError("Alpaca API response was not a JSON object.")
    return parsed, request_id


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_bar(symbol: str, raw_bar: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "timestamp": str(raw_bar.get("t", "")),
        "open": _as_float(raw_bar.get("o")),
        "high": _as_float(raw_bar.get("h")),
        "low": _as_float(raw_bar.get("l")),
        "close": _as_float(raw_bar.get("c")),
        "volume": _as_int(raw_bar.get("v")),
        "trade_count": _as_int(raw_bar.get("n")),
        "vwap": _as_float(raw_bar.get("vw")),
    }


def _build_request(params: dict[str, Any]) -> AlpacaHistoricalBarsRequest:
    symbol = _normalize_symbol(params.get("symbol"))
    start = _normalize_datetime(params.get("start"), "start")
    end = _normalize_datetime(params.get("end"), "end")
    timeframe = str(params.get("timeframe", DEFAULT_TIMEFRAME) or DEFAULT_TIMEFRAME).strip()
    if not timeframe:
        raise ValueError("`timeframe` cannot be empty.")
    feed = params.get("feed")
    adjustment = params.get("adjustment", "raw")
    sort = str(params.get("sort", "asc") or "asc").strip().lower()
    if sort not in {"asc", "desc"}:
        raise ValueError("`sort` must be `asc` or `desc`.")
    currency = params.get("currency")
    return AlpacaHistoricalBarsRequest(
        symbol=symbol,
        start=start,
        end=end,
        timeframe=timeframe,
        feed=str(feed).strip() if feed is not None and str(feed).strip() else None,
        adjustment=str(adjustment).strip() if adjustment is not None and str(adjustment).strip() else None,
        sort=sort,
        limit=_normalize_limit(params.get("limit", DEFAULT_LIMIT)),
        currency=str(currency).strip() if currency is not None and str(currency).strip() else None,
        max_pages=_normalize_max_pages(params.get("max_pages", DEFAULT_MAX_PAGES)),
    )


def fetch_alpaca_historical_bars(params: dict[str, Any]) -> dict[str, Any]:
    skill_name = "alpaca_historical_bars"
    try:
        request_model = _build_request(params)
        headers = _auth_headers()
        base_url = os.getenv("ALPACA_DATA_BASE_URL", DEFAULT_DATA_BASE_URL).strip() or DEFAULT_DATA_BASE_URL

        page_token: str | None = None
        page_count = 0
        request_ids: list[str] = []
        bars: list[dict[str, Any]] = []
        truncated = False

        while True:
            query: dict[str, Any] = {
                "timeframe": request_model.timeframe,
                "start": request_model.start,
                "end": request_model.end,
                "limit": request_model.limit,
                "sort": request_model.sort,
            }
            if request_model.feed:
                query["feed"] = request_model.feed
            if request_model.adjustment:
                query["adjustment"] = request_model.adjustment
            if request_model.currency:
                query["currency"] = request_model.currency
            if page_token:
                query["page_token"] = page_token

            endpoint = f"{base_url.rstrip('/')}/v2/stocks/{request_model.symbol}/bars"
            url = f"{endpoint}?{urlencode(query)}"
            payload, request_id = _http_get_json(url=url, headers=headers)
            if request_id:
                request_ids.append(request_id)

            page_bars = payload.get("bars", [])
            if isinstance(page_bars, list):
                bars.extend(_normalize_bar(request_model.symbol, item) for item in page_bars if isinstance(item, dict))

            page_count += 1
            page_token = payload.get("next_page_token")
            if not isinstance(page_token, str) or not page_token:
                break

            if page_count >= request_model.max_pages:
                truncated = True
                break

        return {
            "skill": skill_name,
            "status": "ok",
            "summary": f"Fetched {len(bars)} bar(s) for {request_model.symbol} from Alpaca.",
            "bars": bars,
            "artifacts": {
                "symbol": request_model.symbol,
                "start": request_model.start,
                "end": request_model.end,
                "timeframe": request_model.timeframe,
                "feed": request_model.feed or "",
                "adjustment": request_model.adjustment or "",
                "sort": request_model.sort,
                "limit": request_model.limit,
                "page_count": page_count,
                "truncated": truncated,
                "request_ids": request_ids,
            },
        }
    except HTTPError as exc:
        message = f"Alpaca HTTP error {exc.code}."
        try:
            raw = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw)
            detail = parsed.get("message") if isinstance(parsed, dict) else ""
            if detail:
                message = f"{message} {detail}"
        except Exception:
            pass
        return {
            "skill": skill_name,
            "status": "fail",
            "summary": message,
            "bars": [],
            "artifacts": {"error_type": "http_error", "status_code": exc.code},
        }
    except URLError as exc:
        return {
            "skill": skill_name,
            "status": "fail",
            "summary": f"Network error while calling Alpaca: {exc.reason}",
            "bars": [],
            "artifacts": {"error_type": "network_error"},
        }
    except Exception as exc:
        return {
            "skill": skill_name,
            "status": "fail",
            "summary": str(exc) or f"{exc.__class__.__name__} while fetching Alpaca bars.",
            "bars": [],
            "artifacts": {"error_type": exc.__class__.__name__},
        }
