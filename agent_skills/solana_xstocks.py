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

DEFAULT_BIRDEYE_BASE_URL = "https://public-api.birdeye.so"
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_INTERVAL = "1D"

VALID_MINT = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")

XSTOCKS_SOLANA_MINTS: dict[str, str] = {
    "AAPLX": "XsbEhLAtcf6HdfpFZ5xEMdqW8nfAvcsP5bdudRLJzJp",
    "ABBVX": "XswbinNKyPmzTa5CskMbCPvMW6G5CMnZXZEeQSSQoie",
    "ABTX": "XsHtf5RpxsQ7jeJ9ivNewouZKJHbPxhPoEy6yYvULr7",
    "ACNX": "Xs5UJzmCRQ8DWZjskExdSQDnbE6iLkRu2jjrRAB1JSU",
    "AMBRX": "XsaQTCgebC2KPbf27KUhdv5JFvHhQ4GDAPURwrEhAzb",
    "AMDX": "XsXcJ6GZ9kVnjqGsjBnktRcuwMBmvKWh8S93RefZ1rF",
    "AMZNX": "Xs3eBt7uRfJX8QUs4suhyU8p2M6DoUDrJyWBa8LLZsg",
    "APPX": "XsPdAVBi8Zc1xvv53k4JcMrQaEDTgkGqKYeh7AYgPHV",
    "AVGOX": "XsgSaSvNSqLTtFuyWPBhK9196Xb9Bbdyjj4fH3cPJGo",
    "AZNX": "Xs3ZFkPYT2BN7qBMqf1j1bfTeTm1rFzEFSsQ1z3wAKU",
    "BACX": "XswsQk4duEQmCbGzfqUUWYmi7pV7xpJ9eEmLHXCaEQP",
    "BMNRX": "XsrBCwaH8c46xiqXBChzobgufRKxQxAWUWbndgBNzFn",
    "BRK.BX": "Xs6B6zawENwAbWVi7w92rjazLuAr5Az59qgWKcNb45x",
    "BTBTX": "XsPLBFy59Q3hY59KLAJur8QyvziMF4xUxGTxXqXE7cT",
    "BTGOX": "XsvHMmbDcd14DHHW16PkxPGW7ks77ehxUv1E9Zmxgj4",
    "CMCSAX": "XsvKCaNsxg2GN8jjUmq71qukMJr7Q1c5R2Mk9P8kcS8",
    "COINX": "Xs7ZdzSHLU9ftNJsii5fCeJhoRWSC32SQGzGQtePxNu",
    "COPXX": "XsybfiKkD4UmjkAGT2uR8X2sq9AWFtvGJM2KTffoALZ",
    "CRCLX": "XsueG8BtpquVJX9LVLLEGuViXUungE6WmK5YZ3p3bd1",
    "CRMX": "XsczbcQ3zfcgAEt9qHQES8pxKAVG5rujPSHQEXi4kaN",
    "CRWDX": "Xs7xXqkcK7K8urEqGg52SECi79dRp2cEKKuYjUePYDw",
    "CSCOX": "Xsr3pdLQyXvDJBFgpR5nexCEZwXvigb8wbPYp4YoNFf",
    "CVXX": "XsNNMt7WTNA2sV3jrb1NNfNgapxRF5i4i6GcnTRRHts",
    "DFDVX": "Xs2yquAgsHByNzx68WJC55WHjHBvG9JsMB7CWjTLyPy",
    "DHRX": "Xseo8tgCZfkHxWS9xbFYeKFyMSbWEvZGFV1Gh53GtCV",
    "GLDX": "Xsv9hRk1z5ystj9MhnA7Lq4vjSsLwzL2nxrwmwtD3re",
    "GMEX": "Xsf9mBktVB9BSU5kf4nHxPq5hCBJ2j2ui3ecFGxPRGc",
    "GOOGLX": "XsCPL9dNWBMvFtTmwcCA5v3xWPSMEBCszbQdiLLq6aN",
    "GSX": "XsgaUyp4jd1fNBCxgtTKkW64xnnhQcvgaxzsbAq5ZD1",
    "HDX": "XszjVtyhowGjSC5odCqBpW1CtXXwXjYokymrk7fGKD3",
    "HONX": "XsRbLZthfABAPAfumWNEJhPyiKDW6TvDVeAeW7oKqA2",
    "HOODX": "XsvNBAYkrDRNhA7wPHQfX3ZUXZyZLdnCQDfHZ56bzpg",
    "IBMX": "XspwhyYPdWVM8XBHZnpS9hgyag9MKjLRyE3tVfmCbSr",
    "IEMGX": "XsFnZawJdLdXfBSEt5Vw29K5vdBiHotdPLjUPafpfHs",
    "IJRX": "XsyZcb97BzETAqi9BoP2C9D196MiMNBisGMVNje2Thz",
    "INTCX": "XshPgPdXFRWB8tP1j82rebb2Q9rPgGX37RuqzohmArM",
    "IWMX": "XsbELVbLGBkn7xfMfyYuUipKGt1iRUc2B7pYRvFTFu3",
    "JNJX": "XsGVi5eo1Dh2zUpic4qACcjuWGjNv8GCt3dm5XcX6Dn",
    "JPMX": "XsMAqkcKsUewDrzVkait4e5u4y8REgtyS7jWgCpLV2C",
    "KOX": "XsaBXg8dU5cPM6ehmVctMkVqoiRG2ZjMo1cyBJ3AykQ",
    "KRAQX": "XsAiRejKuvLAdq9KtedrMSrabz7SWdzKoVK6Qgac1Ki",
    "LINX": "XsSr8anD1hkvNMu8XQiVcmiaTP7XGvYu7Q58LdmtE8Z",
    "LLYX": "Xsnuv4omNoHozR6EEW5mXkw8Nrny5rB3jVfLqi6gKMH",
    "MAX": "XsApJFV9MAktqnAc6jqzsHVujxkGm9xcSUffaBoYLKC",
    "MCDX": "XsqE9cRRpzxcGKDXj1BJ7Xmg4GRhZoyY1KpmGSxAWT2",
    "MDTX": "XsDgw22qRLTv5Uwuzn6T63cW69exG41T6gwQhEK22u2",
    "METAX": "Xsa62P5mvPszXL1krVUnU5ar38bBSVcWAB6fmPCo5Zu",
    "MRKX": "XsnQnU7AdbRZYe2akqqpibDdXjkieGFfSkbkjX1Sd1X",
    "MRVLX": "XsuxRGDzbLjnJ72v74b7p9VY6N66uYgTCyfwwRjVCJA",
    "MSFTX": "XspzcW1PRtgf6Wj92HCiZdjzKCyFekVD8P5Ueh3dRMX",
    "MSTRX": "XsP7xzNPvEHS1m6qfanPUGjNmdnmsLKEoNAnHjdxxyZ",
    "NFLXX": "XsEH7wWfJJu2ZT3UCFeVfALnVA6CP5ur7Ee11KmzVpL",
    "NVDAX": "Xsc9qvGR1efVDFGLrVsmkzv3qi45LTBjeUKSPmx9qEh",
    "NVOX": "XsfAzPzYrYjd4Dpa9BU3cusBsvWfVB9gBcyGC87S57n",
    "OPENX": "XsGtpmjhmC8kyjVSWL4VicGu36ceq9u55PTgF8bhGv6",
    "ORCLX": "XsjFwUPiLofddX5cWFHW35GCbXcSu1BCUGfxoQAQjeL",
    "PALLX": "XsTTtPA5V19YwHKDv4xeVXNM6kdsQNJvg3MyWkRUckt",
    "PEPX": "Xsv99frTRUeornyvCfvhnDesQDWuvns1M852Pez91vF",
    "PFEX": "XsAtbqkAP1HJxy7hFDeq7ok6yM43DQ9mQ1Rh861X8rw",
    "PGX": "XsYdjDjNUygZ7yGKfQaB6TxLh2gC6RRjzLtLAGJrhzV",
    "PLTRX": "XsoBhf2ufR8fTyNSjqfU71DYGaE6Z3SUGAidpzriAA4",
    "PMX": "Xsba6tUnSjDae2VcopDB6FGGDaxRrewFCDa5hKn5vT3",
    "PPLTX": "Xst6eFD4YT6sz9RLMysN9SyvaZWtraSdVJQGu5ZkAme",
    "QQQX": "Xs8S1uUs1zvS2p7iwtsG3b6fkhpvmwz4GYU3gWAmWHZ",
    "SCHFX": "XsWAnFM77x6YvpdaZoos79R12o4Yj4r7EVkaTWddzhU",
    "SLVX": "XsxAd6okt8y1RRK6gNg7iJaqiWNiq5Md5EDf3ZrF2dm",
    "SPYX": "XsoCS1TfEyfFhfvj8EtZ528L3CaKBDBRqRapnBbDF2W",
    "STRCX": "Xs78JED6PFZxWc2wCEPspZW9kL3Se5J7L5TChKgsidH",
    "TBLLX": "XsqBC5tcVQLYt8wqGCHRnAUUecbRYXoJCReD6w7QEKp",
    "TMOX": "Xs8drBWy3Sd5QY3aifG9kt9KFs2K3PGZmx7jWrsrk57",
    "TONXX": "XscE4GUcsYhcyZu5ATiGUMmhxYa1D5fwbpJw4K6K4dp",
    "TQQQX": "XsjQP3iMAaQ3kQScQKthQpx9ALRbjKAjQtHg6TFomoc",
    "TSLAX": "XsDoVfqeBukxuZHWhdvWHBhgEHjGNst4MLodqsJHzoB",
    "UNHX": "XszvaiXGPwvk2nwb3o9C1CX4K6zH8sez11E6uyup6fe",
    "VTIX": "XsssYEQjzxBCFgvYFFNuhJFBeHNdLWYeUSP8F45cDr9",
    "VTX": "XsEdDDTcVGJU6nvdRdVnj53eKTrsCkvtrVfXGmUK68V",
    "VX": "XsqgsbXwWogGJsNcVZ3TyVouy2MbTkfCFhCGGGcQZ2p",
    "WMTX": "Xs151QeqTCiuKtinzfRATnUESM2xTU6V9Wy8Vy538ci",
    "XOMX": "XsaHND8sHyfMfsWPj6kSdd5VwvCayZvjYgKmmcNL5qh",
}


@dataclass
class SolanaXStocksRequest:
    symbol: str
    mint: str
    time_from: int
    time_to: int
    interval: str = DEFAULT_INTERVAL
    currency: str = "usd"


def _to_unix(value: Any, field_name: str) -> int:
    if value is None:
        if field_name == "time_to":
            return int(datetime.now(timezone.utc).timestamp())
        raise ValueError(f"`{field_name}` is required.")
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError(f"`{field_name}` cannot be empty.")
    if text.isdigit():
        return int(text)
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _normalize_interval(value: Any) -> str:
    interval = str(value or DEFAULT_INTERVAL).strip()
    allowed = {"1m", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "6H", "8H", "12H", "1D", "3D", "1W", "1M"}
    if interval not in allowed:
        raise ValueError(f"Unsupported interval `{interval}`.")
    return interval


def _normalize_currency(value: Any) -> str:
    currency = str(value or "usd").strip().lower()
    if currency not in {"usd", "native"}:
        raise ValueError("`currency` must be `usd` or `native`.")
    return currency


def _resolve_symbol_and_mint(params: dict[str, Any]) -> tuple[str, str]:
    symbol = str(params.get("symbol") or params.get("ticker") or "").strip()
    mint = str(params.get("mint") or params.get("address") or "").strip()
    if mint:
        if not VALID_MINT.fullmatch(mint):
            raise ValueError(f"Invalid Solana mint `{mint}`.")
        return symbol or mint, mint
    if not symbol:
        raise ValueError("Provide either `symbol`/`ticker` or `mint`/`address`.")
    mapped = XSTOCKS_SOLANA_MINTS.get(symbol.upper())
    if not mapped:
        raise ValueError(
            f"Unknown xStocks symbol `{symbol}`. Provide a Solana mint in `mint`."
        )
    return symbol.upper(), mapped


def _auth_headers() -> dict[str, str]:
    api_key = os.getenv("BIRDEYE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("BIRDEYE_API_KEY is required.")
    return {
        "Accept": "application/json",
        "X-API-KEY": api_key,
        "x-chain": "solana",
        "User-Agent": "quant-pitch-evaluator/solana-xstocks-skill",
    }


def _http_get_json(url: str, headers: dict[str, str], timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    request = Request(url=url, headers=headers, method="GET")
    with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310 - fixed host from configuration
        body = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError("Birdeye API response was not a JSON object.")
    return parsed


def _build_request(params: dict[str, Any]) -> SolanaXStocksRequest:
    symbol, mint = _resolve_symbol_and_mint(params)
    interval = _normalize_interval(params.get("interval") or params.get("timeframe"))
    currency = _normalize_currency(params.get("currency"))
    time_from = _to_unix(params.get("time_from") or params.get("start"), "time_from")
    time_to = _to_unix(params.get("time_to") or params.get("end"), "time_to")
    if time_to <= time_from:
        raise ValueError("`time_to` must be greater than `time_from`.")
    return SolanaXStocksRequest(
        symbol=symbol,
        mint=mint,
        time_from=time_from,
        time_to=time_to,
        interval=interval,
        currency=currency,
    )


def fetch_solana_xstocks_bars(params: dict[str, Any]) -> dict[str, Any]:
    skill_name = "solana_xstocks_bars"
    try:
        req = _build_request(params)
        headers = _auth_headers()
        base_url = os.getenv("BIRDEYE_BASE_URL", DEFAULT_BIRDEYE_BASE_URL).strip() or DEFAULT_BIRDEYE_BASE_URL
        query = urlencode(
            {
                "address": req.mint,
                "type": req.interval,
                "currency": req.currency,
                "time_from": req.time_from,
                "time_to": req.time_to,
            }
        )
        payload = _http_get_json(url=f"{base_url.rstrip('/')}/defi/v3/ohlcv?{query}", headers=headers)
        if not payload.get("success"):
            raise RuntimeError(str(payload.get("message") or "Birdeye request failed."))
        data = payload.get("data", {}) if isinstance(payload.get("data"), dict) else {}
        is_scaled = bool(data.get("is_scaled_ui_token", False))
        items = data.get("items", []) if isinstance(data.get("items"), list) else []
        bars: list[dict[str, Any]] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            open_px = raw.get("scaled_o") if is_scaled and raw.get("scaled_o") is not None else raw.get("o")
            high_px = raw.get("scaled_h") if is_scaled and raw.get("scaled_h") is not None else raw.get("h")
            low_px = raw.get("scaled_l") if is_scaled and raw.get("scaled_l") is not None else raw.get("l")
            close_px = raw.get("scaled_c") if is_scaled and raw.get("scaled_c") is not None else raw.get("c")
            volume = raw.get("scaled_v") if is_scaled and raw.get("scaled_v") is not None else raw.get("v")
            bars.append(
                {
                    "symbol": req.symbol,
                    "mint": req.mint,
                    "timestamp": int(raw.get("unix_time", 0) or 0),
                    "open": float(open_px) if open_px is not None else None,
                    "high": float(high_px) if high_px is not None else None,
                    "low": float(low_px) if low_px is not None else None,
                    "close": float(close_px) if close_px is not None else None,
                    "volume": float(volume) if volume is not None else None,
                    "volume_usd": float(raw.get("v_usd")) if raw.get("v_usd") is not None else None,
                    "interval": req.interval,
                    "currency": req.currency,
                }
            )
        return {
            "skill": skill_name,
            "status": "ok",
            "summary": f"Fetched {len(bars)} Solana OHLCV bar(s) for {req.symbol}.",
            "bars": bars,
            "artifacts": {
                "symbol": req.symbol,
                "mint": req.mint,
                "time_from": req.time_from,
                "time_to": req.time_to,
                "interval": req.interval,
                "currency": req.currency,
                "is_scaled_ui_token": is_scaled,
            },
        }
    except HTTPError as exc:
        return {
            "skill": skill_name,
            "status": "fail",
            "summary": f"Birdeye HTTP error {exc.code}.",
            "bars": [],
            "artifacts": {"error_type": "http_error", "status_code": exc.code},
        }
    except URLError as exc:
        return {
            "skill": skill_name,
            "status": "fail",
            "summary": f"Network error while calling Birdeye: {exc.reason}",
            "bars": [],
            "artifacts": {"error_type": "network_error"},
        }
    except Exception as exc:
        return {
            "skill": skill_name,
            "status": "fail",
            "summary": str(exc) or "Unknown Solana xStocks fetch error.",
            "bars": [],
            "artifacts": {"error_type": exc.__class__.__name__},
        }
