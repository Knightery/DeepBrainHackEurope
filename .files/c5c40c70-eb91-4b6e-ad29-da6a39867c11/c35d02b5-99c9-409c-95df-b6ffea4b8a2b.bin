from __future__ import annotations

import io
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urljoin

import numpy as np
import pandas as pd
import requests

EIA_WNGSR_URL = "https://ir.eia.gov/ngs/wngsr.csv"
EIA_HISTORY_URL = "https://ir.eia.gov/ngs/ngshistory.xls"
NOAA_WEEKLY_HDD_ARCHIVE_URL = (
    "https://ftp.cpc.ncep.noaa.gov/htdocs/products/analysis_monitoring/"
    "cdus/degree_days/archives/Heating%20degree%20Days/weekly%20states/"
)
STOOQ_EQT_DAILY_URL = "https://stooq.com/q/d/l/?s=eqt.us&i=d"


@dataclass(frozen=True)
class Paths:
    root: Path
    raw_dir: Path
    processed_dir: Path
    reports_dir: Path

    @staticmethod
    def from_script(script_file: Path) -> "Paths":
        root = script_file.resolve().parent.parent
        return Paths(
            root=root,
            raw_dir=root / "data" / "raw",
            processed_dir=root / "data" / "processed",
            reports_dir=root / "reports",
        )


def ensure_dirs(paths: Paths) -> None:
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)


def _fetch_bytes(url: str, timeout: int = 90, retries: int = 4) -> bytes:
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retries:
                break
            time.sleep(0.5 * (2 ** (attempt - 1)))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch bytes from {url}")


def _fetch_text(url: str, timeout: int = 90, retries: int = 4) -> str:
    return _fetch_bytes(url, timeout=timeout, retries=retries).decode(
        "utf-8", errors="replace"
    )


def download_static_files(paths: Paths) -> Dict[str, str]:
    ensure_dirs(paths)
    outputs: Dict[str, str] = {}

    eia_wngsr_path = paths.raw_dir / "eia_wngsr_latest.csv"
    eia_wngsr_path.write_bytes(_fetch_bytes(EIA_WNGSR_URL))
    outputs["eia_wngsr_latest"] = str(eia_wngsr_path)

    eia_history_path = paths.raw_dir / "eia_ngshistory.xls"
    eia_history_path.write_bytes(_fetch_bytes(EIA_HISTORY_URL))
    outputs["eia_ngshistory"] = str(eia_history_path)

    eqt_path = paths.raw_dir / "eqt_prices_stooq_daily.csv"
    eqt_path.write_bytes(_fetch_bytes(STOOQ_EQT_DAILY_URL))
    outputs["eqt_prices"] = str(eqt_path)

    return outputs


def _extract_report_date(text: str, fallback_file_name: str) -> datetime:
    match = re.search(
        r"LAST DATE OF DATA COLLECTION PERIOD IS\s+([A-Z]{3})\s+(\d{1,2}),\s+(\d{4})",
        text,
    )
    if match:
        month, day, year = match.groups()
        return datetime.strptime(f"{month} {int(day):02d} {year}", "%b %d %Y")

    name = unquote(fallback_file_name).replace(".txt", "")
    return datetime.strptime(name, "%b %d, %Y")


def _extract_us_hdd_row(text: str) -> Optional[Dict[str, float]]:
    # First "UNITED STATES" line is population-weighted national summary.
    for line in text.splitlines():
        if not line.startswith(" UNITED STATES"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            return {
                "hdd_week_total": float(parts[2]),
                "hdd_week_dev_norm": float(parts[3]),
                "hdd_week_dev_last_year": float(parts[4]),
                "hdd_cum_total": float(parts[5]),
            }
        except ValueError:
            continue
    return None


def download_noaa_weekly_hdd_csv(
    output_path: Path,
    start_year: int = 2010,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    now_year = datetime.now(timezone.utc).year
    year_cap = now_year if end_year is None else min(end_year, now_year)

    root_html = _fetch_text(NOAA_WEEKLY_HDD_ARCHIVE_URL, timeout=120)
    years = sorted(set(re.findall(r'href="(\d{4})/"', root_html)))

    rows: List[Dict[str, object]] = []
    for year_str in years:
        year = int(year_str)
        if year < start_year or year > year_cap:
            continue

        year_url = urljoin(NOAA_WEEKLY_HDD_ARCHIVE_URL, f"{year_str}/")
        year_html = _fetch_text(year_url, timeout=120)
        file_names = sorted(
            set(re.findall(r'href="([A-Za-z]{3}%20\d{2},%20\d{4}\.txt)"', year_html))
        )

        for file_name in file_names:
            file_url = urljoin(year_url, file_name)
            text = _fetch_text(file_url, timeout=120)
            date = _extract_report_date(text, file_name)
            parsed = _extract_us_hdd_row(text)
            if parsed is None:
                continue

            rows.append(
                {
                    "report_date": date.date().isoformat(),
                    "week_ending": (date - pd.Timedelta(days=1)).date().isoformat(),
                    "source_url": file_url,
                    **parsed,
                }
            )

    noaa = pd.DataFrame(rows).drop_duplicates(subset=["report_date"]).sort_values(
        "report_date"
    )
    noaa.to_csv(output_path, index=False)
    return noaa


def parse_eia_history(eia_xls_path: Path) -> pd.DataFrame:
    content = eia_xls_path.read_bytes()
    stocks = pd.read_excel(
        io.BytesIO(content),
        sheet_name="html_report_history",
        skiprows=6,
    )
    changes = pd.read_excel(
        io.BytesIO(content),
        sheet_name="weekly_net_changes",
        skiprows=6,
    )

    stocks = stocks[["Week ending", "Total Lower 48"]].rename(
        columns={"Week ending": "week_ending", "Total Lower 48": "total_stocks_bcf"}
    )
    changes = changes[["Unnamed: 0", "Total Lower 48"]].rename(
        columns={"Unnamed: 0": "week_ending", "Total Lower 48": "net_change_bcf"}
    )

    for frame in (stocks, changes):
        frame["week_ending"] = pd.to_datetime(frame["week_ending"], errors="coerce")

    stocks["total_stocks_bcf"] = pd.to_numeric(
        stocks["total_stocks_bcf"], errors="coerce"
    )
    changes["net_change_bcf"] = pd.to_numeric(changes["net_change_bcf"], errors="coerce")

    stocks = stocks.dropna(subset=["week_ending", "total_stocks_bcf"])
    changes = changes.dropna(subset=["week_ending", "net_change_bcf"])

    return (
        stocks.merge(changes, on="week_ending", how="inner")
        .drop_duplicates(subset=["week_ending"])
        .sort_values("week_ending")
        .reset_index(drop=True)
    )


def parse_noaa_csv(noaa_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(noaa_csv_path)
    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    for column in [
        "hdd_week_total",
        "hdd_week_dev_norm",
        "hdd_week_dev_last_year",
        "hdd_cum_total",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["week_ending", "hdd_week_dev_norm"]).sort_values(
        "week_ending"
    )


def parse_eqt_prices(eqt_csv_path: Path) -> pd.DataFrame:
    prices = pd.read_csv(eqt_csv_path)
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
    return prices.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(
        drop=True
    )


def _rolling_zscore(
    series: pd.Series,
    lookback: int = 104,
    min_periods: int = 52,
) -> pd.Series:
    mean = series.rolling(lookback, min_periods=min_periods).mean().shift(1)
    std = series.rolling(lookback, min_periods=min_periods).std(ddof=0).shift(1)
    return (series - mean) / std


def build_feature_frame(eia: pd.DataFrame, noaa: pd.DataFrame) -> pd.DataFrame:
    frame = (
        eia.merge(
            noaa[
                [
                    "week_ending",
                    "hdd_week_total",
                    "hdd_week_dev_norm",
                    "hdd_week_dev_last_year",
                    "hdd_cum_total",
                ]
            ],
            on="week_ending",
            how="inner",
        )
        .sort_values("week_ending")
        .reset_index(drop=True)
    )

    frame["week_of_year"] = frame["week_ending"].dt.isocalendar().week.astype(int)

    frame["storage_seasonal_mean_bcf"] = frame.groupby("week_of_year")[
        "net_change_bcf"
    ].transform(lambda x: x.expanding().mean().shift(1))
    frame["storage_surprise_bcf"] = (
        frame["net_change_bcf"] - frame["storage_seasonal_mean_bcf"]
    )

    frame["warmth_proxy"] = -frame["hdd_week_dev_norm"]
    frame["warmth_seasonal_mean"] = frame.groupby("week_of_year")[
        "warmth_proxy"
    ].transform(lambda x: x.expanding().mean().shift(1))
    frame["warmth_surprise"] = frame["warmth_proxy"] - frame["warmth_seasonal_mean"]

    frame["storage_surprise_z"] = _rolling_zscore(frame["storage_surprise_bcf"])
    frame["warmth_surprise_z"] = _rolling_zscore(frame["warmth_surprise"])
    frame["signal_score"] = frame["storage_surprise_z"] + frame["warmth_surprise_z"]

    return frame


def add_forward_returns(
    frame: pd.DataFrame,
    prices: pd.DataFrame,
    hold_days: int = 10,
    release_lag_days: int = 6,
) -> pd.DataFrame:
    enriched = frame.copy()
    returns: List[float] = []
    entries: List[pd.Timestamp] = []
    exits: List[pd.Timestamp] = []

    dates = prices["Date"].values
    closes = prices["Close"].values

    for week_ending in enriched["week_ending"]:
        entry_target = week_ending + pd.Timedelta(days=release_lag_days)
        entry_idx = dates.searchsorted(np.datetime64(entry_target))

        if entry_idx >= len(prices) or entry_idx + hold_days >= len(prices):
            returns.append(np.nan)
            entries.append(pd.NaT)
            exits.append(pd.NaT)
            continue

        entry_price = closes[entry_idx]
        exit_price = closes[entry_idx + hold_days]
        ret = float(exit_price / entry_price - 1.0)
        returns.append(ret)
        entries.append(pd.Timestamp(dates[entry_idx]))
        exits.append(pd.Timestamp(dates[entry_idx + hold_days]))

    enriched[f"eqt_fwd_return_{hold_days}d"] = returns
    enriched["entry_date"] = entries
    enriched["exit_date"] = exits
    return enriched


def _safe_t_stat(values: pd.Series) -> float:
    clean = values.dropna()
    n = clean.shape[0]
    if n < 2:
        return float("nan")
    std = float(clean.std(ddof=1))
    if std == 0.0:
        return float("nan")
    return float(clean.mean() / (std / math.sqrt(n)))


def _welch_t_stat(group_a: pd.Series, group_b: pd.Series) -> float:
    a = group_a.dropna()
    b = group_b.dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = math.sqrt(float(a.var(ddof=1)) / len(a) + float(b.var(ddof=1)) / len(b))
    if se == 0:
        return float("nan")
    return float((b.mean() - a.mean()) / se)


def compute_metrics(
    frame_with_returns: pd.DataFrame,
    hold_days: int,
    threshold: float,
) -> Dict[str, float]:
    return_col = f"eqt_fwd_return_{hold_days}d"
    valid = frame_with_returns.dropna(subset=["signal_score", return_col]).copy()

    metrics: Dict[str, float] = {
        "sample_rows": float(len(valid)),
        "start_date": valid["week_ending"].min().date().isoformat(),
        "end_date": valid["week_ending"].max().date().isoformat(),
        "corr_signal_vs_return": float(valid["signal_score"].corr(valid[return_col])),
        "corr_storage_vs_return": float(
            valid["storage_surprise_z"].corr(valid[return_col])
        ),
        "corr_warmth_vs_return": float(
            valid["warmth_surprise_z"].corr(valid[return_col])
        ),
    }

    quintile_data = valid.copy()
    quintile_data["quintile"] = pd.qcut(
        quintile_data["signal_score"], 5, labels=False, duplicates="drop"
    )
    quintile_means = quintile_data.groupby("quintile")[return_col].mean().to_dict()
    metrics["quintile_0_mean"] = float(quintile_means.get(0, np.nan))
    metrics["quintile_4_mean"] = float(quintile_means.get(4, np.nan))
    metrics["quintile_4_minus_0"] = float(
        metrics["quintile_4_mean"] - metrics["quintile_0_mean"]
    )

    q0 = quintile_data.loc[quintile_data["quintile"] == 0, return_col]
    q4 = quintile_data.loc[quintile_data["quintile"] == 4, return_col]
    metrics["quintile_spread_t_stat"] = _welch_t_stat(q0, q4)

    position = np.where(
        valid["signal_score"] > threshold,
        -1.0,
        np.where(valid["signal_score"] < -threshold, 1.0, 0.0),
    )
    valid["position"] = position
    traded = valid.loc[valid["position"] != 0.0].copy()
    traded["strategy_return"] = traded["position"] * traded[return_col]

    metrics["threshold"] = float(threshold)
    metrics["trade_count"] = float(len(traded))
    metrics["strategy_avg_return"] = float(traded["strategy_return"].mean())
    metrics["strategy_hit_rate"] = float((traded["strategy_return"] > 0).mean())
    metrics["strategy_t_stat"] = _safe_t_stat(traded["strategy_return"])

    short_trades = valid.loc[valid["signal_score"] > threshold, return_col]
    long_trades = valid.loc[valid["signal_score"] < -threshold, return_col]
    metrics["short_bucket_count"] = float(len(short_trades))
    metrics["short_bucket_avg_raw_return"] = float(short_trades.mean())
    metrics["long_bucket_count"] = float(len(long_trades))
    metrics["long_bucket_avg_raw_return"] = float(long_trades.mean())

    return metrics


def write_metrics_json(path: Path, metrics: Dict[str, float]) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def current_signal(latest_row: pd.Series, threshold: float) -> str:
    score = float(latest_row["signal_score"])
    if score >= threshold:
        return "SHORT"
    if score <= -threshold:
        return "LONG"
    return "FLAT"


def pick_latest_row(frame: pd.DataFrame) -> pd.Series:
    usable = frame.dropna(subset=["signal_score"]).sort_values("week_ending")
    if usable.empty:
        raise ValueError("No usable rows with signal_score.")
    return usable.iloc[-1]
