# EQT One-Shot: EIA + NOAA Strategy Folder

This folder implements a one-shot trade workflow for US equity `EQT` using two public macro data streams:

- EIA weekly natural gas storage (`ngshistory.xls` + latest `wngsr.csv`)
- NOAA CPC weekly US heating degree-day archive

It includes:
- raw downloaded data files
- a historical correlation proof report
- an executable trade signal script

## Folder Layout

```text
eqt_one_shot_eia_noaa/
  data/
    raw/
    processed/
  reports/
  scripts/
```

## Quick Start

From repo root:

```powershell
.\.venv\Scripts\python eqt_one_shot_eia_noaa\scripts\download_data.py
.\.venv\Scripts\python eqt_one_shot_eia_noaa\scripts\analyze_correlation.py --hold-days 10 --threshold 1.0
.\.venv\Scripts\python eqt_one_shot_eia_noaa\scripts\trade_signal.py --hold-days 10 --threshold 1.0
```

## Outputs

- `data/raw/download_manifest.json`
- `data/raw/eia_ngshistory.xls`
- `data/raw/eia_wngsr_latest.csv`
- `data/raw/noaa_weekly_hdd_us.csv`
- `data/raw/eqt_prices_stooq_daily.csv`
- `data/processed/feature_dataset.csv`
- `reports/correlation_metrics.json`
- `reports/correlation_proof.md`
- `reports/latest_trade_signal.json`

## Signal Logic (Summary)

- Compute `storage_surprise_bcf` from EIA weekly net-change versus same-week seasonal baseline.
- Compute `warmth_surprise` from NOAA HDD deviation versus same-week seasonal baseline.
- Convert both to rolling z-scores and combine:
  - `signal_score = storage_surprise_z + warmth_surprise_z`
- Trade rule:
  - if `signal_score > threshold`: short `EQT`
  - if `signal_score < -threshold`: long `EQT`
  - else: flat

## Source URLs

- EIA Weekly Underground Natural Gas Storage Report: `https://ir.eia.gov/ngs/wngsr.csv`
- EIA historical storage workbook: `https://ir.eia.gov/ngs/ngshistory.xls`
- NOAA HDD archive root: `https://ftp.cpc.ncep.noaa.gov/htdocs/products/analysis_monitoring/cdus/degree_days/archives/Heating%20degree%20Days/weekly%20states/`
- EQT daily prices (Stooq CSV): `https://stooq.com/q/d/l/?s=eqt.us&i=d`

## Notes

- NOAA archive files are labeled preliminary in NOAA’s own documentation.
- This is a tactical statistical strategy, not investment advice.

