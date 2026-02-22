# EQT One-Shot Correlation Proof

Generated (UTC): 2026-02-22T07:43:13.904554+00:00

## Data Sources
- EIA weekly natural gas storage history: `data/raw/eia_ngshistory.xls`
- NOAA CPC weekly heating degree day archive: `data/raw/noaa_weekly_hdd_us.csv`
- EQT daily prices: `data/raw/eqt_prices_stooq_daily.csv`

## Feature Construction
- `storage_surprise_bcf`: current EIA weekly net change minus prior expanding mean for same week-of-year.
- `warmth_surprise`: negative NOAA HDD deviation from normal minus prior expanding mean for same week-of-year.
- `signal_score = z(storage_surprise_bcf) + z(warmth_surprise)` using a 104-week rolling z-score (shifted by 1 week).

## Historical Results
- Sample rows: 732 (2012-01-06 to 2026-01-30)
- Hold period: 10 trading days
- Entry lag: 6 calendar days after EIA week-ending date (release timing proxy)
- `corr(signal_score, EQT fwd return)`: -0.0607
- `corr(storage_surprise_z, EQT fwd return)`: -0.0708
- `corr(warmth_surprise_z, EQT fwd return)`: -0.0440

### Quintile Spread Check
- Q0 mean fwd return: 1.1850%
- Q4 mean fwd return: -0.5203%
- Q4 - Q0 spread: -1.7053%
- Welch t-stat (Q4 - Q0): -1.915

### Tradable Threshold Rule
- Rule: if `signal_score > 1.0`, short EQT; if `< -1.0`, long EQT; else no trade.
- Trades triggered: 342
- Average strategy return per trade: 0.8653%
- Hit rate: 57.89%
- Strategy t-stat: 2.108

## Latest Read
- Week ending: 2026-02-13
- EIA net change (Bcf): -144.0
- Storage surprise (Bcf): 3.19
- NOAA HDD dev from normal: -15.0
- Warmth surprise: 8.25
- Signal score: 0.486

## Caveats
- This is a tactical statistical relationship, not a structural certainty.
- NOAA archive data are preliminary by source labeling.
- Price history source is Stooq adjusted daily data.