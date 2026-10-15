# Intraday PE Track — 15m / 1h / 4h (V1 rebuild)

Rebuild of the `cursor/pe-intraday-v1-0eae` track, now living in `workers/trendlabs/intraday/`.
Same method conventions as the daily PE track (`trendlabs.prediction`); different data and product metric.

## Goal
Predict price moves at **15m / 1h / 4h** horizons on a liquid US equity/ETF set (same 12 symbols as the daily track, SPY kept as benchmark context).

## Method (unchanged from daily track conventions)
- **Execution rule:** signal at close of bar `t` → enter at open of bar `t+1` → exit at close of bar `t+h` (`h` = horizon in panel bars). `R = ln(close_{t+h}/open_{t+1})`, `z = R/(sigma_vol_window * sqrt(h))`, sigma from past `vol_window` (20) bar log returns at `t` only.
- **Panels:** RTH-only (9:30–16:00 ET), session-anchored at 9:30 ET. Bar label = bar start (UTC). 15m → 26 bars/day, 1h → 7, 4h → 2 (last 4h bar is 2.5h: 13:30–16:00).
- **Purged walk-forward:** expanding train / fixed test in **bars**, purge gap ≥ `horizon_bars` so train labels cannot overlap test. Gate bars imported from `trendlabs.prediction.gate` (same thresholds: MeanIC ≥ 0.03, positive-fold rate ≥ 0.60, Q5−Q1 Sharpe(10bps) ≥ 0.5, ≥ 5 folds, ≥ 30 quintile sessions).
- Baselines: random, per-symbol historical mean, momentum (`MOM_1h`), Ridge on all features, plus the Causal+QAE slice's plan baselines: `persistence` (last bar log return), `short_mom` (last 2 wall-clock hours, floor 1 bar), `range_rv_signed` (close location in trailing 1-session range × `VOL_1d`). All PIT, no train fit.
- Metrics per baseline: per-bar cross-sectional Spearman IC (+ ICIR, positive-fold rate, per-session IC), rank IC (Pearson of ranks), hit-rate `P(sign(score)=sign(z))`, and MAE of the cross-sectionally standardized score vs `z`.

## Data depth (the bottleneck)
| Source | Access | Base grid | History |
|---|---|---|---|
| Alpaca v2 (`data.alpaca.markets/v2/stocks/{sym}/bars`) | free **Basic** plan via paper account (keys in `.env`) | 1m–1h | **SIP historical since 2016**, 200 req/min, 10k bars/page |
| Yahoo (yfinance, keyless fallback) | no key | 1h | ~730d |
| Yahoo | no key | 5m/15m | ~60d |

Current keyless state (2026-10-15):
- `pe_intra_v1_1h`: 60,516 rows / 12 symbols, 2023-11 → 2026-10
- `pe_intra_v1_4h`: 17,142 rows / 12 symbols, 2023-12 → 2026-10 (resampled from 1h)
- `pe_intra_v1_15m`: 17,016 rows / 12 symbols, 2026-07 → 2026-10 (Yahoo 15m cap — thin; treat 15m gate reads as low-power)

## Usage
```bash
# 1) backfill bars (Alpaca needs ALPACA_API_KEY/SECRET in .env — free paper account)
python -m trendlabs.intraday.collect --source alpaca --timeframe 5m --years 10   # 2016 → today
python -m trendlabs.intraday.collect --source yahoo --timeframe 1h               # keyless fallback

# 2) build panel + targets + features per horizon
python -m trendlabs.intraday.dataset --tf 15m   # also: 1h, 4h   (npm run pe-intraday-dataset -- --tf 15m)

# 3) baselines (purged WF, writes data/intraday/baselines_{tf}.json)
python -m trendlabs.intraday.baselines --tf 1h   # (npm run pe-intraday-baselines -- --tf 1h)
```
Source resolution per panel: `15m: alpaca 5m → yahoo 5m → yahoo 15m`; `1h/4h: alpaca 5m → yahoo 1h`. All panels resample from the cached base grid, so an Alpaca 5m backfill upgrades every horizon at once (~10y × 12 symbols ≈ 4,700 requests ≈ 30 min at Basic pacing).

## Features (`intraday/features.py`, all PIT)
Momentum over wall-clock horizons (`MOM_1h/4h/1d/5d`, accel), trend quality (rolling OLS slope/R², 1d & 5d), vol (`VOL_1h/1d`, ratio, `RANGE_1d`, overnight `GAP`/`GAP_Z`), volume (`VolumeZ`, `DollarVolume`, `RVOL_SLOT` = volume vs same time-of-day slot over prior sessions), time-of-day/calendar (`BAR_FRAC`, sin/cos, day-of-week), market context broadcast per bar (`SPY_MOM_1h/1d`, `SPY_VOL_1d`, RS, cross-sectional dispersion, breadth).

## Status
- Pipeline: collect → panel → targets → purged WF → baselines — working (Yahoo 1h/15m, keyless).
- Next: Alpaca 5m backfill (needs keys) → rebuild panels with ~10y depth → re-run baselines; XGBoost per horizon **only** if a baseline/model clears the gate.
- Daily PE FAIL + HAR-vol remain method benchmarks; never mixed into the intraday product metric.

## Tests
`workers/trendlabs/test_intraday.py`: session-anchored resampling (15m/4h), execution-rule targets + NaN tail, purged-fold label/no-overlap invariants, PIT property of `RVOL_SLOT`, PIT value checks + future-bar perturbation invariance for `persistence`/`short_mom`/`range_rv_signed`, and a deterministic end-to-end runner check emitting all seven baselines with the extended metrics.
