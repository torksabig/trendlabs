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
- Pipeline: collect → panel → targets → purged WF → baselines — working (Yahoo 1h/15m/4h panels, keyless). All seven baselines run on all three horizons.
- **Measurement rule (2026-10-15):** per-bar cross-sectional IC clusters within sessions (same ordering repeated across the day's bars), so per-bar MeanIC overstates confidence. Headline reads use session-aggregated IC + session-block bootstrap CI95 + Q5−Q1 spread on raw R in bps vs round-trip cost `2×(FEE_BPS+SLIPPAGE_BPS)` = 20bps. All three are now emitted per baseline in `baselines_{tf}.json` (`session_ic.ci95`, `economic`).
- Deep-dive on the 15m ridge (+0.021 per-bar MeanIC): session-aggregated IC spans zero (CI95 ≈ [−0.03, +0.04]); signal flips by slot (open +0.031 → mid −0.033); concentrated in XOM/CVX/AMD vs JPM/GS over one 60d window; Q5−Q1 = +2.5 bps gross vs 20 bps round trip → **uneconomic, likely clustering + time-of-day artifact**.
- The only session-robust structure is short-horizon **reversal**: `short_mom` session IC −0.039 (CI95 [−0.069, −0.012]) on 15m. Observation; a sign-flipped candidate must be declared before the deep-panel run, never after.
- Next: Alpaca 5m backfill (needs keys) → rebuild panels with ~10y depth → re-run baselines with cluster-robust reads; XGBoost per horizon **only** if a pre-declared score clears the gate AND beats costs at depth.
- Daily PE FAIL + HAR-vol remain method benchmarks; never mixed into the intraday product metric.

## Pre-declared hypotheses (2026-10-15 21:20 UTC+3 — before any Alpaca deep panel exists)

Registered before the 10y backfill so no score can be declared post-hoc. Deep-panel evaluation
re-runs all seven baselines plus the flips below with the same gate bars and the cluster-robust
reads (session IC CI95 + economics vs 20 bps round trip).

| ID | Hypothesis | Score tested | Accept (all of) |
|---|---|---|---|
| H1 | Intraday reversal: negative short-horizon momentum predicts positive next-bar z | `-score_short_mom` (2h wall-clock) | session IC ≥ 0.03, CI95 excludes 0, folds+ ≥ 0.60, Q5−Q1 ≥ 20 bps, ≥ 100 folds |
| H2 | Time-of-day conditioning: cross-sectional ranking skill concentrates at session-open bars | ridge per-slot (open vs rest) | open-slot session IC ≥ 0.03, CI95 excludes 0, and ≥ 2× the rest-of-session IC |
| H3 | Momentum-as-edge (positive short momentum) — **expected to fail** | `score_short_mom`, `score_mom` | rejected if session IC CI95 upper bound < 0.03 |

Rules: no additional sign flips may be introduced after seeing deep-panel OOS; any new candidate
score must be registered in this table before its first deep-panel evaluation. XGBoost per horizon
is unlocked only by H1/H2 clearing on the deep panel (H3 rejection is a result, not a blocker).

## Tests
`workers/trendlabs/test_intraday.py`: session-anchored resampling (15m/4h), execution-rule targets + NaN tail, purged-fold label/no-overlap invariants, PIT property of `RVOL_SLOT`, PIT value checks + future-bar perturbation invariance for `persistence`/`short_mom`/`range_rv_signed`, and a deterministic end-to-end runner check emitting all seven baselines with the extended metrics.
