# Prediction Engine — Intraday V1 (exploratory)

**Status:** Research / exploratory first slice  
**Scope:** 15-minute / 1-hour / 4-hour forward-return **baselines** on a small liquid set  
**Out of scope:** XGBoost, live brokerage, UI, full daily universe  

**Disclaimer:** `EXPLORATORY — SURVIVORSHIP BIAS NOT FULLY CONTROLLED`

This path is **independent** of the daily PE ranking stack and the volatility V1 gate. Those remain untouched benchmarks.

---

## Goal

Can simple, leakage-safe baselines show any usable rank/sign signal for **short-horizon** price moves on liquid US names, using only free/paper-friendly data?

Horizons (decision at bar \(t\), feature prices ≤ \(t\)):

| Horizon | 5m bar count | Target (close-to-close) |
| --- | ---: | --- |
| 15m | 3 | \(R = \ln(P^{\mathrm{close}}_{t+3}/P^{\mathrm{close}}_{t})\) |
| 1h | 12 | \(R = \ln(P^{\mathrm{close}}_{t+12}/P^{\mathrm{close}}_{t})\) |
| 4h | 48 | \(R = \ln(P^{\mathrm{close}}_{t+48}/P^{\mathrm{close}}_{t})\) |

**Price basis:** **close** (not mid). Mid \((H+L)/2\) is a possible later sensitivity check.

---

## Data

| Item | Choice |
| --- | --- |
| Source | Yahoo Finance via `yfinance` (no paid secrets) |
| Interval | **5-minute** bars (preferred over 1m for history length) |
| Yahoo window | Typically **~60 calendar days** for 5m; **~7 days** for 1m (1m not used) |
| Symbols | QQQ, SPY, AAPL, MSFT, NVDA |
| Storage | `data/prediction_engine/intraday/` (`panel_5m.parquet`, `manifest.json`) |
| Results | `data/prediction_engine/intraday_results.json` |

### Honest limits

- Yahoo public intraday history is **short**; purged walk-forward sample sizes are thin, especially at 4h (48-bar purge).
- Fixed mega-cap/ETF list → **survivorship / convenience bias**.
- Bar gaps and extended-hours inclusion vary; we do not reconstruct a perfect RTH grid in V1.
- **Paper research only** — no brokerage integration; the LLM never trades.

---

## Baselines (no XGBoost yet)

1. **Persistence** — last realized return of the same horizon: \(\ln(P_t/P_{t-h})\).
2. **Short momentum** — \(\ln(P_t/P_{t-6})\) (~30 minutes on 5m bars).
3. **Range RV (signed)** — mean \((H-L)/P\) over 12 bars × \(\mathrm{sign}(\mathrm{short\_mom})\).

Each horizon is evaluated separately.

---

## Evaluation

- **Purged walk-forward** on bar timestamps; purge gap = horizon bars (no overlapping labels across train/test).
- Defaults: expanding train ≥ 1500 bars, test 400 bars, step = test.
- Metrics per horizon / model: pooled **Spearman IC**, cross-sectional **Rank IC** (mean over timestamps), **hit-rate** (sign), **MAE**.

---

## How to run

```bash
# Fetch (or reuse cache) + evaluate baselines
npm run pe-intraday

# Force refetch from Yahoo
cd workers && ../.venv/bin/python -m trendlabs.prediction.intraday --refresh

# Unit tests (synthetic series only — not claimed market results)
cd workers && ../.venv/bin/python -m unittest trendlabs.test_prediction_intraday -v
```

---

## Recommended next steps

1. **Better data** if available later (longer 5m/1m history from a paper-friendly vendor) — biggest sample-size lever.
2. Else richer **features** (order-flow proxies, session time, overnight gap) still on Yahoo 5m, then light models.
3. Keep daily PE / volatility benchmarks unchanged for comparison.
