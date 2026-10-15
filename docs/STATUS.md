# TrendLab STATUS (cloud workspace)

**Decision:** Continue in this cloud workspace. You can **kill the stuck Mac `bg_7` / deep-15m baselines job** if it is still spinning with no progress — that track is not needed for the 1d ML path below.

## What’s done (1d) — verified in this cloud VM

- ML pipeline scripts exist and work locally: `features` → `train-ml` → `ml-predict` → `ml-eval`.
- **All 12 universe symbols** have trained daily (`1d`) XGBoost models under `models/{SYMBOL}/1d/…`.
- Fresh live predictions written for every symbol (bar `2026-10-12`); those 12 rows need 4 more daily bars before scoring.
- Historical backfill scored: **480** predictions, **overall direction accuracy ~69.6%**.

### Holdout test metrics (honest chronological test set)

| Symbol | Test acc | Test AUC | WF acc |
| --- | ---: | ---: | ---: |
| AAPL | 0.435 | 0.520 | 0.484 |
| MSFT | 0.472 | 0.521 | 0.505 |
| NVDA | 0.481 | 0.510 | 0.506 |
| SPY | 0.587 | 0.516 | 0.534 |

Most models are near chance / slightly under majority — expected for this MVP label.

### Backfill direction accuracy (40 samples / symbol)

SPY 0.85 · JNJ 0.825 · NVDA/UNH 0.75 · GS/JPM 0.725 · AAPL 0.70 · MSFT/QQQ 0.675 · XOM 0.60 · AMD 0.575 · CVX 0.50

## Recommended next command (in this cloud workspace)

```bash
npm run ml-eval -- --timeframe 1d --backfill
```

Or after more candles land:

```bash
npm run ml-predict -- --timeframe 1d && npm run ml-eval -- --timeframe 1d
```

## Mac note

Ignore / kill Mac `bg_7` deep-15m baselines if still spinning — cloud 1d ML is the active track. No UI / no deployment in this step.
