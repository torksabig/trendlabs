# TrendLab STATUS (cloud workspace)

**Decision:** Continue in this cloud workspace. You can **kill the stuck Mac `bg_7` / deep-15m baselines job** if it is still spinning with no progress — that track is not needed for the 1d ML path below.

## What’s done (1d)

- ML pipeline scripts exist and work: `features` → `train-ml` → `ml-predict` → `ml-eval`.
- **All 12 universe symbols** have trained daily (`1d`) XGBoost models under `models/{SYMBOL}/1d/…` (AAPL, MSFT, NVDA, AMD, JPM, GS, XOM, CVX, JNJ, UNH, SPY, QQQ).
- Fresh live predictions written for every symbol (as of bar `2026-10-12`). Those 12 rows are **not scored yet** (need 4 more daily bars of future price).
- Historical backfill already scored: **480** predictions, **overall direction accuracy ~69.6%**.

### Holdout test metrics (from training `metrics.json`)

These are the honest chronological test-set numbers (model vs majority baseline). Most models are **near chance / slightly under majority** — expected for this MVP label; do not treat backfill accuracy as production edge.

| Symbol | Test accuracy | Test ROC-AUC | Walk-forward acc |
| --- | ---: | ---: | ---: |
| AAPL | 0.435 | 0.520 | 0.484 |
| MSFT | 0.472 | 0.521 | 0.505 |
| NVDA | 0.481 | 0.510 | 0.506 |
| SPY | 0.587 | 0.516 | 0.534 |

### Backfill scored direction accuracy (40 samples / symbol)

| Symbol | Acc | Symbol | Acc |
| --- | ---: | --- | ---: |
| SPY | 0.850 | JNJ | 0.825 |
| NVDA | 0.750 | UNH | 0.750 |
| GS | 0.725 | JPM | 0.725 |
| AAPL | 0.700 | MSFT | 0.675 |
| QQQ | 0.675 | XOM | 0.600 |
| AMD | 0.575 | CVX | 0.500 |

## Recommended next command

Re-score after more candles land (or force a fresh historical backfill):

```bash
npm run ml-eval -- --timeframe 1d --backfill
```

Optional: refresh live predictions first:

```bash
npm run ml-predict -- --timeframe 1d && npm run ml-eval -- --timeframe 1d
```

## Not in scope here

- No UI / no deployment work.
- Ignore Mac deep-15m / `bg_7` baselines; cloud 1d ML path is the active track.
