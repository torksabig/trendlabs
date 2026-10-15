# TrendLab ML training (XGBoost)

Local, free-stack binary classifiers — one model per **symbol + timeframe**. No paid inference.

## Label

At bar `t` (completed candle):

> Will `close[t + 4]` be **higher** than `close[t]`?

Horizon is fixed at **4 bars**. Features at `t` never use bars after `t`.

## Commands

```bash
pip install -r workers/requirements.txt
npm run features -- --timeframe 1d
npm run train-ml -- --timeframe 1d
npm run ml-predict -- --timeframe 1d
npm run ml-eval -- --timeframe 1d --backfill
npm run test:ml
```

See `docs/STATUS.md` for current cloud metrics.
