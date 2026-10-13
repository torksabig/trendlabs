# TrendLab Data Extraction

## Goal

Persist clean historical OHLCV for every stock already configured in TrendLab, ready for feature engineering and per-symbol model training.

## Universe

Symbols are read from `workers/trendlabs/config.py` (`INSTRUMENTS` / `SYMBOLS`). Unknown tickers are rejected.

Current universe: AAPL, MSFT, NVDA, AMD, JPM, GS, XOM, CVX, JNJ, UNH, SPY, QQQ.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r workers/requirements.txt

# All configured symbols + default extract timeframes (1d, 1h, 15m, 4h)
python -m trendlabs.extract

# Daily history only (deepest free history; best starting point for ML)
python -m trendlabs.extract --timeframes 1d

# Subset
python -m trendlabs.extract --symbols AAPL MSFT --timeframes 1d 1h

# Legacy short collect (lookback window)
python -m trendlabs.collect --timeframe 1h
python -m trendlabs.collect --timeframe 1d --full-history
```

npm aliases: `npm run extract`, `npm run collect`, `npm run test:extract`.

## Storage

| Store | Location |
| --- | --- |
| SQLite (primary local) | `data/trendlabs.db` table `candles` |
| Ingestion audit | `ingestion_runs` + `meta.last_extraction` |
| Supabase (optional) | apply `supabase/migrations/005_ml_market_data.sql` |

Candle fields: `symbol`, `timeframe`, `ts` (UTC), `open`, `high`, `low`, `close`, `volume`, `source`, `adjusted`, `provider_updated_at`, `updated_at`.

Primary key / upsert key: `(symbol, timeframe, ts)`.

## Provider limits (free)

| Timeframe | Yahoo practical max | Notes |
| --- | --- | --- |
| `1d` | `max` / ~10y | ML priority |
| `1h` / `4h` | ~730d | `4h` resampled from 1h |
| `15m` | ~60d | Shortest free depth |

Alpaca Basic (optional) uses paginated bars with split adjustment when keys are set.

## Quality rules

1. Never invent missing prices.
2. Drop invalid OHLC rows (NaN, high < low, non-positive prices).
3. Deduplicate on timestamp (`keep=last`).
4. Drop incomplete final candle unless `--include-incomplete`.
5. Record gap counts in the quality report (not filled).
6. Retry transient HTTP failures with exponential backoff.

## Schema for later ML phases

Migration `005_ml_market_data.sql` and SQLite bootstrap also create:

- `market_features`
- `model_versions`
- `ml_predictions` / `ml_prediction_outcomes`
- `paper_trades`

These tables are empty until feature engineering / training phases run.

## Recovery

Re-run `python -m trendlabs.extract` after downtime. Upserts overwrite the same timestamps safely; ingestion runs are append-only audit rows.
