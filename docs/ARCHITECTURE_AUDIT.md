# TrendLab Architecture Audit

**Date:** 2026-10-13  
**Scope:** Repository audit before AI training / data-extraction work  
**Source repo:** `torksabig/trendlabs`  
**Session constraint:** Data extraction focus; UI redesign and deployment deferred

---

## 1. Existing infrastructure

| Layer | Current stack | Notes |
| --- | --- | --- |
| Frontend | Next.js (App Router) + Tailwind + shadcn-style Button | Port `3456`; Arena dashboard, markets, agents, research, backtests, trades |
| Charts | TradingView Lightweight Charts | Live + candle charts |
| Market data (browser/API) | Yahoo Finance chart API via `src/lib/yahoo-candles.ts` | No key; short lookbacks for UI |
| Market data (workers) | Alpaca Basic (optional) → Yahoo Finance fallback (`workers/trendlabs/market_data.py`) | Free / public |
| Agents | Deterministic Python strategies (EMA, breakout, momentum, supertrend) | Paper trading only |
| Research LLM | Switchable free-tier providers + Ollama + rule fallback | Never executes trades |
| Local DB | SQLite at `data/trendlabs.db` | Primary worker store |
| Cloud DB | Supabase PostgreSQL schema in `supabase/migrations/` | Optional sync for reports |
| Automation | GitHub Actions + optional macOS `launchd` | Collect / simulate / research |
| Deploy | Vercel Hobby | Dashboard reads `public/data/snapshot.json` |

```
Alpaca / Yahoo → Python workers (SQLite) → snapshot JSON → Next.js dashboard
                      ↖ optional Supabase (agent_reports)
```

---

## 2. Supported stock symbols

Configured in both `src/lib/instruments.ts` and `workers/trendlabs/config.py` (must stay in sync):

| Symbol | Name | Sector |
| --- | --- | --- |
| AAPL | Apple Inc. | Technology |
| MSFT | Microsoft Corp. | Technology |
| NVDA | NVIDIA Corp. | Semiconductors |
| AMD | Advanced Micro Devices | Semiconductors |
| JPM | JPMorgan Chase | Financial services |
| GS | Goldman Sachs | Financial services |
| XOM | Exxon Mobil | Energy |
| CVX | Chevron | Energy |
| JNJ | Johnson & Johnson | Healthcare |
| UNH | UnitedHealth Group | Healthcare |
| SPY | SPDR S&P 500 ETF | Market benchmarks |
| QQQ | Invesco QQQ Trust | Market benchmarks |

Timeframes already defined: `15m`, `1h`, `4h`, `1d`.

**Rule:** Do not hardcode other example tickers. All pipelines must iterate `SYMBOLS` / `INSTRUMENTS`.

---

## 3. Working functionality

- Candle collection (`python -m trendlabs.collect`) with idempotent SQLite upserts
- Paper-trading simulation across four agents
- Indicator enrichment (EMA, RSI, ATR, ADX, Supertrend)
- Backtest, walk-forward, experiment engine, prediction eval (rule-based forecasts)
- Research LLM reports with provider fallback
- Dashboard pages reading snapshot / public JSON
- Supabase migrations for instruments, candles, agents, trades, reports, backtests, regimes

---

## 4. Missing components (for AI prediction platform)

| Component | Status |
| --- | --- | --- |
| Deep historical extract (5–10y daily) | Partial — `1d` period was `5y`, lookback capped at ~400 bars |
| Multi-timeframe batch extract CLI | Missing — collect is single timeframe |
| Provider pagination / retries / rate limits | Partial — no Alpaca page loop, no backoff |
| Candle quality reporting (gaps, holidays, dupes) | Missing |
| ML feature store (`market_features` + feature versions) | Missing (indicators computed in-memory only) |
| Per-symbol XGBoost models | Missing |
| Immutable ML predictions + outcome scoring for 4-candle horizon | Partial — existing predictions are rule-based / LLM horizons |
| PDF research reports | Missing |
| Controlled retraining / model version registry | Missing |

---

## 5. Reusable code

- `MarketDataClient` — extend rather than replace
- `Store.upsert_candles` / `load_candles` — keep PK `(symbol, timeframe, ts)`
- `INSTRUMENTS` / `SYMBOLS` / `TIMEFRAMES` — single source for universe
- `indicators.py` — base for feature engineering (avoid look-ahead)
- Existing SQLite + Supabase candle tables — extend with `updated_at` and ML tables via migrations
- Free Yahoo + optional Alpaca — no paid APIs

---

## 6. Technical risks

1. **Yahoo history limits:** Intraday (`15m` ~60d, `1h` ~730d) cannot match daily depth; document clearly.
2. **Incomplete bars:** Live/intraday last candle may be unfinished; extract must exclude or label incomplete.
3. **Split adjustments:** Both providers use adjusted history; mixing adjusted/unadjusted would corrupt features.
4. **Rate limits / outages:** Yahoo and Alpaca free tiers can throttle; need retries without inventing prices.
5. **Dual instrument lists:** TS + Python can drift — extract/tests should assert symbol parity.
6. **Supabase credentials:** Not available in this environment; SQLite is the local source of truth for extraction tests.
7. **Cloud agent is not a permanent worker:** Long-running schedulers stay on Mac / Actions (deployment deferred).

---

## 7. Proposed implementation changes (this session)

**In scope now — data extraction**

1. Document audit (this file).
2. Extend timeframe history config (`1d` → up to 10y / `max`).
3. Harden `MarketDataClient`: retries, Alpaca pagination, quality validation, no synthetic fills.
4. Add `python -m trendlabs.extract` for multi-symbol / multi-timeframe historical ingest.
5. Extend SQLite + Supabase schema: `updated_at` on candles, `ingestion_runs`, ML-ready tables (`market_features`, `model_versions`, `ml_predictions`, `ml_prediction_outcomes`, `paper_trades` alias/extension).
6. Automated tests for upserts, duplicates, quality gates.
7. Run extraction against configured universe and record results.

**Explicitly deferred**

- UI redesign / new dashboard sections
- Deployment / Vercel / production worker setup
- Full XGBoost training, PDF reports, continuous learning (next phases after extract is solid)

---

## 8. Environment variables (no secrets)

See `.env.example`. Extraction uses:

- `ALPACA_API_KEY` / `ALPACA_API_SECRET` (optional)
- `ALPACA_FEED` (default `iex`)
- `TRENDLAB_DB` / `TRENDLAB_DATA_DIR` (optional paths)
- `LOOKBACK_BARS` (legacy collect default; extract uses timeframe max history)

Never expose Supabase service-role or Alpaca secrets to the browser.

---

## 9. Extraction verification (2026-10-13)

Live Yahoo Finance extract completed with **0 errors** for all 12 configured symbols:

| Timeframe | Approx. depth | Bars (all symbols) |
| --- | --- | --- |
| `1d` | IPO/`max` history | ~133k |
| `1h` | ~730d | ~60k |
| `15m` | ~60d | ~18k |
| `4h` | resampled from 1h | ~20k |

**Total:** ~233k candle rows across 48 `(symbol, timeframe)` series in local SQLite.

Unit tests: `python -m unittest trendlabs.test_market_data` — 10/10 passed.
