# Trend Arena

Four competing **rule-based** trend-following agents paper-trade Binance public market data. Market data, agents, SQLite, and the Streamlit dashboard run at **$0/month** API cost. An optional AI research agent can evaluate performance once daily if you set an API key.

## What’s included

| Component | Cost |
| --- | --- |
| Binance live / historical candles (public REST) | $0 |
| EMA Crossover, ADX Trend, Momentum, Supertrend agents | $0 |
| Local paper-trading broker + SQLite | $0 |
| Streamlit dashboard | $0 |
| Optional LLM research agent | Variable (off by default) |

Agents calculate EMA, ADX, ROC/SMA, and Supertrend in pure Python — no LLM required for trading decisions.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Seed a backtest (needs outbound HTTPS to api.binance.com)
PYTHONPATH=src python scripts/seed_backtest.py

# Dashboard (default port 8512)
PYTHONPATH=src streamlit run app.py --server.port 8512 --server.headless true
```

Open [http://localhost:8512](http://localhost:8512).

## Dashboard actions

1. **Run full backtest** — fetch candles, reset books, replay every agent bar-by-bar.
2. **Fetch latest & step agents** — sync the newest candle and take one decision per agent.
3. **Daily research report** — rule-based performance review (or LLM if `OPENAI_API_KEY` is set).
4. **Reset paper books** — restore each agent to starting cash.

## Architecture

```
Binance public /api/v3/klines
        │
        ▼
  SimulationEngine ──► four TradingAgent strategies
        │
        ▼
   PaperBroker (fees, position sizing)
        │
        ▼
     SQLite (candles, trades, equity, research_notes)
        │
        ▼
   Streamlit dashboard + optional ResearchAgent
```

## Optional AI research

Leave `OPENAI_API_KEY` unset for a free rule-based daily report. To enable an LLM:

```bash
cp .env.example .env
# edit OPENAI_API_KEY / OPENAI_MODEL
```

Recommended posture: keep the four trading agents deterministic; call the research agent at most once per day with compact stats.

## Notes

- Binance public market-data endpoints do not require API authentication; they are rate-limited.
- Default REST host is `https://data-api.binance.vision` (with failover to `api.binance.us` / `api.binance.com`) because some regions get HTTP 451 from `api.binance.com`.
- This is **local paper trading**, not exchange Demo Mode. Binance Demo Mode is a separate authenticated API if you want exchange-side simulated orders later.
- Nothing here places real orders.
