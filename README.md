# TrendLab

Multi-agent stock market research and **paper-trading** platform targeting **$0/month** infrastructure.

Four independent trend-following agents each manage a simulated $10,000 book across 12 stocks and ETFs. The Next.js dashboard reads a JSON snapshot produced by Python workers — so Vercel Hobby stays free and never needs always-on servers.

## Architecture

| Layer | Technology | Cost posture |
| --- | --- | --- |
| Dashboard | Next.js + Tailwind on Vercel Hobby | Free |
| Charts | TradingView Lightweight Charts | Free |
| Market data | Alpaca Basic (optional) or Yahoo Finance fallback | Free tier / public |
| Agents + simulation | Python (pandas / NumPy) | Free |
| Automation | GitHub Actions | Free within limits |
| Database | Local SQLite for workers + Supabase schema ready | Free tier |
| AI / paid TradingView webhooks | Not included in v1 | $0 |

No component can place real-money trades.

## Universe

AAPL, MSFT, NVDA, AMD, JPM, GS, XOM, CVX, JNJ, UNH, SPY, QQQ

## Agents

1. **EMA Trend** — EMA 20/50/200 confirmation  
2. **Breakout** — 20-bar high entries / 10-bar low exits  
3. **Momentum** — ROC + ADX + volume filter  
4. **Supertrend** — volatility-adjusted trend follower  

Shared rules: long-only, max 3 positions, 1% equity risk budget, fees + slippage, idempotent fills.

## Quick start

```bash
# Dashboard
npm install
npm run seed          # collect candles + simulate agents + write snapshot
npm run dev           # http://localhost:3456

# Workers only
python3 -m venv .venv
source .venv/bin/activate
pip install -r workers/requirements.txt
cd workers && python -m trendlabs.collect --timeframe 1h
cd workers && python -m trendlabs.simulate --timeframe 1h
```

Without `ALPACA_API_KEY` / `ALPACA_API_SECRET`, collection uses Yahoo Finance so local demos stay unblocked. When Alpaca keys are present, the collector prefers Alpaca (`iex` feed by default) and records the source on every candle.

## Supabase

Apply `supabase/migrations/001_trendlabs_schema.sql` in a free Supabase project when you are ready to sync beyond the committed snapshot. The dashboard currently ships with the snapshot pipeline so Phase 1 works before credentials exist.

## Automation

`.github/workflows/trendlabs.yml` collects data and refreshes `public/data/snapshot.json` on a weekday schedule. Add optional `ALPACA_API_KEY` / `ALPACA_API_SECRET` repository secrets.

## Deploy (Vercel Hobby)

1. Push this repo to GitHub.  
2. Import the project in Vercel (personal / non-commercial Hobby).  
3. Keep using the committed snapshot, or let Actions refresh it.  
4. Do **not** add live brokerage trading credentials.

## Phase map

- **Phase 1** ✅ Data → snapshot → dashboard charts  
- **Phase 2** ✅ Indicators + trend labels on markets  
- **Phase 3** ✅ Four paper agents with risk controls  
- **Phase 4** ✅ Initial historical trend-event research  
- **Phase 5** ✅ GitHub Actions workflow scaffold  
- **Phase 6** 🔲 Deeper analytics, Supabase live sync, walk-forward UI  

## Safety

Paper trading only. Delayed/IEX data is labeled in metadata. Free-tier quotas should throttle collection frequency rather than incur spend.
