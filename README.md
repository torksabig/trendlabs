# TrendLab

Multi-agent stock market research and **paper-trading** platform targeting **$0/month** infrastructure.

Four independent trend-following agents each manage a simulated $10,000 book across 12 stocks and ETFs. A fifth **shared research agent** (local Qwen3 14B via Ollama on your Mac) interprets verified Python analytics and proposes experiments — it never executes trades.

## Architecture

| Layer | Technology | Cost posture |
| --- | --- | --- |
| Dashboard | Next.js + Tailwind on Vercel Hobby | Free |
| Charts | TradingView Lightweight Charts | Free |
| Market data | Alpaca Basic (optional) or Yahoo Finance fallback | Free tier / public |
| Trading agents | Deterministic Python strategies | Free |
| Research agent | Qwen3 14B via Ollama on Mac | $0 API fees |
| Automation | GitHub Actions (data + paper trades) | Free within limits |
| Local AI schedule | macOS `launchd` every ~4 hours | Free |
| Database | SQLite workers + Supabase schema | Free tier |

```
Alpaca / Yahoo → Python indicators & agents → snapshot JSON
                      ↓
              Local Qwen3 (Mac, Ollama)
                      ↓
         agent_reports → Supabase / snapshot → Vercel dashboard
```

The LLM stays on your Mac. Vercel only reads published reports. Cloud paper trading continues if the Mac sleeps.

## Universe

AAPL, MSFT, NVDA, AMD, JPM, GS, XOM, CVX, JNJ, UNH, SPY, QQQ

## Agents

**Trading (deterministic)**  
1. EMA Trend — EMA 20/50/200 confirmation  
2. Breakout — 20-bar high entries / 10-bar low exits  
3. Momentum — ROC + ADX + volume filter  
4. Supertrend — volatility-adjusted trend follower  

**Research (local LLM)**  
5. Shared Qwen3 research agent — explains regimes, risks, agent performance, and strategy experiments for Python to backtest

Shared trading rules: long-only, max 3 positions, 1% equity risk budget, fees + slippage, idempotent fills.

## Quick start

```bash
npm install
python3 -m venv .venv && source .venv/bin/activate
pip install -r workers/requirements.txt

npm run seed          # collect + simulate + research fallback report
npm run dev           # http://localhost:3456
```

### Local Qwen3 14B on Mac (recommended)

```bash
# Install Ollama for macOS, then:
ollama pull qwen3:14b-q4_K_M
ollama run qwen3:14b-q4_K_M

# From the TrendLab repo
source .venv/bin/activate
pip install -r workers/requirements.txt
npm run research -- --symbol AAPL --timeframe 1h
```

Without Ollama, `npm run research -- --no-ollama` (or `npm run seed`) writes a validated deterministic report so the dashboard pipeline still works in CI/cloud.

Optional every-4-hours Mac schedule: copy `ops/macos/com.trendlabs.research.plist`, edit paths, then `launchctl load ~/Library/LaunchAgents/com.trendlabs.research.plist`.

## Supabase

Apply:

1. `supabase/migrations/001_trendlabs_schema.sql`  
2. `supabase/migrations/002_agent_reports.sql`  

Set `SUPABASE_URL` + `SUPABASE_SECRET_KEY` in the worker environment to upload research reports. Never put the secret key in the Next.js client bundle.

## Automation split

| Schedule | Where | Job |
| --- | --- | --- |
| Every 15–60m (market hours) | GitHub Actions | Collect candles + run paper agents |
| Every ~4 hours | Mac `launchd` + Ollama | Shared Qwen research report |

## Deploy (Vercel Hobby)

1. Push this repo to GitHub.  
2. Import in Vercel (personal / non-commercial Hobby).  
3. Dashboard reads `public/data/snapshot.json` (Actions or local research refresh it).  
4. Do **not** add live brokerage trading credentials.

### Arena job buttons on Vercel

Vercel cannot run the Python workers, so `/api/jobs` dispatches GitHub Actions instead. Set these **server-only** env vars in the Vercel project:

| Variable | Example | Notes |
| --- | --- | --- |
| `GITHUB_TOKEN` or `GH_JOB_TOKEN` | `ghp_…` / fine-grained PAT | Needs `actions:write` (and `contents:write` on the token used by Actions to push snapshots). Never use `NEXT_PUBLIC_`. |
| `GITHUB_REPOSITORY` | `torksabig/trendlabs` | `owner/repo` that hosts `.github/workflows/trendlabs.yml` |
| `GITHUB_REF` | `main` | Optional; defaults to the Vercel git branch or `main` |

Flow: Arena button → `POST /api/jobs` → `workflow_dispatch` → Python worker on Actions → commit `public/data/snapshot.json` → Vercel redeploy. Locally, jobs still spawn Python when Actions credentials are absent (or `TRENDLAB_JOB_MODE=local`).

## Safety

- Paper trading only.  
- LLM `executes_trades` is always `false` and is schema-enforced.  
- Numerical claims in reports are reconciled against Python-calculated snapshots.  
- Delayed/IEX vs consolidated data is labeled in candle metadata.
