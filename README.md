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

## Quick start

```bash
npm install
python3 -m venv .venv && source .venv/bin/activate
pip install -r workers/requirements.txt

npm run seed
npm run dev
```

## Deploy (Vercel Hobby)

Import this GitHub repository in Vercel. Dashboard reads `public/data/snapshot.json`.
