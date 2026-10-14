# TrendLab

Multi-agent stock market research and **paper-trading** platform targeting **$0/month** infrastructure.

Four independent trend-following agents each manage a simulated $10,000 book across 12 stocks and ETFs. A fifth **shared research agent** (switchable free-tier cloud LLMs, local Ollama, or rule-based fallback) interprets verified Python analytics and proposes experiments — it never executes trades.

## Architecture

| Layer | Technology | Cost posture |
| --- | --- | --- |
| Dashboard | Next.js + Tailwind on Vercel Hobby | Free |
| Charts | TradingView Lightweight Charts | Free |
| Market data | Alpaca Basic (optional) or Yahoo Finance fallback | Free tier / public |
| Trading agents | Deterministic Python strategies | Free |
| Research agent | Free-tier cloud LLMs / Ollama / fallback | $0-friendly |
| Automation | GitHub Actions (data + paper trades + research) | Free within limits |
| Local AI schedule | macOS `launchd` every ~4 hours (optional) | Free |
| Database | SQLite workers + Supabase schema | Free tier |

```
Alpaca / Yahoo → Python indicators & agents → snapshot JSON
                      ↓
     Gemini / Groq / OpenRouter / Workers AI / Cerebras / Mistral
              or Ollama (Mac) or rule-based fallback
                      ↓
         agent_reports → Supabase / snapshot → Vercel dashboard
```

Pick a provider in the Arena sidebar (or `TRENDLAB_LLM_PROVIDER`). On Vercel, research jobs dispatch GitHub Actions with that provider input and use repo secrets for API keys.

## Universe

AAPL, MSFT, NVDA, AMD, JPM, GS, XOM, CVX, JNJ, UNH, SPY, QQQ

## Agents

**Trading (deterministic)**  
1. EMA Trend — EMA 20/50/200 confirmation  
2. Breakout — 20-bar high entries / 10-bar low exits  
3. Momentum — ROC + ADX + volume filter  
4. Supertrend — volatility-adjusted trend follower  

**Research (switchable LLM)**  
5. Shared research agent — explains regimes, risks, agent performance, and strategy experiments for Python to backtest (`executes_trades: false`)

Shared trading rules: long-only, max 3 positions, 1% equity risk budget, fees + slippage, idempotent fills.

## Quick start

```bash
npm install
python3 -m venv .venv && source .venv/bin/activate
pip install -r workers/requirements.txt

npm run extract       # deep historical OHLCV for all configured symbols
npm run seed          # collect + simulate + research fallback report
npm run dev           # http://localhost:3456
```

## Market data extraction

Historical OHLCV for the **configured universe only** (see `workers/trendlabs/config.py` / `src/lib/instruments.ts`).

```bash
# Full practical history: daily (up to max/~10y), then 1h / 15m / 4h within free-provider limits
npm run extract

# One symbol, daily only (recommended first ML path)
npm run extract -- --symbols AAPL --timeframes 1d

# Refresh without rewriting snapshot JSON
npm run extract -- --no-snapshot

# Unit tests (upserts, duplicates, quality filters, extract orchestration)
npm run test:extract
```

Details: [docs/DATA_EXTRACTION.md](docs/DATA_EXTRACTION.md) · Architecture audit: [docs/ARCHITECTURE_AUDIT.md](docs/ARCHITECTURE_AUDIT.md)

Provider notes:

- Prefer existing Alpaca Basic keys when present; otherwise Yahoo Finance (no key).
- Missing / invalid bars are **dropped**, never invented.
- Incomplete in-progress candles are dropped by default.
- Upserts are idempotent on `(symbol, timeframe, ts)`.

### Research LLM providers ($0-friendly)

Set `TRENDLAB_LLM_PROVIDER=auto` (default) to try free-cloud keys in order, then Ollama, then rule-based fallback. Or pick one explicitly: `google`, `groq`, `openrouter`, `cloudflare`, `cerebras`, `mistral`, `ollama`, `fallback`.

| Provider | Env / Actions secret | Default model |
| --- | --- | --- |
| Google AI Studio | `GOOGLE_AI_API_KEY` or `GEMINI_API_KEY` | `gemini-2.0-flash` |
| Groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter/free` |
| Cloudflare Workers AI | `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_API_TOKEN` | `@cf/meta/llama-3.1-8b-instruct` |
| Cerebras | `CEREBRAS_API_KEY` | `llama3.1-8b` |
| Mistral | `MISTRAL_API_KEY` | `mistral-small-latest` |
| Ollama (local) | `OLLAMA_HOST` / `OLLAMA_MODEL` | `qwen3:14b-q4_K_M` |
| Fallback | (none) | deterministic report |

```bash
# Cloud free-tier example
export GROQ_API_KEY=...
npm run research -- --symbol AAPL --provider groq

# Local Mac Ollama
ollama pull qwen3:14b-q4_K_M
npm run research -- --symbol AAPL --provider ollama

# No keys → validated rule-based report (CI-safe)
npm run research -- --symbol AAPL --provider fallback
```

Optional every-4-hours Mac schedule: copy `ops/macos/com.trendlabs.research.plist`, edit paths, then `launchctl load ~/Library/LaunchAgents/com.trendlabs.research.plist`.

## Evaluation (V2)

Deterministic historical replay is separate from forward paper-trading state.

```bash
npm run backtest        # full-period backtest → data/backtests/
npm run walk-forward    # rolling OOS folds (+ full backtest) → data/walk_forward/
npm run experiment      # control vs variant OOS accept/reject → data/experiments/
npm run predict-eval    # score forecasts + calibration table → data/prediction_eval/
npm run repeating-trends  # find historically repeating close-path motifs
```

Repeating-trend method notes: [docs/REPEATING_TRENDS.md](docs/REPEATING_TRENDS.md).

- Next-bar open fills, fees/slippage, Sharpe / Sortino / drawdown / α vs SPY  
- Walk-forward combines **out-of-sample** folds only  
- Experiment engine compares parameter variants on OOS folds (never mutates paper books)  
- Prediction eval stores 1h/4h/24h realized moves and confidence calibration  
- Market regimes + strategy configs in `workers/trendlabs/`

## Supabase

Apply:

1. `supabase/migrations/001_trendlabs_schema.sql`  
2. `supabase/migrations/002_agent_reports.sql`  
3. `supabase/migrations/003_backtests_experiments.sql`  
4. `supabase/migrations/004_market_regimes_walk_forward.sql`  
5. `supabase/migrations/005_ml_market_data.sql` (extraction metadata + ML tables)   

Set `SUPABASE_URL` + `SUPABASE_SECRET_KEY` in the worker environment to upload research reports. Never put the secret key in the Next.js client bundle.

## Automation split

| Schedule | Where | Job |
| --- | --- | --- |
| Every 15–60m (market hours) | GitHub Actions | Collect candles + run paper agents |
| On demand (Arena / workflow_dispatch) | GitHub Actions | Research with selected LLM provider |
| Every ~4 hours (optional) | Mac `launchd` + Ollama | Local research report |

## Deploy (Vercel Hobby)

See **[docs/DEPLOY.md](docs/DEPLOY.md)** for the one-click production checklist (Vercel claim/import, GitHub Actions, env vars).

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

Add the LLM API keys above as **GitHub Actions repository secrets** (used when Arena triggers research). Optionally mirror the same keys (or `TRENDLAB_LLM_PROVIDER`) in Vercel so `/api/llm` can show which providers are configured — secrets are never returned to the browser, only booleans.

Flow: Arena **Research LLM** selector → `POST /api/jobs` with `provider` → `workflow_dispatch` inputs → Python worker on Actions → commit `public/data/snapshot.json` → Vercel redeploy. Locally, jobs still spawn Python when Actions credentials are absent (or `TRENDLAB_JOB_MODE=local`).

## Safety

- Paper trading only.  
- LLM `executes_trades` is always `false` and is schema-enforced.  
- Numerical claims in reports are reconciled against Python-calculated snapshots.  
- Delayed/IEX vs consolidated data is labeled in candle metadata.
