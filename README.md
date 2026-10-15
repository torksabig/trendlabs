# TrendLab

We're building a paper-trading sandbox to test whether simple trend-following rules hold up on a small equities universe. Four deterministic agents each manage a simulated **$10,000** book across a fixed 12-symbol set. A fifth **research LLM** (free-tier cloud, local Ollama, or a rule-based fallback) reads Python-verified analytics and proposes experiment ideas — it **never** places orders.

The default stack aims for low or zero ongoing cost: Vercel Hobby for the dashboard, GitHub Actions for workers, optional free-tier market data and LLMs, SQLite locally, optional Supabase. That is a cost posture, not a performance claim.

## Limits (read this first)

This is exploratory research infrastructure, not a finished edge-finding system.

- **Short history** on some symbols and timeframes — sample sizes can be thin.
- **Survivorship bias** is not controlled; the universe is a fixed mega-cap / ETF list chosen for convenience.
- **ML and information-coefficient results are early and weak** so far; treat them as scaffolding, not evidence.
- Everything here is **paper only**. The LLM never trades. Accepted experiment variants are never auto-promoted into the paper agents.
- Metrics are diagnostics under stated assumptions (fees, slippage heuristics, simple SPY excess return). They do not prove alpha.

## Architecture

| Layer | Stack | Notes |
| --- | --- | --- |
| Dashboard | Next.js + Tailwind on Vercel Hobby | UI only |
| Charts | TradingView Lightweight Charts | |
| Market data | Alpaca Basic (optional) or Yahoo Finance | Free tier / public; check feed quality |
| Trading agents | Deterministic Python strategies | Paper fills only |
| Research agent | Free-tier cloud LLMs / Ollama / fallback | Explains and proposes; does not trade |
| Automation | GitHub Actions (collect, simulate, research) | Within Actions free limits |
| Local AI (optional) | macOS `launchd` + Ollama | |
| Persistence | SQLite workers + optional Supabase schema | |

```
Alpaca / Yahoo → Python indicators & agents → snapshot JSON
                      ↓
     Gemini / Groq / OpenRouter / Workers AI / Cerebras / Mistral
              or Ollama (Mac) or rule-based fallback
                      ↓
         agent_reports → Supabase (optional) / snapshot → Vercel dashboard
```

Python workers own all numbers (fills, metrics, regimes, experiment decisions). Vercel serves the UI and can dispatch GitHub Actions via `/api/jobs` when Arena buttons run remotely. Locally, jobs spawn Python when Actions credentials are absent (or `TRENDLAB_JOB_MODE=local`).

See **[docs/DEPLOY.md](docs/DEPLOY.md)** for Vercel + Actions wiring (`GITHUB_TOKEN` / `GITHUB_REPOSITORY`, LLM secrets on Actions, never `NEXT_PUBLIC_` for tokens).

## Universe

AAPL, MSFT, NVDA, AMD, JPM, GS, XOM, CVX, JNJ, UNH, SPY, QQQ

Fixed list for repeatable runs. Not a survivorship-adjusted universe.

## What runs

### Trading agents (deterministic, paper only)

| ID | Strategy | Default logic |
| --- | --- | --- |
| `ema_trend` | EMA Trend | Long when EMA 20 > 50 > 200 and close > fast EMA; optional ADX gate for experiments |
| `breakout` | Breakout | Long on close above prior 20-bar high; flatten below 10-bar low |
| `momentum` | Momentum | Positive N-bar ROC + ADX ≥ 22 + EMA20 > EMA50 + volume > 20-bar SMA |
| `supertrend` | Supertrend | Follow ATR-based Supertrend direction (period 10, mult 3) |

Shared risk defaults: **long-only**, **max 3 positions**, **1% equity** risk budget per entry, ATR-based stops, **5 bps fee + 5 bps slippage**, idempotent fills. Agent `signal_strength` is a fixed heuristic in `[0, 1]` — **not** calibrated hit-rate confidence.

### Research LLM

Switch provider in the Arena sidebar or via `TRENDLAB_LLM_PROVIDER` (`auto` tries free-cloud keys → Ollama → fallback). It summarizes regimes, risks, and agent stats, and can propose structured experiment variants for Python to evaluate. Schema forces `executes_trades: false`. Numerical claims are reconciled against Python snapshots when possible.

### Backtesting & walk-forward

Historical replay is **separate** from the forward paper books. Full-period backtests write `data/backtests/`; rolling OOS folds write `data/walk_forward/` (the leaderboard stitches **test folds only**). Useful for comparing rules under one protocol — not a guarantee that anything transfers live.

### Regimes

Rule-based labels on each bar (`risk_off`, `high_volatility`, `trending_bullish` / `trending_bearish`, `low_volatility`, `sideways`, `unknown`), plus trade attribution and heuristic fit scores. These are labels we compute for slicing results, not ground-truth market regimes.

### Experiment engine

Control vs variant on walk-forward OOS folds → `accept` / `reject` / `insufficient`. Catalog experiments run without an LLM; `npm run llm-experiment` asks the research LLM for 1–3 structured proposals, then Python evaluates them. **Accepted variants are never auto-promoted** into live paper agents.

### Predictions & calibration

Rule-based hourly outlooks (`indicator_score_v1`: up / down / sideways) from EMA/RSI/ADX/trend score. Confidence is a **heuristic** (score magnitude + ADX boost), not a probability from historical calibration. `predict-eval` scores 1h / 4h / 24h realized moves and builds a confidence-bucket calibration table so we can see how wrong the heuristic is.

### XGBoost ML (research)

Per-symbol classifiers on candle features, time-ordered holdout (no shuffle). Artifacts under `data/ml/`. Research only — `executes_trades: false`. Early holdout numbers have been weak; do not treat them as a trading signal.

### Prediction Engine V1 (cross-sectional ranking)

Separate from the ternary classifier: daily features → purged walk-forward \(\hat{z}\) → OOS quintiles and a simple paper book (enter next open, exit close \(t+5\)), with 10/20 bps cost stress and a fixed V1 gate. Artifacts under `data/prediction_engine/` (`pe-dataset`, `pe-baselines`, `pe-train`, `pe-eval`, `pe-volatility`, `pe-intraday`). Volatility V1 compares 5-day RV persistence vs HAR log-vol (`volatility_results.json`). Intraday V1 is a separate exploratory slice: Yahoo **5m** bars on a small liquid set → 15m/1h/4h close-to-close baselines (`intraday_results.json`); Yahoo 5m history is short (~60d) — treat as scaffolding. All outputs carry **EXPLORATORY — SURVIVORSHIP BIAS NOT FULLY CONTROLLED**. Not live trading.

---

## Math & methodology

What we compute, and why — so runs are comparable. None of this is offered as proof of edge.

### Execution model (backtest + paper sim)

1. Indicators and signals use completed bars through index `i` (no look-ahead).
2. Orders fill at bar `i+1` **open**, never the signal-bar close.
3. Buy: `fill = open × (1 + slippage_bps/10_000)`; sell: `fill = open × (1 − slippage_bps/10_000)`.
4. Fee = `notional × fee_bps/10_000` (default **5 bps** each side; slippage default **5 bps**).
5. Stop-losses checked against the fill bar’s low/high before discretionary exits.
6. Position size from risk budget: `qty ≈ (equity × risk_per_trade) / risk_per_share`, capped by cash, max positions, and min notional ($25).

### Performance metrics

Computed on the equity curve and paired BUY→SELL round-trips (`workers/trendlabs/metrics.py`):

| Metric | Definition used here |
| --- | --- |
| **Total return** | Ending equity / starting cash − 1 |
| **CAGR** | `(end/start)^(1/years) − 1` using wall-clock span of the curve |
| **Sharpe** | `(mean r / std r) × √periods_per_year` (sample std); periods inferred from timeframe (e.g. 1h ≈ 252×6.5) |
| **Sortino** | Same with downside-only std; `None` if no negative returns |
| **Max drawdown** | Min of equity / running peak − 1 |
| **Profit factor** | Gross winning round-trips / |gross losing| |
| **Exposure** | Fraction of equity points with invested capital (equity − cash > 0) |
| **α vs SPY** | Strategy total return − buy-and-hold SPY return over the same window (**simple excess return**, not CAPM β-adjusted alpha) |
| **Win rate / streaks** | From fee-adjusted round-trip PnL |

We use these to compare agents and experiment variants under one protocol. High Sharpe in a thin sample is not evidence by itself.

### Walk-forward OOS

- Rolling folds: warm-up → **calibrate** window (recorded for future tuning) → **test** window.
- Defaults (typical CLI): warm-up 120, calibrate 80, test 60 bars; step defaults to `test_bars`.
- Indicators may use history before `test_start`; **metrics combine OOS test periods only**.
- Fold equity curves are **stitched by chaining period returns** onto continuous capital (each fold restarts relative scale from its own start).
- Leaderboard prefers `mean_fold_sharpe`, then Sharpe, then total return.

### Regime classification

Mutually exclusive labels, **priority order**:

1. **risk_off** — 20-bar return ≤ −6%, or (close below EMA200 and vol ≥ high threshold and ret < −2%)
2. **high_volatility** — vol ≥ expanding-median×1.35 and ADX < 22
3. **trending_bullish** — ADX ≥ 22, EMA20 > EMA50, close > EMA50
4. **trending_bearish** — ADX ≥ 22, EMA20 < EMA50, close < EMA50
5. **low_volatility** — vol ≤ median×0.65 and ADX < 18
6. **sideways** — otherwise (including low ADX chop)

Vol thresholds are adaptive (expanding median of 20-bar annualized vol). Regime fit labels (`excellent` / `good` / `mediocre` / `poor` / `insufficient`) are heuristics from per-regime trade count, win rate, and average PnL.

### Experiment accept / reject

Defaults in `AcceptanceCriteria`:

| Gate | Default | Effect |
| --- | --- | --- |
| Min trades | 12 | Else `insufficient` |
| Sharpe lift | +0.15 vs control | Uses `mean_fold_sharpe` when present |
| Max DD worsen | ≤ 3 pp | Variant may not deepen DD beyond this |
| Fold robustness | ≥ 50% of folds | Variant Sharpe ≥ control |

Fail any (except trade count → `insufficient`) → `reject`. Pass all → `accept`. Gates only decide whether a variant looks better than control under this protocol. They never mutate paper books.

### Prediction scoring & calibration

- Realized move: percent change from signal-bar close to a forward bar (1 / 4 / 24 hour offsets on 1h data; wall-clock fallback if sparse).
- Directional “hit” uses a ±0.05% band for sideways.
- Calibration table buckets predicted confidence (50–60% … 90%+) vs realized accuracy; flags **overconfident** when accuracy lags bucket midpoint by >5pp (n ≥ 5). Rule-based confidence is **not** historically calibrated.

### ML features, labels, split

- **Features**: returns (1–20), EMA spreads / price vs EMA, RSI (+norm), ADX, ATR%, vol, volume z/ratio, range/location, Supertrend direction, momentum, regime one-hots (`features.py`).
- **Label**: forward return over `horizon` bars → `up` / `down` / `sideways` using a timeframe band (e.g. 1h ≈ 0.08%, 1d ≈ 0.2%). Label uses `shift(-h)` only — no look-ahead in features.
- **Split**: last ~20% of rows (time-ordered, min test size enforced) held out; no shuffle. XGBoost (`hist`, depth 4, 120 trees) reports accuracy, macro-F1, directional accuracy on non-sideways preds, and feature importance.

---

## Quick start

Scripts below match `package.json` (Python via `.venv`).

```bash
npm install
python3 -m venv .venv && source .venv/bin/activate
pip install -r workers/requirements.txt

npm run seed          # collect + simulate + research fallback report
npm run dev           # http://localhost:3456
```

### Workers

```bash
npm run collect
npm run simulate
npm run research -- --symbol AAPL --provider fallback
npm run predict

npm run backtest
npm run walk-forward
npm run experiment
npm run llm-experiment
npm run predict-eval
npm run ml-train -- --symbols QQQ AAPL --timeframe 1d
npm run ml-predict -- --score-holdout
npm run ml-predict -- --timeframe 1d

npm run pe-dataset
npm run pe-baselines
npm run pe-train
npm run pe-eval
npm run pe-volatility
npm run pe-intraday
```

### Research LLM providers

`TRENDLAB_LLM_PROVIDER=auto` (default) tries free-cloud keys in order, then Ollama, then rule-based fallback. Explicit: `google`, `groq`, `openrouter`, `cloudflare`, `cerebras`, `mistral`, `ollama`, `fallback`.

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
export GROQ_API_KEY=...
npm run research -- --symbol AAPL --provider groq

ollama pull qwen3:14b-q4_K_M
npm run research -- --symbol AAPL --provider ollama

npm run research -- --symbol AAPL --provider fallback
```

Optional Mac schedule: copy `ops/macos/com.trendlabs.research.plist`, edit paths, `launchctl load ~/Library/LaunchAgents/com.trendlabs.research.plist`.

### Supabase (optional)

Apply migrations `001`–`004` under `supabase/migrations/`. Set `SUPABASE_URL` + `SUPABASE_SECRET_KEY` in the **worker** environment to upload research reports. Never put the secret key in the Next.js client bundle.

### Automation split

| Schedule | Where | Job |
| --- | --- | --- |
| Every 15–60m (market hours) | GitHub Actions | Collect candles + run paper agents |
| On demand (Arena / `workflow_dispatch`) | GitHub Actions | Research with selected LLM provider |
| Every ~4 hours (optional) | Mac `launchd` + Ollama | Local research report |

---

## Safety / non-goals

- **Paper trading only** — no live brokerage, no order routing, no real capital.
- LLM and ML paths set `executes_trades: false` (schema-enforced where applicable).
- Experiment accept ≠ promote; paper agent params stay on defaults unless you change code/config yourself.
- Delayed/IEX vs consolidated feeds are labeled in candle metadata; do not treat IEX as SIP.
- Not a broker, not investment advice, not a claim of edge. Metrics are research diagnostics under the assumptions above.
