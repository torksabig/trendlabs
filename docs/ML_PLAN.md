# ML Plan — per-stock prediction pipeline (paper research only)

Goal: real per-stock prediction pipeline on free data + local XGBoost. No live trading,
no UI redesign, no paid deploy.

## Scheduling (operational, since 2026-10-16)

launchd jobs (`~/Library/LaunchAgents/com.trendlabs.{daily,weekly}-ml.plist`), runners in
`ops/macos/`:

| Job | Schedule | Pipeline |
|---|---|---|
| `com.trendlabs.daily-ml` | 16:10 ET daily | collect → ml-features → ml-predict --persist → ml-paper |
| `com.trendlabs.weekly-ml` | Fri 17:00 ET | + ml-train → ml-scorecard (approval registry) → predict → ml-paper |

Logs: `data/ml/logs/daily_YYYY-MM-DD.log`. No auto-retrain outside the weekly slot.

## fv2 feature/label candidates (2026-10-16, pre-declared before evaluation)

- **+8 market-context columns** (`fv2-bda1540a`): SPY ret1/5/20, SPY vol20, breadth20,
  RS-SPY 5/20, cross-sectional dispersion — built on a memoized 12-symbol panel
  (`build_panel_features`), PIT throughout.
- **Vol-scaled labels** (`label_scheme="vol_scaled"`): band = 0.5 × trailing 20d σ (shifted 1
  bar, PIT) instead of fixed 0.002 — kills the near-binary label degeneracy, gives 'sideways'
  a real class share and lowers the majority-class accuracy ceiling.

### fv2 result — gate 1/12 → 6/12 approved

| Symbol | Acc | Majority | Persistence | Gate |
|---|---:|---:|---:|---|
| QQQ | 0.444 | 0.402 | 0.335 | **approved** (+0.042) |
| SPY | 0.454 | 0.416 | 0.352 | **approved** (+0.038) |
| JPM | 0.432 | 0.423 | 0.368 | **approved** |
| AAPL | 0.454 | 0.439 | 0.369 | **approved** |
| GS | 0.438 | 0.433 | 0.381 | **approved** |
| JNJ | 0.439 | 0.430 | 0.363 | **approved** |
| NVDA | 0.414 | 0.415 | 0.349 | watch (hairline) |
| others | 0.389–0.449 | | | watch |

Brier worsens under vol-scaled labels (harder task, ~0.64 vs ~0.60) — not comparable to the
fixed-band scorecard. Macro-F1 improved broadly. Persistence still beats accuracy everywhere:
context features moved the ceiling, models remain weak — but 6/12 now clear their own
majority bar, so the paper book is allowed to trade. First positions: GS bearish, JPM bullish
(open, maturing via the daily job).

Decision rule going forward: a change is kept only if the embargoed scorecard gate improves;
regressions revert. h=5 regression candidate stays pre-declared for the next session if
directional accuracy stays under persistence after two fv2 weekly re-runs.

## Pre-declared candidate: h=5 regression on vol-normalized target (registered 2026-10-16,
## BEFORE evaluation)

Motivation: fv2 cleared the majority gate on 6/12, but accuracy margins are thin and
persistence still wins — classification on a 3-way band throws away magnitude information.
Candidate (registered before any run, no post-hoc tuning):

- **Target**: `z = fwd_ret(h=5) / (σ20 · √5)`; σ20 = trailing 20-day simple-return std (PIT,
  shift-safe). XGBRegressor (300×lr0.05×depth4, subsample 0.9, colsample 0.8, λ=1).
- **Data**: the same fv2 context panel (`fv2-bda1540a`), embargo = 5 bars, chronological 80/20.
- **Class mapping for the gate**: predicted class = up if `pred_z > 0.5/√5` (= 0.2236), down if
  `< −0.2236`, else sideways — exactly consistent with the vol-scaled band (0.5σ_daily) applied
  to the 5-day forward return.
- **Accept (all of)**: (a) holdout accuracy > majority baseline on the same window; (b) holdout
  Spearman IC(pred_z, realized_z) ≥ 0.05. Regressions in the gate vs the h=1 fv2 track revert.
- Serving impact: none this session — h=5 reg bundles are research-only (`*_reg` suffix, not
  matched by the h=1 serving glob); promotion to the paper book is a separate pre-declared step.

### h=5 regression verdict (2026-10-16): REJECTED 0/12 — with a signal-bearing diagnostic

| Symbol | Acc | Majority | Spearman IC | Gate |
|---|---:|---:|---:|---|
| SPY | 0.333 | 0.504 | **+0.124** | watch |
| MSFT | 0.332 | 0.466 | **+0.113** | watch |
| UNH | 0.329 | 0.426 | +0.080 | watch |
| CVX | 0.328 | 0.451 | +0.077 | watch |
| NVDA | 0.312 | 0.486 | +0.068 | watch |
| QQQ | 0.294 | 0.487 | +0.067 | watch |
| others | 0.258–0.343 | 0.426–0.482 | −0.044…+0.049 | watch |

Criterion (a) fails on all 12 (mapped-class accuracy 0.26–0.34, far below majority — [INFERENCE]
the |pred_z|>0.2236 class cut is almost never met by an MSE-regressed z, so the model maps nearly
everything to 'sideways'); criterion (b) passes on 8/12 — **Spearman IC up to +0.124 is the
strongest single-number signal measured anywhere in this project** (best intraday session IC was
+0.0064; best classifier margin +0.042 accuracy).

Status: candidate REJECTED as registered. The IC finding motivates — but does not by itself
authorize — a follow-up. **Registered next candidate (C2, not yet evaluated):** rank-based
policy evaluated by purged walk-forward (expanding-window folds over the full history, fresh
OOS per fold — no reuse of the burned holdout): trade sign of pred_z only when |pred_z| ≥ a
train-fold-chosen quantile threshold; report per-fold directional hit-rate + net-of-20bps P&L.
### C2 verdict (2026-10-16): 5/12 APPROVED as registered — with two recorded caveats

Walk-forward: 47–61 folds/symbol (expanding train ≥ 750 sessions, 126-session test, purge 5),
|pred_z| ≥ τ (80th pct of TRAIN preds) → ~1,000–2,000 calls per symbol over 2016–2026:

| Symbol | Folds | Calls | Hit-rate | CI95 | Bull share | ΣPnL% | Gate |
|---|---:|---:|---:|---|---:|---:|---|
| UNH | 60 | 1415 | 0.606 | [.565, .650] | 0.74 | +768 | **approved** |
| MSFT | 61 | 1377 | 0.621 | [.573, .667] | 0.83 | +727 | **approved** |
| CVX | 60 | 1403 | 0.590 | [.546, .634] | 0.67 | +230 | **approved** |
| SPY | 60 | 1078 | 0.599 | [.554, .646] | 0.69 | +195 | **approved** |
| XOM | 61 | 1587 | 0.568 | [.529, .607] | 0.73 | +66 | **approved** |
| JNJ | 61 | 1185 | 0.547 | [.498, .597] | 0.76 | +153 | watch |
| GS | 48 | 907 | 0.546 | [.492, .602] | 0.64 | −27 | watch |
| JPM | 61 | 1189 | 0.535 | [.489, .582] | 0.69 | −63 | watch |
| AAPL | 61 | 2074 | 0.532 | [.491, .568] | 0.85 | +137 | watch |
| QQQ | 47 | 789 | 0.546 | [.483, .606] | 0.48 | +74 | watch |
| AMD | 60 | 1030 | 0.506 | [.452, .560] | 0.69 | +45 | watch |
| NVDA | 47 | 1046 | 0.500 | [.446, .561] | 0.75 | +438 | watch |

**Caveat 1 — long bias:** calls are 48–85% bullish (10y bull tape). Part of the hit-rate is
drift beta, not alpha; QQQ (the most balanced book, bull 0.48) has a CI spanning 0.5 — the
control case did NOT clear. **Caveat 2 — overlap:** 5-day holds entered on consecutive days
overlap; ΣPnL double-counts concurrent exposure (per-trade edge is real, capital-curve
interpretation requires a portfolio sim).

Approved-as-registered: UNH, MSFT, CVX, SPY, XOM — research paper-book only, serving still
locked. **Registered next (C3):** beta-controlled version — per fold, compare policy hit-rate
against (i) always-long hit-rate on the same test rows and (ii) a symmetrized book (equal
long/short call quotas); gate requires the policy to beat BOTH, plus positive net P&L.
Nothing in C2 auto-promotes anything to serving.

### C3 verdict (2026-10-16): REJECTED — the C2 edge is long-market beta, not direction alpha

Same folds, same calls, three comparisons per symbol (registered gate: policy beats BOTH
controls AND pnl > 0):

| Symbol | Policy hit | Always-long (same rows) | Δ | Δ CI95 | Sym book | Gate |
|---|---:|---:|---:|---|---:|---|
| UNH | 0.5845 | 0.5569 | +0.0276 | [−0.009, +0.088] | 0.548 | approved (noise-margin) |
| SPY | 0.5863 | 0.5835 | **+0.0028** | [−0.076, +0.011] | 0.586 | approved (noise-margin) |
| AMD | 0.512 | 0.507 | +0.005 | [−0.092, +0.061] | 0.540 | watch |
| all others | | | **negative** | | | watch |

- Policy beats always-long on the same rows in **3/12** (by +0.005, +0.003, +0.028 — all inside
  noise; no symbol's Δ CI95 excludes 0 on the positive side; JPM and QQQ are significantly
  NEGATIVE).
- Symmetrized (50/50) book performs on par with or better than the policy on most symbols —
  under equal long/short exposure the direction selection adds nothing.
- Conclusion: **the C2 hit-rates were long-market beta** — the model learned "this decade drifted
  up", not cross-sectional or timing skill. C2's approvals are vacated for strategy purposes.

### Where the evidence stands (end of the pre-declared ladder)

1. Daily h=1 classifiers: 6/12 beat majority — margins hair-thin, persistence still better.
2. Intraday 15m/1h/4h at 10y depth: session-IC ≈ 0.006, Q5−Q1 ≈ 0.1–0.3 bps — microscopic.
3. h=5 regression: rank IC up to +0.124 held out — but maps to no tradable class edge.
4. C2 rank policy: 5/12 walk-forward approved — long-beta artifact per C3.
5. C3 beta control: REJECTED. **Direction-selection alpha on this universe, with free daily
   data and XGBoost, is not established.** The honest product answer is drift/long-beta, which
   is not a model result.

The daily paper book (h=1 track, 2 open trades) keeps running as the one remaining live
experiment; weekly scorecard governs approvals. Any further candidates (e.g., cross-sectional
market-neutral formulations, universe expansion, alternative targets) must be registered in
this file before evaluation, same rules.

## Phase A — lock daily models (done 2026-10-15)

1. **Data extract** — daily OHLCV for all 12 symbols in the local SQLite store:

   ```bash
   npm run collect -- --timeframe 1d --full-history      # 134,072 bars, yfinance, 0 errors
   ```

2. **Train** per-symbol XGBoost (horizon 1 day, band 0.002, chronological 80/20 split):

   ```bash
   npm run ml-train -- --timeframe 1d
   ```

3. **Predict** next-bar direction and persist into the predictions table:

   ```bash
   npm run ml-predict -- --timeframe 1d --horizon 1 --persist
   ```

4. **Scorecard** — held-out accuracy / macro-F1 / multiclass Brier vs majority,
   persistence, and always-up baselines, with the Phase A gate:

   ```bash
   npm run ml-scorecard -- --timeframe 1d      # wrapper: see below
   ```

   Direct: `PYTHONPATH=workers .venv/bin/python -m trendlabs.ml_scorecard --timeframe 1d`

5. **Outcome eval** on completed horizons (rule-based forecasts backfilled now;
   ML predictions score after their 1-day horizon completes):

   ```bash
   npm run predict-eval -- --timeframe 1d --backfill
   ```

### Phase A result (2026-10-15)

| Symbol | Acc | MacroF1 | Brier | Majority | Persistence | Gate |
|---|---:|---:|---:|---:|---:|---|
| AMD | 0.481 | 0.332 | 0.562 | 0.480 | 0.434 | **approved** |
| QQQ | 0.463 | 0.308 | 0.615 | 0.461 | 0.392 | **approved** |
| UNH | 0.467 | 0.301 | 0.596 | 0.461 | 0.402 | **approved** |
| AAPL | 0.454 | 0.299 | 0.599 | 0.475 | 0.396 | watch |
| NVDA | 0.480 | 0.313 | 0.558 | 0.511 | 0.469 | watch |
| others | 0.414–0.452 | | 0.596–0.639 | | | watch |

**3/12 approved** (AMD, QQQ, UNH), margins +0.001…+0.006 over majority. Persistence beats
every model on directional accuracy. Full table: `data/ml/scorecard_1d.md`, raw:
`data/ml/scorecard_1d.json`.

Gate rule: model stays "watch" unless held-out accuracy > majority baseline accuracy.
Approved models are the only ones allowed to serve predictions downstream.

### Rule-forecast backfill eval (separate from ML, method benchmark)

`npm run predict-eval -- --timeframe 1d --backfill --backfill-bars 120`
(new `--backfill-bars` caps the scan to recent completed horizons; legacy
`--backfill` walks full history and takes >10 min locally):

- 240 rule-based forecasts inserted, 500 predictions scored, 0 skipped.
- Rule agents realized accuracy **0.402** with mean confidence **0.902** —
  badly overconfident at every bucket (acc 0.00–0.45 vs claimed 0.64–0.92).
- Implication for Phase D: thresholds must come from calibrated ML probabilities,
  not raw rule-agent confidence; rule agents stay method benchmarks only.

## Phase B — features & leakage hardening (done 2026-10-15)

1. **`feature_version` frozen** — `trendlabs.features.FEATURE_VERSION` = `fv1-` + sha1 of the
   FEATURE_COLUMNS list (any add/remove/reorder bumps it). Recorded in model bundles, train
   reports, scorecard rows, and every `--persist`-ed prediction.
2. **Embargo** — `ml_train._time_split(embargo=h)`: `embargo ≥ horizon` rows dropped between
   train end and test start (train labels can't share their forward window with test features).
   Bundle carries `embargo`; the scorecard reproduces the identical split.
3. **Leakage tests** — `test_ml.py::PhaseBTests`: (a) perturbing the last candle must not change
   any earlier feature row (caught and fixed a real warm-up lookahead in
   `regimes.classify_series`'s full-series-median fallback → now expanding-median); (b) embargo
   gap invariants; (c) `market_features` roundtrip + prediction lineage. 12 tests pass.
4. **`market_features` persistence** — SQLite table keyed `(symbol, timeframe, ts, feature_version)`;
   `npm run ml-features` backfills snapshots for the whole candle history (133,200 rows @1d,
   `fv1-1ff97c69`); `ml-predict --persist` writes the as-of bar's snapshot with each prediction.

### Phase B effect on the gate (honest change)

Embargo removed the 1-row label/feature overlap between train and test. Re-trained all 12 models:

- Gate drops **3/12 → 1/12**: only UNH (acc 0.471 vs majority 0.461). AMD (0.477 vs 0.480) and
  QQQ (0.460 vs 0.461) were split artifacts — Phase A "approvals" revoked.
- NVDA still has the best Brier (0.558) but sits below its 0.511 majority class share.
- Persistence still beats every model directionally. Current table: `data/ml/scorecard_1d.md`.
- Weekly retrain cadence (Phase D) re-runs the embargoed split; a symbol is approved only while
  its holdout beats majority on the current window.

## Phase C — intraday only if daily gate passes

- Train `1h` next (`npm run ml-train -- --timeframe 1h` after
  `npm run collect -- --timeframe 1h --full-history`).
- Deep 15m / Alpaca 5m status: the 10y Alpaca backfill exists
  (`data/intraday/bars/alpaca/`, 2016→now × 12 symbols). Purged-WF baselines on the deep
  15m panel (497 folds, 2,486 sessions) show statistically real but economically
  microscopic structure — best session IC +0.0064 (CI [0.0032, 0.0094]), Q5−Q1 ≤ 0.26 bps
  vs 20 bps round-trip cost; reversal family −0.007…−0.011 (CIs exclude zero).
  Pre-declared H1/H2/H3 all REJECTED (`docs/INTRADAY_V1.md`) → boosting stays locked.
  Treat intraday as an optional upgrade, not the product metric.

## Phase D — paper signals (implemented 2026-10-15, still not live)

1. **Calibrated policy** — one-vs-rest isotonic on up/down fitted on the TRAIN window only;
   calibrators kept **only if they improve train-window Brier** (causal keep/drop — holdout
   never informs it). Policy: Bullish if calibrated P(up) ≥ θ, Bearish if calibrated P(down) ≥ θ,
   else Watch; θ swept {0.40…0.60} on train precision with ≥ 30 calls, ties → lowest θ
   (`ml_train._choose_policy_thresholds`). All 12 models kept their calibrators (train Brier
   0.52–0.57 → 0.38–0.53), θ = 0.55–0.60. Honest caveat: isotonic's train gain does NOT
   generalize (holdout Brier rises 0.56–0.63 → 0.62–0.71 in the report columns) — the report
   keeps both numbers visible; θ being conservative (0.6) is partly the correct response.
2. **Paper trades, separate from accuracy** — `paper_trades` SQLite table;
   `ml-predict --persist` opens a trade only for **scorecard-approved** symbols with a
   non-watch signal (execution: enter open(t+1), exit close(t+h)); `npm run ml-paper`
   matures open trades against candle history and writes `data/ml/paper_pnl.json`
   (win-rate / mean / total P&L, by policy). Today's live book: **0 trades** — the only
   approved symbol (UNH) signals Watch (P(up)=0.53 < θ=0.60); the bullish/bearish signals
   from unapproved symbols are shadow-only. This is the gate working as designed.
3. **Weekly cadence, no auto-retrain** — nothing retrains on a bar; the only approval
   path is a scorecard re-run writing `data/ml/approved_models.json`
   (`ml-train → ml-scorecard → ml-predict --persist → ml-paper`). Candidate models
   become approved only while their embargoed holdout beats the majority baseline.

Phase D commands:
```bash
DYLD_FALLBACK_LIBRARY_PATH="$HOME/libomp/lib" npm run ml-train -- --timeframe 1d
npm run ml-scorecard -- --timeframe 1d
DYLD_FALLBACK_LIBRARY_PATH="$HOME/libomp/lib" npm run ml-predict -- --timeframe 1d --horizon 1 --persist
npm run ml-paper
```

## Phase E — later (deferred)

Dashboard AI cards / charts, PDF research reports, Vercel / always-on worker deploy.

## Local environment (macOS, no Homebrew)

The repo targets Python 3.12 (CI: `.github/workflows/trendlabs.yml`). Local setup:

```bash
# Python 3.12 venv at .venv (npm scripts call ../.venv/bin/python)
python3 -m pip install --user uv          # uv is a single static binary
uv python install 3.12
rm -rf .venv && uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r workers/requirements.txt pytest

# XGBoost needs OpenMP (no brew on this Mac): fetch libomp from conda-forge once,
# extract libomp.dylib to ~/libomp/lib, then set at LAUNCH time (dyld ignores
# runtime-set env):
export DYLD_FALLBACK_LIBRARY_PATH="$HOME/libomp/lib"
```

`xgboost 3.4.1` verified working locally with this setup. CI (Linux) needs none of it.

## Layout mapping (plan ↔ repo)

| Plan | Repo reality |
|---|---|
| `models/{SYMBOL}/1d/` | `data/ml/models/{SYMBOL}_1d_h1.joblib` (+ encoder/feature_columns bundle) |
| approved model | gate = "approved" in `data/ml/scorecard_1d.json` |
| `npm run extract / train-ml / ml-eval` | `npm run collect / ml-train / ml-scorecard + predict-eval` |
| majority/trend baselines | computed in `trendlabs.ml_scorecard` on the identical holdout |
