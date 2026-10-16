# ML Plan — per-stock prediction pipeline (paper research only)

Goal: real per-stock prediction pipeline on free data + local XGBoost. No live trading,
no UI redesign, no paid deploy.

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

## Phase D — paper signals (not live)

1. Bullish / Bearish / Watch from calibrated probs + thresholds.
2. Paper trades logged separately from model accuracy.
3. No auto-retrain every bar — weekly candidate vs approved model only.

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
