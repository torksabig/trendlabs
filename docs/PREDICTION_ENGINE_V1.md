# TrendLab Prediction Engine V1 — Engineering Spec

**Status:** Research / exploratory  
**Scope this pass:** Phases A + B (+ light Phase C scaffold)  
**Out of scope:** Dashboard UI, conformal calibration, full quantile Q10/Q50/Q90 (interfaces may be stubbed), point-in-time S&P 500 membership, live capital, intraday, brokerage deploy

**Disclaimer (all outputs):** `EXPLORATORY — SURVIVORSHIP BIAS NOT FULLY CONTROLLED`

---

## Purpose

Build a **cross-sectional equity prediction research stack** that ranks liquid names by expected risk-adjusted forward return, under a realistic next-session execution rule. The engine is separate from TrendLab’s existing rule agents and from the older ternary XGBoost classifier (`ml_train.py`).

V1 answers: *Given only information available at the close of day \(t\), can we rank names so that higher ranks earn higher volatility-normalized 5-day actionable returns when entered at the next open?*

Rule-based trading agents remain in the lab as **deterministic benchmarks**, not as the prediction model.

---

## Problem Framing

| Item | Definition |
| --- | --- |
| Decision time | End of session \(t\) (features use bars ≤ \(t\) only) |
| Entry | Next session open: \(P^{\text{open}}_{t+1}\) |
| Exit | Close of session \(t+5\): \(P^{\text{close}}_{t+5}\) |
| Horizon | Actionable ~5 trading-day hold after next open |
| Output | Cross-sectional score / rank of expected \(z\) (and later calibrated intervals) |
| Capital | Research ranking only — no live allocation this pass |

Agents (`ema_trend`, `breakout`, `momentum`, `supertrend`) are **rule benchmarks** for comparing economic usefulness of ranks vs simple strategies, not predictors of \(z\).

---

## Target Definition

### Actionable log return

\[
R_{i,t} = \ln\left(\frac{P^{\text{close}}_{i,t+5}}{P^{\text{open}}_{i,t+1}}\right)
\]

### Volatility-normalized target

Let \(\sigma_{i,t}\) be the sample standard deviation of the past **20** daily close-to-close log returns ending at \(t\) (no future bars):

\[
r_{i,u} = \ln\left(\frac{P^{\text{close}}_{i,u}}{P^{\text{close}}_{i,u-1}}\right), \quad
\sigma_{i,t} = \operatorname{std}(r_{i,t-19},\ldots,r_{i,t})
\]

\[
z_{i,t} = \frac{R_{i,t}}{\sigma_{i,t}\sqrt{5}}
\]

Rows with insufficient history, missing open/close, or \(\sigma_{i,t}=0\) are dropped.

---

## Next-Session Execution Rule

Documented in code comments and dataset manifests:

1. Signal / features computed on completed daily bar \(t\).
2. Hypothetical long entered at **open of \(t+1\)** (no same-bar fill).
3. Position exited at **close of \(t+5\)**.
4. Evaluation targets use that P&L path; no intraday fills in V1.
5. Walk-forward folds **purge ≥ 5 trading days** between train end and test start so label windows do not leak into training.

---

## Feature Families (Point-in-Time)

All features use information ≤ \(t\) only.

### Momentum

| Feature | Definition (approx.) |
| --- | --- |
| `MOM20` | \(P^{\text{close}}_t / P^{\text{close}}_{t-20} - 1\) |
| `MOM60` | \(P^{\text{close}}_t / P^{\text{close}}_{t-60} - 1\) |
| `MOM12_1` | 12−1 momentum: \(P^{\text{close}}_{t-21} / P^{\text{close}}_{t-252} - 1\) (skip last ~1 month) |

### Volatility

| Feature | Definition |
| --- | --- |
| `VOL20` | \(\sigma_{i,t}\) (20-day log-return std) |
| `VOL120` | 120-day log-return std |
| `VOLRatio` | `VOL20 / VOL120` |
| `ATRNorm` | ATR(14) / close |

### Volume / liquidity

| Feature | Definition |
| --- | --- |
| `DollarVolume` | close × volume |
| `VolumeZ` | (volume − SMA20(volume)) / STD20(volume) |

### Relative strength

| Feature | Definition |
| --- | --- |
| `RS_SPY_20` | `MOM20` − SPY `MOM20` on same date |

### Market context (broadcast per date)

| Feature | Definition |
| --- | --- |
| `SPY_MOM20` | SPY 20-day momentum |
| `SPY_VOL20` | SPY 20-day vol |
| `XS_Dispersion` | Cross-sectional std of `MOM20` across universe on date \(t\) |

---

## Universe & Bias Control

- **V1 fixed liquid universe:** symbols available in TrendLab daily OHLCV store (e.g. AAPL, MSFT, NVDA, AMD, JPM, GS, XOM, CVX, JNJ, UNH, SPY, QQQ — subject to data coverage).
- SPY is the market benchmark for relative/context features; SPY may be excluded from ranking panels when scoring stock-only spreads.
- **No** point-in-time S&P 500 membership reconstruction in this pass.
- Every dataset export and metrics JSON must carry:

  `EXPLORATORY — SURVIVORSHIP BIAS NOT FULLY CONTROLLED`

---

## Phase A — Research Dataset

1. Load daily OHLCV from existing collect/store.
2. Build targets \(R\), \(z\) and feature matrix.
3. Versioned export under `data/prediction_engine/`:
   - panel table (parquet preferred, csv fallback)
   - `manifest.json` with schema version, feature list, universe, purge/execution notes, git sha, timestamps, disclaimer
4. Dataset version id (e.g. `pe_dataset_v1`).

---

## Phase B — Baselines

Deterministic baselines on the **same** dataset and purged walk-forward:

| Baseline | Score |
| --- | --- |
| Random ranking | Fixed-seed random scores (reproducible) |
| Historical mean | Per-symbol train-set mean of \(z\) (cross-sectional rank; global mean fallback) |
| 12−1 momentum | Rank by `MOM12_1` |
| Ridge regression | Ridge on standardized feature matrix; scaler + model **fit on train only** |

### Metrics (cross-sectional, chronological)

Per test day with ≥2 names:

- Spearman IC(\(\hat{z}\), \(z\))
- Aggregate: **MeanIC**, **ICIR** (MeanIC / std(IC)), **PositiveICRate**
- If cross-section \(N\) allows (≥10 preferred): simple **Q5−Q1** mean \(z\) spread (top vs bottom quintile of predicted scores)

**No random train/test splits** — chronological purged walk-forward only.

---

## Phase C — Model (Light Scaffold This Pass)

- XGBoost **regressor** predicting \(z\) (not ternary classification).
- Same purged walk-forward and IC metrics as baselines.
- Simple regularized hyperparameters; **no** hyperparameter hunt.
- Stub interfaces for future quantile heads (Q10 / Q50 / Q90) and conformal calibration — not fully implemented.

---

## Evaluation Protocol

1. Sort dates chronologically.
2. Expanding or rolling train window; default expanding with minimum train length.
3. Purge gap \(G \ge 5\) trading days after last train label-origin date before first test date.
4. Fit scalers/models on train fold only; apply to test.
5. Report IC table for all baselines + XGB.

---

## CLI

| Command | Action |
| --- | --- |
| `npm run pe-dataset` | Build/export research dataset + manifest |
| `npm run pe-baselines` | Run baselines; write IC results JSON |
| `npm run pe-train` | Train XGB regressor; write IC results JSON |

Artifacts under `data/prediction_engine/`.

---

## Tests (Required)

- Features contain **no future bars** (shift/leakage checks).
- Target uses **open\_{t+1}** and **close\_{t+5}**.
- Walk-forward **purge gap** ≥ 5.
- Baseline **determinism** (same seed → same scores/IC).
- Scaling / Ridge / XGB fit on **train only**.

---

## Explicit Non-Goals (This Pass)

- Dashboard redesign or prediction UI
- Live brokerage / capital allocation
- Deploy / production serving
- Full conformal intervals
- Production quantile model
- Point-in-time index membership
- Intraday features or labels

---

## Relationship to Existing Code

| Component | Role after V1 |
| --- | --- |
| `agents.py` | Rule-based **benchmarks** (keep) |
| `ml_train.py` / ternary XGB | Legacy classification research — do not overload; PE uses new modules |
| `workers/trendlabs/prediction/` | Prediction Engine V1 package |
| `data/prediction_engine/` | Versioned PE artifacts |

---

## Final Research Question

**Can a point-in-time daily feature set, under a next-open → close\(t+5\) execution rule and purged walk-forward evaluation, produce out-of-sample cross-sectional Spearman IC (and Q5−Q1 spread) that reliably beats random, historical-mean, and 12−1 momentum baselines on TrendLab’s liquid universe — acknowledging that survivorship bias is not fully controlled in V1?**

Success for this engineering pass is not “beat the market,” but: (1) correct leakage-safe dataset, (2) comparable baseline IC table, (3) light XGB scaffold on the same protocol, (4) tests and versioned artifacts that make the research question answerable in subsequent iterations.
