# TrendLab Prediction Engine V2
## Implementation Plan

## 1. Objective

Upgrade TrendLab from a single cross-sectional XGBoost predictor into a multi-output forecasting system that separately estimates:

1. Trend direction
2. Expected risk-adjusted return
3. Expected volatility
4. Expected downside and upside range
5. Cross-sectional stock ranking
6. Prediction uncertainty
7. Economic usefulness after transaction costs

The system must preserve TrendLab's existing principle:

> Models generate forecasts. Deterministic code evaluates them. Trading logic consumes forecasts separately.

The immediate objective is not automated live trading.

The objective is to determine whether TrendLab can produce statistically reliable forward forecasts on unseen data.

---

# 2. Target Architecture

```text
                    MARKET DATA
                         │
                         ▼
                  FEATURE ENGINE
                         │
       ┌─────────────────┼──────────────────┐
       │                 │                  │
       ▼                 ▼                  ▼
 TREND ENGINE      VOLATILITY ENGINE   QUANTILE ENGINE
       │                 │                  │
       ▼                 ▼                  ▼
 predicted z       predicted σ        Q10 / Q50 / Q90
 expected return                       return interval
       │                 │                  │
       └─────────────────┼──────────────────┘
                         ▼
                    ENSEMBLE
                         │
                         ▼
               UNCERTAINTY ENGINE
                         │
                         ▼
             CROSS-SECTIONAL RANKING
                         │
                         ▼
                  FORECAST OBJECT
                         │
                         ▼
                ECONOMIC SIMULATOR
                         │
                         ▼
                 PAPER PORTFOLIO
                         │
                         ▼
                     DASHBOARD
```

---

# 3. Existing Foundation

The following existing Prediction Engine components should be preserved:

```text
workers/trendlabs/prediction/
```

Existing modules:

```text
targets.py
features.py
dataset.py
baselines.py
metrics_ic.py
walk_forward.py
train.py
quantile_stub.py
```

Existing functionality includes:

- five-day actionable return
- volatility-normalized target
- daily features
- SPY-relative information
- purged walk-forward validation
- Ridge baseline
- momentum baseline
- historical-mean baseline
- random baseline
- XGBoost regression
- Spearman IC
- ICIR
- Q5-Q1 analysis

V2 should extend these components rather than replace them.

---

# 4. Phase 1 — Expand the Feature Engine

## Goal

Increase the amount of independent information available to the models without blindly adding technical indicators.

Modify:

```text
workers/trendlabs/prediction/features.py
```

---

## 4.1 Momentum structure

Add:

```text
MOM5
MOM10
MOM20
MOM60
MOM120
MOM12_1
```

Use log or simple returns consistently.

Example:

\[
MOM_{20}
=
\ln(P_t/P_{t-20})
\]

Add momentum acceleration:

\[
MOMACCEL_{10,20}
=
MOM_{10}-MOM_{20}
\]

and:

\[
MOMACCEL_{20,60}
=
MOM_{20}-MOM_{60}
\]

Purpose:

Distinguish an accelerating trend from a decelerating one.

---

## 4.2 Trend quality

Add rolling linear regression against time.

For a 20-day window:

\[
P_t=\alpha+\beta t+\epsilon
\]

Store:

```text
TREND_SLOPE20
TREND_R2_20
```

Repeat if useful for:

```text
TREND_SLOPE60
TREND_R2_60
```

Interpretation:

- slope = trend direction
- \(R^2\) = trend consistency

A strong slope with low \(R^2\) means a noisy trend.

---

## 4.3 Relative strength

Current SPY-relative strength should be expanded.

Add:

```text
RS_SPY_5
RS_SPY_20
RS_SPY_60
```

Formula:

\[
RS_{SPY,20}
=
R_{stock,20}-R_{SPY,20}
\]

Later add sector-relative strength:

```text
RS_SECTOR_5
RS_SECTOR_20
RS_SECTOR_60
```

This requires a symbol → sector ETF mapping.

Example:

```text
Technology → XLK
Financials → XLF
Energy → XLE
Healthcare → XLV
Industrials → XLI
```

---

## 4.4 Market breadth

Calculate:

```text
BREADTH20
BREADTH50
BREADTH200
```

Example:

\[
Breadth_{20}
=
\frac{
\#(P_i>MA20_i)
}
{N}
\]

Add:

```text
ADVANCE_DECLINE_RATIO
```

optional later.

---

## 4.5 Return decomposition

Add:

### Overnight return

\[
R_{overnight,t}
=
\ln(Open_t/Close_{t-1})
\]

### Intraday return

\[
R_{intraday,t}
=
\ln(Close_t/Open_t)
\]

Store rolling totals:

```text
OVERNIGHT_RET5
OVERNIGHT_RET20

INTRADAY_RET5
INTRADAY_RET20
```

Purpose:

Separate overnight repricing from active-session demand.

---

# 5. Phase 2 — Improve Volatility Features

## Goal

Use the full OHLC information rather than relying mainly on close-to-close returns.

Add:

```text
workers/trendlabs/prediction/volatility_features.py
```

---

## 5.1 Close-to-close volatility

Keep:

```text
VOL20
VOL120
VOLRatio
```

---

## 5.2 Parkinson volatility

Implement:

\[
\sigma^2_P
=
\frac{1}
{4\ln(2)}
\left[
\ln(H/L)
\right]^2
\]

Rolling features:

```text
PARKINSON_VOL5
PARKINSON_VOL20
```

---

## 5.3 Garman-Klass volatility

Implement OHLC-based Garman-Klass estimator.

Expose:

```text
GK_VOL5
GK_VOL20
```

---

## 5.4 Rogers-Satchell volatility

Add:

```text
RS_VOL5
RS_VOL20
```

This provides another estimator useful when price drift is present.

---

## 5.5 Volatility acceleration

Add:

\[
VOLACCEL
=
VOL20/VOL120
\]

Existing `VOLRatio` can serve this role.

Also add:

```text
VOL5_VOL20_RATIO
```

to detect rapidly expanding volatility.

---

# 6. Phase 3 — Dedicated Volatility Model

Create:

```text
workers/trendlabs/prediction/volatility_model.py
```

The volatility model must be independent of the return model.

---

## 6.1 Volatility target

Define future realized volatility over five days.

For example:

\[
RV_{t,t+5}
=
\sqrt{
\sum_{j=1}^{5}
r_{t+j}^2
}
\]

Store:

```text
future_rv_5
```

Alternatively use log variance:

\[
\log(RV+\epsilon)
\]

to stabilize the distribution.

---

## 6.2 HAR baseline

Implement a HAR-style baseline:

\[
RV_{t+1}
=
\beta_0
+
\beta_dRV_t
+
\beta_wRV_t^{week}
+
\beta_mRV_t^{month}
+\epsilon
\]

This becomes the volatility benchmark.

---

## 6.3 XGBoost volatility model

Train XGBoost using:

```text
VOL5
VOL20
VOL120
VOLRatio
ATRNorm
Parkinson
Garman-Klass
Rogers-Satchell
VolumeZ
market volatility
SPY volatility
dispersion
```

Compare:

```text
HAR
XGBoost volatility
```

Metrics:

```text
MAE
RMSE
QLIKE
rank correlation
```

---

# 7. Phase 4 — Quantile Forecasting

Replace:

```text
quantile_stub.py
```

with actual quantile models.

Prefer creating:

```text
quantile_model.py
```

---

## 7.1 Quantiles

Predict:

\[
Q_{0.10}
\]

\[
Q_{0.50}
\]

\[
Q_{0.90}
\]

of five-day actionable return.

Outputs:

```text
q10
q50
q90
```

---

## 7.2 Interpretation

Example:

```text
q10 = -0.031
q50 = +0.008
q90 = +0.047
```

means:

```text
10th percentile: -3.1%
Median: +0.8%
90th percentile: +4.7%
```

Do not describe these as guaranteed bounds.

---

## 7.3 Validation

Measure:

```text
pinball loss
coverage
interval width
```

For Q10-Q90:

\[
Coverage
=
P(Q10\leq R\leq Q90)
\]

Expected approximate coverage:

\[
80\%
\]

before calibration.

---

# 8. Phase 5 — Model Competition

Create:

```text
workers/trendlabs/prediction/models/
```

Structure:

```text
models/
    ridge.py
    xgboost.py
    lightgbm.py
    random_forest.py
```

Keep model interfaces consistent.

Example:

```python
fit(train)
predict(test)
```

---

## 8.1 Models

Evaluate:

```text
Ridge
XGBoost
LightGBM
Random Forest
```

Do not add neural networks yet.

---

## 8.2 Identical validation

Every model must receive:

- identical features
- identical walk-forward folds
- identical purge periods
- identical universe
- identical targets

Otherwise comparisons are invalid.

---

# 9. Phase 6 — Ensemble Engine

Create:

```text
workers/trendlabs/prediction/ensemble.py
```

Inputs:

```text
ridge_prediction
xgb_prediction
lightgbm_prediction
random_forest_prediction
```

Start with simple equal weighting.

\[
\hat z_{ensemble}
=
\frac{1}{K}
\sum_{k=1}^{K}\hat z_k
\]

Do not optimize ensemble weights initially.

---

## 9.1 Model disagreement

Calculate:

\[
Disagreement
=
Std
(
\hat z_1,\hat z_2,...,\hat z_K
)
\]

Store:

```text
model_disagreement
```

Low disagreement:

```text
models mostly agree
```

High disagreement:

```text
forecast uncertain
```

---

# 10. Phase 7 — Confidence Engine

Create:

```text
workers/trendlabs/prediction/confidence.py
```

Confidence must not be an arbitrary model probability.

It should combine:

```text
model agreement
signal strength
historical calibration
historical performance in similar conditions
data completeness
```

Initial implementation:

\[
AgreementScore
=
\frac{1}
{1+Disagreement}
\]

Then combine with normalized signal magnitude.

Example:

\[
ConfidenceScore
=
AgreementScore
\times
SignalStrength
\]

Later calibrate this empirically.

---

# 11. Phase 8 — Selective Prediction

Add abstention capability.

TrendLab should be allowed to output:

```text
NO STRONG FORECAST
```

rather than predicting every stock.

---

## 11.1 Coverage thresholds

Example:

```text
Top 10% strongest positive predictions
Bottom 10% strongest negative predictions
Middle 80% = no strong directional call
```

Test:

```text
coverage vs accuracy
coverage vs IC
coverage vs economic return
```

Example:

| Coverage | Direction Accuracy |
|---:|---:|
| 100% | 52.8% |
| 50% | 54.1% |
| 25% | 56.3% |
| 10% | 59.0% |

Illustrative only.

The goal is to determine whether confidence contains useful information.

---

# 12. Phase 9 — Expand IC and Ranking Metrics

Modify:

```text
metrics_ic.py
```

Current Q5-Q1 should be expanded.

Calculate:

```text
Q1
Q2
Q3
Q4
Q5
```

average realized returns.

Test monotonicity:

\[
R_{Q1}
<
R_{Q2}
<
R_{Q3}
<
R_{Q4}
<
R_{Q5}
\]

Store:

```text
quantile_returns
Q5_Q1_spread
monotonicity_score
```

---

## 12.1 Fold-level metrics

Store by fold:

```text
MeanIC
MedianIC
PositiveICRate
ICIR
Q5-Q1
```

This prevents one period from hiding weak performance elsewhere.

---

# 13. Phase 10 — Economic Simulator

Create:

```text
workers/trendlabs/prediction/economic_simulator.py
```

Purpose:

Translate rankings into hypothetical economic results.

---

## 13.1 Initial portfolio

Keep it deliberately simple.

```text
Rank universe
↓
Select top 20%
↓
Equal weight
↓
Enter next open
↓
Hold five trading days
↓
Rebalance
```

Do not optimize position sizes yet.

---

## 13.2 Costs

Test at minimum:

```text
0 bps
10 bps
20 bps
```

round-trip cost assumptions.

Calculate:

```text
gross return
net return
turnover
Sharpe
Sortino
CAGR
max drawdown
profit factor
```

---

# 14. Phase 11 — Forecast Object

Create:

```text
workers/trendlabs/prediction/forecast.py
```

Standard forecast object:

```python
{
    "symbol": "NVDA",
    "timestamp": "...",

    "trend_score": 0.47,
    "expected_return": 0.016,

    "predicted_volatility": 0.058,

    "q10": -0.034,
    "q50": 0.016,
    "q90": 0.062,

    "rank_percentile": 0.91,

    "model_predictions": {
        "ridge": 0.38,
        "xgboost": 0.51,
        "lightgbm": 0.48
    },

    "model_disagreement": 0.055,

    "confidence_score": 0.76,

    "forecast_state": "bullish"
}
```

This becomes the common interface consumed by the dashboard and later by trading logic.

---

# 15. Phase 12 — Trend Classification

Trend classification should be derived from the forecast.

Initial experimental mapping:

```text
z > +0.50
STRONG_BULLISH

+0.15 < z <= +0.50
BULLISH

-0.15 <= z <= +0.15
NEUTRAL

-0.50 <= z < -0.15
BEARISH

z < -0.50
STRONG_BEARISH
```

These thresholds are temporary.

They must eventually be determined from validation data.

---

# 16. Phase 13 — Earnings Features

Create:

```text
workers/trendlabs/prediction/events.py
```

Add:

```text
days_to_earnings
days_since_earnings
earnings_next_1d
earnings_next_5d
earnings_recent_5d
```

Do not initially use earnings surprise unless a reliable point-in-time dataset is available.

The first objective is simply to tell the fluctuation model:

> A scheduled high-volatility event is approaching.

---

# 17. Phase 14 — Market Regime Diagnostics

Do not make regime detection the main model.

Use it for analysis.

Store:

```text
bull
bear
sideways
high_vol
low_vol
```

Then evaluate:

```text
IC by regime
Sharpe by regime
accuracy by regime
quantile calibration by regime
```

This answers:

> When does TrendLab work?

rather than assuming regime models automatically improve forecasts.

---

# 18. Phase 15 — Conformal Calibration

Once quantile forecasting works, implement:

```text
ConformalCalibrator
```

Goal:

Improve empirical interval coverage.

For intended 80% interval:

\[
P(
R\in[L,U]
)
\approx0.80
\]

Measure actual coverage walk-forward.

Do not claim statistical guarantees unless the implemented conformal assumptions genuinely support them.

---

# 19. Phase 16 — Prediction Persistence

Every prediction must be saved before its outcome becomes known.

Create persistence table:

```text
predictions
```

Fields:

```text
prediction_id
symbol
prediction_timestamp
model_version
feature_version

trend_score
expected_return
predicted_volatility

q10
q50
q90

rank_percentile

model_disagreement
confidence_score

created_at
```

Predictions must be immutable.

---

# 20. Phase 17 — Realized Outcomes

Create:

```text
prediction_outcomes
```

Fields:

```text
prediction_id

entry_open
exit_close

realized_return
realized_z
realized_rank

forecast_error
direction_correct

evaluated_at
```

This creates a permanent forward-testing history.

---

# 21. Phase 18 — Drift Monitoring

Create:

```text
workers/trendlabs/prediction/drift.py
```

Monitor:

```text
RollingIC60
RollingMAE60
RollingSharpe60
prediction distribution
feature distribution
model disagreement
```

Trigger warnings such as:

```text
MODEL_HEALTHY

MODEL_DEGRADING

MODEL_DRIFT

INSUFFICIENT_DATA
```

Do not automatically retrain based on one bad period.

---

# 22. Phase 19 — Dashboard

Update the Vercel dashboard only after the backend forecast object is stable.

The main screen should show:

```text
Symbol

Trend
Expected return
Predicted volatility
Q10 / Q50 / Q90
Cross-sectional rank
Confidence
Model agreement
```

Example:

```text
NVDA

Trend
Bullish

Expected 5-day return
+1.6%

Expected volatility
5.8%

Forecast range
-3.4% → +6.2%

Relative ranking
91st percentile

Confidence
High

Model agreement
Strong
```

---

# 23. Research Dashboard

Separate from the user forecast screen.

Display:

```text
Mean IC
Rolling IC
ICIR
positive IC rate

Q1-Q5 returns
Q5-Q1 spread

model comparison

Ridge
XGBoost
LightGBM
Ensemble

coverage vs accuracy

quantile coverage

economic return after costs

model performance by regime
```

---

# 24. Acceptance Tests

No model should progress unless it passes engineering tests.

Required tests:

```text
No future information enters features

Signals using close[t] cannot enter at close[t]

Purging removes overlapping target periods

Feature transforms are fitted on training data only

Predictions are immutable after creation

Quantile ordering always satisfies:

Q10 <= Q50 <= Q90

Economic costs are deducted correctly

Models use identical folds

Model disagreement is calculated consistently

No missing feature silently becomes zero
```

---

# 25. V2 Research Gates

Do not decide thresholds after seeing results.

Initial gates:

### Trend model

\[
MeanIC>0.03
\]

and:

```text
Positive IC in ≥60% of folds
```

---

### Quantile model

Approximately correct coverage.

For Q10-Q90:

```text
target coverage ≈ 80%
```

---

### Economic test

At 10 bps round trip:

```text
positive net expectancy
positive Sharpe
```

Initial research target:

\[
Sharpe>0.5
\]

---

### Ensemble

Ensemble must outperform or materially stabilize the strongest individual model.

If it doesn't:

do not use it.

---

# 26. Implementation Order

Do not build everything simultaneously.

## Sprint 1

Build:

```text
momentum structure
trend slope / R²
relative strength improvements
market breadth
overnight/intraday returns
```

Then rerun existing XGBoost and baselines.

Question:

> Did IC improve?

---

## Sprint 2

Build:

```text
Parkinson volatility
Garman-Klass
Rogers-Satchell
future realized-vol target
HAR volatility model
XGBoost volatility model
```

Question:

> Can we predict future volatility better than historical volatility?

---

## Sprint 3

Replace quantile stub.

Build:

```text
Q10
Q50
Q90
pinball-loss evaluation
coverage evaluation
```

Question:

> Can we produce useful price-range estimates?

---

## Sprint 4

Add:

```text
LightGBM
Random Forest
ensemble
model disagreement
confidence score
```

Question:

> Does model diversity improve OOS forecasting?

---

## Sprint 5

Build:

```text
full quintile analysis
economic simulator
turnover
transaction costs
```

Question:

> Does predictive information survive costs?

---

## Sprint 6

Build:

```text
prediction persistence
outcome persistence
forward paper predictions
drift monitoring
```

Question:

> Does performance survive genuine forward testing?

---

# 27. What Not to Build Yet

Do not add:

```text
LSTM
Transformers
reinforcement learning
quantum models
autonomous AI trading
complex portfolio optimization
options datasets
high-frequency execution
```

until the classical system demonstrates reliable predictive information.

---

# 28. Final V2 Output

For each stock, TrendLab should ultimately produce:

```text
TREND

Bullish / Neutral / Bearish


EXPECTED RETURN

+1.4%


EXPECTED VOLATILITY

4.9%


EXPECTED RANGE

Q10   -2.8%
Q50   +1.1%
Q90   +5.3%


RELATIVE RANK

87th percentile


MODEL AGREEMENT

High


CONFIDENCE

Moderate / High


RISK FLAGS

Earnings in 2 days
Elevated volatility
Model disagreement
```

The algorithm should never reduce all of this information to a single BUY/SELL prediction.

---

# 29. Core Design Principle

TrendLab should model:

\[
\boxed{
Direction
+
ExpectedReturn
+
Volatility
+
Distribution
+
RelativeRank
+
Uncertainty
}
\]

separately.

The final decision layer can then determine whether a particular forecast is sufficiently strong to act on.

This architecture gives TrendLab a much better chance of identifying useful predictive information than attempting to directly guess tomorrow's price with one model.