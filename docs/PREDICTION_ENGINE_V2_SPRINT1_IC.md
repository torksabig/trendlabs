# Prediction Engine V2 — Sprint 1 IC Report

**Disclaimer:** `EXPLORATORY — SURVIVORSHIP BIAS NOT FULLY CONTROLLED`

## Question

> Did MeanIC improve vs prior ~0.025?

**Answer: No.** XGB MeanIC moved from 0.023401 (V1 features) to 0.020207 (Sprint 1 features); neither exceeds the ~0.025 reference (prior mom12_1 / rounded XGB ballpark).

## IC comparison (identical purged walk-forward)

| Model | MeanIC before | MeanIC after | Δ MeanIC | ICIR before | ICIR after |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 0.005327 | 0.002191 | -0.003136 | 0.016214 | 0.006699 |
| historical_mean | 0.017277 | 0.021086 | 0.003809 | 0.047585 | 0.058525 |
| mom12_1 | 0.025366 | 0.025586 | 0.000219 | 0.065587 | 0.066141 |
| ridge | 0.020674 | 0.011613 | -0.009061 | 0.053466 | 0.031576 |
| xgboost | 0.023401 | 0.020207 | -0.003194 | 0.064759 | 0.056061 |

## Notes

- Before: `pe_dataset_v1` / V1 feature set (simple-return moms).
- After: `pe_dataset_v2` / Sprint 1 log-return moms + trend/RS/breadth/ON-ID.
- Protocol unchanged: expanding train, test=63d, step=63d, purge≥5, rank excludes SPY.
- Ridge MeanIC fell with the wider feature set; mom12_1 baseline nearly unchanged.
- Sprint 2–6 (vol models, quantiles, LightGBM, ensemble, dashboard) not implemented.
