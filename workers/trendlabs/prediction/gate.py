"""V1 research gate — fixed bars; do not lower after seeing results."""

from __future__ import annotations

from typing import Any

from trendlabs.prediction import DISCLAIMER

# Fixed V1 bars (do not revise downward after peeking at OOS).
GATE_MEAN_IC_MIN = 0.03
GATE_POSITIVE_FOLD_RATE_MIN = 0.60
GATE_Q5_Q1_SHARPE_10BPS_MIN = 0.5
GATE_MIN_FOLDS_FOR_DECISION = 5
GATE_MIN_QUINTILE_DAYS = 30


def evaluate_v1_gate(
    *,
    xgb_ic: dict[str, Any],
    mom_ic: dict[str, Any] | None,
    quintiles: dict[str, Any],
    economic: dict[str, Any],
    mom_economic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return PASS / FAIL / INSUFFICIENT with structured reasons.

    Criteria (all required for PASS):
      - MeanIC > 0.03
      - positive WF folds >= 60%
      - broadly monotonic quantiles (z or R)
      - beat momentum predictively (MeanIC) OR economically (10bps long Q5 Sharpe)
      - Q5−Q1 net Sharpe at 10bps > 0.5
      - 20bps stress positive expectancy (LS Q5−Q1 mean period return > 0)
    Always stamp exploratory survivorship disclaimer.
    """
    reasons: list[dict[str, Any]] = []
    fold_ic = xgb_ic.get("fold_ic") or {}
    n_folds = int(fold_ic.get("n_folds") or 0)
    n_q_days = int(quintiles.get("n_days") or 0)
    mean_ic = xgb_ic.get("MeanIC")
    pos_fold = fold_ic.get("positive_fold_rate")

    insufficient = n_folds < GATE_MIN_FOLDS_FOR_DECISION or n_q_days < GATE_MIN_QUINTILE_DAYS
    if insufficient:
        reasons.append(
            {
                "check": "sample_size",
                "passed": False,
                "detail": f"n_folds={n_folds} (min {GATE_MIN_FOLDS_FOR_DECISION}), "
                f"quintile_days={n_q_days} (min {GATE_MIN_QUINTILE_DAYS})",
            }
        )

    # 1) MeanIC
    ic_ok = mean_ic is not None and float(mean_ic) > GATE_MEAN_IC_MIN
    reasons.append(
        {
            "check": "mean_ic",
            "passed": bool(ic_ok),
            "threshold": GATE_MEAN_IC_MIN,
            "value": mean_ic,
            "detail": f"MeanIC={mean_ic} > {GATE_MEAN_IC_MIN}",
        }
    )

    # 2) Positive WF folds
    folds_ok = pos_fold is not None and float(pos_fold) >= GATE_POSITIVE_FOLD_RATE_MIN
    reasons.append(
        {
            "check": "positive_wf_folds",
            "passed": bool(folds_ok),
            "threshold": GATE_POSITIVE_FOLD_RATE_MIN,
            "value": pos_fold,
            "detail": f"positive_fold_rate={pos_fold} >= {GATE_POSITIVE_FOLD_RATE_MIN}",
        }
    )

    # 3) Broadly monotonic
    mono = bool(quintiles.get("broadly_monotonic"))
    reasons.append(
        {
            "check": "broadly_monotonic_quantiles",
            "passed": mono,
            "value": {
                "broadly_monotonic_z": quintiles.get("broadly_monotonic_z"),
                "broadly_monotonic_R": quintiles.get("broadly_monotonic_R"),
            },
            "detail": "Q1..Q5 mean z or R broadly non-decreasing",
        }
    )

    # 4) Beat momentum predictively or economically
    mom_mean = (mom_ic or {}).get("MeanIC")
    beat_pred = (
        mean_ic is not None and mom_mean is not None and float(mean_ic) > float(mom_mean)
    )
    xgb_long_10 = ((economic.get("long_Q5") or {}).get("10bps") or {})
    mom_long_10 = ((mom_economic or {}).get("long_Q5") or {}).get("10bps") or {}
    xgb_sh = xgb_long_10.get("Sharpe")
    mom_sh = mom_long_10.get("Sharpe")
    beat_econ = (
        xgb_sh is not None and mom_sh is not None and float(xgb_sh) > float(mom_sh)
    ) or (xgb_sh is not None and mom_sh is None and float(xgb_sh) > 0)
    beat_mom = bool(beat_pred or beat_econ)
    reasons.append(
        {
            "check": "beat_momentum",
            "passed": beat_mom,
            "value": {
                "xgb_MeanIC": mean_ic,
                "mom_MeanIC": mom_mean,
                "xgb_long_Q5_Sharpe_10bps": xgb_sh,
                "mom_long_Q5_Sharpe_10bps": mom_sh,
                "beat_predictively": beat_pred,
                "beat_economically": beat_econ,
            },
            "detail": "XGB MeanIC > MOM MeanIC OR long-Q5 Sharpe@10bps > MOM",
        }
    )

    # 5) Q5−Q1 net Sharpe @ 10bps
    ls_10 = ((economic.get("long_Q5_short_Q1") or {}).get("10bps") or {})
    ls_sharpe = ls_10.get("Sharpe")
    ls_ok = ls_sharpe is not None and float(ls_sharpe) > GATE_Q5_Q1_SHARPE_10BPS_MIN
    reasons.append(
        {
            "check": "q5_q1_net_sharpe_10bps",
            "passed": bool(ls_ok),
            "threshold": GATE_Q5_Q1_SHARPE_10BPS_MIN,
            "value": ls_sharpe,
            "detail": f"Q5-Q1 Sharpe@10bps={ls_sharpe} > {GATE_Q5_Q1_SHARPE_10BPS_MIN}",
        }
    )

    # 6) 20bps stress positive expectancy
    ls_20 = ((economic.get("long_Q5_short_Q1") or {}).get("20bps") or {})
    stress_ok = bool(ls_20.get("positive_expectancy"))
    reasons.append(
        {
            "check": "stress_20bps_positive_expectancy",
            "passed": stress_ok,
            "value": {
                "positive_expectancy": ls_20.get("positive_expectancy"),
                "mean_period_return": ls_20.get("mean_period_return"),
            },
            "detail": "Q5-Q1 mean net period return > 0 at 20bps RT",
        }
    )

    checks_ok = all(r["passed"] for r in reasons if r["check"] != "sample_size")
    if insufficient:
        decision = "INSUFFICIENT"
    elif checks_ok:
        decision = "PASS"
    else:
        decision = "FAIL"

    return {
        "disclaimer": DISCLAIMER,
        "decision": decision,
        "gates": {
            "MeanIC_min": GATE_MEAN_IC_MIN,
            "positive_wf_folds_min": GATE_POSITIVE_FOLD_RATE_MIN,
            "q5_q1_net_sharpe_10bps_min": GATE_Q5_Q1_SHARPE_10BPS_MIN,
            "min_folds_for_decision": GATE_MIN_FOLDS_FOR_DECISION,
            "min_quintile_days": GATE_MIN_QUINTILE_DAYS,
        },
        "reasons": reasons,
        "passed_all_research_checks": checks_ok,
    }
