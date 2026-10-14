"""Phase D/E CLI — OOS quintile analysis, economic simulator, V1 gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trendlabs.prediction import DATASET_VERSION, DISCLAIMER, PURGE_GAP_DAYS
from trendlabs.prediction.baselines import score_mom12_1
from trendlabs.prediction.dataset import PE_DIR, load_exported_dataset
from trendlabs.prediction.economic import run_economic_simulation
from trendlabs.prediction.gate import evaluate_v1_gate
from trendlabs.prediction.metrics_ic import evaluate_scores, fold_mean_ic_stats, full_quintile_analysis
from trendlabs.prediction.train import run_xgb_walk_forward
from trendlabs.prediction.walk_forward import generate_purged_folds, mask_dates, unique_sorted_dates


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            return float(obj) if obj == obj else None  # noqa: PLR0124
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return obj


def _mom_oos_panel(
    panel: pd.DataFrame,
    *,
    min_train_days: int,
    test_days: int,
    step_days: int | None,
    purge_gap: int,
    rank_exclude_spy: bool = True,
) -> pd.DataFrame:
    """MOM12_1 scores on the same purged OOS dates as XGB (no fitting)."""
    work = panel.copy()
    work["ts"] = pd.to_datetime(work["ts"], utc=True)
    rank_panel = work.loc[work["symbol"] != "SPY"].copy() if rank_exclude_spy else work
    dates = unique_sorted_dates(rank_panel)
    folds = generate_purged_folds(
        dates,
        min_train_days=min_train_days,
        test_days=test_days,
        step_days=step_days,
        purge_gap=purge_gap,
    )
    parts: list[pd.DataFrame] = []
    for fold in folds:
        test = mask_dates(rank_panel, fold.test_dates)
        if test.empty:
            continue
        keep = [c for c in ["symbol", "ts", "z", "R", "MOM12_1"] if c in test.columns]
        chunk = test[keep].copy()
        chunk["fold_id"] = fold.fold_id
        chunk["z_hat"] = score_mom12_1(test)  # reuse score_col name for sim helpers
        parts.append(chunk)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def run_economic_eval(
    panel: pd.DataFrame,
    *,
    min_train_days: int = 252,
    test_days: int = 63,
    step_days: int | None = None,
    purge_gap: int = PURGE_GAP_DAYS,
) -> dict[str, Any]:
    """Full Phase D/E pipeline → report dict (no UI)."""
    xgb = run_xgb_walk_forward(
        panel,
        min_train_days=min_train_days,
        test_days=test_days,
        step_days=step_days,
        purge_gap=purge_gap,
        return_oos=True,
    )
    oos: pd.DataFrame = xgb.pop("oos", pd.DataFrame())
    if oos is None or oos.empty:
        return {
            "disclaimer": DISCLAIMER,
            "error": xgb.get("error", "empty_oos"),
            "gate": {
                "disclaimer": DISCLAIMER,
                "decision": "INSUFFICIENT",
                "reasons": [{"check": "oos", "passed": False, "detail": "no OOS predictions"}],
            },
        }

    # Ensure MOM12_1 on OOS for baseline comparison if present on panel
    if "MOM12_1" not in oos.columns and "MOM12_1" in panel.columns:
        # Merge from panel
        m = panel[["symbol", "ts", "MOM12_1"]].copy()
        m["ts"] = pd.to_datetime(m["ts"], utc=True)
        oos["ts"] = pd.to_datetime(oos["ts"], utc=True)
        oos = oos.merge(m, on=["symbol", "ts"], how="left")

    quintiles = full_quintile_analysis(oos, score_col="z_hat")
    economic = run_economic_simulation(oos, score_col="z_hat")

    mom_oos = _mom_oos_panel(
        panel,
        min_train_days=min_train_days,
        test_days=test_days,
        step_days=step_days,
        purge_gap=purge_gap,
    )
    mom_ic = evaluate_scores(mom_oos, score_col="z_hat") if not mom_oos.empty else {}
    if not mom_oos.empty:
        mom_ic["fold_ic"] = fold_mean_ic_stats(mom_oos, score_col="z_hat")
    mom_econ = run_economic_simulation(mom_oos, score_col="z_hat") if not mom_oos.empty else {}
    mom_quintiles = full_quintile_analysis(mom_oos, score_col="z_hat") if not mom_oos.empty else {}

    gate = evaluate_v1_gate(
        xgb_ic=xgb.get("xgboost") or {},
        mom_ic=mom_ic,
        quintiles=quintiles,
        economic=economic,
        mom_economic=mom_econ,
    )

    # Compact metric table for humans / README paste
    def _row(name: str, ic: dict[str, Any], econ: dict[str, Any], q: dict[str, Any]) -> dict[str, Any]:
        long10 = (econ.get("long_Q5") or {}).get("10bps") or {}
        ls10 = (econ.get("long_Q5_short_Q1") or {}).get("10bps") or {}
        ls20 = (econ.get("long_Q5_short_Q1") or {}).get("20bps") or {}
        return {
            "model": name,
            "MeanIC": ic.get("MeanIC"),
            "ICIR": ic.get("ICIR"),
            "positive_fold_rate": (ic.get("fold_ic") or {}).get("positive_fold_rate"),
            "Q1_mean_R": (q.get("mean_R_by_quintile") or {}).get("Q1"),
            "Q5_mean_R": (q.get("mean_R_by_quintile") or {}).get("Q5"),
            "Q5_Q1_mean_R": q.get("Q5_Q1_mean_R"),
            "broadly_monotonic": q.get("broadly_monotonic"),
            "long_Q5_Sharpe_10bps": long10.get("Sharpe"),
            "long_Q5_CAGR_10bps": long10.get("CAGR"),
            "long_Q5_maxDD_10bps": long10.get("max_drawdown"),
            "long_Q5_turnover_10bps": long10.get("mean_turnover"),
            "LS_Sharpe_10bps": ls10.get("Sharpe"),
            "LS_Sharpe_20bps": ls20.get("Sharpe"),
            "LS_positive_expectancy_20bps": ls20.get("positive_expectancy"),
        }

    metric_table = [
        _row("xgboost", xgb.get("xgboost") or {}, economic, quintiles),
        _row("mom12_1", mom_ic, mom_econ, mom_quintiles),
    ]

    return {
        "disclaimer": DISCLAIMER,
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "D/E — OOS Evaluation + Economic Simulator",
        "protocol": xgb.get("protocol"),
        "xgboost": xgb.get("xgboost"),
        "mom12_1": mom_ic,
        "quintiles": {
            "xgboost": quintiles,
            "mom12_1": mom_quintiles,
        },
        "economic": {
            "xgboost": economic,
            "mom12_1": mom_econ,
        },
        "metric_table": metric_table,
        "gate": gate,
        "hyperparameters": xgb.get("hyperparameters"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PE V1 Phase D/E — OOS eval + economic gate")
    parser.add_argument("--data-dir", type=Path, default=PE_DIR)
    parser.add_argument("--min-train-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--step-days", type=int, default=None)
    parser.add_argument("--purge-gap", type=int, default=PURGE_GAP_DAYS)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: data/prediction_engine/economic_eval.json)",
    )
    args = parser.parse_args()

    panel = load_exported_dataset(args.data_dir)
    report = run_economic_eval(
        panel,
        min_train_days=args.min_train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        purge_gap=args.purge_gap,
    )
    out_path = Path(args.out or (Path(args.data_dir) / "economic_eval.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

    gate = report.get("gate") or {}
    summary = {
        "wrote": str(out_path),
        "disclaimer": DISCLAIMER,
        "decision": gate.get("decision"),
        "metric_table": report.get("metric_table"),
    }
    print(json.dumps(_json_safe(summary), indent=2))
    # Exit 0 even on FAIL — research artifact written; gate is in JSON
    return 0 if "gate" in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
