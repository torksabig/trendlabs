"""Experiment engine: compare control vs variant on out-of-sample folds.

The research LLM may propose hypotheses; this module evaluates them with the
deterministic backtester / walk-forward engine. It never places trades or
mutates forward paper-trading books.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from trendlabs.config import DEFAULT_TIMEFRAME, ROOT, SYMBOLS
from trendlabs.db import Store
from trendlabs.strategy_config import (
    StrategyConfig,
    config_for,
    override_config,
)
from trendlabs.walk_forward import WalkForwardEngine


EXPERIMENT_DIR = Path(ROOT) / "data" / "experiments"
Decision = Literal["accept", "reject", "insufficient"]


@dataclass(frozen=True)
class AcceptanceCriteria:
    min_trades: int = 12
    min_sharpe_lift: float = 0.15
    max_drawdown_worsen_pp: float = 3.0  # percentage points worse DD allowed
    min_fold_win_rate: float = 0.5  # fraction of folds where variant Sharpe >= control


@dataclass
class ExperimentSpec:
    experiment_id: str
    hypothesis: str
    control: StrategyConfig
    variant: StrategyConfig
    source: str = "catalog"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "control": self.control.to_dict(),
            "variant": self.variant.to_dict(),
            "source": self.source,
            "notes": self.notes,
        }


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:  # noqa: BLE001
        return None


def catalog_experiments() -> list[ExperimentSpec]:
    """Built-in candidate experiments (no LLM required)."""
    ema = config_for("ema_trend")
    mom = config_for("momentum")
    brk = config_for("breakout")
    specs = [
        ExperimentSpec(
            experiment_id="ema_adx20",
            hypothesis=(
                "Adding ADX>=20 to EMA Trend entries should reduce sideways whipsaws "
                "and improve out-of-sample Sharpe."
            ),
            control=ema,
            variant=override_config(
                ema,
                params={"adx_min": 20.0},
                name="EMA Trend + ADX>=20",
            ),
            notes="Classic trend-filter experiment from research fallback proposals.",
        ),
        ExperimentSpec(
            experiment_id="ema_adx25",
            hypothesis="Stricter ADX>=25 gate on EMA Trend improves OOS risk-adjusted returns.",
            control=ema,
            variant=override_config(
                ema,
                params={"adx_min": 25.0},
                name="EMA Trend + ADX>=25",
            ),
        ),
        ExperimentSpec(
            experiment_id="ema_atr_stop_15",
            hypothesis="Tighter ATR stop (1.5x vs 2.0x) reduces max drawdown for EMA Trend.",
            control=ema,
            variant=override_config(
                ema,
                risk={"atr_stop_mult": 1.5},
                name="EMA Trend ATR 1.5x",
            ),
        ),
        ExperimentSpec(
            experiment_id="mom_adx28",
            hypothesis="Raising Momentum ADX floor from 22 to 28 filters weak trends.",
            control=mom,
            variant=override_config(
                mom,
                params={"adx_min": 28.0},
                name="Momentum ADX>=28",
            ),
        ),
        ExperimentSpec(
            experiment_id="breakout_entry_30",
            hypothesis="Longer 30-bar breakout lookback reduces false breakouts vs 20-bar.",
            control=brk,
            variant=override_config(
                brk,
                params={"entry_lookback": 30},
                name="Breakout 30/10",
            ),
        ),
    ]
    return specs


def decide(
    control_metrics: dict[str, Any],
    variant_metrics: dict[str, Any],
    *,
    criteria: AcceptanceCriteria,
    fold_comparisons: list[dict[str, Any]] | None = None,
) -> tuple[Decision, list[str]]:
    reasons: list[str] = []
    v_trades = int(variant_metrics.get("trade_count") or 0)
    if v_trades < criteria.min_trades:
        reasons.append(f"insufficient trades ({v_trades} < {criteria.min_trades})")
        return "insufficient", reasons

    c_sharpe = control_metrics.get("mean_fold_sharpe")
    if c_sharpe is None:
        c_sharpe = control_metrics.get("sharpe")
    v_sharpe = variant_metrics.get("mean_fold_sharpe")
    if v_sharpe is None:
        v_sharpe = variant_metrics.get("sharpe")
    if v_sharpe is None or c_sharpe is None:
        reasons.append("missing Sharpe on control or variant")
        return "insufficient", reasons

    lift = float(v_sharpe) - float(c_sharpe)
    if lift < criteria.min_sharpe_lift:
        reasons.append(
            f"Sharpe lift {lift:.3f} < required {criteria.min_sharpe_lift:.3f} "
            f"(control={c_sharpe:.3f}, variant={v_sharpe:.3f})"
        )

    c_dd = float(control_metrics.get("max_drawdown_pct") or 0.0)
    v_dd = float(variant_metrics.get("max_drawdown_pct") or 0.0)
    # drawdowns are negative; worsen means more negative
    dd_worsen = (c_dd - v_dd)  # positive if variant DD is worse
    if dd_worsen > criteria.max_drawdown_worsen_pp:
        reasons.append(
            f"max DD worsened by {dd_worsen:.2f}pp "
            f"(limit {criteria.max_drawdown_worsen_pp:.2f}pp; "
            f"control={c_dd:.2f}%, variant={v_dd:.2f}%)"
        )

    if fold_comparisons:
        wins = sum(1 for f in fold_comparisons if f.get("variant_ge_control"))
        rate = wins / len(fold_comparisons)
        if rate < criteria.min_fold_win_rate:
            reasons.append(
                f"fold robustness {rate:.0%} < {criteria.min_fold_win_rate:.0%} "
                f"({wins}/{len(fold_comparisons)} folds)"
            )

    if reasons:
        return "reject", reasons

    reasons.append(
        f"OOS Sharpe lift {lift:.3f}; trades={v_trades}; "
        f"DD control={c_dd:.2f}% variant={v_dd:.2f}%"
    )
    return "accept", reasons


class ExperimentEngine:
    def __init__(
        self,
        store: Store,
        *,
        criteria: AcceptanceCriteria | None = None,
    ) -> None:
        self.store = store
        self.criteria = criteria or AcceptanceCriteria()

    def evaluate(
        self,
        spec: ExperimentSpec,
        *,
        symbols: list[str],
        timeframe: str = "1h",
        warm_up: int = 120,
        calibrate_bars: int = 80,
        test_bars: int = 60,
        step_bars: int | None = None,
        benchmark_symbol: str = "SPY",
    ) -> dict[str, Any]:
        control_engine = WalkForwardEngine(store=self.store, risk=spec.control.risk)
        control_engine.configs = [spec.control]
        variant_engine = WalkForwardEngine(store=self.store, risk=spec.variant.risk)
        variant_engine.configs = [spec.variant]

        control_run = control_engine.run(
            symbols=symbols,
            timeframe=timeframe,
            warm_up=warm_up,
            calibrate_bars=calibrate_bars,
            test_bars=test_bars,
            step_bars=step_bars,
            benchmark_symbol=benchmark_symbol,
        )
        variant_run = variant_engine.run(
            symbols=symbols,
            timeframe=timeframe,
            warm_up=warm_up,
            calibrate_bars=calibrate_bars,
            test_bars=test_bars,
            step_bars=step_bars,
            benchmark_symbol=benchmark_symbol,
        )

        control_agent = control_run["agents"][0]
        variant_agent = variant_run["agents"][0]

        # Per-fold robustness: compare mean fold Sharpes where both exist
        fold_comparisons: list[dict[str, Any]] = []
        c_folds = {f["fold_id"]: f for f in control_agent.get("folds") or []}
        for vf in variant_agent.get("folds") or []:
            cf = c_folds.get(vf["fold_id"])
            if not cf:
                continue
            cs = cf["metrics"].get("sharpe")
            vs = vf["metrics"].get("sharpe")
            if cs is None or vs is None:
                continue
            fold_comparisons.append(
                {
                    "fold_id": vf["fold_id"],
                    "control_sharpe": cs,
                    "variant_sharpe": vs,
                    "variant_ge_control": float(vs) >= float(cs),
                }
            )

        decision, reasons = decide(
            control_agent["metrics"],
            variant_agent["metrics"],
            criteria=self.criteria,
            fold_comparisons=fold_comparisons,
        )

        raw = (
            f"{spec.experiment_id}|{timeframe}|{','.join(sorted(symbols))}|"
            f"{control_run['run_id']}|{variant_run['run_id']}|{_git_sha() or 'nogit'}"
        )
        run_id = hashlib.sha1(raw.encode()).hexdigest()[:12]

        comparison = {
            "control": {
                "total_return_pct": control_agent["metrics"].get("total_return_pct"),
                "sharpe": control_agent["metrics"].get("sharpe"),
                "mean_fold_sharpe": control_agent["metrics"].get("mean_fold_sharpe"),
                "max_drawdown_pct": control_agent["metrics"].get("max_drawdown_pct"),
                "trade_count": control_agent["metrics"].get("trade_count"),
                "alpha_pct": control_agent["metrics"].get("alpha_pct"),
            },
            "variant": {
                "total_return_pct": variant_agent["metrics"].get("total_return_pct"),
                "sharpe": variant_agent["metrics"].get("sharpe"),
                "mean_fold_sharpe": variant_agent["metrics"].get("mean_fold_sharpe"),
                "max_drawdown_pct": variant_agent["metrics"].get("max_drawdown_pct"),
                "trade_count": variant_agent["metrics"].get("trade_count"),
                "alpha_pct": variant_agent["metrics"].get("alpha_pct"),
            },
        }

        payload = {
            "run_id": run_id,
            "kind": "experiment",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "code_sha": _git_sha(),
            "experiment": spec.to_dict(),
            "decision": decision,
            "reasons": reasons,
            "criteria": asdict(self.criteria),
            "timeframe": timeframe,
            "symbols": sorted(symbols),
            "walk_forward": {
                "control_run_id": control_run["run_id"],
                "variant_run_id": variant_run["run_id"],
                "fold_count": control_run["fold_count"],
                "warm_up": control_run["warm_up"],
                "calibrate_bars": control_run["calibrate_bars"],
                "test_bars": control_run["test_bars"],
            },
            "comparison": comparison,
            "fold_comparisons": fold_comparisons,
            "separates_paper_trading": True,
            "mutates_production_strategy": False,
            "llm_executes_trades": False,
        }
        # Persist locally for registry
        self.store.upsert_experiment_result(payload)
        return payload


def write_experiment_result(payload: dict[str, Any], directory: Path = EXPERIMENT_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{payload['run_id']}.json"
    path.write_text(json.dumps(payload, indent=2))

    # Append / update registry latest
    registry_path = directory / "latest.json"
    registry: dict[str, Any]
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text())
        except json.JSONDecodeError:
            registry = {"kind": "experiment_registry", "experiments": []}
    else:
        registry = {"kind": "experiment_registry", "experiments": []}

    summary = {
        "run_id": payload["run_id"],
        "experiment_id": payload["experiment"]["experiment_id"],
        "hypothesis": payload["experiment"]["hypothesis"],
        "decision": payload["decision"],
        "reasons": payload["reasons"],
        "comparison": payload["comparison"],
        "created_at": payload["created_at"],
        "code_sha": payload.get("code_sha"),
    }
    experiments = [
        e
        for e in registry.get("experiments", [])
        if e.get("experiment_id") != summary["experiment_id"]
    ]
    experiments.insert(0, summary)
    registry = {
        "kind": "experiment_registry",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "experiments": experiments[:50],
    }
    registry_path.write_text(json.dumps(registry, indent=2))
    return path


def _auto_windows(store: Store, symbols: list[str], timeframe: str) -> tuple[int, int, int]:
    max_len = 0
    for sym in symbols:
        max_len = max(max_len, len(store.load_candles(sym, timeframe)))
    warm_up, calibrate, test = 120, 80, 60
    if max_len < warm_up + calibrate + test + 2:
        warm_up = max(60, max_len // 4)
        calibrate = max(30, max_len // 8)
        test = max(25, max_len // 8)
    return warm_up, calibrate, test


def main() -> int:
    parser = argparse.ArgumentParser(description="TrendLab experiment engine (OOS control vs variant)")
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Run one catalog id (default: run all catalog experiments)",
    )
    parser.add_argument("--list", action="store_true", help="List catalog experiments and exit")
    parser.add_argument("--min-trades", type=int, default=12)
    parser.add_argument("--min-sharpe-lift", type=float, default=0.15)
    args = parser.parse_args()

    catalog = {e.experiment_id: e for e in catalog_experiments()}
    if args.list:
        print(
            json.dumps(
                [
                    {
                        "experiment_id": e.experiment_id,
                        "hypothesis": e.hypothesis,
                        "control": e.control.strategy_id,
                        "variant_params": e.variant.params,
                        "variant_risk": asdict(e.variant.risk),
                    }
                    for e in catalog.values()
                ],
                indent=2,
            )
        )
        return 0

    store = Store()
    available = store.list_symbols_with_data(args.timeframe)
    symbols = [s for s in args.symbols if s in available]
    if not symbols:
        raise SystemExit(f"No candle data for {args.timeframe}. Run collect first.")

    warm_up, calibrate, test = _auto_windows(store, symbols, args.timeframe)
    criteria = AcceptanceCriteria(
        min_trades=args.min_trades,
        min_sharpe_lift=args.min_sharpe_lift,
    )
    engine = ExperimentEngine(store, criteria=criteria)

    selected = (
        [catalog[args.experiment_id]]
        if args.experiment_id
        else list(catalog.values())
    )
    if args.experiment_id and args.experiment_id not in catalog:
        raise SystemExit(f"Unknown experiment_id. Known: {sorted(catalog)}")

    results = []
    for spec in selected:
        print(f"Evaluating {spec.experiment_id}…")
        payload = engine.evaluate(
            spec,
            symbols=symbols,
            timeframe=args.timeframe,
            warm_up=warm_up,
            calibrate_bars=calibrate,
            test_bars=test,
        )
        path = write_experiment_result(payload)
        results.append(
            {
                "experiment_id": spec.experiment_id,
                "run_id": payload["run_id"],
                "decision": payload["decision"],
                "reasons": payload["reasons"],
                "comparison": payload["comparison"],
                "path": str(path),
            }
        )
        print(f"  → {payload['decision']}: {payload['reasons'][0]}")

    print(json.dumps({"count": len(results), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
