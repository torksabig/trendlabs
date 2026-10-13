"""Walk-forward / out-of-sample evaluation over rolling time windows.

Calibration windows are recorded for future parameter tuning. Only out-of-sample
(test) periods contribute to the combined leaderboard metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trendlabs.backtest import BacktestEngine, write_backtest_result
from trendlabs.config import DEFAULT_TIMEFRAME, ROOT, SYMBOLS
from trendlabs.db import Store
from trendlabs.metrics import compute_metrics
from trendlabs.strategy_config import RiskConfig, default_strategy_configs


WALK_FORWARD_DIR = Path(ROOT) / "data" / "walk_forward"


@dataclass(frozen=True)
class FoldSpec:
    fold_id: int
    calibrate_start_ts: str
    calibrate_end_ts: str
    test_start_ts: str
    test_end_ts: str
    calibrate_bars: int
    test_bars: int


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


def generate_bar_folds(
    timestamps: list[str],
    *,
    warm_up: int,
    calibrate_bars: int,
    test_bars: int,
    step_bars: int | None = None,
) -> list[FoldSpec]:
    """Build rolling folds from a master timestamp index.

    Indicators may use bars before test_start (including calibrate + warm-up).
    Trading/evaluation only uses [test_start, test_end].
    """
    step = step_bars if step_bars is not None else test_bars
    n = len(timestamps)
    if n < warm_up + calibrate_bars + test_bars + 2:
        return []

    folds: list[FoldSpec] = []
    # First test window starts after warm-up + calibrate
    test_start = warm_up + calibrate_bars
    fold_id = 0
    while test_start + test_bars <= n - 1:
        test_end = test_start + test_bars - 1
        cal_start = max(0, test_start - calibrate_bars)
        cal_end = test_start - 1
        folds.append(
            FoldSpec(
                fold_id=fold_id,
                calibrate_start_ts=timestamps[cal_start],
                calibrate_end_ts=timestamps[cal_end],
                test_start_ts=timestamps[test_start],
                test_end_ts=timestamps[test_end],
                calibrate_bars=cal_end - cal_start + 1,
                test_bars=test_end - test_start + 1,
            )
        )
        fold_id += 1
        test_start += step
        if step <= 0:
            break

    # Tail fold: remaining bars after last full window (if meaningful)
    if folds:
        last_end_ts = folds[-1].test_end_ts
        last_end_idx = timestamps.index(last_end_ts)
        remaining = (n - 1) - last_end_idx
        if remaining >= max(20, test_bars // 3):
            test_start = last_end_idx + 1
            test_end = n - 1
            cal_start = max(0, test_start - calibrate_bars)
            cal_end = test_start - 1
            folds.append(
                FoldSpec(
                    fold_id=fold_id,
                    calibrate_start_ts=timestamps[cal_start],
                    calibrate_end_ts=timestamps[cal_end],
                    test_start_ts=timestamps[test_start],
                    test_end_ts=timestamps[test_end],
                    calibrate_bars=cal_end - cal_start + 1,
                    test_bars=test_end - test_start + 1,
                )
            )
    return folds


def _stitch_equity_curves(
    fold_curves: list[list[dict[str, Any]]],
    starting_cash: float,
) -> list[dict[str, Any]]:
    """Compound fold equity by chaining period returns onto a continuous curve."""
    if not fold_curves:
        return []
    stitched: list[dict[str, Any]] = []
    equity = starting_cash
    for curve in fold_curves:
        if not curve:
            continue
        base = float(curve[0]["equity"])
        if base <= 0:
            continue
        for point in curve:
            ret = float(point["equity"]) / base
            # Scale cash proportionally when present
            cash_ratio = float(point.get("cash", 0)) / base if base else 0.0
            stitched.append(
                {
                    "ts": point["ts"],
                    "equity": equity * ret,
                    "cash": equity * cash_ratio,
                }
            )
        equity = stitched[-1]["equity"] if stitched else equity
        # Next fold restarts relative scale from its own start
    return stitched


def _aggregate_agent_oos(
    agent_id: str,
    name: str,
    fold_results: list[dict[str, Any]],
    *,
    starting_cash: float,
    timeframe: str,
    benchmark_folds: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    curves = []
    trades: list[dict[str, Any]] = []
    fold_summaries = []
    regime_acc: dict[str, dict[str, Any]] = {}

    for fr in fold_results:
        agent = next((a for a in fr["agents"] if a["agent_id"] == agent_id), None)
        if not agent:
            continue
        curves.append(agent.get("equity_curve") or [])
        trades.extend(agent.get("trades") or [])
        fold_summaries.append(
            {
                "fold_id": fr["fold_id"],
                "test_start_ts": fr["test_start_ts"],
                "test_end_ts": fr["test_end_ts"],
                "metrics": agent["metrics"],
                "regime_fit": agent.get("regime_fit") or {},
            }
        )
        for regime, stats in (agent.get("regime_performance") or {}).items():
            bucket = regime_acc.setdefault(
                regime,
                {"trade_count": 0, "wins": 0, "total_pnl": 0.0, "pnls": []},
            )
            n = int(stats.get("trade_count") or 0)
            wr = float(stats.get("win_rate") or 0)
            bucket["trade_count"] += n
            bucket["wins"] += int(round(wr * n))
            bucket["total_pnl"] += float(stats.get("total_pnl") or 0)
            if n:
                bucket["pnls"].append(float(stats.get("avg_pnl") or 0))

    stitched = _stitch_equity_curves(curves, starting_cash)
    bench_stitched = _stitch_equity_curves(benchmark_folds, starting_cash)
    metrics = compute_metrics(
        equity_curve=stitched,
        trades=trades,
        starting_cash=starting_cash,
        timeframe=timeframe,
        benchmark_curve=bench_stitched,
    )

    # Mean fold Sharpe (ignores None)
    sharpes = [
        f["metrics"]["sharpe"]
        for f in fold_summaries
        if f["metrics"].get("sharpe") is not None
    ]
    metrics["mean_fold_sharpe"] = (sum(sharpes) / len(sharpes)) if sharpes else None
    metrics["folds_with_trades"] = sum(
        1 for f in fold_summaries if (f["metrics"].get("trade_count") or 0) > 0
    )

    regime_performance: dict[str, Any] = {}
    for regime, bucket in regime_acc.items():
        n = bucket["trade_count"]
        if n <= 0:
            continue
        regime_performance[regime] = {
            "trade_count": n,
            "win_rate": bucket["wins"] / n,
            "total_pnl": bucket["total_pnl"],
            "avg_pnl": bucket["total_pnl"] / n,
        }

    return {
        "agent_id": agent_id,
        "name": name,
        "metrics": metrics,
        "regime_performance": regime_performance,
        "folds": fold_summaries,
        "equity_curve": stitched,
        "trade_count_oos": len(trades),
    }


class WalkForwardEngine:
    def __init__(
        self,
        store: Store,
        *,
        risk: RiskConfig | None = None,
    ) -> None:
        self.store = store
        self.risk = risk or RiskConfig()
        self.configs = default_strategy_configs()

    def run(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        warm_up: int = 120,
        calibrate_bars: int = 80,
        test_bars: int = 60,
        step_bars: int | None = None,
        benchmark_symbol: str = "SPY",
    ) -> dict[str, Any]:
        # Discover master timeline from the longest symbol series
        lengths: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df = self.store.load_candles(symbol, timeframe)
            if len(df) >= warm_up + calibrate_bars + test_bars:
                lengths[symbol] = df
        if not lengths:
            raise RuntimeError(
                "Not enough candle history for walk-forward "
                f"(need warm_up+calibrate+test ≥ {warm_up + calibrate_bars + test_bars})"
            )

        master_symbol = max(lengths, key=lambda s: len(lengths[s]))
        timestamps = lengths[master_symbol]["ts"].astype(str).tolist()
        folds = generate_bar_folds(
            timestamps,
            warm_up=warm_up,
            calibrate_bars=calibrate_bars,
            test_bars=test_bars,
            step_bars=step_bars,
        )
        if not folds:
            raise RuntimeError("Could not generate any walk-forward folds from available bars")

        engine = BacktestEngine(self.store, risk=self.risk, configs=self.configs)
        fold_payloads: list[dict[str, Any]] = []
        for fold in folds:
            result = engine.run(
                symbols=list(lengths.keys()),
                timeframe=timeframe,
                warm_up=warm_up,
                benchmark_symbol=benchmark_symbol,
                test_start_ts=fold.test_start_ts,
                test_end_ts=fold.test_end_ts,
            )
            fold_payloads.append(
                {
                    "fold_id": fold.fold_id,
                    **asdict(fold),
                    "run_id": result["run_id"],
                    "bars_tested": result["bars_tested"],
                    "market_regime_distribution": result.get("market_regime_distribution") or {},
                    "agents": [
                        {
                            "agent_id": a["agent_id"],
                            "name": a["name"],
                            "metrics": a["metrics"],
                            "regime_performance": a.get("regime_performance") or {},
                            "regime_fit": a.get("regime_fit") or {},
                            "trades": a.get("trades") or [],
                            "equity_curve": a.get("equity_curve") or [],
                            "strategy_config": a.get("strategy_config"),
                        }
                        for a in result["agents"]
                    ],
                    "benchmark_curve": result.get("benchmark_curve") or [],
                }
            )

        agent_ids = [c.strategy_id for c in self.configs]
        names = {c.strategy_id: c.name for c in self.configs}
        bench_folds = [fp.get("benchmark_curve") or [] for fp in fold_payloads]
        agents_oos = [
            _aggregate_agent_oos(
                aid,
                names[aid],
                fold_payloads,
                starting_cash=self.risk.starting_cash,
                timeframe=timeframe,
                benchmark_folds=bench_folds,
            )
            for aid in agent_ids
        ]
        agents_oos.sort(
            key=lambda r: (
                r["metrics"].get("mean_fold_sharpe") is not None
                or r["metrics"].get("sharpe") is not None,
                r["metrics"].get("mean_fold_sharpe")
                if r["metrics"].get("mean_fold_sharpe") is not None
                else (r["metrics"].get("sharpe") if r["metrics"].get("sharpe") is not None else -999),
                r["metrics"].get("total_return_pct") or -999,
            ),
            reverse=True,
        )

        raw = (
            f"wf|{timeframe}|{','.join(sorted(lengths.keys()))}|"
            f"{warm_up}|{calibrate_bars}|{test_bars}|{len(folds)}|{_git_sha() or 'nogit'}"
        )
        run_id = hashlib.sha1(raw.encode()).hexdigest()[:12]

        return {
            "run_id": run_id,
            "kind": "walk_forward",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "code_sha": _git_sha(),
            "timeframe": timeframe,
            "symbols": sorted(lengths.keys()),
            "benchmark_symbol": benchmark_symbol
            if benchmark_symbol in lengths
            else master_symbol,
            "warm_up": warm_up,
            "calibrate_bars": calibrate_bars,
            "test_bars": test_bars,
            "step_bars": step_bars if step_bars is not None else test_bars,
            "fold_count": len(folds),
            "folds": [
                {
                    "fold_id": f.fold_id,
                    "calibrate_start_ts": f.calibrate_start_ts,
                    "calibrate_end_ts": f.calibrate_end_ts,
                    "test_start_ts": f.test_start_ts,
                    "test_end_ts": f.test_end_ts,
                    "calibrate_bars": f.calibrate_bars,
                    "test_bars": f.test_bars,
                }
                for f in folds
            ],
            "agents": agents_oos,
            "fold_details": [
                {
                    "fold_id": fp["fold_id"],
                    "test_start_ts": fp["test_start_ts"],
                    "test_end_ts": fp["test_end_ts"],
                    "bars_tested": fp["bars_tested"],
                    "market_regime_distribution": fp["market_regime_distribution"],
                    "leaderboard": [
                        {
                            "agent_id": a["agent_id"],
                            "sharpe": a["metrics"].get("sharpe"),
                            "total_return_pct": a["metrics"].get("total_return_pct"),
                            "max_drawdown_pct": a["metrics"].get("max_drawdown_pct"),
                            "trade_count": a["metrics"].get("trade_count"),
                        }
                        for a in sorted(
                            fp["agents"],
                            key=lambda x: (
                                x["metrics"].get("sharpe") is not None,
                                x["metrics"].get("sharpe")
                                if x["metrics"].get("sharpe") is not None
                                else -999,
                            ),
                            reverse=True,
                        )
                    ],
                }
                for fp in fold_payloads
            ],
            "separates_paper_trading": True,
            "out_of_sample_only": True,
            "look_ahead_bias": False,
            "risk": asdict(self.risk),
        }


def write_walk_forward_result(payload: dict[str, Any], directory: Path = WALK_FORWARD_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{payload['run_id']}.json"
    # Full detail on disk
    path.write_text(json.dumps(payload, indent=2))

    compact = json.loads(json.dumps(payload))
    for agent in compact.get("agents", []):
        curve = agent.get("equity_curve") or []
        if len(curve) > 400:
            step = max(1, len(curve) // 400)
            agent["equity_curve"] = curve[::step]
        # drop per-fold heavy curves from compact
        for fold in agent.get("folds") or []:
            fold.pop("equity_curve", None)
    compact.pop("fold_details_full", None)

    latest = directory / "latest.json"
    latest.write_text(json.dumps(compact, indent=2))
    public = Path(ROOT) / "public" / "data" / "walk_forward" / "latest.json"
    public.parent.mkdir(parents=True, exist_ok=True)
    # API-friendly: drop stitched curves further
    api_payload = json.loads(json.dumps(compact))
    for agent in api_payload.get("agents", []):
        agent.pop("equity_curve", None)
    public.write_text(json.dumps(api_payload))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="TrendLab walk-forward OOS evaluation")
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--warm-up", type=int, default=120)
    parser.add_argument("--calibrate-bars", type=int, default=80)
    parser.add_argument("--test-bars", type=int, default=60)
    parser.add_argument("--step-bars", type=int, default=None)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument(
        "--also-full-backtest",
        action="store_true",
        help="Also write a full-period backtest JSON (does not mix into OOS metrics).",
    )
    args = parser.parse_args()

    store = Store()
    available = store.list_symbols_with_data(args.timeframe)
    symbols = [s for s in args.symbols if s in available]
    if not symbols:
        raise SystemExit(f"No candle data for timeframe {args.timeframe}. Run collect first.")

    # Auto-shrink windows when history is short (hourly free-tier lookback ~months).
    max_len = 0
    for sym in symbols:
        max_len = max(max_len, len(store.load_candles(sym, args.timeframe)))
    warm_up = args.warm_up
    calibrate = args.calibrate_bars
    test = args.test_bars
    need = warm_up + calibrate + test + 2
    if max_len < need:
        # Fit three folds if possible
        warm_up = min(warm_up, max(80, max_len // 5))
        calibrate = min(calibrate, max(40, max_len // 6))
        test = min(test, max(30, max_len // 6))
        if max_len < warm_up + calibrate + test + 2:
            warm_up = max(60, max_len // 4)
            calibrate = max(30, max_len // 8)
            test = max(25, max_len // 8)

    engine = WalkForwardEngine(store)
    result = engine.run(
        symbols=symbols,
        timeframe=args.timeframe,
        warm_up=warm_up,
        calibrate_bars=calibrate,
        test_bars=test,
        step_bars=args.step_bars,
        benchmark_symbol=args.benchmark,
    )
    path = write_walk_forward_result(result)

    if args.also_full_backtest:
        bt = BacktestEngine(store)
        full = bt.run(
            symbols=symbols,
            timeframe=args.timeframe,
            warm_up=warm_up,
            benchmark_symbol=args.benchmark,
        )
        write_backtest_result(full)

    summary = {
        "run_id": result["run_id"],
        "path": str(path),
        "fold_count": result["fold_count"],
        "warm_up": warm_up,
        "calibrate_bars": calibrate,
        "test_bars": test,
        "leaderboard_oos": [
            {
                "agent_id": a["agent_id"],
                "total_return_pct": a["metrics"].get("total_return_pct"),
                "sharpe": a["metrics"].get("sharpe"),
                "mean_fold_sharpe": a["metrics"].get("mean_fold_sharpe"),
                "max_drawdown_pct": a["metrics"].get("max_drawdown_pct"),
                "trade_count": a["metrics"].get("trade_count"),
                "alpha_pct": a["metrics"].get("alpha_pct"),
            }
            for a in result["agents"]
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
