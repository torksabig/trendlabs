"""Deterministic historical backtester — separate from forward paper-trading state.

Execution model (no look-ahead):
1. Indicators/signals use candles through bar i (completed bar).
2. Orders fill at bar i+1 open (+/- slippage), never the signal bar close.
3. Stop-losses checked against bar i+1 low/high before other exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trendlabs.agents import BaseAgent, Signal, build_agents
from trendlabs.config import DEFAULT_TIMEFRAME, ROOT, SYMBOLS
from trendlabs.db import Store
from trendlabs.indicators import enrich
from trendlabs.metrics import compute_metrics
from trendlabs.strategy_config import RiskConfig, StrategyConfig, default_strategy_configs


BACKTEST_DIR = Path(ROOT) / "data" / "backtests"


@dataclass
class Book:
    cash: float
    positions: dict[str, dict] = field(default_factory=dict)


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


def _buy_and_hold_curve(
    prices: pd.Series,
    timestamps: pd.Series,
    starting_cash: float,
) -> list[dict[str, Any]]:
    if prices.empty or float(prices.iloc[0]) <= 0:
        return []
    qty = starting_cash / float(prices.iloc[0])
    return [
        {
            "ts": str(timestamps.iloc[i]),
            "equity": float(qty * float(prices.iloc[i])),
            "cash": 0.0,
        }
        for i in range(len(prices))
    ]


class BacktestEngine:
    """Multi-symbol, multi-agent historical replay with next-open fills."""

    def __init__(
        self,
        store: Store,
        *,
        risk: RiskConfig | None = None,
        agents: list[BaseAgent] | None = None,
        configs: list[StrategyConfig] | None = None,
    ) -> None:
        self.store = store
        self.risk = risk or RiskConfig()
        self.agents = agents or build_agents()
        self.configs = {c.strategy_id: c for c in (configs or default_strategy_configs())}

    def run(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        warm_up: int = 220,
        benchmark_symbol: str = "SPY",
    ) -> dict[str, Any]:
        risk = self.risk
        series: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df = self.store.load_candles(symbol, timeframe)
            if len(df) < warm_up + 5:
                continue
            series[symbol] = enrich(df).reset_index(drop=True)
        if not series:
            raise RuntimeError("No symbols have enough candle history to backtest")

        master_symbol = max(series, key=lambda s: len(series[s]))
        master = series[master_symbol]
        n = len(master)
        start = min(max(warm_up, 50), n - 2)

        symbol_ts = {sym: df["ts"].astype(str).tolist() for sym, df in series.items()}
        symbol_index_at: dict[str, list[int | None]] = {sym: [None] * n for sym in series}
        for sym, ts_list in symbol_ts.items():
            j = 0
            for i, ts in enumerate(master["ts"].astype(str).tolist()):
                while j + 1 < len(ts_list) and ts_list[j + 1] <= ts:
                    j += 1
                if ts_list[j] <= ts:
                    symbol_index_at[sym][i] = j

        books = {a.id: Book(cash=risk.starting_cash) for a in self.agents}
        trades: dict[str, list[dict[str, Any]]] = {a.id: [] for a in self.agents}
        equity: dict[str, list[dict[str, Any]]] = {a.id: [] for a in self.agents}
        pending: dict[str, list[Signal]] = {a.id: [] for a in self.agents}

        for i in range(start, n - 1):
            # Fill pending signals from previous bar at this bar's open
            for agent in self.agents:
                book = books[agent.id]
                fills = pending[agent.id]
                pending[agent.id] = []
                if fills:
                    self._execute_pending(
                        agent,
                        book,
                        fills,
                        series,
                        symbol_index_at,
                        i,
                        timeframe,
                        trades[agent.id],
                    )

            # Mark equity at this bar close
            marks = self._marks(series, symbol_index_at, i)
            ts = str(master["ts"].iloc[i])
            for agent in self.agents:
                book = books[agent.id]
                eq = book.cash + sum(
                    pos["qty"] * marks.get(sym, pos["avg_entry"])
                    for sym, pos in book.positions.items()
                )
                equity[agent.id].append({"ts": ts, "equity": eq, "cash": book.cash})

            # New signals from completed bar i → execute next bar
            for agent in self.agents:
                signals: list[Signal] = []
                for sym, idxs in symbol_index_at.items():
                    j = idxs[i]
                    if j is None or j < 30:
                        continue
                    hist = series[sym].iloc[: j + 1]
                    signals.append(agent.evaluate(sym, hist))
                pending[agent.id] = signals

        # Final mark
        marks = self._marks(series, symbol_index_at, n - 1)
        ts = str(master["ts"].iloc[n - 1])
        for agent in self.agents:
            book = books[agent.id]
            eq = book.cash + sum(
                pos["qty"] * marks.get(sym, pos["avg_entry"])
                for sym, pos in book.positions.items()
            )
            equity[agent.id].append({"ts": ts, "equity": eq, "cash": book.cash})

        # Benchmark buy & hold on SPY (or first symbol)
        bench_sym = benchmark_symbol if benchmark_symbol in series else master_symbol
        bdf = series[bench_sym].iloc[start:]
        benchmark_curve = _buy_and_hold_curve(bdf["close"], bdf["ts"], risk.starting_cash)

        results = []
        for agent in self.agents:
            cfg = self.configs.get(agent.id)
            metrics = compute_metrics(
                equity_curve=equity[agent.id],
                trades=trades[agent.id],
                starting_cash=risk.starting_cash,
                timeframe=timeframe,
                benchmark_curve=benchmark_curve,
            )
            results.append(
                {
                    "agent_id": agent.id,
                    "name": agent.name,
                    "strategy_config": cfg.to_dict() if cfg else {"strategy_id": agent.id},
                    "metrics": metrics,
                    "trades": trades[agent.id],
                    "equity_curve": equity[agent.id],
                    "open_positions": {
                        sym: {
                            "qty": pos["qty"],
                            "avg_entry": pos["avg_entry"],
                            "stop_loss": pos.get("stop_loss"),
                            "opened_at": pos["opened_at"],
                        }
                        for sym, pos in books[agent.id].positions.items()
                    },
                }
            )

        results.sort(
            key=lambda r: (
                r["metrics"]["sharpe"] is not None,
                r["metrics"]["sharpe"] if r["metrics"]["sharpe"] is not None else -999,
                r["metrics"]["total_return_pct"],
            ),
            reverse=True,
        )

        payload = {
            "run_id": self._run_id(symbols, timeframe, start, n),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "code_sha": _git_sha(),
            "timeframe": timeframe,
            "symbols": sorted(series.keys()),
            "benchmark_symbol": bench_sym,
            "warm_up": warm_up,
            "bars_tested": (n - 1) - start,
            "start_ts": str(master["ts"].iloc[start]),
            "end_ts": str(master["ts"].iloc[n - 1]),
            "execution_model": "signal_on_bar_close_fill_next_open",
            "risk": {
                "starting_cash": risk.starting_cash,
                "fee_bps": risk.fee_bps,
                "slippage_bps": risk.slippage_bps,
                "risk_per_trade": risk.risk_per_trade,
                "max_positions": risk.max_positions,
                "atr_stop_mult": risk.atr_stop_mult,
            },
            "benchmark_curve": benchmark_curve,
            "agents": results,
            "separates_paper_trading": True,
            "look_ahead_bias": False,
        }
        return payload

    def _run_id(self, symbols: list[str], timeframe: str, start: int, n: int) -> str:
        raw = f"{timeframe}|{','.join(sorted(symbols))}|{start}|{n}|{_git_sha() or 'nogit'}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def _marks(
        self,
        series: dict[str, pd.DataFrame],
        symbol_index_at: dict[str, list[int | None]],
        i: int,
    ) -> dict[str, float]:
        marks: dict[str, float] = {}
        for sym, idxs in symbol_index_at.items():
            j = idxs[i]
            if j is not None:
                marks[sym] = float(series[sym].iloc[j]["close"])
        return marks

    def _bar(
        self,
        series: dict[str, pd.DataFrame],
        symbol_index_at: dict[str, list[int | None]],
        symbol: str,
        i: int,
    ) -> pd.Series | None:
        j = symbol_index_at[symbol][i]
        if j is None:
            return None
        return series[symbol].iloc[j]

    def _execute_pending(
        self,
        agent: BaseAgent,
        book: Book,
        signals: list[Signal],
        series: dict[str, pd.DataFrame],
        symbol_index_at: dict[str, list[int | None]],
        i: int,
        timeframe: str,
        trade_rows: list[dict[str, Any]],
    ) -> None:
        risk = self.risk
        by_symbol = {s.symbol: s for s in signals}
        # Stops / flats first
        for symbol in list(book.positions.keys()):
            bar = self._bar(series, symbol_index_at, symbol, i)
            if bar is None:
                continue
            pos = book.positions[symbol]
            open_px = float(bar["open"])
            low_px = float(bar["low"])
            signal = by_symbol.get(symbol)
            stop = pos.get("stop_loss")
            stop_hit = stop is not None and low_px <= float(stop)
            flatten = signal is None or signal.side == "flat" or stop_hit
            if not flatten:
                # trail stop update from signal if still long
                if signal and signal.stop_loss is not None:
                    pos["stop_loss"] = max(float(pos.get("stop_loss") or 0), float(signal.stop_loss))
                continue
            fill_px = min(open_px, float(stop)) if stop_hit and stop is not None else open_px
            reason = "Stop loss triggered" if stop_hit else (signal.reason if signal else "Signal flat")
            self._sell(agent.id, book, symbol, fill_px, str(bar["ts"]), timeframe, reason, trade_rows)

        # Entries
        longs = [
            s
            for s in signals
            if s.side == "long" and s.symbol not in book.positions and s.symbol in series
        ]
        longs.sort(key=lambda s: s.signal_strength, reverse=True)
        for signal in longs:
            if len(book.positions) >= risk.max_positions:
                break
            bar = self._bar(series, symbol_index_at, signal.symbol, i)
            if bar is None:
                continue
            self._buy(agent.id, book, signal, float(bar["open"]), str(bar["ts"]), timeframe, trade_rows)

    def _buy(
        self,
        agent_id: str,
        book: Book,
        signal: Signal,
        price: float,
        ts: str,
        timeframe: str,
        trade_rows: list[dict[str, Any]],
    ) -> None:
        risk = self.risk
        slip = price * (risk.slippage_bps / 10_000)
        fill = price + slip
        stop = signal.stop_loss or (fill * (1 - 0.01 * risk.atr_stop_mult))
        risk_per_share = max(fill - float(stop), fill * 0.01)
        risk_budget = book.cash * risk.risk_per_trade
        slots = max(1, risk.max_positions - len(book.positions))
        max_notional = book.cash / slots
        qty = min(risk_budget / risk_per_share, max_notional / fill)
        qty = max(0.0, qty)
        if qty * fill < risk.min_notional:
            return
        notional = qty * fill
        fee = notional * (risk.fee_bps / 10_000)
        if notional + fee > book.cash:
            return
        book.cash -= notional + fee
        book.positions[signal.symbol] = {
            "qty": qty,
            "avg_entry": fill,
            "stop_loss": float(stop),
            "opened_at": ts,
        }
        trade_rows.append(
            {
                "agent_id": agent_id,
                "symbol": signal.symbol,
                "side": "BUY",
                "qty": qty,
                "price": fill,
                "fee": fee,
                "slippage": slip,
                "stop_loss": float(stop),
                "reason": signal.reason,
                "timeframe": timeframe,
                "ts": ts,
                "signal_strength": signal.signal_strength,
            }
        )

    def _sell(
        self,
        agent_id: str,
        book: Book,
        symbol: str,
        price: float,
        ts: str,
        timeframe: str,
        reason: str,
        trade_rows: list[dict[str, Any]],
    ) -> None:
        risk = self.risk
        pos = book.positions.pop(symbol, None)
        if not pos:
            return
        slip = price * (risk.slippage_bps / 10_000)
        fill = price - slip
        notional = pos["qty"] * fill
        fee = notional * (risk.fee_bps / 10_000)
        book.cash += notional - fee
        trade_rows.append(
            {
                "agent_id": agent_id,
                "symbol": symbol,
                "side": "SELL",
                "qty": pos["qty"],
                "price": fill,
                "fee": fee,
                "slippage": slip,
                "stop_loss": pos.get("stop_loss"),
                "reason": reason,
                "timeframe": timeframe,
                "ts": ts,
                "signal_strength": None,
            }
        )


def write_backtest_result(payload: dict[str, Any], directory: Path = BACKTEST_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{payload['run_id']}.json"
    path.write_text(json.dumps(payload, indent=2))
    latest = directory / "latest.json"
    # Compact latest for API (drop full equity points beyond 400)
    compact = json.loads(json.dumps(payload))
    for agent in compact.get("agents", []):
        curve = agent.get("equity_curve") or []
        if len(curve) > 400:
            step = max(1, len(curve) // 400)
            agent["equity_curve"] = curve[::step]
        # keep trade count but trim trade list in latest
        trades = agent.get("trades") or []
        if len(trades) > 80:
            agent["trades"] = trades[-80:]
    if compact.get("benchmark_curve") and len(compact["benchmark_curve"]) > 400:
        step = max(1, len(compact["benchmark_curve"]) // 400)
        compact["benchmark_curve"] = compact["benchmark_curve"][::step]
    latest.write_text(json.dumps(compact, indent=2))
    public = Path(ROOT) / "public" / "data" / "backtests" / "latest.json"
    public.parent.mkdir(parents=True, exist_ok=True)
    public.write_text(json.dumps(compact))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="TrendLab deterministic backtest engine")
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--warm-up", type=int, default=220)
    parser.add_argument("--benchmark", default="SPY")
    args = parser.parse_args()

    store = Store()
    available = store.list_symbols_with_data(args.timeframe)
    symbols = [s for s in args.symbols if s in available]
    if not symbols:
        raise SystemExit(f"No candle data for timeframe {args.timeframe}. Run collect first.")

    engine = BacktestEngine(store)
    result = engine.run(
        symbols=symbols,
        timeframe=args.timeframe,
        warm_up=args.warm_up,
        benchmark_symbol=args.benchmark,
    )
    path = write_backtest_result(result)
    summary = {
        "run_id": result["run_id"],
        "path": str(path),
        "start_ts": result["start_ts"],
        "end_ts": result["end_ts"],
        "bars_tested": result["bars_tested"],
        "leaderboard": [
            {
                "agent_id": a["agent_id"],
                "total_return_pct": a["metrics"]["total_return_pct"],
                "sharpe": a["metrics"]["sharpe"],
                "max_drawdown_pct": a["metrics"]["max_drawdown_pct"],
                "trade_count": a["metrics"]["trade_count"],
                "alpha_pct": a["metrics"]["alpha_pct"],
            }
            for a in result["agents"]
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
