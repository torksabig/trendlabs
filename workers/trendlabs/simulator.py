"""Long-only paper broker with risk limits, fees, slippage, and idempotent fills."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from trendlabs.agents import BaseAgent, Signal, build_agents
from trendlabs.config import (
    FEE_BPS,
    MAX_POSITIONS,
    RISK_PER_TRADE,
    SLIPPAGE_BPS,
    STARTING_CASH,
)
from trendlabs.db import Store
from trendlabs.indicators import enrich
from trendlabs.trends import detect_trend_events


@dataclass
class Book:
    cash: float
    positions: dict[str, dict] = field(default_factory=dict)


class PaperSimulator:
    def __init__(self, store: Store) -> None:
        self.store = store
        self.agents = build_agents()

    def run(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        warm_up: int = 220,
    ) -> dict:
        self.store.reset_simulation()
        books = {a.id: Book(cash=STARTING_CASH) for a in self.agents}
        trade_rows: list[tuple] = []
        equity_rows: list[tuple] = []
        marks: dict[str, float] = {}

        series: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df = self.store.load_candles(symbol, timeframe)
            if len(df) < warm_up + 5:
                continue
            series[symbol] = enrich(df).reset_index(drop=True)

        if not series:
            raise RuntimeError("No symbols have enough candle history to simulate")

        # Master clock = longest series
        master_symbol = max(series, key=lambda s: len(series[s]))
        master = series[master_symbol]
        n = len(master)
        start = min(max(warm_up, 50), n - 1)

        # Map master timestamps to each symbol's last available index
        symbol_ts = {sym: df["ts"].astype(str).tolist() for sym, df in series.items()}
        symbol_index_at: dict[str, list[int | None]] = {sym: [None] * n for sym in series}
        for sym, ts_list in symbol_ts.items():
            j = 0
            for i, ts in enumerate(master["ts"].astype(str).tolist()):
                while j + 1 < len(ts_list) and ts_list[j + 1] <= ts:
                    j += 1
                if ts_list[j] <= ts:
                    symbol_index_at[sym][i] = j

        for i in range(start, n):
            ts = str(master["ts"].iloc[i])
            for sym, idxs in symbol_index_at.items():
                j = idxs[i]
                if j is not None:
                    marks[sym] = float(series[sym].iloc[j]["close"])

            for agent in self.agents:
                book = books[agent.id]
                signals: list[Signal] = []
                for sym, idxs in symbol_index_at.items():
                    j = idxs[i]
                    if j is None or j < 30:
                        continue
                    hist = series[sym].iloc[: j + 1]
                    signals.append(agent.evaluate(sym, hist))

                self._apply_exits(agent, book, signals, marks, ts, timeframe, trade_rows)
                self._apply_entries(agent, book, signals, marks, ts, timeframe, trade_rows)

                equity = book.cash + sum(
                    pos["qty"] * marks.get(sym, pos["avg_entry"]) for sym, pos in book.positions.items()
                )
                equity_rows.append((agent.id, ts, book.cash, equity))

        with self.store.connect() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO trades (
                    agent_id, symbol, side, qty, price, fee, slippage, stop_loss,
                    reason, timeframe, ts, idempotency_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                trade_rows,
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO equity_snapshots (agent_id, ts, cash, equity)
                VALUES (?, ?, ?, ?)
                """,
                equity_rows,
            )
            conn.execute("DELETE FROM positions")
            for agent_id, book in books.items():
                conn.execute(
                    "UPDATE agents SET cash = ?, updated_at = ? WHERE id = ?",
                    (book.cash, datetime.now(timezone.utc).isoformat(), agent_id),
                )
                for symbol, pos in book.positions.items():
                    conn.execute(
                        """
                        INSERT INTO positions (agent_id, symbol, qty, avg_entry, stop_loss, opened_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            agent_id,
                            symbol,
                            pos["qty"],
                            pos["avg_entry"],
                            pos.get("stop_loss"),
                            pos["opened_at"],
                        ),
                    )

        for symbol, df in series.items():
            for event in detect_trend_events(symbol, timeframe, df):
                self.store.insert_trend_event(event)

        self.store.set_meta("last_simulation", datetime.now(timezone.utc).isoformat())
        self.store.set_meta("simulation_timeframe", timeframe)

        leaderboard = []
        for agent in self.agents:
            book = books[agent.id]
            equity = book.cash + sum(
                pos["qty"] * marks.get(sym, pos["avg_entry"]) for sym, pos in book.positions.items()
            )
            leaderboard.append(
                {
                    "agent_id": agent.id,
                    "name": agent.name,
                    "equity": equity,
                    "return_pct": (equity / STARTING_CASH - 1) * 100,
                    "positions": len(book.positions),
                    "cash": book.cash,
                }
            )
        leaderboard.sort(key=lambda x: x["equity"], reverse=True)
        return {
            "symbols": list(series.keys()),
            "timeframe": timeframe,
            "bars": n - start,
            "trades": len(trade_rows),
            "leaderboard": leaderboard,
        }

    def _apply_exits(
        self,
        agent: BaseAgent,
        book: Book,
        signals: list[Signal],
        marks: dict[str, float],
        ts: str,
        timeframe: str,
        trade_rows: list[tuple],
    ) -> None:
        by_symbol = {s.symbol: s for s in signals}
        for symbol in list(book.positions.keys()):
            pos = book.positions[symbol]
            price = marks.get(symbol)
            if price is None:
                continue
            signal = by_symbol.get(symbol)
            stop_hit = pos.get("stop_loss") is not None and price <= float(pos["stop_loss"])
            flatten = signal is None or signal.side == "flat" or stop_hit
            if not flatten:
                continue
            reason = "Stop loss triggered" if stop_hit else (signal.reason if signal else "Signal flat")
            self._sell(agent.id, book, symbol, price, ts, timeframe, reason, trade_rows)

    def _apply_entries(
        self,
        agent: BaseAgent,
        book: Book,
        signals: list[Signal],
        marks: dict[str, float],
        ts: str,
        timeframe: str,
        trade_rows: list[tuple],
    ) -> None:
        longs = [s for s in signals if s.side == "long" and s.symbol not in book.positions]
        longs.sort(key=lambda s: s.signal_strength, reverse=True)
        for signal in longs:
            if len(book.positions) >= MAX_POSITIONS:
                break
            price = marks.get(signal.symbol)
            if price is None or price <= 0:
                continue
            self._buy(agent.id, book, signal, price, ts, timeframe, trade_rows)

    def _buy(
        self,
        agent_id: str,
        book: Book,
        signal: Signal,
        price: float,
        ts: str,
        timeframe: str,
        trade_rows: list[tuple],
    ) -> None:
        slip = price * (SLIPPAGE_BPS / 10_000)
        fill = price + slip
        stop = signal.stop_loss or (fill * 0.97)
        risk_per_share = max(fill - stop, fill * 0.01)
        risk_budget = book.cash * RISK_PER_TRADE
        max_notional = book.cash / max(1, MAX_POSITIONS - len(book.positions))
        qty = min(risk_budget / risk_per_share, max_notional / fill)
        qty = max(0.0, qty)
        if qty * fill < 25:
            return
        notional = qty * fill
        fee = notional * (FEE_BPS / 10_000)
        if notional + fee > book.cash:
            return
        book.cash -= notional + fee
        book.positions[signal.symbol] = {
            "qty": qty,
            "avg_entry": fill,
            "stop_loss": stop,
            "opened_at": ts,
        }
        key = f"{agent_id}:{signal.symbol}:BUY:{ts}:{timeframe}"
        trade_rows.append(
            (
                agent_id,
                signal.symbol,
                "BUY",
                qty,
                fill,
                fee,
                slip,
                stop,
                signal.reason,
                timeframe,
                ts,
                key,
            )
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
        trade_rows: list[tuple],
    ) -> None:
        pos = book.positions.pop(symbol, None)
        if not pos:
            return
        slip = price * (SLIPPAGE_BPS / 10_000)
        fill = price - slip
        notional = pos["qty"] * fill
        fee = notional * (FEE_BPS / 10_000)
        book.cash += notional - fee
        key = f"{agent_id}:{symbol}:SELL:{ts}:{timeframe}"
        trade_rows.append(
            (
                agent_id,
                symbol,
                "SELL",
                pos["qty"],
                fill,
                fee,
                slip,
                pos.get("stop_loss"),
                reason,
                timeframe,
                ts,
                key,
            )
        )
