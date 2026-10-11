"""Simulation engine: fetch candles, run agents, paper-trade, record equity."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_sim.agents import AGENT_DESCRIPTIONS, build_agents
from trading_sim.config import (
    AGENT_LABELS,
    DEFAULT_INTERVAL,
    DEFAULT_LOOKBACK,
    DEFAULT_SYMBOL,
    STARTING_CASH,
)
from trading_sim.db import Database
from trading_sim.market_data import BinancePublicClient
from trading_sim.paper_broker import AgentBook, PaperBroker


@dataclass
class RunResult:
    symbol: str
    interval: str
    bars_processed: int
    trades: int
    latest_prices: dict[str, float]
    equity: dict[str, float]
    signals: dict[str, str]


class SimulationEngine:
    def __init__(
        self,
        db: Database | None = None,
        client: BinancePublicClient | None = None,
    ) -> None:
        self.db = db or Database()
        self.client = client or BinancePublicClient()
        self.broker = PaperBroker(self.db)
        self.agents = build_agents()
        self.db.ensure_agents(AGENT_LABELS, AGENT_DESCRIPTIONS)

    def sync_market_data(
        self,
        symbol: str = DEFAULT_SYMBOL,
        interval: str = DEFAULT_INTERVAL,
        limit: int = DEFAULT_LOOKBACK,
    ) -> pd.DataFrame:
        candles = self.client.fetch_klines(symbol=symbol, interval=interval, limit=limit)
        self.db.upsert_candles(candles)
        self.db.set_meta("symbol", symbol.upper())
        self.db.set_meta("interval", interval)
        self.db.set_meta("last_sync", candles["close_time"].iloc[-1].isoformat())
        return candles

    def reset(self) -> None:
        self.db.clear_simulation()

    def backtest(
        self,
        symbol: str = DEFAULT_SYMBOL,
        interval: str = DEFAULT_INTERVAL,
        limit: int = DEFAULT_LOOKBACK,
        refetch: bool = True,
        warm_up: int = 80,
    ) -> RunResult:
        if refetch:
            candles = self.sync_market_data(symbol, interval, limit)
        else:
            candles = self.db.load_candles(symbol, interval)
            if candles.empty:
                candles = self.sync_market_data(symbol, interval, limit)

        self.reset()
        self.db.ensure_agents(AGENT_LABELS, AGENT_DESCRIPTIONS)

        trade_count = 0
        signals: dict[str, str] = {}
        series = {aid: agent.generate_signal_series(candles) for aid, agent in self.agents.items()}
        books = {aid: AgentBook(cash=STARTING_CASH) for aid in self.agents}
        equity_rows: list[tuple[str, str, float, float, float, float]] = []
        trade_rows: list[tuple] = []

        start = min(max(warm_up, 30), len(candles) - 1)
        fee_rate = self.broker.fee_bps / 10_000
        size_pct = self.broker.position_size_pct
        symbol_u = symbol.upper()

        for i in range(start, len(candles)):
            bar = candles.iloc[i]
            price = float(bar["close"])
            ts = bar["close_time"].isoformat()

            for agent_id in self.agents:
                signal = series[agent_id][i]
                book = books[agent_id]
                fill = None
                if signal.side == "long" and book.position_qty <= 0:
                    notional = book.cash * size_pct
                    if notional > 0 and price > 0:
                        buy_qty = notional / price
                        fee = notional * fee_rate
                        if fee < book.cash:
                            book.cash -= notional + fee
                            book.position_qty = buy_qty
                            book.avg_entry = price
                            fill = ("BUY", buy_qty, price, fee, signal.reason)
                elif signal.side == "flat" and book.position_qty > 0:
                    qty = book.position_qty
                    proceeds = qty * price
                    fee = proceeds * fee_rate
                    book.cash += proceeds - fee
                    book.position_qty = 0.0
                    book.avg_entry = 0.0
                    fill = ("SELL", qty, price, fee, signal.reason)

                if fill:
                    trade_count += 1
                    trade_rows.append(
                        (agent_id, symbol_u, fill[0], fill[1], fill[2], fill[3], ts, fill[4])
                    )
                eq = book.cash + book.position_qty * price
                equity_rows.append(
                    (agent_id, ts, book.cash, book.position_qty, price, eq)
                )
                if i == len(candles) - 1:
                    signals[agent_id] = f"{signal.side.upper()}: {signal.reason}"

        with self.db.connect() as conn:
            conn.executemany(
                """
                INSERT INTO trades (agent_id, symbol, side, qty, price, fee, ts, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                trade_rows,
            )
            conn.executemany(
                """
                INSERT INTO equity (agent_id, ts, cash, position_qty, mark_price, equity)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                equity_rows,
            )
            for agent_id, book in books.items():
                conn.execute(
                    """
                    UPDATE agents
                    SET cash = ?, position_qty = ?, avg_entry = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (book.cash, book.position_qty, book.avg_entry, ts, agent_id),
                )

        equity = {}
        latest_prices = {}
        mark = float(candles["close"].iloc[-1])
        for agent_id, book in books.items():
            equity[agent_id] = book.cash + book.position_qty * mark
            latest_prices[agent_id] = mark

        return RunResult(
            symbol=symbol.upper(),
            interval=interval,
            bars_processed=len(candles) - start,
            trades=trade_count,
            latest_prices=latest_prices,
            equity=equity,
            signals=signals,
        )

    def step_live(
        self,
        symbol: str = DEFAULT_SYMBOL,
        interval: str = DEFAULT_INTERVAL,
        limit: int = DEFAULT_LOOKBACK,
    ) -> RunResult:
        """Fetch latest candles and advance each agent one decision on the newest bar."""
        candles = self.sync_market_data(symbol, interval, limit)
        price = float(candles["close"].iloc[-1])
        ts = candles["close_time"].iloc[-1].isoformat()
        trade_count = 0
        signals: dict[str, str] = {}
        equity: dict[str, float] = {}

        for agent_id, agent in self.agents.items():
            signal = agent.generate_signal(candles)
            fill = self.broker.reconcile(
                agent_id=agent_id,
                symbol=symbol.upper(),
                desired_side=signal.side,
                price=price,
                ts=ts,
                reason=signal.reason,
            )
            if fill:
                trade_count += 1
            eq = self.broker.mark_equity(agent_id, price, ts)
            equity[agent_id] = eq
            signals[agent_id] = f"{signal.side.upper()}: {signal.reason}"

        return RunResult(
            symbol=symbol.upper(),
            interval=interval,
            bars_processed=1,
            trades=trade_count,
            latest_prices={aid: price for aid in self.agents},
            equity=equity,
            signals=signals,
        )
