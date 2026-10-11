"""Local paper-trading broker — fills at bar close with simple fee model."""

from __future__ import annotations

from dataclasses import dataclass

from trading_sim.config import FEE_BPS, POSITION_SIZE_PCT
from trading_sim.db import Database


@dataclass
class Fill:
    side: str
    qty: float
    price: float
    fee: float
    reason: str


@dataclass
class AgentBook:
    cash: float
    position_qty: float = 0.0
    avg_entry: float = 0.0


class PaperBroker:
    def __init__(
        self,
        db: Database,
        fee_bps: float = FEE_BPS,
        position_size_pct: float = POSITION_SIZE_PCT,
    ) -> None:
        self.db = db
        self.fee_bps = fee_bps
        self.position_size_pct = position_size_pct

    def mark_equity(self, agent_id: str, mark_price: float, ts: str) -> float:
        state = self.db.get_agent(agent_id)
        equity = state["cash"] + state["position_qty"] * mark_price
        self.db.insert_equity(
            agent_id=agent_id,
            ts=ts,
            cash=state["cash"],
            position_qty=state["position_qty"],
            mark_price=mark_price,
            equity=equity,
        )
        return equity

    def reconcile(
        self,
        agent_id: str,
        symbol: str,
        desired_side: str,
        price: float,
        ts: str,
        reason: str,
        book: AgentBook | None = None,
        persist: bool = True,
    ) -> Fill | None:
        if book is None:
            state = self.db.get_agent(agent_id)
            book = AgentBook(
                cash=float(state["cash"]),
                position_qty=float(state["position_qty"]),
                avg_entry=float(state["avg_entry"]),
            )
        fee_rate = self.fee_bps / 10_000
        fill: Fill | None = None

        if desired_side == "long" and book.position_qty <= 0:
            notional = book.cash * self.position_size_pct
            if notional > 0 and price > 0:
                buy_qty = notional / price
                fee = notional * fee_rate
                if fee < book.cash:
                    book.cash = book.cash - notional - fee
                    book.position_qty = buy_qty
                    book.avg_entry = price
                    fill = Fill("BUY", buy_qty, price, fee, reason)
                    if persist:
                        self.db.insert_trade(
                            agent_id, symbol, "BUY", buy_qty, price, fee, ts, reason
                        )

        elif desired_side == "flat" and book.position_qty > 0:
            qty = book.position_qty
            proceeds = qty * price
            fee = proceeds * fee_rate
            book.cash = book.cash + proceeds - fee
            book.position_qty = 0.0
            book.avg_entry = 0.0
            fill = Fill("SELL", qty, price, fee, reason)
            if persist:
                self.db.insert_trade(agent_id, symbol, "SELL", qty, price, fee, ts, reason)

        if persist:
            self.db.update_agent_state(
                agent_id, book.cash, book.position_qty, book.avg_entry
            )
        return fill
