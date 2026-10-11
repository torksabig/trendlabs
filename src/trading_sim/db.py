"""SQLite persistence for candles, trades, equity, and research notes."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from trading_sim.config import AGENTS, DB_PATH, STARTING_CASH


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path | str = DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    open_time TEXT NOT NULL,
                    close_time TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (symbol, interval, open_time)
                );

                CREATE TABLE IF NOT EXISTS agents (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    description TEXT NOT NULL,
                    cash REAL NOT NULL,
                    position_qty REAL NOT NULL DEFAULT 0,
                    avg_entry REAL NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    price REAL NOT NULL,
                    fee REAL NOT NULL,
                    ts TEXT NOT NULL,
                    reason TEXT,
                    FOREIGN KEY (agent_id) REFERENCES agents(id)
                );

                CREATE TABLE IF NOT EXISTS equity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    cash REAL NOT NULL,
                    position_qty REAL NOT NULL,
                    mark_price REAL NOT NULL,
                    equity REAL NOT NULL,
                    FOREIGN KEY (agent_id) REFERENCES agents(id)
                );

                CREATE TABLE IF NOT EXISTS research_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    proposals_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    def ensure_agents(self, labels: dict[str, str], descriptions: dict[str, str]) -> None:
        with self.connect() as conn:
            for agent_id in AGENTS:
                exists = conn.execute(
                    "SELECT 1 FROM agents WHERE id = ?", (agent_id,)
                ).fetchone()
                if exists:
                    continue
                conn.execute(
                    """
                    INSERT INTO agents (id, label, description, cash, position_qty, avg_entry, updated_at)
                    VALUES (?, ?, ?, ?, 0, 0, ?)
                    """,
                    (
                        agent_id,
                        labels[agent_id],
                        descriptions[agent_id],
                        STARTING_CASH,
                        utc_now(),
                    ),
                )

    def upsert_candles(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        rows = [
            (
                r.symbol,
                r.interval,
                r.open_time.isoformat(),
                r.close_time.isoformat(),
                float(r.open),
                float(r.high),
                float(r.low),
                float(r.close),
                float(r.volume),
            )
            for r in df.itertuples(index=False)
        ]
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO candles (
                    symbol, interval, open_time, close_time, open, high, low, close, volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, interval, open_time) DO UPDATE SET
                    close_time=excluded.close_time,
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    volume=excluded.volume
                """,
                rows,
            )
        return len(rows)

    def load_candles(self, symbol: str, interval: str) -> pd.DataFrame:
        with self.connect() as conn:
            df = pd.read_sql_query(
                """
                SELECT * FROM candles
                WHERE symbol = ? AND interval = ?
                ORDER BY open_time ASC
                """,
                conn,
                params=(symbol.upper(), interval),
            )
        if df.empty:
            return df
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
        return df

    def get_agents(self) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query("SELECT * FROM agents ORDER BY id", conn)

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        if row is None:
            raise KeyError(agent_id)
        return dict(row)

    def update_agent_state(
        self,
        agent_id: str,
        cash: float,
        position_qty: float,
        avg_entry: float,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE agents
                SET cash = ?, position_qty = ?, avg_entry = ?, updated_at = ?
                WHERE id = ?
                """,
                (cash, position_qty, avg_entry, utc_now(), agent_id),
            )

    def insert_trade(
        self,
        agent_id: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        fee: float,
        ts: str,
        reason: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO trades (agent_id, symbol, side, qty, price, fee, ts, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (agent_id, symbol, side, qty, price, fee, ts, reason),
            )

    def insert_equity(
        self,
        agent_id: str,
        ts: str,
        cash: float,
        position_qty: float,
        mark_price: float,
        equity: float,
    ) -> None:
        self.insert_equity_many(
            [(agent_id, ts, cash, position_qty, mark_price, equity)]
        )

    def insert_equity_many(
        self,
        rows: list[tuple[str, str, float, float, float, float]],
    ) -> None:
        if not rows:
            return
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO equity (agent_id, ts, cash, position_qty, mark_price, equity)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def get_trades(self, agent_id: str | None = None) -> pd.DataFrame:
        with self.connect() as conn:
            if agent_id:
                return pd.read_sql_query(
                    "SELECT * FROM trades WHERE agent_id = ? ORDER BY ts DESC",
                    conn,
                    params=(agent_id,),
                )
            return pd.read_sql_query("SELECT * FROM trades ORDER BY ts DESC", conn)

    def get_equity(self, agent_id: str | None = None) -> pd.DataFrame:
        with self.connect() as conn:
            if agent_id:
                return pd.read_sql_query(
                    "SELECT * FROM equity WHERE agent_id = ? ORDER BY ts ASC",
                    conn,
                    params=(agent_id,),
                )
            return pd.read_sql_query("SELECT * FROM equity ORDER BY ts ASC", conn)

    def clear_simulation(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM trades")
            conn.execute("DELETE FROM equity")
            conn.execute("DELETE FROM research_notes")
            conn.execute(
                "UPDATE agents SET cash = ?, position_qty = 0, avg_entry = 0, updated_at = ?",
                (STARTING_CASH, utc_now()),
            )

    def save_research_note(self, source: str, summary: str, proposals: list[str]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO research_notes (created_at, source, summary, proposals_json)
                VALUES (?, ?, ?, ?)
                """,
                (utc_now(), source, summary, json.dumps(proposals)),
            )

    def get_research_notes(self, limit: int = 20) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                """
                SELECT * FROM research_notes
                ORDER BY created_at DESC
                LIMIT ?
                """,
                conn,
                params=(limit,),
            )

    def set_meta(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default
