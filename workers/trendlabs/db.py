from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from trendlabs.config import AGENTS, DB_PATH, INSTRUMENTS, STARTING_CASH


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()
        self.seed_reference_data()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
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
                CREATE TABLE IF NOT EXISTS instruments (
                    symbol TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    sector TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    source TEXT NOT NULL,
                    adjusted INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (symbol, timeframe, ts)
                );
                CREATE TABLE IF NOT EXISTS agents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    description TEXT NOT NULL,
                    cash REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS positions (
                    agent_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    qty REAL NOT NULL,
                    avg_entry REAL NOT NULL,
                    stop_loss REAL,
                    opened_at TEXT NOT NULL,
                    PRIMARY KEY (agent_id, symbol)
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    price REAL NOT NULL,
                    fee REAL NOT NULL,
                    slippage REAL NOT NULL,
                    stop_loss REAL,
                    reason TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    agent_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    cash REAL NOT NULL,
                    equity REAL NOT NULL,
                    PRIMARY KEY (agent_id, ts)
                );
                CREATE TABLE IF NOT EXISTS trend_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    strength REAL,
                    volatility REAL,
                    support REAL,
                    resistance REAL,
                    features_json TEXT NOT NULL,
                    forward_return_5 REAL,
                    forward_return_10 REAL,
                    forward_return_20 REAL
                );
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    model TEXT NOT NULL,
                    source TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    last_close REAL NOT NULL,
                    drivers_json TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    model TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    next_update_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS prediction_outcomes (
                    prediction_id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    predicted_at TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    expected_move_pct REAL,
                    last_close REAL NOT NULL,
                    actual_1h_pct REAL,
                    actual_4h_pct REAL,
                    actual_24h_pct REAL,
                    hit_1h INTEGER,
                    hit_4h INTEGER,
                    hit_24h INTEGER,
                    scored_at TEXT NOT NULL,
                    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
                );
                CREATE TABLE IF NOT EXISTS experiment_results (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def seed_reference_data(self) -> None:
        with self.connect() as conn:
            for item in INSTRUMENTS:
                conn.execute(
                    """
                    INSERT INTO instruments (symbol, name, sector)
                    VALUES (?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET name=excluded.name, sector=excluded.sector
                    """,
                    (item["symbol"], item["name"], item["sector"]),
                )
            for agent in AGENTS:
                exists = conn.execute("SELECT 1 FROM agents WHERE id = ?", (agent["id"],)).fetchone()
                if not exists:
                    conn.execute(
                        """
                        INSERT INTO agents (id, name, strategy, description, cash, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            agent["id"],
                            agent["name"],
                            agent["strategy"],
                            agent["description"],
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
                r.timeframe,
                pd.Timestamp(r.ts).isoformat(),
                float(r.open),
                float(r.high),
                float(r.low),
                float(r.close),
                float(r.volume),
                r.source,
                1 if bool(getattr(r, "adjusted", True)) else 0,
            )
            for r in df.itertuples(index=False)
        ]
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO candles (
                    symbol, timeframe, ts, open, high, low, close, volume, source, adjusted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timeframe, ts) DO UPDATE SET
                    open=excluded.open, high=excluded.high, low=excluded.low,
                    close=excluded.close, volume=excluded.volume, source=excluded.source,
                    adjusted=excluded.adjusted
                """,
                rows,
            )
        return len(rows)

    def load_candles(self, symbol: str, timeframe: str) -> pd.DataFrame:
        with self.connect() as conn:
            df = pd.read_sql_query(
                """
                SELECT * FROM candles
                WHERE symbol = ? AND timeframe = ?
                ORDER BY ts ASC
                """,
                conn,
                params=(symbol.upper(), timeframe),
            )
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    def list_symbols_with_data(self, timeframe: str) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM candles WHERE timeframe = ? ORDER BY symbol",
                (timeframe,),
            ).fetchall()
        return [r["symbol"] for r in rows]

    def reset_simulation(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM trades")
            conn.execute("DELETE FROM positions")
            conn.execute("DELETE FROM equity_snapshots")
            conn.execute("UPDATE agents SET cash = ?, updated_at = ?", (STARTING_CASH, utc_now()))

    def get_agents(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM agents ORDER BY id").fetchall()]

    def get_positions(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM positions").fetchall()]

    def get_trades(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM trades ORDER BY ts DESC LIMIT ?", (limit,)
                ).fetchall()
            ]

    def get_equity(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM equity_snapshots ORDER BY ts ASC"
                ).fetchall()
            ]

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

    def insert_trend_event(self, event: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO trend_events (
                    symbol, timeframe, detected_at, direction, strength, volatility,
                    support, resistance, features_json, forward_return_5,
                    forward_return_10, forward_return_20
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["symbol"],
                    event["timeframe"],
                    event["detected_at"],
                    event["direction"],
                    event.get("strength"),
                    event.get("volatility"),
                    event.get("support"),
                    event.get("resistance"),
                    json.dumps(event.get("features", {})),
                    event.get("forward_return_5"),
                    event.get("forward_return_10"),
                    event.get("forward_return_20"),
                ),
            )

    def get_trend_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trend_events ORDER BY detected_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["features"] = json.loads(item.pop("features_json"))
            out.append(item)
        return out

    def insert_agent_report(self, payload: dict[str, Any]) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO agent_reports (
                    symbol, timeframe, model, source, report_json, snapshot_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["symbol"],
                    payload["timeframe"],
                    payload["model"],
                    payload.get("source", "unknown"),
                    json.dumps(payload["report"]),
                    json.dumps(payload.get("snapshot", {})),
                    payload.get("created_at", utc_now()),
                ),
            )
            return int(cur.lastrowid)

    def get_agent_reports(self, limit: int = 20, symbol: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if symbol:
                rows = conn.execute(
                    """
                    SELECT * FROM agent_reports
                    WHERE symbol = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (symbol.upper(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM agent_reports
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["report"] = json.loads(item.pop("report_json"))
            item["snapshot"] = json.loads(item.pop("snapshot_json"))
            out.append(item)
        return out

    def insert_prediction(self, payload: dict[str, Any]) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO predictions (
                    symbol, timeframe, horizon, direction, confidence, last_close,
                    drivers_json, summary, model, source, created_at, next_update_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["symbol"],
                    payload["timeframe"],
                    payload["horizon"],
                    payload["direction"],
                    float(payload["confidence"]),
                    float(payload["last_close"]),
                    json.dumps(payload.get("drivers", {})),
                    payload["summary"],
                    payload.get("model", "rule_based"),
                    payload.get("source", "indicators"),
                    payload.get("created_at", utc_now()),
                    payload["next_update_at"],
                ),
            )
            return int(cur.lastrowid)

    def get_latest_predictions(self, timeframe: str | None = None) -> list[dict[str, Any]]:
        """Latest prediction per symbol (optionally filtered by timeframe)."""
        with self.connect() as conn:
            if timeframe:
                rows = conn.execute(
                    """
                    SELECT p.* FROM predictions p
                    INNER JOIN (
                        SELECT symbol, MAX(id) AS max_id
                        FROM predictions
                        WHERE timeframe = ?
                        GROUP BY symbol
                    ) latest ON p.id = latest.max_id
                    ORDER BY p.symbol ASC
                    """,
                    (timeframe,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT p.* FROM predictions p
                    INNER JOIN (
                        SELECT symbol, MAX(id) AS max_id
                        FROM predictions
                        GROUP BY symbol
                    ) latest ON p.id = latest.max_id
                    ORDER BY p.symbol ASC
                    """
                ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["drivers"] = json.loads(item.pop("drivers_json"))
            out.append(item)
        return out

    def get_predictions(self, limit: int = 48) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM predictions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["drivers"] = json.loads(item.pop("drivers_json"))
            out.append(item)
        return out

    def get_predictions_for_eval(self, limit: int = 500) -> list[dict[str, Any]]:
        """Oldest-first batch for outcome scoring (skips already scored)."""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT p.* FROM predictions p
                LEFT JOIN prediction_outcomes o ON o.prediction_id = p.id
                WHERE o.prediction_id IS NULL
                ORDER BY p.created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["drivers"] = json.loads(item.pop("drivers_json"))
            out.append(item)
        return out

    def upsert_prediction_outcome(self, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO prediction_outcomes (
                    prediction_id, symbol, timeframe, predicted_at, direction,
                    confidence, expected_move_pct, last_close,
                    actual_1h_pct, actual_4h_pct, actual_24h_pct,
                    hit_1h, hit_4h, hit_24h, scored_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(prediction_id) DO UPDATE SET
                    actual_1h_pct=excluded.actual_1h_pct,
                    actual_4h_pct=excluded.actual_4h_pct,
                    actual_24h_pct=excluded.actual_24h_pct,
                    hit_1h=excluded.hit_1h,
                    hit_4h=excluded.hit_4h,
                    hit_24h=excluded.hit_24h,
                    scored_at=excluded.scored_at,
                    expected_move_pct=excluded.expected_move_pct
                """,
                (
                    int(payload["prediction_id"]),
                    payload["symbol"],
                    payload["timeframe"],
                    payload["predicted_at"],
                    payload["direction"],
                    float(payload["confidence"]),
                    payload.get("expected_move_pct"),
                    float(payload["last_close"]),
                    payload.get("actual_1h_pct"),
                    payload.get("actual_4h_pct"),
                    payload.get("actual_24h_pct"),
                    payload.get("hit_1h"),
                    payload.get("hit_4h"),
                    payload.get("hit_24h"),
                    payload.get("scored_at", utc_now()),
                ),
            )

    def get_prediction_outcomes(self, limit: int = 500) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM prediction_outcomes
                ORDER BY predicted_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def upsert_experiment_result(self, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO experiment_results (run_id, experiment_id, decision, created_at, payload_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    decision=excluded.decision,
                    created_at=excluded.created_at,
                    payload_json=excluded.payload_json
                """,
                (
                    payload["run_id"],
                    payload["experiment"]["experiment_id"],
                    payload["decision"],
                    payload["created_at"],
                    json.dumps(payload),
                ),
            )

    def get_experiment_results(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT payload_json FROM experiment_results
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]
