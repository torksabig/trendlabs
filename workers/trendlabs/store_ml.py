"""ML persistence methods for Store (installed on import)."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from trendlabs.db import Store, utc_now


class _MLMethods:
    def upsert_market_features(
        self,
        df: pd.DataFrame,
        *,
        feature_version: str,
    ) -> int:
        """Persist feature rows. Replaces same (symbol, timeframe, ts, version)."""
        if df.empty:
            return 0
        now = utc_now()
        rows: list[tuple[Any, ...]] = []
        feature_cols = [
            c
            for c in df.columns
            if c not in {"symbol", "timeframe", "ts", "feature_version", "created_at"}
        ]
        for r in df.itertuples(index=False):
            payload = {c: float(getattr(r, c)) for c in feature_cols}
            rows.append(
                (
                    str(getattr(r, "symbol")).upper(),
                    str(getattr(r, "timeframe")),
                    pd.Timestamp(getattr(r, "ts")).isoformat(),
                    feature_version,
                    json.dumps(payload),
                    now,
                )
            )
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO market_features (
                    symbol, timeframe, ts, feature_version, features_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timeframe, ts, feature_version) DO UPDATE SET
                    features_json=excluded.features_json,
                    created_at=excluded.created_at
                """,
                rows,
            )
        return len(rows)

    def load_market_features(
        self,
        symbol: str,
        timeframe: str,
        *,
        feature_version: str,
    ) -> pd.DataFrame:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT ts, features_json, created_at
                FROM market_features
                WHERE symbol = ? AND timeframe = ? AND feature_version = ?
                ORDER BY ts ASC
                """,
                (symbol.upper(), timeframe, feature_version),
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        records = []
        for r in rows:
            item = json.loads(r["features_json"])
            item["ts"] = r["ts"]
            item["created_at"] = r["created_at"]
            item["symbol"] = symbol.upper()
            item["timeframe"] = timeframe
            item["feature_version"] = feature_version
            records.append(item)
        df = pd.DataFrame(records)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    def upsert_model_version(self, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO model_versions (
                    model_version, symbol, timeframe, feature_version, algorithm,
                    status, artifact_path, metrics_json, trained_at,
                    dataset_start, dataset_end, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_version) DO UPDATE SET
                    status=excluded.status,
                    artifact_path=excluded.artifact_path,
                    metrics_json=excluded.metrics_json,
                    trained_at=excluded.trained_at,
                    dataset_start=excluded.dataset_start,
                    dataset_end=excluded.dataset_end,
                    notes=excluded.notes
                """,
                (
                    payload["model_version"],
                    payload["symbol"].upper(),
                    payload["timeframe"],
                    payload["feature_version"],
                    payload.get("algorithm", "xgboost"),
                    payload.get("status", "candidate"),
                    payload["artifact_path"],
                    json.dumps(payload.get("metrics", {})),
                    payload["trained_at"],
                    payload.get("dataset_start"),
                    payload.get("dataset_end"),
                    payload.get("notes"),
                ),
            )

    def get_model_version(self, model_version: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE model_version = ?",
                (model_version,),
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["metrics"] = json.loads(item.pop("metrics_json") or "{}")
        return item

    def list_model_versions(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        if timeframe:
            clauses.append("timeframe = ?")
            params.append(timeframe)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM model_versions
                {where}
                ORDER BY trained_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["metrics"] = json.loads(item.pop("metrics_json") or "{}")
            out.append(item)
        return out

    def latest_model_version(
        self,
        symbol: str,
        timeframe: str,
        *,
        status: str = "candidate",
    ) -> dict[str, Any] | None:
        rows = self.list_model_versions(
            symbol=symbol, timeframe=timeframe, status=status, limit=1
        )
        return rows[0] if rows else None

    def insert_ml_prediction(self, payload: dict[str, Any]) -> str:
        import uuid
        pred_id = payload.get("id") or str(uuid.uuid4())
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ml_predictions (
                    id, symbol, timeframe, model_version, feature_version,
                    predicted_at, horizon_bars, reference_close, prob_up,
                    prob_not_up, features_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timeframe, model_version, predicted_at, horizon_bars)
                DO NOTHING
                """,
                (
                    pred_id,
                    payload["symbol"].upper(),
                    payload["timeframe"],
                    payload["model_version"],
                    payload["feature_version"],
                    payload["predicted_at"],
                    int(payload["horizon_bars"]),
                    float(payload["reference_close"]),
                    float(payload["prob_up"]),
                    float(payload["prob_not_up"]),
                    json.dumps(payload.get("features", {})),
                    payload.get("created_at", utc_now()),
                ),
            )
            row = conn.execute(
                """
                SELECT id FROM ml_predictions
                WHERE symbol = ? AND timeframe = ? AND model_version = ?
                  AND predicted_at = ? AND horizon_bars = ?
                """,
                (
                    payload["symbol"].upper(),
                    payload["timeframe"],
                    payload["model_version"],
                    payload["predicted_at"],
                    int(payload["horizon_bars"]),
                ),
            ).fetchone()
        return str(row["id"]) if row else pred_id

    def get_ml_predictions(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        limit: int = 500,
        unscored_only: bool = False,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if symbol:
            clauses.append("p.symbol = ?")
            params.append(symbol.upper())
        if timeframe:
            clauses.append("p.timeframe = ?")
            params.append(timeframe)
        join = ""
        if unscored_only:
            join = "LEFT JOIN ml_prediction_outcomes o ON o.prediction_id = p.id"
            clauses.append("o.prediction_id IS NULL")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT p.* FROM ml_predictions p
                {join}
                {where}
                ORDER BY p.predicted_at ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["features"] = json.loads(item.pop("features_json") or "{}")
            out.append(item)
        return out

    def insert_ml_prediction_outcome(self, payload: dict[str, Any]) -> None:
        """Persist outcome once. Existing outcomes are not overwritten (immutable)."""
        with self.connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM ml_prediction_outcomes WHERE prediction_id = ?",
                (payload["prediction_id"],),
            ).fetchone()
            if exists:
                return
            conn.execute(
                """
                INSERT INTO ml_prediction_outcomes (
                    prediction_id, actual_close, actual_ts, realized_return,
                    direction_correct, evaluated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["prediction_id"],
                    float(payload["actual_close"]),
                    payload["actual_ts"],
                    float(payload["realized_return"]),
                    1 if payload["direction_correct"] else 0,
                    payload.get("evaluated_at", utc_now()),
                ),
            )

    def get_ml_prediction_outcomes(self, limit: int = 500) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT o.*, p.symbol, p.timeframe, p.model_version, p.predicted_at,
                       p.horizon_bars, p.prob_up, p.reference_close
                FROM ml_prediction_outcomes o
                JOIN ml_predictions p ON p.id = o.prediction_id
                ORDER BY o.evaluated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def install_ml_store_methods() -> None:
    for name, value in _MLMethods.__dict__.items():
        if callable(value) and not name.startswith("_"):
            setattr(Store, name, value)


install_ml_store_methods()
