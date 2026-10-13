"""Market data collectors: Alpaca Basic when keyed, otherwise Yahoo Finance fallback.

Never invents missing OHLCV prices. Incomplete in-progress candles are dropped
unless explicitly requested. Retries with backoff on transient provider failures.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import pandas as pd
import requests
import yfinance as yf

from trendlabs.config import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_DATA_URL,
    ALPACA_FEED,
    LOOKBACK_BARS,
    TIMEFRAMES,
)

Source = Literal["alpaca", "yfinance"]

REQUIRED_OHLCV = ("open", "high", "low", "close", "volume")


@dataclass
class QualityReport:
    symbol: str
    timeframe: str
    source: str
    rows_raw: int = 0
    rows_kept: int = 0
    duplicates_removed: int = 0
    invalid_ohlc_dropped: int = 0
    incomplete_dropped: int = 0
    gap_count: int = 0
    first_ts: str | None = None
    last_ts: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "source": self.source,
            "rows_raw": self.rows_raw,
            "rows_kept": self.rows_kept,
            "duplicates_removed": self.duplicates_removed,
            "invalid_ohlc_dropped": self.invalid_ohlc_dropped,
            "incomplete_dropped": self.incomplete_dropped,
            "gap_count": self.gap_count,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "notes": list(self.notes),
        }


class MarketDataError(RuntimeError):
    """Raised when a provider cannot return usable market data."""


class MarketDataClient:
    def __init__(
        self,
        *,
        max_retries: int = 4,
        base_backoff_sec: float = 1.0,
        request_pause_sec: float = 0.35,
    ) -> None:
        self.alpaca_ready = bool(ALPACA_API_KEY and ALPACA_API_SECRET)
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec
        self.request_pause_sec = request_pause_sec
        self.last_quality: QualityReport | None = None

    @property
    def preferred_source(self) -> Source:
        return "alpaca" if self.alpaca_ready else "yfinance"

    def fetch(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = LOOKBACK_BARS,
        *,
        full_history: bool = False,
        include_incomplete: bool = False,
    ) -> pd.DataFrame:
        if timeframe not in TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        errors: list[str] = []
        if self.alpaca_ready:
            try:
                return self._fetch_with_retries(
                    "alpaca",
                    lambda: self._fetch_alpaca(
                        symbol,
                        timeframe,
                        limit,
                        full_history=full_history,
                        include_incomplete=include_incomplete,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"alpaca: {exc}")

        try:
            return self._fetch_with_retries(
                "yfinance",
                lambda: self._fetch_yfinance(
                    symbol,
                    timeframe,
                    limit,
                    full_history=full_history,
                    include_incomplete=include_incomplete,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"yfinance: {exc}")
            raise MarketDataError(
                f"No market data for {symbol} {timeframe}: {'; '.join(errors)}"
            ) from exc

    def _fetch_with_retries(self, label: str, fn: Any) -> pd.DataFrame:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if attempt > 0 or self.request_pause_sec:
                    time.sleep(self.request_pause_sec + random.uniform(0, 0.15))
                return fn()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.max_retries - 1:
                    break
                backoff = self.base_backoff_sec * (2**attempt) + random.uniform(0, 0.5)
                time.sleep(backoff)
        assert last_exc is not None
        raise MarketDataError(f"{label} failed after retries: {last_exc}") from last_exc

    def _fetch_alpaca(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        *,
        full_history: bool,
        include_incomplete: bool,
    ) -> pd.DataFrame:
        cfg = TIMEFRAMES[timeframe]
        tf = cfg["alpaca"]
        request_tf = "1Hour" if timeframe == "4h" else tf
        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        }

        # Paginate oldest→newest. For full history, omit limit and walk pages.
        page_limit = 10000
        if not full_history:
            page_limit = min(limit * (4 if timeframe == "4h" else 1), 10000)

        params: dict[str, Any] = {
            "timeframe": request_tf,
            "limit": page_limit,
            "adjustment": "split",
            "feed": ALPACA_FEED,
            "sort": "asc",
        }
        if full_history:
            start = cfg.get("history_start")
            if start:
                params["start"] = start

        url = f"{ALPACA_DATA_URL.rstrip('/')}/v2/stocks/{symbol.upper()}/bars"
        bars: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            page_params = dict(params)
            if page_token:
                page_params["page_token"] = page_token
            resp = requests.get(url, headers=headers, params=page_params, timeout=45)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise MarketDataError(f"Alpaca HTTP {resp.status_code}")
            resp.raise_for_status()
            payload = resp.json()
            chunk = payload.get("bars") or []
            bars.extend(chunk)
            page_token = payload.get("next_page_token")
            if not page_token:
                break
            if not full_history and len(bars) >= page_limit:
                break
            time.sleep(self.request_pause_sec)

        if not bars:
            raise MarketDataError(f"No Alpaca bars for {symbol}")

        df = pd.DataFrame(bars)
        df = df.rename(
            columns={
                "t": "ts",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
        )
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df[["ts", "open", "high", "low", "close", "volume"]].sort_values("ts")
        if timeframe == "4h":
            df = self._resample_ohlcv(df, "4h")
        if not full_history:
            df = df.tail(limit)
        df = df.reset_index(drop=True)
        df["symbol"] = symbol.upper()
        df["timeframe"] = timeframe
        df["source"] = f"alpaca:{ALPACA_FEED}"
        df["adjusted"] = True
        df["provider_updated_at"] = datetime.now(timezone.utc).isoformat()
        return self._finalize(df, include_incomplete=include_incomplete)

    def _fetch_yfinance(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        *,
        full_history: bool,
        include_incomplete: bool,
    ) -> pd.DataFrame:
        cfg = TIMEFRAMES[timeframe]
        interval = cfg["yf"]
        period = cfg.get("max_period", cfg["period"]) if full_history else cfg["period"]
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval, auto_adjust=True)
        if hist.empty:
            raise MarketDataError(f"No Yahoo Finance data for {symbol} {timeframe}")

        hist = hist.reset_index()
        ts_col = "Datetime" if "Datetime" in hist.columns else "Date"
        df = pd.DataFrame(
            {
                "ts": pd.to_datetime(hist[ts_col], utc=True),
                "open": hist["Open"].astype(float),
                "high": hist["High"].astype(float),
                "low": hist["Low"].astype(float),
                "close": hist["Close"].astype(float),
                "volume": hist["Volume"].astype(float),
            }
        )
        df = df.sort_values("ts")
        if timeframe == "4h":
            df = self._resample_ohlcv(df, "4h")
        if not full_history:
            df = df.tail(limit)
        df = df.reset_index(drop=True)
        df["symbol"] = symbol.upper()
        df["timeframe"] = timeframe
        df["source"] = "yfinance"
        df["adjusted"] = True
        df["provider_updated_at"] = datetime.now(timezone.utc).isoformat()
        return self._finalize(df, include_incomplete=include_incomplete)

    def _finalize(self, df: pd.DataFrame, *, include_incomplete: bool) -> pd.DataFrame:
        symbol = str(df["symbol"].iloc[0]) if not df.empty else ""
        timeframe = str(df["timeframe"].iloc[0]) if not df.empty else ""
        source = str(df["source"].iloc[0]) if not df.empty else ""
        report = QualityReport(
            symbol=symbol,
            timeframe=timeframe,
            source=source,
            rows_raw=len(df),
        )

        cleaned = self._drop_invalid_ohlc(df, report)
        before_dupes = len(cleaned)
        cleaned = self._drop_duplicates(cleaned)
        report.duplicates_removed = before_dupes - len(cleaned)

        if not include_incomplete:
            before_inc = len(cleaned)
            cleaned = self._drop_incomplete_candle(cleaned, timeframe)
            report.incomplete_dropped = before_inc - len(cleaned)

        report.gap_count = self._count_gaps(cleaned, timeframe)
        report.rows_kept = len(cleaned)
        if not cleaned.empty:
            report.first_ts = pd.Timestamp(cleaned["ts"].iloc[0]).isoformat()
            report.last_ts = pd.Timestamp(cleaned["ts"].iloc[-1]).isoformat()
        else:
            report.notes.append("no rows after quality filters")

        self.last_quality = report
        if cleaned.empty:
            raise MarketDataError(
                f"All candles rejected for {symbol} {timeframe} ({source})"
            )
        return cleaned.reset_index(drop=True)

    @staticmethod
    def _drop_invalid_ohlc(df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        numeric_ok = out[list(REQUIRED_OHLCV)].apply(pd.to_numeric, errors="coerce")
        for col in REQUIRED_OHLCV:
            out[col] = numeric_ok[col]
        mask = (
            out["open"].notna()
            & out["high"].notna()
            & out["low"].notna()
            & out["close"].notna()
            & out["volume"].notna()
            & (out["high"] >= out["low"])
            & (out["high"] >= out["open"])
            & (out["high"] >= out["close"])
            & (out["low"] <= out["open"])
            & (out["low"] <= out["close"])
            & (out["open"] > 0)
            & (out["high"] > 0)
            & (out["low"] > 0)
            & (out["close"] > 0)
            & (out["volume"] >= 0)
        )
        dropped = int((~mask).sum())
        report.invalid_ohlc_dropped = dropped
        if dropped:
            report.notes.append(f"dropped {dropped} invalid OHLCV rows (not filled)")
        return out.loc[mask].copy()

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=["ts"], keep="last")

    @staticmethod
    def _drop_incomplete_candle(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Drop the latest bar if its period has not closed yet (UTC)."""
        if df.empty or timeframe not in TIMEFRAMES:
            return df
        seconds = int(TIMEFRAMES[timeframe].get("seconds", 0))
        if seconds <= 0:
            return df
        now = pd.Timestamp.now(tz="UTC")
        last_ts = pd.Timestamp(df["ts"].iloc[-1])
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        # A candle starting at last_ts is complete only after last_ts + duration.
        if last_ts + pd.Timedelta(seconds=seconds) > now:
            return df.iloc[:-1].copy()
        return df

    @staticmethod
    def _count_gaps(df: pd.DataFrame, timeframe: str) -> int:
        if len(df) < 2 or timeframe not in TIMEFRAMES:
            return 0
        seconds = int(TIMEFRAMES[timeframe].get("seconds", 0))
        if seconds <= 0:
            return 0
        # Only count gaps for equities sessions roughly: > 1.5x expected spacing
        # for intraday; for daily allow weekends (gap > 3 days).
        deltas = df["ts"].sort_values().diff().dt.total_seconds().dropna()
        if timeframe == "1d":
            return int((deltas > 3.5 * 86400).sum())
        return int((deltas > 1.5 * seconds).sum())

    @staticmethod
    def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        indexed = df.set_index("ts")
        out = indexed.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        ).dropna(subset=["open", "high", "low", "close"])
        return out.reset_index()
