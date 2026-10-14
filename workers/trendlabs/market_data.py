"""Market data collectors: Alpaca Basic when keyed, otherwise Yahoo Finance fallback."""

from __future__ import annotations

from typing import Literal

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

# Yahoo Finance hard caps by interval (as of 2024/2025 public API behavior).
YFINANCE_MAX_PERIOD = {
    "15m": "60d",
    "1h": "730d",
    "4h": "730d",  # resampled from 60m
    "1d": "max",
}


class MarketDataClient:
    def __init__(self) -> None:
        self.alpaca_ready = bool(ALPACA_API_KEY and ALPACA_API_SECRET)

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
    ) -> pd.DataFrame:
        if timeframe not in TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        if self.alpaca_ready and not full_history:
            try:
                return self._fetch_alpaca(symbol, timeframe, limit)
            except Exception:
                # Fall through to Yahoo so local/CI runs stay $0 and unblocked.
                pass
        return self._fetch_yfinance(
            symbol,
            timeframe,
            limit,
            full_history=full_history,
        )

    def _fetch_alpaca(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        tf = TIMEFRAMES[timeframe]["alpaca"]
        # For 4h we pull 1Hour and resample.
        request_tf = "1Hour" if timeframe == "4h" else tf
        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        }
        params = {
            "timeframe": request_tf,
            "limit": min(limit * (4 if timeframe == "4h" else 1), 10000),
            "adjustment": "split",
            "feed": ALPACA_FEED,
            "sort": "asc",
        }
        url = f"{ALPACA_DATA_URL.rstrip('/')}/v2/stocks/{symbol.upper()}/bars"
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        bars = resp.json().get("bars") or []
        if not bars:
            raise ValueError(f"No Alpaca bars for {symbol}")

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
        df = df.tail(limit).reset_index(drop=True)
        df["symbol"] = symbol.upper()
        df["timeframe"] = timeframe
        df["source"] = f"alpaca:{ALPACA_FEED}"
        df["adjusted"] = True
        return df

    def _fetch_yfinance(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        *,
        full_history: bool = False,
    ) -> pd.DataFrame:
        cfg = TIMEFRAMES[timeframe]
        interval = cfg["yf"]
        period = YFINANCE_MAX_PERIOD.get(timeframe, cfg["period"]) if full_history else cfg["period"]
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval, auto_adjust=True)
        if hist.empty and full_history and timeframe == "1d":
            # Explicit start for QQQ-era ETFs if period=max fails
            hist = ticker.history(start="1990-01-01", interval=interval, auto_adjust=True)
        if hist.empty:
            raise ValueError(f"No Yahoo Finance data for {symbol} {timeframe}")

        hist = hist.reset_index()
        # Column may be Datetime or Date
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
        ).dropna()
        df = df.sort_values("ts")
        if timeframe == "4h":
            df = self._resample_ohlcv(df, "4h")
        df = self._drop_duplicates(df)
        if not full_history:
            df = df.tail(limit)
        df = df.reset_index(drop=True)
        df["symbol"] = symbol.upper()
        df["timeframe"] = timeframe
        df["source"] = "yfinance"
        df["adjusted"] = True
        return df

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
        ).dropna()
        return out.reset_index()

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=["ts"], keep="last")
