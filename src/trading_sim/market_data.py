"""Binance public market-data client (no API key required)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from trading_sim.config import (
    BINANCE_REST,
    BINANCE_REST_FALLBACKS,
    DEFAULT_INTERVAL,
    DEFAULT_LOOKBACK,
    DEFAULT_SYMBOL,
)


class BinancePublicClient:
    """Fetch klines from Binance public REST endpoints with endpoint failover."""

    def __init__(
        self,
        base_url: str = BINANCE_REST,
        fallbacks: tuple[str, ...] = BINANCE_REST_FALLBACKS,
        timeout: float = 20.0,
    ) -> None:
        self.base_urls = []
        for url in (base_url, *fallbacks):
            cleaned = url.rstrip("/")
            if cleaned and cleaned not in self.base_urls:
                self.base_urls.append(cleaned)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "trading-sim/0.1"})
        self.active_base = self.base_urls[0]

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        errors: list[str] = []
        for base in self.base_urls:
            try:
                resp = self.session.get(
                    f"{base}{path}",
                    params=params,
                    timeout=self.timeout,
                )
                if resp.status_code == 451:
                    errors.append(f"{base}: 451 restricted")
                    continue
                resp.raise_for_status()
                self.active_base = base
                return resp.json()
            except requests.RequestException as exc:
                errors.append(f"{base}: {exc}")
        raise RuntimeError("All Binance public endpoints failed: " + "; ".join(errors))

    def ping(self) -> bool:
        self._get("/api/v3/ping")
        return True

    def fetch_klines(
        self,
        symbol: str = DEFAULT_SYMBOL,
        interval: str = DEFAULT_INTERVAL,
        limit: int = DEFAULT_LOOKBACK,
    ) -> pd.DataFrame:
        params = {"symbol": symbol.upper(), "interval": interval, "limit": min(limit, 1000)}
        rows: list[list[Any]] = self._get("/api/v3/klines", params=params)
        if not rows:
            raise ValueError(f"No klines returned for {symbol} {interval}")

        df = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df["symbol"] = symbol.upper()
        df["interval"] = interval
        return df[
            [
                "symbol",
                "interval",
                "open_time",
                "close_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ].reset_index(drop=True)
