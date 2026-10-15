"""Intraday bar collection and parquet cache.

Sources:
  - alpaca: Alpaca Market Data v2 bars (requires ALPACA_API_KEY/ALPACA_API_SECRET).
    Free Basic plan: historical SIP bars since 2016, 200 requests/min, 10k bars/page.
  - yahoo: yfinance (no key). Hard caps: 5m/15m ~60d, 1h ~730d, 1d max.

Cache layout: data/intraday/bars/{source}/{SYMBOL}_{tf}.parquet  (columns ts,o,h,l,c,v UTC)
              data/intraday/bars/{source}/{SYMBOL}_{tf}_meta.json
All bars are filtered to regular trading hours (9:30 <= bar_start < 16:00 ET).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from trendlabs.config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_DATA_URL
from trendlabs.intraday import BARS_DIR, DEFAULT_UNIVERSE

Source = str  # "alpaca" | "yahoo"

# Alpaca v2 timeframe codes for the base grids we cache.
ALPACA_TF = {"5m": "5Min", "15m": "15Min", "1h": "1Hour"}
YF_MAX_PERIOD = {"5m": "60d", "15m": "60d", "1h": "730d"}

_BARS_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]
_ET = "America/New_York"
# Free Basic plan allows 200 calls/min; pace at <=180 for headroom.
_ALPACA_MIN_INTERVAL_S = 60.0 / 180.0


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only bars whose start time is inside 09:30–16:00 ET."""
    if df.empty:
        return df
    et = df["ts"].dt.tz_convert(_ET)
    clock = et.dt.hour * 60 + et.dt.minute
    mask = (clock >= 9 * 60 + 30) & (clock < 16 * 60)
    return df.loc[mask].copy()


def _normalize_yahoo(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out = out.reset_index()
    ts_col = out.columns[0]  # yfinance index name varies by version (Date/Datetime)
    out = out.rename(
        columns={ts_col: "ts", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    cols = [c for c in _BARS_COLUMNS if c in out.columns]
    return out[cols]


def fetch_yahoo_bars(symbol: str, timeframe: str = "1h", period: str | None = None) -> pd.DataFrame:
    """Yahoo intraday bars, RTH-filtered. Period capped per interval."""
    import yfinance as yf

    period = period or YF_MAX_PERIOD.get(timeframe, "60d")
    raw = yf.Ticker(symbol).history(period=period, interval=timeframe, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame(columns=_BARS_COLUMNS)
    df = filter_rth(_normalize_yahoo(raw))
    return df.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)


def fetch_alpaca_bars(
    symbol: str,
    timeframe: str = "5m",
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    feed: str = "sip",
    max_requests: int = 10_000,
) -> pd.DataFrame:
    """Alpaca v2 bars with pagination and Basic-plan pacing. Raises on auth failure."""
    if not (ALPACA_API_KEY and ALPACA_API_SECRET):
        raise RuntimeError("Alpaca keys missing — set ALPACA_API_KEY / ALPACA_API_SECRET in .env")

    tf_code = ALPACA_TF[timeframe]
    url = f"{ALPACA_DATA_URL.rstrip('/')}/v2/stocks/{symbol}/bars"
    headers = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_API_SECRET}
    params: dict[str, Any] = {"timeframe": tf_code, "limit": 10000, "feed": feed, "sort": "asc"}
    if start:
        params["start"] = start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if end:
        params["end"] = end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows: list[dict[str, Any]] = []
    session = requests.Session()
    last_call = 0.0
    for _ in range(max_requests):
        wait = _ALPACA_MIN_INTERVAL_S - (time.monotonic() - last_call)
        if wait > 0:
            time.sleep(wait)
        last_call = time.monotonic()
        resp = session.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(float(resp.headers.get("Retry-After", "30") or 30))
            continue
        if resp.status_code == 401:
            raise RuntimeError("Alpaca auth failed (401) — check API keys")
        resp.raise_for_status()
        payload = resp.json()
        bars = payload.get("bars") or []
        rows.extend(bars)
        token = payload.get("next_page_token")
        if token:
            params["page_token"] = token
        else:
            break

    if not rows:
        return pd.DataFrame(columns=_BARS_COLUMNS)
    df = pd.DataFrame(rows).rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = filter_rth(df[_BARS_COLUMNS])
    return df.sort_values("ts").reset_index(drop=True)


def cache_path(source: Source, symbol: str, timeframe: str) -> Path:
    return BARS_DIR / source / f"{symbol}_{timeframe}.parquet"


def merge_cache(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, bool]:
    """Merge new bars into the cache file; returns (merged, is_new_file)."""
    if path.exists():
        old = pd.read_parquet(path)
        merged = (
            pd.concat([old, df], ignore_index=True)
            .drop_duplicates(subset=["ts"], keep="last")
            .sort_values("ts")
            .reset_index(drop=True)
        )
        is_new = False
    else:
        merged = df.sort_values("ts").reset_index(drop=True)
        is_new = True
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    return merged, is_new


def collect_symbol(source: Source, symbol: str, timeframe: str, *, years: float, feed: str = "sip") -> dict[str, Any]:
    """Fetch and cache bars for one symbol; returns a lineage manifest."""
    path = cache_path(source, symbol, timeframe)
    if source == "alpaca":
        start = datetime.now(timezone.utc) - timedelta(days=int(years * 365.25))
        # Free Basic plan cannot query the most recent 15 minutes of SIP data
        # (docs: historical `end` must be >= 15 min old without a subscription).
        end = datetime.now(timezone.utc) - timedelta(minutes=16)
        df = fetch_alpaca_bars(symbol, timeframe, start=start, end=end, feed=feed)
    elif source == "yahoo":
        df = fetch_yahoo_bars(symbol, timeframe)
    else:
        raise ValueError(f"unknown source {source!r}")

    merged, is_new = merge_cache(df, path)
    meta = {
        "source": source,
        "symbol": symbol,
        "timeframe": timeframe,
        "feed": feed if source == "alpaca" else "yahoo",
        "rows": int(len(merged)),
        "ts_min": str(merged["ts"].min()) if not merged.empty else None,
        "ts_max": str(merged["ts"].max()) if not merged.empty else None,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "new_cache": is_new,
        "path": str(path),
    }
    meta_path = path.parent / (path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect intraday bars into the parquet cache")
    parser.add_argument("--source", choices=["alpaca", "yahoo"], default="yahoo")
    parser.add_argument("--timeframe", choices=list(ALPACA_TF), default="1h", help="base bar grid to cache")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--years", type=float, default=10.0, help="lookback for alpaca backfill")
    parser.add_argument("--feed", default="sip", help="alpaca feed: sip (Basic historical) or iex")
    args = parser.parse_args()

    symbols = args.symbols or DEFAULT_UNIVERSE
    metas = []
    for sym in symbols:
        meta = collect_symbol(args.source, sym, args.timeframe, years=args.years, feed=args.feed)
        metas.append(meta)
        print(f"{sym}: {meta['rows']} rows [{meta['ts_min']} → {meta['ts_max']}]")
    ok = any(m["rows"] > 0 for m in metas)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
