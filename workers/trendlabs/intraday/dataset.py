"""Build and export the intraday research panel for a horizon timeframe.

Source resolution per panel tf (deepest history first):
  15m: alpaca 5m → yahoo 5m → yahoo 15m
  1h : alpaca 5m → yahoo 1h
  4h : alpaca 5m → yahoo 1h

Panels are RTH-only and session-anchored at 9:30 ET; bar label = bar START time (UTC).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trendlabs.intraday import (
    BARS_DIR,
    BARS_PER_DAY,
    BENCHMARK_SYMBOL,
    DATASET_VERSION,
    DEFAULT_HORIZON_BARS,
    DEFAULT_UNIVERSE,
    DISCLAIMER,
    EXECUTION_RULE,
    INTRA_DIR,
    PURGE_NOTE,
    TF_MINUTES,
    VOL_WINDOW,
)
from trendlabs.intraday.collect import cache_path, filter_rth
from trendlabs.intraday.features import FEATURE_COLUMNS, attach_market_context, symbol_features
from trendlabs.prediction.dataset import _git_sha

_ET = "America/New_York"
_SESSION_OPEN_MIN = 9 * 60 + 30

_SOURCE_PREFERENCE = {
    "15m": [("alpaca", "5m"), ("yahoo", "5m"), ("yahoo", "15m")],
    "1h": [("alpaca", "5m"), ("yahoo", "1h")],
    "4h": [("alpaca", "5m"), ("yahoo", "1h")],
}


def resample_bars(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate RTH bars into session-anchored bars labeled at their start time."""
    if df.empty:
        return df
    tf_min = TF_MINUTES[tf]
    et = df["ts"].dt.tz_convert(_ET)
    minutes_of_day = et.dt.hour * 60 + et.dt.minute
    slot = np.floor((minutes_of_day - _SESSION_OPEN_MIN) / tf_min).astype("int64")
    day = et.dt.floor("D")
    bar_start_et = day + pd.to_timedelta(_SESSION_OPEN_MIN + slot * tf_min, unit="m")
    bar_start = bar_start_et.dt.tz_convert("UTC")

    work = pd.DataFrame(
        {
            "ts": bar_start,
            "slot": slot,
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "volume": df["volume"].astype(float),
        }
    )
    agg = work.groupby("ts", sort=True).agg(
        slot=("slot", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return agg.reset_index()


def attach_targets(
    panel: pd.DataFrame,
    *,
    horizon_bars: int = DEFAULT_HORIZON_BARS,
    vol_window: int = VOL_WINDOW,
) -> pd.DataFrame:
    """Intraday execution-rule targets, per symbol, matching the daily-track convention.

    R = ln(close_{t+h} / open_{t+1}); z = R / (sigma_vol_window * sqrt(h)); sigma from
    past vol_window bar log returns at t only. Trailing sigma uses bar returns across
    session boundaries (overnight contribution included) — documented simplification.
    """
    out = panel.sort_values(["symbol", "ts"]).copy()
    g = out.groupby("symbol", sort=False)
    close = out["close"].astype(float)
    open_ = out["open"].astype(float)
    log_ret = np.log(close / close.groupby(out["symbol"], sort=False).shift(1).replace(0.0, np.nan))

    # rolling sigma per symbol on the aligned log_ret series
    sigma = log_ret.groupby(out["symbol"], sort=False).transform(
        lambda s: s.rolling(vol_window, min_periods=vol_window).std()
    )
    future_close = g["close"].shift(-horizon_bars)
    next_open = g["open"].shift(-1)
    out["log_ret"] = log_ret
    out["sigma"] = sigma
    out["R"] = np.log(future_close.astype(float) / next_open.astype(float))
    out["z"] = out["R"] / (sigma * np.sqrt(float(horizon_bars))).replace(0.0, np.nan)
    return out.sort_values(["ts", "symbol"]).reset_index(drop=True)


def resolve_base(symbol: str, tf: str) -> tuple[str, str] | None:
    """Pick the deepest cached base grid for a panel tf."""
    for source, base_tf in _SOURCE_PREFERENCE[tf]:
        if cache_path(source, symbol, base_tf).exists():
            return source, base_tf
    return None


def load_base_bars(symbol: str, source: str, base_tf: str) -> pd.DataFrame:
    df = pd.read_parquet(cache_path(source, symbol, base_tf))
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return filter_rth(df)


def build_panel(
    tf: str,
    *,
    symbols: list[str] | None = None,
    horizon_bars: int = DEFAULT_HORIZON_BARS,
    vol_window: int = VOL_WINDOW,
) -> pd.DataFrame:
    symbols = list(symbols or DEFAULT_UNIVERSE)
    # SPY carries broadcast market-context features; without it every row is NaN-context
    # and silently filtered. Always include it (ranking stages exclude it themselves).
    if BENCHMARK_SYMBOL not in symbols:
        symbols.append(BENCHMARK_SYMBOL)
    bars_per_day = BARS_PER_DAY[tf]
    frames: list[pd.DataFrame] = []
    source_map: dict[str, str] = {}
    for sym in symbols:
        base = resolve_base(sym, tf)
        if base is None:
            print(f"skip {sym}: no cached bars (run collect first)")
            continue
        source, base_tf = base
        bars = load_base_bars(sym, source, base_tf)
        if len(bars) < 10 * bars_per_day:
            print(f"skip {sym}: only {len(bars)} base bars")
            continue
        panel_tf = resample_bars(bars, tf)
        panel_tf.insert(0, "symbol", sym)
        feats = symbol_features(panel_tf, tf=tf, bars_per_day=bars_per_day)
        frames.append(feats)
        source_map[sym] = f"{source}:{base_tf}->{tf}"

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    panel = attach_market_context(panel, spy_symbol=BENCHMARK_SYMBOL)
    panel = attach_targets(panel, horizon_bars=horizon_bars, vol_window=vol_window)

    keep = [
        "symbol",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "R",
        "z",
        "sigma",
        *FEATURE_COLUMNS,
    ]
    keep = list(dict.fromkeys(keep))
    panel = panel[keep].copy()
    feature_ok = panel[FEATURE_COLUMNS].notna().all(axis=1)
    target_ok = panel["z"].notna() & panel["R"].notna()
    panel = panel.loc[feature_ok & target_ok].sort_values(["ts", "symbol"]).reset_index(drop=True)
    panel["disclaimer"] = DISCLAIMER
    panel["dataset_version"] = f"{DATASET_VERSION}_{tf}"
    panel.attrs["source_map"] = source_map
    return panel


def write_dataset(
    panel: pd.DataFrame,
    tf: str,
    *,
    out_dir: Path | None = None,
    symbols_requested: list[str] | None = None,
    horizon_bars: int = DEFAULT_HORIZON_BARS,
    vol_window: int = VOL_WINDOW,
) -> dict[str, Any]:
    out_dir = Path(out_dir or INTRA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    version = f"{DATASET_VERSION}_{tf}"
    parquet_path = out_dir / f"{version}.parquet"
    manifest_path = out_dir / f"{version}_manifest.json"

    export_cols = [c for c in panel.columns if c != "disclaimer"]
    table = panel[export_cols].copy()
    table.to_parquet(parquet_path, index=False)

    symbols_present = sorted(panel["symbol"].unique().tolist()) if not panel.empty else []
    source_map = panel.attrs.get("source_map", {})

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(Path(INTRA_DIR).resolve().parent.parent))
        except ValueError:
            return str(p)

    manifest: dict[str, Any] = {
        "dataset_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "disclaimer": DISCLAIMER,
        "execution_rule": EXECUTION_RULE,
        "timeframe": tf,
        "horizon_bars": horizon_bars,
        "purge_note": PURGE_NOTE,
        "vol_window": vol_window,
        "universe_requested": symbols_requested or DEFAULT_UNIVERSE,
        "universe_present": symbols_present,
        "n_rows": int(len(panel)),
        "n_symbols": len(symbols_present),
        "ts_min": str(panel["ts"].min()) if not panel.empty else None,
        "ts_max": str(panel["ts"].max()) if not panel.empty else None,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": ["R", "z", "sigma"],
        "source_map": source_map,
        "files": {"parquet": _rel(parquet_path)},
        "survivorship_note": "Fixed liquid universe matching the daily PE track; PIT index membership not reconstructed.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "latest_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build intraday research panel")
    parser.add_argument("--tf", choices=list(BARS_PER_DAY), default="1h")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_BARS)
    args = parser.parse_args()

    symbols = args.symbols or DEFAULT_UNIVERSE
    panel = build_panel(args.tf, symbols=symbols, horizon_bars=args.horizon)
    if panel.empty:
        print("ERROR: empty panel — run collect first (python -m trendlabs.intraday.collect)")
        return 1
    manifest = write_dataset(panel, args.tf, symbols_requested=symbols, horizon_bars=args.horizon)
    print(
        f"Wrote {manifest['n_rows']} rows / {manifest['n_symbols']} symbols "
        f"[{manifest['ts_min']} → {manifest['ts_max']}] tf={args.tf}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
