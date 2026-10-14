"""Phase A — build and export versioned Prediction Engine research dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trendlabs.config import ROOT, SYMBOLS
from trendlabs.db import Store
from trendlabs.prediction import DATASET_VERSION, DISCLAIMER, HORIZON_DAYS, PURGE_GAP_DAYS, VOL_WINDOW
from trendlabs.prediction.features import FEATURE_COLUMNS, attach_market_context, symbol_features
from trendlabs.prediction.targets import attach_targets

PE_DIR = Path(ROOT) / "data" / "prediction_engine"
TIMEFRAME = "1d"

# Fixed liquid universe for V1 — subset of TrendLab symbols with daily coverage preferred.
# Ranked names exclude pure benchmark when computing stock-only panels (SPY kept for context).
DEFAULT_UNIVERSE = list(SYMBOLS)

EXECUTION_RULE = (
    "Signal at close of day t; enter at open of t+1; exit at close of t+5. "
    "Target R = ln(close_{t+5}/open_{t+1}); z = R/(sigma_20*sqrt(5)) with sigma from "
    "past 20 daily log returns at t only. Purge gap ≥ 5 trading days in walk-forward."
)


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:  # noqa: BLE001
        return None


def load_daily_panel(
    store: Store,
    symbols: list[str] | None = None,
    *,
    timeframe: str = TIMEFRAME,
) -> pd.DataFrame:
    """Load daily OHLCV for universe; skip symbols with insufficient bars."""
    symbols = symbols or DEFAULT_UNIVERSE
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        df = store.load_candles(symbol, timeframe)
        if df is None or df.empty or len(df) < 300:
            continue
        df = df.sort_values("ts").reset_index(drop=True)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_research_dataset(
    store: Store | None = None,
    *,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Construct PIT feature + target panel."""
    store = store or Store()
    raw = load_daily_panel(store, symbols)
    if raw.empty:
        return raw

    parts: list[pd.DataFrame] = []
    for symbol, g in raw.groupby("symbol", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        g = attach_targets(g, horizon=HORIZON_DAYS, vol_window=VOL_WINDOW)
        g = symbol_features(g)
        parts.append(g)

    panel = pd.concat(parts, ignore_index=True)
    panel = attach_market_context(panel, spy_symbol="SPY")

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
        "sigma_20",
        *FEATURE_COLUMNS,
    ]
    # Deduplicate if any column appears twice
    keep = list(dict.fromkeys(keep))
    panel = panel[keep].copy()

    # Drop rows missing target or core features needed for training
    feature_ok = panel[FEATURE_COLUMNS].notna().all(axis=1)
    target_ok = panel["z"].notna() & panel["R"].notna()
    panel = panel.loc[feature_ok & target_ok].reset_index(drop=True)
    panel["disclaimer"] = DISCLAIMER
    panel["dataset_version"] = DATASET_VERSION
    return panel


def write_dataset(
    panel: pd.DataFrame,
    *,
    out_dir: Path | None = None,
    symbols_requested: list[str] | None = None,
) -> dict[str, Any]:
    """Export parquet (+ csv fallback companion) and manifest JSON."""
    out_dir = Path(out_dir or PE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / f"{DATASET_VERSION}.parquet"
    csv_path = out_dir / f"{DATASET_VERSION}.csv"
    manifest_path = out_dir / f"{DATASET_VERSION}_manifest.json"

    export_cols = [c for c in panel.columns if c != "disclaimer"]
    table = panel[export_cols].copy() if export_cols else panel.copy()

    wrote_parquet = False
    try:
        table.to_parquet(parquet_path, index=False)
        wrote_parquet = True
    except Exception:  # noqa: BLE001
        # Always write CSV if parquet engine missing
        pass
    table.to_csv(csv_path, index=False)

    symbols_present = sorted(panel["symbol"].unique().tolist()) if not panel.empty else []
    date_min = str(panel["ts"].min()) if not panel.empty else None
    date_max = str(panel["ts"].max()) if not panel.empty else None

    def _rel(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(Path(ROOT).resolve()))
        except ValueError:
            return str(path)

    manifest: dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "disclaimer": DISCLAIMER,
        "execution_rule": EXECUTION_RULE,
        "horizon_days": HORIZON_DAYS,
        "purge_gap_days": PURGE_GAP_DAYS,
        "vol_window": VOL_WINDOW,
        "timeframe": TIMEFRAME,
        "universe_requested": symbols_requested or DEFAULT_UNIVERSE,
        "universe_present": symbols_present,
        "n_rows": int(len(panel)),
        "n_symbols": len(symbols_present),
        "date_min": date_min,
        "date_max": date_max,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": ["R", "z", "sigma_20"],
        "files": {
            "parquet": _rel(parquet_path) if wrote_parquet else None,
            "csv": _rel(csv_path),
        },
        "survivorship_note": (
            "Fixed liquid universe from available TrendLab symbols; "
            "point-in-time index membership not reconstructed."
        ),
        "agents_note": (
            "TrendLab rule agents (ema_trend, breakout, momentum, supertrend) "
            "are rule-based benchmarks, not PE predictors."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Convenience pointer
    (out_dir / "latest_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_exported_dataset(out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = Path(out_dir or PE_DIR)
    parquet_path = out_dir / f"{DATASET_VERSION}.parquet"
    csv_path = out_dir / f"{DATASET_VERSION}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df
    raise FileNotFoundError(f"No dataset at {parquet_path} or {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Prediction Engine V1 research dataset")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=PE_DIR)
    args = parser.parse_args()

    symbols = args.symbols or DEFAULT_UNIVERSE
    panel = build_research_dataset(symbols=symbols)
    if panel.empty:
        print("ERROR: empty panel — collect daily OHLCV first (npm run collect -- --timeframe 1d --full-history)")
        return 1
    manifest = write_dataset(panel, out_dir=args.out_dir, symbols_requested=symbols)
    print(
        f"Wrote {manifest['n_rows']} rows / {manifest['n_symbols']} symbols "
        f"[{manifest['date_min']} → {manifest['date_max']}]"
    )
    print(f"disclaimer={DISCLAIMER}")
    print(json.dumps({k: manifest[k] for k in ('dataset_version', 'files', 'n_rows')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
