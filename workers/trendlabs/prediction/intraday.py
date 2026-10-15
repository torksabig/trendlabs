"""PE intraday V1 — 15m / 1h / 4h forward-return baselines (paper research only).

Uses Yahoo Finance 5-minute bars (free / no paid secrets). Short history window
(~60 calendar days for 5m on Yahoo). Separate from the daily PE / volatility paths.

Targets (close-to-close log returns on 5m bars):
  R_15m = ln(close_{t+3}  / close_t)
  R_1h  = ln(close_{t+12} / close_t)
  R_4h  = ln(close_{t+48} / close_t)

Baselines (features at t use prices ≤ t only):
  persistence  — last realized return of the same horizon
  short_mom    — short lookback momentum (6 bars ≈ 30m)
  range_rv     — recent range / close as a signed magnitude via mom sign

Disclaimer: EXPLORATORY — SURVIVORSHIP BIAS NOT FULLY CONTROLLED
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from trendlabs.config import ROOT
from trendlabs.prediction import DISCLAIMER
from trendlabs.prediction.dataset import PE_DIR
from trendlabs.prediction.metrics_ic import daily_spearman_ic, summarize_ic
from trendlabs.prediction.walk_forward import generate_purged_folds, mask_dates, unique_sorted_dates

INTRADAY_DIR = PE_DIR / "intraday"
RESULTS_PATH = PE_DIR / "intraday_results.json"

# Small liquid set — not the full daily PE universe.
DEFAULT_SYMBOLS = ["QQQ", "SPY", "AAPL", "MSFT", "NVDA"]

# Prefer 5m: Yahoo 1m history is ~7 days (too short/noisy for WF); 5m ≈ 60d.
BAR_INTERVAL = "5m"
YFINANCE_PERIOD = "60d"
YFINANCE_INTERVAL = "5m"

# Bar counts on 5m: 15m→3, 1h→12, 4h→48
HORIZON_BARS: dict[str, int] = {"15m": 3, "1h": 12, "4h": 48}
SHORT_MOM_BARS = 6  # ≈ 30 minutes on 5m bars
RANGE_RV_WINDOW = 12  # ≈ 1 hour of range

PRICE_BASIS = "close"  # documented choice vs mid=(high+low)/2

SCHEMA_COLS = ["symbol", "ts", "open", "high", "low", "close", "volume"]


def forward_log_return(close: pd.Series, horizon_bars: int) -> pd.Series:
    """R_t = ln(close_{t+h} / close_t). Close-to-close; NaN near series end."""
    c = close.astype(float)
    return np.log(c.shift(-horizon_bars) / c)


def past_log_return(close: pd.Series, lookback_bars: int) -> pd.Series:
    """ln(close_t / close_{t-lookback}); uses prices ≤ t only."""
    c = close.astype(float)
    return np.log(c / c.shift(lookback_bars))


def range_rv_feature(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Mean (high-low)/close over the past `window` bars ending at t (inclusive)."""
    rng = (high.astype(float) - low.astype(float)) / close.astype(float).replace(0.0, np.nan)
    return rng.rolling(window=window, min_periods=window).mean()


def attach_intraday_targets_and_features(
    df: pd.DataFrame,
    *,
    horizons: dict[str, int] | None = None,
    short_mom_bars: int = SHORT_MOM_BARS,
    range_window: int = RANGE_RV_WINDOW,
) -> pd.DataFrame:
    """Per-symbol forward targets + leakage-safe baseline features."""
    horizons = horizons or HORIZON_BARS
    parts: list[pd.DataFrame] = []
    for _, g in df.groupby("symbol", sort=False):
        g = g.sort_values("ts").reset_index(drop=True).copy()
        close = g["close"]
        for name, h in horizons.items():
            g[f"R_{name}"] = forward_log_return(close, h)
            g[f"persistence_{name}"] = past_log_return(close, h)
        g["short_mom"] = past_log_return(close, short_mom_bars)
        g["range_rv"] = range_rv_feature(g["high"], g["low"], close, range_window)
        # Signed range: magnitude from recent range, direction from short momentum
        g["range_rv_signed"] = g["range_rv"] * np.sign(g["short_mom"].fillna(0.0))
        parts.append(g)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out


def fetch_yahoo_5m(
    symbol: str,
    *,
    period: str = YFINANCE_PERIOD,
    interval: str = YFINANCE_INTERVAL,
) -> pd.DataFrame:
    """Pull OHLCV from Yahoo; returns schema columns + source metadata."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval, auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No Yahoo Finance {interval} data for {symbol} period={period}")
    hist = hist.reset_index()
    ts_col = "Datetime" if "Datetime" in hist.columns else "Date"
    df = pd.DataFrame(
        {
            "symbol": symbol.upper(),
            "ts": pd.to_datetime(hist[ts_col], utc=True),
            "open": hist["Open"].astype(float),
            "high": hist["High"].astype(float),
            "low": hist["Low"].astype(float),
            "close": hist["Close"].astype(float),
            "volume": hist["Volume"].astype(float),
        }
    ).dropna(subset=["ts", "open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)
    df["source"] = "yfinance"
    df["interval"] = interval
    df["period_requested"] = period
    return df


def load_or_fetch_panel(
    symbols: list[str] | None = None,
    *,
    data_dir: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load cached parquet panel or fetch from Yahoo and write cache."""
    symbols = [s.upper() for s in (symbols or DEFAULT_SYMBOLS)]
    data_dir = Path(data_dir or INTRADAY_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    panel_path = data_dir / "panel_5m.parquet"
    meta_path = data_dir / "manifest.json"

    if panel_path.exists() and not refresh:
        panel = pd.read_parquet(panel_path)
        panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
        return panel

    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    for sym in symbols:
        try:
            frames.append(fetch_yahoo_5m(sym))
        except Exception as exc:  # noqa: BLE001 — collect per-symbol failures
            errors[sym] = str(exc)

    if not frames:
        raise RuntimeError(f"Failed to fetch any intraday bars: {errors}")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    # Persist schema subset + metadata columns used for auditing
    keep = SCHEMA_COLS + [c for c in ("source", "interval", "period_requested") if c in panel.columns]
    panel[keep].to_parquet(panel_path, index=False)
    # Also write a small CSV sample for human inspection (head)
    sample_path = data_dir / "sample_head.csv"
    panel[SCHEMA_COLS].head(50).to_csv(sample_path, index=False)

    meta = {
        "disclaimer": DISCLAIMER,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bar_interval": BAR_INTERVAL,
        "yfinance_period": YFINANCE_PERIOD,
        "yfinance_interval": YFINANCE_INTERVAL,
        "symbols_requested": symbols,
        "symbols_ok": sorted(panel["symbol"].unique().tolist()),
        "symbols_failed": errors,
        "n_rows": int(len(panel)),
        "ts_min": panel["ts"].min().isoformat() if len(panel) else None,
        "ts_max": panel["ts"].max().isoformat() if len(panel) else None,
        "schema": SCHEMA_COLS,
        "price_basis": PRICE_BASIS,
        "data_limits": {
            "yahoo_5m_typical_max": "~60 calendar days",
            "yahoo_1m_typical_max": "~7 calendar days (not used; too short for WF)",
            "survivorship": "Fixed mega-cap/ETF list; no delisting adjustment",
            "gaps": "Yahoo may omit some bars; extended-hours inclusion varies",
        },
        "panel_path": str(panel_path.relative_to(ROOT)) if panel_path.is_relative_to(ROOT) else str(panel_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return panel


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return None
    sx, sy = x[mask], y[mask]
    if np.unique(sx).size < 2 or np.unique(sy).size < 2:
        return None
    return float(pd.Series(sx).corr(pd.Series(sy), method="spearman"))


def evaluate_horizon_predictions(
    oos: pd.DataFrame,
    *,
    pred_col: str,
    target_col: str,
) -> dict[str, Any]:
    """Pooled Spearman IC, cross-sectional Rank IC, hit-rate, MAE."""
    sub = oos[[pred_col, target_col, "ts", "symbol"]].dropna()
    if sub.empty:
        return {
            "n": 0,
            "SpearmanIC": None,
            "RankIC": None,
            "RankICIR": None,
            "hit_rate": None,
            "MAE": None,
        }

    spearman = _spearman(sub[pred_col].to_numpy(), sub[target_col].to_numpy())
    pred = sub[pred_col].to_numpy(dtype=float)
    tgt = sub[target_col].to_numpy(dtype=float)
    # Hit-rate on sign; zeros in target count as miss unless pred also ~0
    signs_match = np.sign(pred) == np.sign(tgt)
    nonzero = tgt != 0
    hit = float(signs_match[nonzero].mean()) if nonzero.any() else None
    mae = float(np.mean(np.abs(pred - tgt)))

    rank_ic = daily_spearman_ic(sub, score_col=pred_col, target_col=target_col, min_names=2)
    rank_summary = summarize_ic(rank_ic)

    return {
        "n": int(len(sub)),
        "SpearmanIC": spearman,
        "RankIC": rank_summary.get("MeanIC"),
        "RankICIR": rank_summary.get("ICIR"),
        "RankIC_PositiveRate": rank_summary.get("PositiveICRate"),
        "RankIC_n_timestamps": rank_summary.get("n_days"),
        "hit_rate": hit,
        "MAE": mae,
    }


def run_intraday_walk_forward(
    panel: pd.DataFrame,
    *,
    horizons: dict[str, int] | None = None,
    min_train_bars: int = 1500,
    test_bars: int = 400,
    step_bars: int | None = None,
) -> dict[str, Any]:
    """Purged bar-level walk-forward; purge gap = horizon bars (label overlap)."""
    horizons = horizons or HORIZON_BARS
    work = attach_intraday_targets_and_features(panel, horizons=horizons)
    if work.empty:
        return {"disclaimer": DISCLAIMER, "error": "empty_panel", "horizons": {}}

    dates = unique_sorted_dates(work)
    results: dict[str, Any] = {}
    max_h = max(horizons.values())

    for h_name, h_bars in horizons.items():
        target_col = f"R_{h_name}"
        pers_col = f"persistence_{h_name}"
        need = [target_col, pers_col, "short_mom", "range_rv_signed"]
        sub = work.dropna(subset=need).copy()
        if sub.empty:
            results[h_name] = {"error": "no_valid_rows", "horizon_bars": h_bars}
            continue

        sub_dates = unique_sorted_dates(sub)
        # Purge overlapping label horizon so train labels do not use test-period prices
        purge = h_bars
        folds = generate_purged_folds(
            sub_dates,
            min_train_days=min_train_bars,
            test_days=test_bars,
            step_days=step_bars if step_bars is not None else test_bars,
            purge_gap=purge,
        )
        if not folds:
            results[h_name] = {
                "error": "no_folds",
                "horizon_bars": h_bars,
                "n_timestamps": len(sub_dates),
                "min_train_bars": min_train_bars,
                "test_bars": test_bars,
                "purge_gap_bars": purge,
            }
            continue

        oos_parts: list[pd.DataFrame] = []
        for fold in folds:
            test = mask_dates(sub, fold.test_dates)
            if test.empty:
                continue
            chunk = test[["symbol", "ts", target_col]].copy()
            chunk["fold_id"] = fold.fold_id
            chunk["pred_persistence"] = test[pers_col].astype(float)
            chunk["pred_short_mom"] = test["short_mom"].astype(float)
            chunk["pred_range_rv"] = test["range_rv_signed"].astype(float)
            oos_parts.append(chunk)

        if not oos_parts:
            results[h_name] = {"error": "empty_oos", "horizon_bars": h_bars}
            continue

        oos = pd.concat(oos_parts, ignore_index=True)
        models = {
            "persistence": evaluate_horizon_predictions(
                oos, pred_col="pred_persistence", target_col=target_col
            ),
            "short_mom": evaluate_horizon_predictions(
                oos, pred_col="pred_short_mom", target_col=target_col
            ),
            "range_rv_signed": evaluate_horizon_predictions(
                oos, pred_col="pred_range_rv", target_col=target_col
            ),
        }
        results[h_name] = {
            "horizon_bars": h_bars,
            "target": f"R = ln(close_{{t+{h_bars}}} / close_t)",
            "price_basis": PRICE_BASIS,
            "protocol": {
                "min_train_bars": min_train_bars,
                "test_bars": test_bars,
                "step_bars": step_bars if step_bars is not None else test_bars,
                "purge_gap_bars": purge,
                "n_folds": len(folds),
                "n_oos_rows": int(len(oos)),
                "chronological": True,
                "random_splits": False,
            },
            "models": models,
        }

    ts_min = work["ts"].min()
    ts_max = work["ts"].max()
    return {
        "disclaimer": DISCLAIMER,
        "experiment": "pe_intraday_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bar_interval": BAR_INTERVAL,
        "price_basis": PRICE_BASIS,
        "symbols": sorted(work["symbol"].unique().tolist()),
        "n_rows": int(len(work)),
        "n_timestamps": len(dates),
        "ts_min": ts_min.isoformat() if pd.notna(ts_min) else None,
        "ts_max": ts_max.isoformat() if pd.notna(ts_max) else None,
        "yfinance_period": YFINANCE_PERIOD,
        "horizons_bars": horizons,
        "max_purge_bars": max_h,
        "baselines_described": {
            "persistence": "hat = ln(close_t / close_{t-h}) — last realized same horizon",
            "short_mom": f"hat = ln(close_t / close_{{t-{SHORT_MOM_BARS}}}) — ~30m momentum",
            "range_rv_signed": (
                f"hat = mean((high-low)/close, {RANGE_RV_WINDOW} bars) * sign(short_mom)"
            ),
        },
        "data_limits": {
            "yahoo_5m": "~60 calendar days public history",
            "yahoo_1m_not_used": "~7 days; too short for purged WF at 4h",
            "survivorship": "Fixed liquid list; not universe-complete",
            "paper_only": True,
            "brokerage": False,
        },
        "horizons": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PE intraday V1: 15m/1h/4h baselines on Yahoo 5m bars (paper only)"
    )
    parser.add_argument("--data-dir", type=Path, default=INTRADAY_DIR)
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--refresh", action="store_true", help="Refetch Yahoo bars")
    parser.add_argument("--min-train-bars", type=int, default=1500)
    parser.add_argument("--test-bars", type=int, default=400)
    parser.add_argument("--step-bars", type=int, default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    try:
        panel = load_or_fetch_panel(args.symbols, data_dir=args.data_dir, refresh=args.refresh)
    except Exception as exc:  # noqa: BLE001
        err = {
            "disclaimer": DISCLAIMER,
            "error": f"fetch_failed: {exc}",
            "experiment": "pe_intraday_v1",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(err, indent=2), encoding="utf-8")
        print(json.dumps(err, indent=2))
        return 1

    result = run_intraday_walk_forward(
        panel,
        min_train_bars=args.min_train_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Compact stdout summary
    summary: dict[str, Any] = {
        "wrote": str(args.out),
        "disclaimer": DISCLAIMER,
        "bar_interval": BAR_INTERVAL,
        "symbols": result.get("symbols"),
        "ts_min": result.get("ts_min"),
        "ts_max": result.get("ts_max"),
        "n_rows": result.get("n_rows"),
        "horizons": {},
    }
    for h_name, block in (result.get("horizons") or {}).items():
        if "models" not in block:
            summary["horizons"][h_name] = {"error": block.get("error")}
            continue
        summary["horizons"][h_name] = {
            k: {
                "SpearmanIC": v.get("SpearmanIC"),
                "RankIC": v.get("RankIC"),
                "hit_rate": v.get("hit_rate"),
                "MAE": v.get("MAE"),
                "n": v.get("n"),
            }
            for k, v in block["models"].items()
        }
    print(json.dumps(summary, indent=2))
    ok = bool(result.get("horizons")) and all(
        "models" in (result["horizons"].get(h) or {}) for h in HORIZON_BARS
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
