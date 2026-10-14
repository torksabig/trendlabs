"""Find historically repeating price-trend shapes in TrendLab candle data.

Algorithm (no look-ahead in match formation):
1. Build completed close-path windows of fixed length.
2. Z-score normalize each window (shape only, not absolute price).
3. Match windows by Pearson correlation above a threshold.
4. For each match set, measure the *subsequent* forward return after the
   historical window ended (never using future data inside the shape).
5. Rank patterns by occurrence count × continuation consistency.

This extends TrendLab's regime labels with shape-level motif discovery.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trendlabs.config import DATA_DIR, DEFAULT_TIMEFRAME, SYMBOLS
from trendlabs.db import Store
from trendlabs.indicators import enrich

OUT_DIR = Path(DATA_DIR) / "repeating_trends"


@dataclass
class RepeatingTrend:
    symbol: str
    timeframe: str
    pattern_id: str
    window_bars: int
    horizon_bars: int
    occurrences: int
    mean_shape_corr: float
    direction: str  # up / down / mixed (based on window net return)
    avg_window_return_pct: float
    avg_forward_return_pct: float
    forward_hit_rate: float
    continuation_rate: float  # fraction where forward return shares window sign
    latest_match_ts: str
    latest_window_return_pct: float
    sample_starts: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _zscore_path(closes: np.ndarray) -> np.ndarray | None:
    if len(closes) < 3:
        return None
    mu = float(np.mean(closes))
    sigma = float(np.std(closes))
    if sigma < 1e-12:
        return None
    return (closes - mu) / sigma


def _window_matrix(closes: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (paths[n, window], end_indices[n]) for completed windows only."""
    n = len(closes) - window + 1
    if n <= 0:
        return np.empty((0, window)), np.empty(0, dtype=int)
    idx = np.arange(window)[None, :] + np.arange(n)[:, None]
    paths = closes[idx]
    ends = np.arange(n) + (window - 1)
    return paths, ends


def find_repeating_trends(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    window: int = 20,
    horizon: int = 10,
    min_corr: float = 0.90,
    min_occurrences: int = 5,
    max_patterns: int = 8,
    recent_anchors: int = 40,
) -> list[RepeatingTrend]:
    """Discover repeating close-path motifs for one symbol/timeframe."""
    if df.empty or len(df) < window + horizon + 30:
        return []

    work = df.sort_values("ts").reset_index(drop=True)
    if "ema_20" not in work.columns:
        work = enrich(work)

    closes = work["close"].astype(float).to_numpy()
    ts = pd.to_datetime(work["ts"], utc=True)
    paths, ends = _window_matrix(closes, window)
    if len(ends) < min_occurrences + recent_anchors:
        return []

    # Only use anchors from the last `recent_anchors` completed windows so we
    # answer: "does the recent trend shape repeat in history?"
    anchor_start = max(0, len(ends) - recent_anchors)
    patterns: list[RepeatingTrend] = []
    used_anchor_ends: set[int] = set()

    for a in range(anchor_start, len(ends)):
        end_a = int(ends[a])
        if end_a in used_anchor_ends:
            continue
        # Need room after the window for forward evaluation on the anchor itself
        # only when scoring historical peers (anchor forward optional).
        shape_a = _zscore_path(paths[a])
        if shape_a is None:
            continue

        # Compare only to windows that fully precede this anchor (no overlap leakage).
        max_hist_end = end_a - window
        if max_hist_end < window:
            continue
        hist_mask = ends <= max_hist_end
        hist_idx = np.where(hist_mask)[0]
        if len(hist_idx) < min_occurrences - 1:
            continue

        hist_paths = paths[hist_idx]
        # Vectorized Pearson correlation of anchor shape vs all prior windows.
        hist_mu = hist_paths.mean(axis=1, keepdims=True)
        hist_sd = hist_paths.std(axis=1, keepdims=True)
        valid = hist_sd[:, 0] > 1e-12
        if valid.sum() < min_occurrences - 1:
            continue
        hist_z = (hist_paths[valid] - hist_mu[valid]) / hist_sd[valid]
        valid_idx = hist_idx[valid]
        sims = (hist_z * shape_a).mean(axis=1)  # corr vs z-scored anchor
        match_pos = np.where(sims >= min_corr)[0]
        corrs = [(int(valid_idx[k]), float(sims[k])) for k in match_pos]

        if len(corrs) + 1 < min_occurrences:
            continue

        # Forward returns after each historical match window (completed only).
        fwd_returns: list[float] = []
        win_returns: list[float] = []
        sample_starts: list[str] = []
        for j, _corr in corrs:
            end_j = int(ends[j])
            start_j = end_j - window + 1
            if end_j + horizon >= len(closes):
                continue
            w_ret = closes[end_j] / closes[start_j] - 1.0
            f_ret = closes[end_j + horizon] / closes[end_j] - 1.0
            win_returns.append(w_ret)
            fwd_returns.append(f_ret)
            if len(sample_starts) < 8:
                sample_starts.append(ts.iloc[start_j].isoformat())

        if len(fwd_returns) < min_occurrences - 1:
            continue

        win_arr = np.asarray(win_returns, dtype=float)
        fwd_arr = np.asarray(fwd_returns, dtype=float)
        mean_win = float(win_arr.mean())
        mean_fwd = float(fwd_arr.mean())
        if mean_win > 0.005:
            direction = "up"
        elif mean_win < -0.005:
            direction = "down"
        else:
            direction = "mixed"

        sign_win = np.sign(win_arr)
        continuation = float(np.mean((fwd_arr * sign_win) > 0)) if len(fwd_arr) else 0.0
        hit = float(np.mean(fwd_arr > 0))

        anchor_start_i = end_a - window + 1
        anchor_win_ret = float(closes[end_a] / closes[anchor_start_i] - 1.0)
        pattern_id = (
            f"{symbol}_{timeframe}_w{window}_h{horizon}_"
            f"{ts.iloc[anchor_start_i].strftime('%Y%m%d')}"
        )

        notes: list[str] = []
        if continuation >= 0.55 and abs(mean_fwd) >= 0.005:
            notes.append("historical matches tended to continue in the same direction")
        elif continuation <= 0.45:
            notes.append("historical matches often mean-reverted after the pattern")
        else:
            notes.append("continuation after matches is near coin-flip historically")

        patterns.append(
            RepeatingTrend(
                symbol=symbol,
                timeframe=timeframe,
                pattern_id=pattern_id,
                window_bars=window,
                horizon_bars=horizon,
                occurrences=len(fwd_returns) + 1,  # include current anchor shape
                mean_shape_corr=float(np.mean([c for _, c in corrs])),
                direction=direction,
                avg_window_return_pct=mean_win * 100.0,
                avg_forward_return_pct=mean_fwd * 100.0,
                forward_hit_rate=hit,
                continuation_rate=continuation,
                latest_match_ts=ts.iloc[anchor_start_i].isoformat(),
                latest_window_return_pct=anchor_win_ret * 100.0,
                sample_starts=sample_starts,
                notes=notes,
            )
        )
        used_anchor_ends.add(end_a)

    # Rank: more occurrences, stronger continuation edge, higher shape agreement.
    def score(p: RepeatingTrend) -> float:
        edge = abs(p.continuation_rate - 0.5)
        return p.occurrences * (0.5 + edge) * p.mean_shape_corr

    patterns.sort(key=score, reverse=True)

    # Deduplicate highly overlapping anchors (same direction + similar start week).
    deduped: list[RepeatingTrend] = []
    seen_weeks: set[str] = set()
    for p in patterns:
        week = p.latest_match_ts[:10]
        key = f"{p.direction}:{week}"
        if key in seen_weeks:
            continue
        seen_weeks.add(key)
        deduped.append(p)
        if len(deduped) >= max_patterns:
            break
    return deduped


def scan_universe(
    *,
    symbols: list[str] | None = None,
    timeframe: str = "1d",
    window: int = 20,
    horizon: int = 10,
    min_corr: float = 0.90,
    min_occurrences: int = 5,
    store: Store | None = None,
) -> dict[str, Any]:
    symbols = [s.upper() for s in (symbols or SYMBOLS)]
    store = store or Store()
    per_symbol: dict[str, list[dict[str, Any]]] = {}
    all_patterns: list[dict[str, Any]] = []

    for symbol in symbols:
        df = store.load_candles(symbol, timeframe)
        if df.empty:
            per_symbol[symbol] = []
            continue
        found = find_repeating_trends(
            df,
            symbol=symbol,
            timeframe=timeframe,
            window=window,
            horizon=horizon,
            min_corr=min_corr,
            min_occurrences=min_occurrences,
        )
        rows = [p.to_dict() for p in found]
        per_symbol[symbol] = rows
        all_patterns.extend(rows)
        top = rows[0] if rows else None
        if top:
            print(
                f"{symbol}: {len(rows)} repeating pattern(s); "
                f"top={top['direction']} occ={top['occurrences']} "
                f"cont={top['continuation_rate']:.0%} "
                f"fwd={top['avg_forward_return_pct']:.2f}% "
                f"corr={top['mean_shape_corr']:.3f}"
            )
        else:
            print(f"{symbol}: no repeating trends above thresholds")

    all_patterns.sort(
        key=lambda p: p["occurrences"] * (0.5 + abs(p["continuation_rate"] - 0.5)) * p["mean_shape_corr"],
        reverse=True,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timeframe": timeframe,
        "window_bars": window,
        "horizon_bars": horizon,
        "min_corr": min_corr,
        "min_occurrences": min_occurrences,
        "symbols": symbols,
        "pattern_count": len(all_patterns),
        "top_patterns": all_patterns[:20],
        "by_symbol": per_symbol,
        "method": {
            "name": "normalized_close_path_motif",
            "description": (
                "Z-scored close windows matched by Pearson correlation; "
                "forward returns evaluated only after each historical window."
            ),
        },
    }


def write_report(payload: dict[str, Any], directory: Path = OUT_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / "latest.json"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = directory / f"repeating_trends_{stamp}.json"
    text = json.dumps(payload, indent=2)
    latest.write_text(text)
    archived.write_text(text)
    # Mirror for dashboard-style public data if present.
    public = Path(__file__).resolve().parents[2] / "public" / "data" / "repeating_trends"
    public.mkdir(parents=True, exist_ok=True)
    (public / "latest.json").write_text(text)
    return latest


def main() -> int:
    parser = argparse.ArgumentParser(description="Find repeating trends in extracted candles")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--window", type=int, default=20, help="Shape window in bars")
    parser.add_argument("--horizon", type=int, default=10, help="Forward evaluation bars")
    parser.add_argument("--min-corr", type=float, default=0.90)
    parser.add_argument("--min-occurrences", type=int, default=5)
    args = parser.parse_args()

    if args.symbols:
        unknown = sorted(set(s.upper() for s in args.symbols) - set(SYMBOLS))
        if unknown:
            raise SystemExit(f"Unknown symbols: {', '.join(unknown)}")

    payload = scan_universe(
        symbols=args.symbols,
        timeframe=args.timeframe or DEFAULT_TIMEFRAME,
        window=args.window,
        horizon=args.horizon,
        min_corr=args.min_corr,
        min_occurrences=args.min_occurrences,
    )
    path = write_report(payload)
    print(
        json.dumps(
            {
                "patterns": payload["pattern_count"],
                "report": str(path),
                "top": [
                    {
                        "symbol": p["symbol"],
                        "direction": p["direction"],
                        "occurrences": p["occurrences"],
                        "continuation_rate": p["continuation_rate"],
                        "avg_forward_return_pct": p["avg_forward_return_pct"],
                        "latest_match_ts": p["latest_match_ts"],
                    }
                    for p in payload["top_patterns"][:10]
                ],
            },
            indent=2,
        )
    )
    return 0 if payload["pattern_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
