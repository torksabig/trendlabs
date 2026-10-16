"""Feature matrix for TrendLab ML models (no look-ahead labels)."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from trendlabs.indicators import enrich


FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ema_spread_20_50",
    "ema_spread_50_200",
    "price_vs_ema20",
    "price_vs_ema50",
    "price_vs_ema200",
    "rsi_14",
    "rsi_norm",
    "adx_14",
    "atr_pct",
    "volatility_20",
    "volume_z",
    "volume_ratio",
    "hl_range_pct",
    "close_location",
    "st_direction",
    "momentum_10",
    "regime_bull",
    "regime_bear",
    "regime_side",
    "regime_hivol",
    "regime_riskoff",
    # Market context (broadcast per bar ts from the SPY daily panel)
    "SPY_RET1",
    "SPY_RET5",
    "SPY_RET20",
    "SPY_VOL20",
    "SPY_BREADTH20",
    "RS_SPY_5",
    "RS_SPY_20",
    "XS_DISP20",
]

# Broadcast/panel columns — allowed NaN per row (XGBoost missing routing) when no
# SPY panel is attached; required complete when context is attached.
_CONTEXT_COLUMNS = frozenset({
    "SPY_RET1", "SPY_RET5", "SPY_RET20", "SPY_VOL20", "SPY_BREADTH20",
    "RS_SPY_5", "RS_SPY_20", "XS_DISP20",
})

# Frozen feature-set identity: deterministic function of the column list, so ANY
# change to FEATURE_COLUMNS (add/remove/reorder) bumps the version and invalidates
# old snapshots/models rather than silently mixing feature definitions.
# fv1 = base 26; fv2 = +8 market-context columns (2026-10-16).
FEATURE_VERSION = "fv" + ("2" if len(FEATURE_COLUMNS) > 26 else "1") + "-" + hashlib.sha1("|".join(FEATURE_COLUMNS).encode("utf-8")).hexdigest()[:8]


def build_feature_frame(candles: pd.DataFrame) -> pd.DataFrame:
    """Enrich OHLCV and compute model features. Index aligned to input rows."""
    if candles.empty:
        return pd.DataFrame()

    df = enrich(candles).copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["ret_20"] = close.pct_change(20)
    df["ema_spread_20_50"] = (df["ema_20"] - df["ema_50"]) / close
    df["ema_spread_50_200"] = (df["ema_50"] - df["ema_200"]) / close
    df["price_vs_ema20"] = close / df["ema_20"] - 1.0
    df["price_vs_ema50"] = close / df["ema_50"] - 1.0
    df["price_vs_ema200"] = close / df["ema_200"] - 1.0
    df["rsi_norm"] = (df["rsi_14"] - 50.0) / 50.0
    df["atr_pct"] = df["atr_14"] / close
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std().replace(0, np.nan)
    df["volume_z"] = (volume - vol_mean) / vol_std
    df["volume_ratio"] = volume / vol_mean.replace(0, np.nan)
    df["hl_range_pct"] = (high - low) / close.replace(0, np.nan)
    df["close_location"] = (close - low) / (high - low).replace(0, np.nan)
    df["momentum_10"] = close / close.shift(10) - 1.0

    regime = df.get("market_regime", pd.Series("unknown", index=df.index)).astype(str)
    df["regime_bull"] = regime.str.contains("bull").astype(float)
    df["regime_bear"] = regime.str.contains("bear").astype(float)
    df["regime_side"] = (regime == "sideways").astype(float)
    df["regime_hivol"] = (regime == "high_volatility").astype(float)
    df["regime_riskoff"] = (regime == "risk_off").astype(float)
    df["st_direction"] = df["st_direction"].astype(float)

    # Context columns always exist (NaN until attach_market_context / panel attach).
    for c in _CONTEXT_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    return df


def attach_market_context(
    df: pd.DataFrame,
    *,
    spy_candles: pd.DataFrame | None = None,
    spy_symbol: str = "SPY",
) -> pd.DataFrame:
    """Broadcast per-ts market context from the SPY daily panel + relative strength.

    SPY_RET{k}: SPY momentum over k daily bars (PIT at the same ts).
    SPY_VOL20:  SPY trailing 20-bar daily return std (annualized).
    SPY_BREADTH20 / XS_DISP20: filled by attach_panel_context (need the multi-symbol panel).
    RS_SPY_{k}: symbol's ret_{k} minus SPY ret over the same window.

    Called on the ALREADY-built per-symbol feature frame (has ret_5/ret_20/volatility_20).
    """
    out = df.copy()
    for c in ("SPY_RET1", "SPY_RET5", "SPY_RET20", "SPY_VOL20", "SPY_BREADTH20", "RS_SPY_5", "RS_SPY_20", "XS_DISP20"):
        out[c] = np.nan

    is_spy_panel = bool(len(out) and (out["symbol"].astype(str) == spy_symbol).any())
    if spy_candles is not None and not spy_candles.empty:
        spy_feat = build_feature_frame(spy_candles)
        # Drop the placeholder columns first — merging into existing SPY_* columns
        # would suffix-collide (_x/_y) and lose the broadcast.
        out = out.drop(columns=[c for c in ("SPY_RET1", "SPY_RET5", "SPY_RET20", "SPY_VOL20") if c in out.columns])
        spy = pd.DataFrame(
            {
                "ts": pd.to_datetime(spy_feat["ts"], utc=True),
                "SPY_RET1": spy_feat["ret_1"].to_numpy(),
                "SPY_RET5": spy_feat["ret_5"].to_numpy(),
                "SPY_RET20": spy_feat["ret_20"].to_numpy(),
                "SPY_VOL20": spy_feat["volatility_20"].to_numpy(),
            }
        ).drop_duplicates(subset="ts")
        out = out.merge(spy, on="ts", how="left")
        out["RS_SPY_5"] = out["ret_5"] - out["SPY_RET5"]
        out["RS_SPY_20"] = out["ret_20"] - out["SPY_RET20"]
    elif is_spy_panel:
        # Symbol panel IS the SPY panel (context = own momentum features).
        out["SPY_RET1"] = out["ret_1"]
        out["SPY_RET5"] = out["ret_5"]
        out["SPY_RET20"] = out["ret_20"]
        out["SPY_VOL20"] = out["volatility_20"]
    return out


def attach_panel_context(panel: pd.DataFrame) -> pd.DataFrame:
    """Breadth + cross-sectional dispersion on a multi-symbol feature panel."""
    if panel.empty:
        return panel
    out = panel.copy()
    close = out["close"].astype(float)
    ma20 = close.groupby(out["symbol"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).mean())
    above = (close > ma20).astype(float).where(ma20.notna())
    # breadth per ts from the above-MA flags (no temp column — Series.groupby aligns on index)
    out["SPY_BREADTH20"] = above.groupby(out["ts"], sort=False).transform(
        lambda s: float(s.mean()) if s.notna().any() else np.nan
    )
    out["XS_DISP20"] = out.groupby("ts", sort=False)["ret_20"].transform(
        lambda s: float(s.std(ddof=1)) if s.notna().sum() >= 2 else np.nan
    )
    return out


_PANEL_MEMO: dict[tuple[str, str, object], pd.DataFrame] = {}


def build_panel_features(store: Any, timeframe: str) -> pd.DataFrame:
    """Feature panel for ALL symbols with data, with full market context attached.

    Memoized on (db path, timeframe, newest candle ts) so per-symbol training doesn't
    rebuild the 12-symbol panel 12 times. Returns columns [symbol, ts, close, *FEATURE_COLUMNS].
    """
    symbols = store.list_symbols_with_data(timeframe)
    if not symbols:
        return pd.DataFrame()
    latest: dict[str, Any] = {}
    frames = []
    for sym in symbols:
        candles = store.load_candles(sym, timeframe)
        if candles.empty:
            continue
        latest[sym] = candles["ts"].max()
        feat = build_feature_frame(candles)
        frames.append(feat)
    if not frames:
        return pd.DataFrame()
    key = (str(getattr(store, "path", "")), timeframe, tuple(sorted((s, pd.Timestamp(t).isoformat()) for s, t in latest.items())))
    hit = _PANEL_MEMO.get(key)
    if hit is not None:
        return hit
    panel = pd.concat(frames, ignore_index=True)
    spy_candles = None if "SPY" in latest else None
    if "SPY" in latest:
        spy_rows = panel.loc[panel["symbol"] == "SPY"]
        # attach SPY broadcast + RS using the SPY rows already in the panel
        spy_bc = pd.DataFrame(
            {
                "ts": pd.to_datetime(spy_rows["ts"], utc=True),
                "SPY_RET1": spy_rows["ret_1"].to_numpy(),
                "SPY_RET5": spy_rows["ret_5"].to_numpy(),
                "SPY_RET20": spy_rows["ret_20"].to_numpy(),
                "SPY_VOL20": spy_rows["volatility_20"].to_numpy(),
            }
        ).drop_duplicates("ts")
        panel = panel.drop(columns=[c for c in ("SPY_RET1", "SPY_RET5", "SPY_RET20", "SPY_VOL20") if c in panel.columns])
        panel = panel.merge(spy_bc, on="ts", how="left")
        panel["RS_SPY_5"] = panel["ret_5"] - panel["SPY_RET5"]
        panel["RS_SPY_20"] = panel["ret_20"] - panel["SPY_RET20"]
    panel = attach_panel_context(panel)
    panel = panel.sort_values(["ts", "symbol"]).reset_index(drop=True)
    _PANEL_MEMO.clear()
    _PANEL_MEMO[key] = panel
    return panel


def make_labels(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    band: float = 0.001,
) -> pd.Series:
    """Forward return classification: up / down / sideways. Label uses future bars only via shift(-h)."""
    fwd = df["close"].astype(float).shift(-horizon) / df["close"].astype(float) - 1.0
    labels = pd.Series("sideways", index=df.index, dtype=object)
    labels = labels.mask(fwd > band, "up")
    labels = labels.mask(fwd < -band, "down")
    labels[fwd.isna()] = np.nan
    return labels


def make_labels_vol_scaled(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    k: float = 0.5,
) -> pd.Series:
    """Vol-scaled labels: band = k × trailing 20-bar return std (per bar, PIT).

    Fixes the degenerate-labels problem: a fixed 0.002 band on ~1–2% daily vol
    dumps most days into 'sideways', making the 3-class task nearly a coin flip.
    The band scales with each bar's own trailing vol, so up/down/sideways stay
    balanced in quiet AND loud regimes. k=0.5 → band ≈ half a daily sigma.
    """
    close = df["close"].astype(float)
    fwd = close.shift(-horizon) / close - 1.0
    sigma = close.pct_change().rolling(20, min_periods=20).std()
    band = (k * sigma.shift(1)).replace(0.0, np.nan)  # shift(1): sigma known at t
    labels = pd.Series("sideways", index=df.index, dtype=object)
    labels = labels.mask(fwd > band, "up")
    labels = labels.mask(fwd < -band, "down")
    labels[fwd.isna() | band.isna()] = np.nan
    return labels


def dataset_from_candles(
    candles: pd.DataFrame,
    *,
    horizon: int = 1,
    band: float = 0.001,
    min_history: int = 60,
    label_scheme: str = "fixed",
    spy_candles: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X, y, and meta frame (ts/close) with NaNs dropped. No shuffle.

    label_scheme: "fixed" (historical, band=const) or "vol_scaled" (band = 0.5×trailing
    20-bar sigma, PIT). When spy_candles is provided (and candles are not SPY itself),
    the 8 market-context columns are attached before feature selection.
    """
    feat = build_feature_frame(candles)
    if feat.empty:
        return pd.DataFrame(), pd.Series(dtype=object), pd.DataFrame()

    if spy_candles is not None:
        feat = attach_market_context(feat, spy_candles=spy_candles)

    if label_scheme == "vol_scaled":
        y = make_labels_vol_scaled(feat, horizon=horizon)
    else:
        y = make_labels(feat, horizon=horizon, band=band)
    meta = pd.DataFrame(
        {
            "ts": feat["ts"],
            "close": feat["close"].astype(float),
            "symbol": feat["symbol"] if "symbol" in feat.columns else None,
            "timeframe": feat["timeframe"] if "timeframe" in feat.columns else None,
        }
    )
    X = feat[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)
    # Market-context columns are OPTIONAL rows-wise: when no SPY panel is attached
    # (or SPY's history starts later), they stay NaN and XGBoost routes NaNs
    # natively. Core per-symbol features still must be complete. When SPY candles
    # ARE attached, the SPY_*/RS_* broadcasts must be complete; BREADTH/XS need a
    # multi-symbol panel and stay optional here (filled by the panel path).
    core = [c for c in FEATURE_COLUMNS if c not in _CONTEXT_COLUMNS]
    broadcast = [c for c in ("SPY_RET1", "SPY_RET5", "SPY_RET20", "SPY_VOL20", "RS_SPY_5", "RS_SPY_20") if c in X.columns]
    mask = X[core].notna().all(axis=1) & y.notna()
    if spy_candles is not None and not spy_candles.empty:
        mask &= X[broadcast].notna().all(axis=1)
    # Drop warm-up head for indicator stability
    if min_history > 0:
        idx = np.arange(len(X))
        mask = mask & (idx >= min_history)
    # Drop last `horizon` rows (no label)
    if horizon > 0:
        mask.iloc[-horizon:] = False

    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True), meta.loc[mask].reset_index(drop=True)


def dataset_from_panel(
    panel: pd.DataFrame,
    symbol: str,
    *,
    horizon: int = 1,
    label_scheme: str = "vol_scaled",
    min_history: int = 60,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """fv2 dataset for one symbol from the context-attached panel.

    Expects panel columns [symbol, ts, close, ret_*, ..., *context]. Rows require
    complete CORE features; context must also be complete (panel build guarantees
    it after warm-up).
    """
    sub = panel.loc[panel["symbol"].astype(str) == symbol.upper()].copy().reset_index(drop=True)
    if sub.empty:
        return pd.DataFrame(), pd.Series(dtype=object), pd.DataFrame()
    if label_scheme == "vol_scaled":
        y = make_labels_vol_scaled(sub, horizon=horizon)
    else:
        y = make_labels(sub, horizon=horizon)
    meta = pd.DataFrame({"ts": sub["ts"], "close": sub["close"].astype(float), "symbol": sub["symbol"], "timeframe": sub.get("timeframe")})
    X = sub[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)
    mask = X.notna().all(axis=1) & y.notna()
    if min_history > 0:
        idx = np.arange(len(X))
        mask = mask & (idx >= min_history)
    if horizon > 0:
        mask.iloc[-horizon:] = False
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True), meta.loc[mask].reset_index(drop=True)


def default_band(timeframe: str) -> float:
    return {
        "15m": 0.0005,
        "1h": 0.0008,
        "4h": 0.0015,
        "1d": 0.002,
    }.get(timeframe, 0.001)


def feature_importance_frame(booster_or_model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    try:
        importances = booster_or_model.feature_importances_
    except Exception:  # noqa: BLE001
        return []
    pairs = sorted(zip(feature_names, importances, strict=True), key=lambda x: -float(x[1]))
    return [{"feature": n, "importance": float(v)} for n, v in pairs]
