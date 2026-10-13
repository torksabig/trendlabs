from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr_vals = atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_vals
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_vals
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    atr_vals = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + multiplier * atr_vals
    lower = hl2 - multiplier * atr_vals
    final_upper = upper.copy()
    final_lower = lower.copy()
    direction = pd.Series(1, index=df.index, dtype=int)
    st = pd.Series(np.nan, index=df.index, dtype=float)

    for i in range(1, len(df)):
        if upper.iloc[i] < final_upper.iloc[i - 1] or df["close"].iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if lower.iloc[i] > final_lower.iloc[i - 1] or df["close"].iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if direction.iloc[i - 1] == 1:
            direction.iloc[i] = -1 if df["close"].iloc[i] < final_lower.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if df["close"].iloc[i] > final_upper.iloc[i] else -1
        st.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    return pd.DataFrame({"supertrend": st, "st_direction": direction})


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_20"] = ema(out["close"], 20)
    out["ema_50"] = ema(out["close"], 50)
    out["ema_200"] = ema(out["close"], 200)
    out["rsi_14"] = rsi(out["close"], 14)
    out["adx_14"] = adx(out, 14)
    out["atr_14"] = atr(out, 14)
    out["volatility_20"] = out["close"].pct_change().rolling(20).std() * np.sqrt(252)
    out["volume_sma_20"] = out["volume"].rolling(20).mean()
    st = supertrend(out)
    out = out.join(st)

    # Quantitative regime engine (TrendLab V2). trend_label kept as alias for UI/snapshot.
    from trendlabs.regimes import classify_series

    out["market_regime"] = classify_series(out)
    out["trend_label"] = out["market_regime"]
    return out
