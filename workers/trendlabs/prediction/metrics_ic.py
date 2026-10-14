"""Cross-sectional IC metrics for Prediction Engine V1."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def daily_spearman_ic(
    panel: pd.DataFrame,
    *,
    score_col: str,
    target_col: str = "z",
    date_col: str = "ts",
    min_names: int = 2,
) -> pd.Series:
    """Per-date Spearman rank correlation between score and target."""

    def _ic(g: pd.DataFrame) -> float:
        sub = g[[score_col, target_col]].dropna()
        if len(sub) < min_names:
            return np.nan
        if sub[score_col].nunique() < 2 or sub[target_col].nunique() < 2:
            return np.nan
        return float(sub[score_col].corr(sub[target_col], method="spearman"))

    grouped = panel.groupby(date_col, sort=True)
    ics = grouped.apply(_ic, include_groups=False)
    ics.name = "ic"
    return ics


def summarize_ic(daily_ic: pd.Series) -> dict[str, Any]:
    s = daily_ic.dropna()
    if s.empty:
        return {
            "n_days": 0,
            "MeanIC": None,
            "ICIR": None,
            "PositiveICRate": None,
            "IC_std": None,
        }
    mean_ic = float(s.mean())
    std_ic = float(s.std(ddof=1)) if len(s) > 1 else None
    icir = (mean_ic / std_ic) if std_ic and std_ic > 0 else None
    return {
        "n_days": int(len(s)),
        "MeanIC": mean_ic,
        "ICIR": icir,
        "PositiveICRate": float((s > 0).mean()),
        "IC_std": std_ic,
    }


def quintile_spread(
    panel: pd.DataFrame,
    *,
    score_col: str,
    target_col: str = "z",
    date_col: str = "ts",
    min_names: int = 10,
) -> dict[str, Any]:
    """Mean Q5−Q1 target spread across days with enough names.

    Q5 = top quintile by predicted score; Q1 = bottom.
    """
    spreads: list[float] = []

    for _, g in panel.groupby(date_col, sort=True):
        sub = g[[score_col, target_col]].dropna()
        if len(sub) < min_names:
            continue
        try:
            q = pd.qcut(sub[score_col], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        if q.nunique() < 5:
            continue
        top = sub.loc[q == q.max(), target_col].mean()
        bot = sub.loc[q == q.min(), target_col].mean()
        spreads.append(float(top - bot))

    if not spreads:
        return {"n_days": 0, "Q5_Q1_mean_z": None, "min_names_required": min_names}
    return {
        "n_days": len(spreads),
        "Q5_Q1_mean_z": float(np.mean(spreads)),
        "min_names_required": min_names,
    }


def evaluate_scores(
    panel: pd.DataFrame,
    *,
    score_col: str,
    target_col: str = "z",
    date_col: str = "ts",
    min_names_ic: int = 2,
    min_names_quintile: int = 10,
) -> dict[str, Any]:
    daily = daily_spearman_ic(
        panel, score_col=score_col, target_col=target_col, date_col=date_col, min_names=min_names_ic
    )
    summary = summarize_ic(daily)
    summary["quintile"] = quintile_spread(
        panel,
        score_col=score_col,
        target_col=target_col,
        date_col=date_col,
        min_names=min_names_quintile,
    )
    return summary
