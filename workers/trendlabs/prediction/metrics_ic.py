"""Cross-sectional IC and quintile metrics for Prediction Engine V1."""

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


def assign_quintiles(
    scores: pd.Series,
    *,
    n_quantiles: int = 5,
) -> pd.Series:
    """Deterministic quintile labels 1..5 (5 = highest score). NaN where not assignable."""
    out = pd.Series(np.nan, index=scores.index, dtype=float)
    valid = scores.dropna()
    if len(valid) < n_quantiles:
        return out
    try:
        # rank method average + qcut on ranks keeps ties stable across equal scores
        ranks = valid.rank(method="first")
        labels = pd.qcut(ranks, n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        return out
    if labels.nunique() < n_quantiles:
        return out
    # labels 0..4 → quintiles 1..5
    out.loc[valid.index] = labels.astype(float) + 1.0
    return out


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
        q = assign_quintiles(sub[score_col])
        if q.notna().sum() < min_names or q.nunique(dropna=True) < 5:
            continue
        top = sub.loc[q == 5, target_col].mean()
        bot = sub.loc[q == 1, target_col].mean()
        spreads.append(float(top - bot))

    if not spreads:
        return {"n_days": 0, "Q5_Q1_mean_z": None, "min_names_required": min_names}
    return {
        "n_days": len(spreads),
        "Q5_Q1_mean_z": float(np.mean(spreads)),
        "min_names_required": min_names,
    }


def full_quintile_analysis(
    panel: pd.DataFrame,
    *,
    score_col: str,
    date_col: str = "ts",
    min_names: int = 10,
) -> dict[str, Any]:
    """OOS quintile means for z and tradable R; monotonicity; Q5−Q1 spreads.

    R = ln(close[t+5]/open[t+1]) — already on the panel as column ``R``.
    """
    z_means: dict[int, list[float]] = {q: [] for q in range(1, 6)}
    r_means: dict[int, list[float]] = {q: [] for q in range(1, 6)}
    spreads_z: list[float] = []
    spreads_r: list[float] = []
    n_days = 0

    for _, g in panel.groupby(date_col, sort=True):
        need = [score_col, "z", "R"]
        if any(c not in g.columns for c in need):
            continue
        sub = g[need].dropna()
        if len(sub) < min_names:
            continue
        q = assign_quintiles(sub[score_col])
        if q.nunique(dropna=True) < 5:
            continue
        n_days += 1
        day_z: dict[int, float] = {}
        day_r: dict[int, float] = {}
        for qi in range(1, 6):
            mask = q == float(qi)
            if not mask.any():
                continue
            mz = float(sub.loc[mask, "z"].mean())
            mr = float(sub.loc[mask, "R"].mean())
            z_means[qi].append(mz)
            r_means[qi].append(mr)
            day_z[qi] = mz
            day_r[qi] = mr
        if 1 in day_z and 5 in day_z:
            spreads_z.append(day_z[5] - day_z[1])
            spreads_r.append(day_r[5] - day_r[1])

    mean_z_by_q = {f"Q{q}": (float(np.mean(z_means[q])) if z_means[q] else None) for q in range(1, 6)}
    mean_r_by_q = {f"Q{q}": (float(np.mean(r_means[q])) if r_means[q] else None) for q in range(1, 6)}

    def _broadly_monotonic(means: dict[str, float | None]) -> bool | None:
        vals = [means[f"Q{q}"] for q in range(1, 6)]
        if any(v is None for v in vals):
            return None
        # Broadly: each step non-decreasing (allow tiny float noise)
        return all(vals[i] <= vals[i + 1] + 1e-12 for i in range(4))

    return {
        "n_days": n_days,
        "min_names_required": min_names,
        "mean_z_by_quintile": mean_z_by_q,
        "mean_R_by_quintile": mean_r_by_q,
        "Q5_Q1_mean_z": float(np.mean(spreads_z)) if spreads_z else None,
        "Q5_Q1_mean_R": float(np.mean(spreads_r)) if spreads_r else None,
        "broadly_monotonic_z": _broadly_monotonic(mean_z_by_q),
        "broadly_monotonic_R": _broadly_monotonic(mean_r_by_q),
        "broadly_monotonic": bool(
            (_broadly_monotonic(mean_z_by_q) is True) or (_broadly_monotonic(mean_r_by_q) is True)
        ),
    }


def fold_mean_ic_stats(
    panel: pd.DataFrame,
    *,
    score_col: str,
    target_col: str = "z",
    date_col: str = "ts",
    fold_col: str = "fold_id",
    min_names: int = 2,
) -> dict[str, Any]:
    """Per-fold MeanIC and fraction of folds with positive MeanIC."""
    if fold_col not in panel.columns:
        return {"n_folds": 0, "positive_fold_rate": None, "fold_mean_ics": []}
    fold_ics: list[float] = []
    for _, g in panel.groupby(fold_col, sort=True):
        daily = daily_spearman_ic(
            g, score_col=score_col, target_col=target_col, date_col=date_col, min_names=min_names
        )
        s = daily.dropna()
        if s.empty:
            continue
        fold_ics.append(float(s.mean()))
    if not fold_ics:
        return {"n_folds": 0, "positive_fold_rate": None, "fold_mean_ics": []}
    return {
        "n_folds": len(fold_ics),
        "positive_fold_rate": float(np.mean([ic > 0 for ic in fold_ics])),
        "fold_mean_ics": fold_ics,
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
