"""Purged chronological walk-forward for Prediction Engine V1.

No random splits. Purge gap ≥ HORIZON (default 5 trading days) so that
labels originating near the end of train (which look ahead to t+5) do not
overlap the test period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trendlabs.prediction import PURGE_GAP_DAYS


@dataclass(frozen=True)
class PurgedFold:
    fold_id: int
    train_dates: tuple  # sorted unique dates in train
    test_dates: tuple
    train_end_date: object
    test_start_date: object
    purge_gap_days: int


def unique_sorted_dates(panel: pd.DataFrame, date_col: str = "ts") -> list:
    dates = pd.to_datetime(panel[date_col], utc=True)
    return sorted(dates.drop_duplicates().tolist())


def generate_purged_folds(
    dates: list,
    *,
    min_train_days: int = 252,
    test_days: int = 63,
    step_days: int | None = None,
    purge_gap: int = PURGE_GAP_DAYS,
) -> list[PurgedFold]:
    """Expanding train, fixed test, with purge between train_end and test_start.

    Train uses dates[0 : train_end_idx+1] (inclusive) where train_end is at least
    `purge_gap` trading days before test_start.
    """
    step = step_days if step_days is not None else test_days
    n = len(dates)
    folds: list[PurgedFold] = []
    if n < min_train_days + purge_gap + test_days:
        return folds

    # First test window starts after min_train + purge
    test_start_idx = min_train_days + purge_gap
    fold_id = 0
    while test_start_idx + test_days <= n:
        test_end_idx = test_start_idx + test_days - 1
        train_end_idx = test_start_idx - purge_gap - 1
        if train_end_idx + 1 < min_train_days:
            break
        train_dates = tuple(dates[: train_end_idx + 1])
        test_dates = tuple(dates[test_start_idx : test_end_idx + 1])
        # Verify purge
        gap = test_start_idx - train_end_idx - 1
        if gap < purge_gap:
            raise RuntimeError(f"Purge gap {gap} < required {purge_gap}")
        folds.append(
            PurgedFold(
                fold_id=fold_id,
                train_dates=train_dates,
                test_dates=test_dates,
                train_end_date=dates[train_end_idx],
                test_start_date=dates[test_start_idx],
                purge_gap_days=gap,
            )
        )
        fold_id += 1
        test_start_idx += step
    return folds


def mask_dates(panel: pd.DataFrame, dates: tuple | list, date_col: str = "ts") -> pd.DataFrame:
    ts = pd.to_datetime(panel[date_col], utc=True)
    date_set = set(pd.to_datetime(list(dates), utc=True))
    return panel.loc[ts.isin(date_set)].copy()


def purge_gap_ok(train_end, test_start, all_dates: list, *, min_gap: int = PURGE_GAP_DAYS) -> bool:
    """Return True if trading-day gap between train_end and test_start ≥ min_gap."""
    dates = list(all_dates)
    try:
        i_train = dates.index(pd.Timestamp(train_end))
        i_test = dates.index(pd.Timestamp(test_start))
    except ValueError:
        # Try matching via normalized timestamps
        norm = [pd.Timestamp(d) for d in dates]
        i_train = norm.index(pd.Timestamp(train_end))
        i_test = norm.index(pd.Timestamp(test_start))
    gap = i_test - i_train - 1
    return gap >= min_gap
