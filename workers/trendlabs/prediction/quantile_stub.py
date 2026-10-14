"""Stub interfaces for future quantile heads (Q10 / Q50 / Q90) and conformal calibration.

Not implemented in V1 Phase A/B (+ light C). Kept so later phases can plug in
without reshaping call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuantilePrediction:
    q10: float
    q50: float
    q90: float

    def as_dict(self) -> dict[str, float]:
        return {"q10": self.q10, "q50": self.q50, "q90": self.q90}


class QuantileModel(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any: ...

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return columns q10, q50, q90."""
        ...


class StubQuantileModel:
    """Placeholder: returns NaN quantiles. Replace in a later phase."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> StubQuantileModel:
        self.n_train_ = len(X)
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        n = len(X)
        return pd.DataFrame(
            {
                "q10": np.full(n, np.nan),
                "q50": np.full(n, np.nan),
                "q90": np.full(n, np.nan),
            },
            index=X.index,
        )


class ConformalCalibrator:
    """Stub for future conformal intervals around point or quantile forecasts."""

    def fit(self, y_true: pd.Series, y_pred: pd.Series) -> ConformalCalibrator:
        self.fitted_ = True
        self.n_ = int(y_true.notna().sum())
        return self

    def interval(self, y_pred: pd.Series, *, alpha: float = 0.1) -> pd.DataFrame:
        raise NotImplementedError(
            "Conformal calibration is out of scope for Prediction Engine V1 Phase A/B"
        )
