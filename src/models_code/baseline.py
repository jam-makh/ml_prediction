"""The trivial predictor every later number is measured against.

Predicts the mean of the last ``n_months`` closing balances. Configured twice in
the config -- ``n_months: 3`` is the 3-month average, ``n_months: 1`` is
persistence (last month's balance).

The only thing ``fit`` learns is a fallback for a user's first month, where no
lag exists: the median of the training targets, taken from training rows only.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.data.data import Dataset
from src.models_code.base_class import Model

# The lag columns averaged, newest first.
BALANCE_LAGS = (
    "prev_1m_closing_balance_usd",
    "prev_2m_closing_balance_usd",
    "prev_3m_closing_balance_usd",
)


class LagAverageBaseline(Model):
    """Predict the mean of whichever of the last ``n_months`` balances exist."""

    def __init__(self, n_months: int = 3, name: str | None = None) -> None:
        super().__init__(name or f"mean_last_{n_months}m")
        self.columns = BALANCE_LAGS[:n_months]
        self.fallback = float("nan")

    def _fit(self, dataset: Dataset) -> None:
        self.fallback = float(dataset.target.median())

    def _predict(self, dataset: Dataset) -> npt.NDArray[np.float64]:
        # mean skips NaN, so 2 lags present -> mean of 2; none -> the fallback.
        average = dataset.frame[list(self.columns)].mean(axis=1)
        return average.fillna(self.fallback).to_numpy(dtype="float64")
