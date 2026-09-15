"""The trivial predictor every later number is measured against.

One class, configured twice. ``n_months: 3`` averages the last three closing
balances -- the headline baseline, the bar the fitted models have to clear to
have earned their complexity. ``n_months: 1`` predicts last month's balance
unchanged, which on a strongly autocorrelated series like a bank balance is
usually the harder of the two to beat. Running both makes it obvious whether a
fitted model is adding anything beyond "the balance does not move much".

Both live in the config rather than in subclasses here, because a class whose
entire body binds one integer is a class that earns nothing:

.. code-block:: yaml

    - {name: three_month_average, kind: lag_average, params: {n_months: 3}}
    - {name: persistence,         kind: lag_average, params: {n_months: 1}}

Neither reads a column from inside the month being predicted. Both work from
``prev_1m`` and friends, which are the closing balances of months already
finished on the day the prediction is made.

A word on what "fitting" means here, because it is not nothing. The baseline
learns exactly one number: a fallback for the rows where the lag columns are
empty, which happens for every user's first months. That fallback is the median
of the training targets, and it is computed from the training rows only, for the
same reason everything else in this project is. A baseline that quietly took its
fallback from the whole panel would be a leaking baseline, and the bar it set
would be too high for the wrong reason.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.models_code.base_class import Model
from src.data.data import Dataset

# The lag columns the baselines read, newest first. Named here rather than
# derived, because a baseline is defined by the specific columns it averages:
# if the feature table renames them, this should break loudly.
BALANCE_LAGS = (
    "prev_1m_closing_balance_usd",
    "prev_2m_closing_balance_usd",
    "prev_3m_closing_balance_usd",
)


class LagAverageBaseline(Model):
    """Predict the mean of the last ``n_months`` closing balances.

    The shared machinery behind both baselines. Averages whichever of the
    requested lag columns are present for a row, so a user with only one month
    of history is predicted from that one month rather than dropped or filled.
    Rows with no history at all fall back to the training median.

    Parameters
    ----------
    n_months : int, optional
        How many lag columns to average, counting back from last month. Must be
        between 1 and the number of columns in ``BALANCE_LAGS``. Default 3.
    name : str, optional
        Label for result tables. Defaults to ``mean_last_{n}m``.

    Raises
    ------
    ValueError
        If ``n_months`` is outside the available lags.
    """

    def __init__(self, n_months: int = 3, name: str | None = None) -> None:
        if not 1 <= n_months <= len(BALANCE_LAGS):
            raise ValueError(
                f"n_months must be between 1 and {len(BALANCE_LAGS)}, "
                f"got {n_months}"
            )
        super().__init__(name or f"mean_last_{n_months}m")
        self.n_months = n_months
        self._columns = BALANCE_LAGS[:n_months]
        # The one fitted quantity. NaN until fit() has run.
        self._fallback = float("nan")

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return the lag columns this baseline reads.

        Overridden so a renamed or missing lag column is reported by name at
        fit time, rather than as a KeyError from inside the averaging.

        Returns
        -------
        tuple of str
            The lag column names, newest first.
        """
        return self._columns

    def _fit(self, dataset: Dataset) -> None:
        """Learn the fallback value from the training targets.

        Parameters
        ----------
        dataset : Dataset
            Training rows.

        Returns
        -------
        None
        """
        # Median rather than mean: the target is heavy-tailed enough that a
        # mean fallback would predict a value no ordinary user ever holds.
        self._fallback = float(dataset.target.median())

    def _predict(self, dataset: Dataset) -> npt.NDArray[np.float64]:
        """Average the available lag columns, row by row.

        Parameters
        ----------
        dataset : Dataset
            Rows to predict for.

        Returns
        -------
        numpy.ndarray of float
            One prediction per row, in row order.
        """
        lags = dataset.frame.loc[:, list(self._columns)].to_numpy(dtype="float64")

        # nanmean over the row: a user with two of three lags present is
        # predicted from those two. The all-NaN rows warn, so they are silenced
        # here and handled explicitly on the next line.
        with np.errstate(invalid="ignore"):
            predictions = np.nanmean(lags, axis=1)

        # Every lag missing means this is one of the user's first months. There
        # is nothing to average, so the training median stands in.
        no_history = ~np.isfinite(predictions)
        predictions[no_history] = self._fallback
        return np.asarray(predictions, dtype="float64")

    def describe(self) -> str:
        """Return a one-line description for the run log.

        Returns
        -------
        str
            The averaging window and the fitted fallback.
        """
        if not self.is_fitted:
            return f"{self.name} (not fitted)"
        return (
            f"{self.name}: mean of {', '.join(self._columns)}; "
            f"fallback {self._fallback:,.2f}"
        )
