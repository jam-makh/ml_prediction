"""
*Ridge rather than plain least squares. The feature table carries three
closing-balance lags that move together closely, and until the redundant
aggregates are dropped from the config it is also exactly rank deficient --
``delta_prev_1m_2m`` is the difference of two lags that are both still present,
and ``prev_1m_net_flow`` is credited minus debited. At rank deficiency ordinary
least squares has no unique solution: scikit-learn does not error, it falls back
to the pseudo-inverse, so coefficients come out but which ones is arbitrary
within the null space. Ridge is defined either way, and with merely correlated
columns it spreads the weight across the group instead of picking one at random.

Ridge rather than lasso, for the same reason in reverse: lasso keeps one
member of a correlated group and zeroes the rest, and which member it keeps
flips between folds. That makes it a poor basis for deciding what to drop, and
on standardised money columns its coordinate descent needs a raised iteration
cap to stop warning on every fit.
]
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.data import Dataset
from src.models_code.anchored_model import AnchoredModel


class RidgeRegression(AnchoredModel):
    """Ridge regression over the feature table, with its own preprocessing.

    Parameters
    ----------
    name : str, optional
        Label for result tables. Default ``ridge``.
    anchor_column : str, optional
        Column holding last month's balance.
    alpha : float
        Regularisation strength, from Optuna or the config's ``params``.
    impute_strategy : str, optional
        Passed to ``SimpleImputer``. Default ``median``: the money columns are
        skewed enough that a mean fill would insert values no user ever held.
    random_state : int, optional
        Accepted for interface symmetry with the booster. Ridge is
        deterministic, so it changes nothing.

    Attributes
    ----------
    best_params_ : dict or None
        ``{"alpha": ...}`` once fitted.

    Raises
    ------
    ValueError
        If ``alpha`` is not set.
    """

    def __init__(
        self,
        name: str = "ridge",
        anchor_column: str = "prev_1m_closing_balance_usd",
        alpha: float | None = None,
        impute_strategy: str = "median",
        random_state: int = 42,
    ) -> None:
        super().__init__(name, anchor_column=anchor_column)
        # Empty means the search has not been run yet, so say so instead of fitting a guess.
        if alpha is None:
            raise ValueError(self._unset_message(name, "alpha"))
        self.alpha = alpha
        self.impute_strategy = impute_strategy
        self.random_state = random_state

        # Populated by _fit.
        self._pipeline: Pipeline | None = None

    @staticmethod
    def _unset_message(name: str, params: str) -> str:
        """Return the error text for penalty settings left empty in the config.

        Parameters
        ----------
        name : str
            The model's name.
        params : str
            The unset settings, e.g. ``alpha`` or ``alpha / l1_ratio``.

        Returns
        -------
        str
            What is empty and how to fill it.
        """
        return (
            f"{name}: {params} not set. Run with optuna.enabled: true, then paste "
            f"best_params from {name}_best_params.json into this model's params "
            f"in the config."
        )

    def _build_estimator(self) -> Any:
        """Return the unfitted linear estimator with the configured penalty.

        Returns
        -------
        sklearn.linear_model.Ridge
            Unfitted.
        """
        return Ridge(alpha=self.alpha)

    def _penalty_params(self) -> dict[str, float]:
        """Return the penalty settings this model fits with.

        Returns
        -------
        dict
            ``{"alpha": ...}``.
        """
        return {"alpha": float(self.alpha)}

    def _build_pipeline(self) -> Pipeline:
        """Return the unfitted impute -> scale -> regressor pipeline.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Unfitted.
        """
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy=self.impute_strategy)),
                ("scale", StandardScaler()),
                ("regressor", self._build_estimator()),
            ]
        )

    def _fit(self, dataset: Dataset) -> None:
        """Fit the pipeline on the rows that have a target.

        Parameters
        ----------
        dataset : Dataset
            Training rows.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no rows have a usable target.
        """
        target = self._training_target(dataset)
        usable = self._usable_rows(target)

        features = dataset.features.loc[usable]
        values = target.to_numpy(dtype="float64")[usable]

        self._pipeline = self._build_pipeline()
        self._pipeline.fit(features, values)
        # Recorded so train.py saves and logs the settings like any tuned model.
        self.best_params_ = self._penalty_params()

    def _predict(self, dataset: Dataset) -> npt.NDArray[np.float64]:
        """Predict the change and turn it back into a balance.

        Parameters
        ----------
        dataset : Dataset
            Rows to predict for.

        Returns
        -------
        numpy.ndarray of float
            One prediction per row, on the balance scale.
        """
        assert self._pipeline is not None  # guaranteed by Model.predict
        # The columns this model was fitted on, not the dataset's own feature
        # list: each model family can drop different columns, and the dataset
        # handed in at scoring time carries the widest list.
        predicted = np.asarray(
            self._pipeline.predict(dataset.frame.loc[:, list(self.feature_columns)]),
            dtype="float64",
        )
        return self._restore_level(dataset, predicted)

    def feature_importance(self) -> pd.Series | None:
        """Return absolute standardised coefficients, largest first.

        Absolute because the question is which columns carry the model, not
        which push the prediction up. Standardised because the pipeline scales
        first, so the coefficients are already on one scale and comparable --
        which is exactly what raw coefficients on money columns are not.

        Returns
        -------
        pandas.Series or None
            One entry per feature, descending, or None before fitting.
        """
        if not self.is_fitted or self._pipeline is None:
            return None

        coefficients = np.abs(
            np.asarray(self._pipeline.named_steps["regressor"].coef_, dtype="float64")
        )
        if coefficients.shape[0] != len(self._feature_columns):
            # The imputer can drop an all-NaN column, which would misalign the
            # names. Better to report nothing than to report the wrong names.
            return None

        series = pd.Series(
            coefficients, index=list(self._feature_columns), dtype="float64"
        )
        series.name = "abs_coefficient"
        return series.sort_values(ascending=False)

    def describe(self) -> str:
        """Return a one-line description for the run log.

        Returns
        -------
        str
            The chosen alpha and the feature count.
        """
        if not self.is_fitted:
            return f"{self.name} (not fitted)"
        return (
            f"{self.name}: ridge alpha={self.alpha:g}, "
            f"{len(self._feature_columns)} features"
        )
