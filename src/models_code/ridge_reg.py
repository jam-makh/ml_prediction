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
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.data import Dataset
from src.models_code.anchored_model import AnchoredModel
from src.window import MonthlyExpandingSplit

# Spans nine orders of magnitude so the CV curve has a genuine interior minimum
# to find rather than being pinned at an end. The top matters here: with the
# balance lags as correlated as they are, a first run capped at 1e3 chose 1e3 --
# the boundary, which means the search wanted more penalty than it was offered
# and the value it reported was the grid's edge, not an answer. A chosen alpha
# still sitting at either end is the signal to widen this again.
DEFAULT_ALPHAS: tuple[float, ...] = (
    0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0,
)


class RidgeRegression(AnchoredModel):
    """Ridge regression over the feature table, with its own preprocessing.

    Parameters
    ----------
    name : str, optional
        Label for result tables. Default ``ridge``.
    anchor_column : str, optional
        Column holding last month's balance.
    alpha : float, optional
        Fixed regularisation strength. Ignored when ``alphas`` is given, which
        is the default -- set this only to pin a value and skip the inner CV.
    alphas : sequence of float, optional
        Candidates for ``RidgeCV``. Default :data:`DEFAULT_ALPHAS`. Pass None to
        use the fixed ``alpha`` instead.
    impute_strategy : str, optional
        Passed to ``SimpleImputer``. Default ``median``: the money columns are
        skewed enough that a mean fill would insert values no user ever held.
    random_state : int, optional
        Accepted for interface symmetry with the booster. Ridge is
        deterministic, so it changes nothing.
    clip, market_scale, recency_half_life : optional
        Training-target treatment, see ``AnchoredModel``.

    Attributes
    ----------
    best_params_ : dict or None
        ``{"alpha": ...}`` once fitted through ``RidgeCV``.
    """

    def __init__(
        self,
        name: str = "ridge",
        anchor_column: str = "prev_1m_closing_balance_usd",
        alpha: float = 1.0,
        alphas: tuple[float, ...] | None = DEFAULT_ALPHAS,
        impute_strategy: str = "median",
        random_state: int = 42,
        clip: float | str | None = None,
        market_scale: bool = False,
        recency_half_life: float | None = None,
    ) -> None:
        super().__init__(
            name,
            anchor_column=anchor_column,
            clip=clip,
            market_scale=market_scale,
            recency_half_life=recency_half_life,
        )
        self.alpha = alpha
        self.alphas = tuple(alphas) if alphas else None
        self.impute_strategy = impute_strategy
        self.random_state = random_state

        # Populated by _fit.
        self._pipeline: Pipeline | None = None

    def _build_pipeline(self, folds: Any = None) -> Pipeline:
        """Return the unfitted impute -> scale -> ridge pipeline.

        Parameters
        ----------
        folds : list of (numpy.ndarray, numpy.ndarray), optional
            Month-aware folds for the inner alpha search. When omitted,
            ``RidgeCV`` falls back to its own efficient leave-one-out, which is
            fine for choosing a scalar but is not month-aware -- so the caller
            passes folds whenever it has them.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Unfitted.
        """
        estimator: Any
        if self.alphas:
            estimator = RidgeCV(alphas=self.alphas, cv=folds)
        else:
            estimator = Ridge(alpha=self.alpha)

        return Pipeline(
            [
                ("impute", SimpleImputer(strategy=self.impute_strategy)),
                ("scale", StandardScaler()),
                ("ridge", estimator),
            ]
        )

    def _fit(self, dataset: Dataset) -> None:
        """Fit the pipeline, choosing alpha by cross-validation.

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

        # Folds are built from the surviving rows' months, not the dataset's, or
        # the index pairs would not line up with the matrix being fitted.
        folds = self._inner_folds(dataset, usable)

        weights = self._training_weights(dataset)
        fit_params = {} if weights is None else {"ridge__sample_weight": weights[usable]}

        self._pipeline = self._build_pipeline(folds)
        self._pipeline.fit(features, values, **fit_params)

        chosen = self._pipeline.named_steps["ridge"]
        if hasattr(chosen, "alpha_"):
            picked = float(chosen.alpha_)
            self.best_params_ = {"alpha": picked}
            # An alpha at either end of the grid is not a choice, it is the
            # search running out of room, and the two ends mean opposite things.
            # Said out loud rather than left in the summary for someone to spot.
            if self.alphas and picked >= max(self.alphas):
                logger.warning(
                    f"  {self.name}: alpha settled on {picked:g}, the top of the "
                    f"grid -- cross-validation wants every coefficient at zero, "
                    f"so this model is predicting the mean movement and nothing "
                    f"else. Widening the grid will not change that; it is a "
                    f"finding about the features, not about alpha."
                )
            elif self.alphas and picked <= min(self.alphas):
                logger.warning(
                    f"  {self.name}: alpha settled on {picked:g}, the bottom of "
                    f"the grid -- the fit wants less penalty than it was "
                    f"offered. Widen DEFAULT_ALPHAS downward in ridge_reg.py."
                )
        else:
            self.best_params_ = {"alpha": float(self.alpha)}

    def _inner_folds(self, dataset: Dataset, usable: npt.NDArray[np.bool_]) -> Any:
        """Return month-aware folds for the alpha search, or None.

        Parameters
        ----------
        dataset : Dataset
            Training rows, before the usable-target filter.
        usable : numpy.ndarray of bool
            Mask of rows that survive into the fit.

        Returns
        -------
        list or None
            Fold index pairs, or None when the training region is too short to
            produce any -- in which case ``RidgeCV`` uses leave-one-out.
        """
        months = dataset.frame[dataset.time_column].loc[usable].reset_index(drop=True)
        splitter = MonthlyExpandingSplit(n_folds=3, test_months=3)
        # X is only used for its length; the months go in as groups.
        folds = list(splitter.split(np.empty((len(months), 1)), groups=months))
        return folds or None

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
            np.asarray(self._pipeline.named_steps["ridge"].coef_, dtype="float64")
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
        alpha = (self.best_params_ or {}).get("alpha", self.alpha)
        return (
            f"{self.name}: ridge alpha={alpha:g}, "
            f"{len(self._feature_columns)} features"
        )
