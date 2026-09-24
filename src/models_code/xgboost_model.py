"""Gradient boosted trees, kept deliberately small.

The model that can find things a linear fit cannot: a threshold effect, an
interaction between spend and balance, a rule that only applies to users with
many accounts. Those are plausible here, which is why it earns a place. A model
that cannot be explained to a stakeholder is a liability in a bank, so the bar is
not "does it fit" but "does it beat the linear model by enough to be worth the
loss of explanation".

The defaults below are shallow trees, a low learning rate and subsampling --
the shape that generalises on a few thousand rows. They are named constructor
arguments rather than a module-level dict, so there is exactly one place to read
them from, and the config carries only what a given run overrides.

**No preprocessing.** XGBoost handles missing values natively: each split learns
a default direction for NaN, which is strictly more informative than replacing
the gap with a median first. Trees split on order rather than magnitude, so
scaling changes nothing. Running the features through a pipeline anyway would
add a fitted median the model does not need and throw away the signal that a
value was missing at all -- which here means "this user is new".

**Parameters come from the config or from Optuna.** Every value, including
``n_estimators``, is either fixed in ``params`` or searched by the model's
``optuna`` block on the training-region folds.

**Importance is gain.** The total improvement in the loss that each feature's
splits bought. Preferred over ``weight``, which counts how often a feature was
split on and so rewards high-cardinality columns for being easy to split rather
than useful, and over ``cover``, which counts rows touched.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from xgboost import XGBRegressor

from src.data.data import Dataset
from src.models_code.anchored_model import AnchoredModel

# The loss. XGBoost's own default is reg:squarederror, and because this class
# never named an objective it was silently running it -- the worst available
# choice on this panel. Squared error weights a row by the square of its error,
# so on a target where one account moves by 500,000 and another by 500, a
# single enterprise row counts for a million times what a small one does: the
# booster was being fitted almost entirely to a handful of users.
#
# reg:absoluteerror weights every row by the size of its error instead, which
# is linear, so a whale is merely large rather than decisive.
#
# reg:pseudohubererror is the middle option, quadratic near zero and linear in
# the tail; it needs huber_slope, set at fit time to the median |training
# target|, so roughly "one typical monthly movement is still an ordinary error".
DEFAULT_OBJECTIVE = "reg:absoluteerror"

# The evaluation metric matched to each objective, so XGBoost reports the loss it minimises.
OBJECTIVE_EVAL_METRICS: dict[str, str] = {
    "reg:absoluteerror": "mae",
    "reg:squarederror": "rmse",
    "reg:pseudohubererror": "mphe",
}


class XGBoostModel(AnchoredModel):
    """Gradient boosted trees over the feature table.

    Parameters
    ----------
    name : str, optional
        Label for result tables. Default ``xgboost``.
    anchor_column : str, optional
        Column holding last month's balance.
    n_estimators : int, optional
        Boosting rounds. Default 300.
    max_depth : int, optional
        Tree depth. Default 4 -- deep enough for an interaction, shallow enough
        not to memorise five thousand rows.
    learning_rate : float, optional
        Default 0.05.
    subsample : float, optional
        Row sample per tree. Default 0.8.
    colsample_bytree : float, optional
        Column sample per tree. Default 0.8.
    min_child_weight : float, optional
        Minimum summed instance weight in a leaf. Default 5.0.
    reg_lambda : float, optional
        L2 penalty on leaf weights. Default 1.0.
    random_state : int, optional
        Default 42.
    objective : str, optional
        The loss. Default ``reg:absoluteerror``; see ``DEFAULT_OBJECTIVE``.
        Named explicitly rather than left to the library, because the library's
        default is the one choice this data cannot afford.
    eval_metric : str, optional
        Metric XGBoost reports. Defaults to the one matching ``objective``.
    n_jobs : int, optional
        Default 1. Single threaded so two runs of one config produce the same
        numbers: tree building is order-dependent across threads, and a model
        comparison that shifts between runs is one nobody can act on.
    clip, market_scale, recency_half_life : optional
        Training-target treatment, see ``AnchoredModel``.
    **params : Any
        Anything else, passed straight to ``XGBRegressor``. ``huber_slope``,
        when omitted under ``reg:pseudohubererror``, is set at fit time to the
        median absolute training target, so "one typical movement" is where
        the loss turns linear whatever axis the target is on.
    """

    def __init__(
        self,
        name: str = "xgboost",
        anchor_column: str = "prev_1m_closing_balance_usd",
        objective: str = DEFAULT_OBJECTIVE,
        eval_metric: str | None = None,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 5.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = 1,
        clip: float | str | None = None,
        market_scale: bool = False,
        recency_half_life: float | None = None,
        **params: Any,
    ) -> None:
        super().__init__(
            name,
            anchor_column=anchor_column,
            clip=clip,
            market_scale=market_scale,
            recency_half_life=recency_half_life,
        )
        self.random_state = random_state
        self.params: dict[str, Any] = {
            "objective": objective,
            "eval_metric": eval_metric or OBJECTIVE_EVAL_METRICS.get(objective, "rmse"),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "n_jobs": n_jobs,
            **params,
        }
        self._estimator: XGBRegressor | None = None

    def _build(self, **overrides: Any) -> XGBRegressor:
        """Return an unfitted regressor with the current parameters.

        Parameters
        ----------
        **overrides : Any
            Applied on top of the stored parameters.

        Returns
        -------
        xgboost.XGBRegressor
            Unfitted.
        """
        return XGBRegressor(**{**self.params, **overrides})

    def _fit(self, dataset: Dataset) -> None:
        """Fit the booster on every usable training row.

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
        weights = self._training_weights(dataset)
        if weights is not None:
            weights = weights[usable]

        if (
            self.params["objective"] == "reg:pseudohubererror"
            and "huber_slope" not in self.params
        ):
            self.params["huber_slope"] = max(float(np.median(np.abs(values))), 1e-9)

        self._estimator = self._build()
        self._estimator.fit(features, values, sample_weight=weights)

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
        assert self._estimator is not None  # guaranteed by Model.predict
        # The columns this model was fitted on; see RidgeRegression._predict.
        predicted = np.asarray(
            self._estimator.predict(dataset.frame.loc[:, list(self.feature_columns)]),
            dtype="float64",
        )
        return self._restore_level(dataset, predicted)

    def feature_importance(self, importance_type: str = "gain") -> pd.Series | None:
        """Return the booster's importance per feature, largest first.

        Parameters
        ----------
        importance_type : str, optional
            Any type XGBoost's ``get_score`` accepts: ``gain`` (average loss
            reduction per split), ``total_gain`` (summed over every split),
            ``weight`` (number of splits), ``cover`` (average rows reached per
            split) or ``total_cover``. Default ``gain``.

        Returns
        -------
        pandas.Series or None
            Value per feature name, descending, or None before fitting.
            Features the booster never split on appear at zero rather than
            being dropped, so the series always covers every input column --
            which is what makes it an answer to "which columns carry the model"
            rather than only a list of the ones that do.
        """
        if not self.is_fitted or self._estimator is None:
            return None

        scores = self._estimator.get_booster().get_score(importance_type=importance_type)
        values = pd.Series(
            {name: float(value) for name, value in scores.items()}, dtype="float64"
        )
        complete = values.reindex(list(self._feature_columns), fill_value=0.0)
        complete.name = importance_type
        return complete.sort_values(ascending=False)

    def describe(self) -> str:
        """Return a one-line description for the run log.

        Returns
        -------
        str
            Shape of the booster and how many features it used.
        """
        if not self.is_fitted:
            return f"{self.name} (not fitted)"
        importance = self.feature_importance()
        used = int((importance > 0).sum()) if importance is not None else 0
        return (
            f"{self.name}: {self.params['n_estimators']} trees, depth "
            f"{self.params['max_depth']}, lr {self.params['learning_rate']:g}, "
            f"split on {used} of "
            f"{len(self._feature_columns)} features"
        )
