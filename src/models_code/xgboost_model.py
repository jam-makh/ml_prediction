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

**Two things are chosen rather than guessed.** ``n_estimators`` comes from early
stopping, not from the search: searching a parameter with a monotone best answer
wastes draws the other six axes could use. It is stopped once per fold and the
median count kept, because a single fold's window is 450 rows and stopping on
one of them chose a single round -- a model that predicts the average movement
and nothing else. Everything else comes from a randomised search over the same
expanding month folds. Both are skipped when the config gives no ``search``
block, which makes a fixed-parameter run a one-line config change.

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
from src.models_code.base_class import AnchoredModel, TargetMode

# Upper bound while early stopping is running, not a target. High enough that a
# low learning rate can still converge; the stopper decides where to stop.
EARLY_STOPPING_CAP = 2_000

# Rounds without improvement before the fit gives up. Fifty is loose enough to
# ride out the noise of a three-month eval window.
EARLY_STOPPING_ROUNDS = 50

# The loss. XGBoost's own default is reg:squarederror, and because this class
# never named an objective it was silently running it -- the worst available
# choice on this panel. Squared error weights a row by the square of its error,
# so on a target where one account moves by 500,000 and another by 500, a
# single enterprise row counts for a million times what a small one does: the
# booster was being fitted almost entirely to a handful of users.
#
# reg:absoluteerror weights every row by the size of its error instead, which
# is linear, so a whale is merely large rather than decisive. It pairs with the
# scaled target rather than substituting for it -- scaling makes the rows
# comparable, the objective stops the remaining outliers from dominating what
# is left.
#
# reg:pseudohubererror is the middle option, quadratic near zero and linear in
# the tail; it needs huber_slope, which is on the scaled axis, so roughly
# "how many typical monthly movements is still an ordinary error".
DEFAULT_OBJECTIVE = "reg:absoluteerror"

# The metric early stopping watches, matched to the objective. These have to
# agree: stopping on RMSE while fitting MAE picks the round count that is best
# for a loss the model is not minimising, and the run before this change showed
# exactly that kind of instability -- stopping rounds of 129, 20 and 1 across
# three folds.
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
    target_mode : str, optional
        ``level`` or ``change``. Default ``change``.
    anchor_column : str, optional
        Column holding last month's balance, read only in change mode.
    search : dict, optional
        The spec's ``search`` block. None means fixed parameters, no search and
        no early stopping.
    n_estimators : int, optional
        Boosting rounds when nothing is searched. Default 300. Ignored once
        early stopping has chosen a count.
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
        What early stopping watches. Defaults to the metric that matches
        ``objective``, which is almost always what is wanted -- pass it only to
        deliberately stop on something other than the loss being fitted.
    n_jobs : int, optional
        Default 1. Single threaded so two runs of one config produce the same
        numbers: tree building is order-dependent across threads, and a model
        comparison that shifts between runs is one nobody can act on.
    clip, market_scale, recency_half_life : optional
        Training-target treatment, see ``AnchoredModel``. Recency weights
        reach the final fit and the early-stopping fits, not the randomised
        search.
    **params : Any
        Anything else, passed straight to ``XGBRegressor``. ``huber_slope``,
        when omitted under ``reg:pseudohubererror``, is set at fit time to the
        median absolute training target, so "one typical movement" is where
        the loss turns linear whatever axis the target is on.

    Attributes
    ----------
    best_params_ : dict or None
        What the search settled on, or None when no search ran.
    best_iteration_ : int or None
        Rounds early stopping chose -- the median across folds -- or None when
        it did not run.
    stopping_rounds_ : list of int
        The per-fold counts that median came from. Worth reading: if they
        disagree wildly, the round count is not a stable property of this data
        and the booster is being sized by whichever quarter it happened to see.
    """

    def __init__(
        self,
        name: str = "xgboost",
        target_mode: TargetMode = "scaled_change",
        anchor_column: str = "prev_1m_closing_balance_usd",
        search: dict[str, Any] | None = None,
        objective: str = DEFAULT_OBJECTIVE,
        eval_metric: str | None = None,
        n_estimators: int = 300,
        min_estimators: int = 1,
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
            target_mode=target_mode,
            anchor_column=anchor_column,
            search=search,
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

        self.min_estimators = int(min_estimators)
        self.best_iteration_: int | None = None
        self.stopping_rounds_: list[int] = []
        self.stopping_floored_ = False
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
        """Search, early-stop, then refit on every training row.

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
        months = dataset.frame[dataset.time_column].loc[usable].reset_index(drop=True)
        weights = self._training_weights(dataset)
        if weights is not None:
            weights = weights[usable]

        if (
            self.params["objective"] == "reg:pseudohubererror"
            and "huber_slope" not in self.params
        ):
            self.params["huber_slope"] = max(float(np.median(np.abs(values))), 1e-9)

        if self.search:
            self._search_and_stop(features, values, months, weights)

        # The final fit, on every training row. When early stopping ran, the
        # round count it found is fixed here and the eval set is gone -- the
        # model that produces the reported number should see all the history
        # there is, right up to the cut.
        rounds = self.best_iteration_ or self.params["n_estimators"]
        self._estimator = self._build(
            n_estimators=rounds, early_stopping_rounds=None
        )
        self._estimator.fit(features, values, sample_weight=weights)

    def _search_and_stop(
        self,
        features: pd.DataFrame,
        values: npt.NDArray[np.float64],
        months: pd.Series,
        weights: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """Pick parameters by randomised search, then a round count by stopping.

        Parameters
        ----------
        features : pandas.DataFrame
            Training features, usable rows only.
        values : numpy.ndarray of float
            Training target, aligned to ``features``.
        months : pandas.Series
            Month per row, aligned to ``features``, index reset.
        weights : numpy.ndarray of float, optional
            Recency weights, aligned to ``features``.

        Returns
        -------
        None
        """
        # Imported here rather than at module scope: tuning imports window,
        # and a top-level import would make this module depend on the search
        # machinery just to define the class.
        from src.tuning import month_folds, run_search

        folds = month_folds(months, n_folds=3, test_months=3)
        if not folds:
            # Too little history to fold. Better a fixed-parameter model than a
            # search over one arbitrary split.
            return

        best, score = run_search(
            estimator=self._build(),
            features=features,
            target=values,
            folds=folds,
            search=self.search,
            random_state=self.random_state,
        )
        self.params.update(best)
        self.best_params_ = best
        self.search_cv_score_ = score

        # Early stopping, once per fold, and the median of the counts is kept.
        #
        # Not the last fold alone, which is what this did first: a fold's test
        # window is 450 rows over three months, and on a target where a handful
        # of users own most of the error that is far too little to stop on. It
        # showed: stopping on the last window alone chose 1 round, which is a
        # model that predicts the mean movement and nothing else -- while the
        # cross-validated score over all five folds says the booster does beat
        # a zero-change prediction. One window was not measuring what the
        # comparison was measuring. Three windows and a median is still cheap,
        # since this is one fit per fold, and it cannot be swung by one quiet
        # quarter.
        rounds: list[int] = []
        for train_idx, eval_idx in folds:
            stopper = self._build(
                n_estimators=EARLY_STOPPING_CAP,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            )
            stopper.fit(
                features.iloc[train_idx],
                values[train_idx],
                sample_weight=None if weights is None else weights[train_idx],
                eval_set=[(features.iloc[eval_idx], values[eval_idx])],
                verbose=False,
            )
            # +1 because best_iteration is a zero-based index into the rounds.
            rounds.append(int(stopper.best_iteration) + 1)

        # Early stopping on this panel collapses. Across runs it has chosen 4,
        # 5 and 6 rounds -- a booster of six stumps, which cannot represent an
        # interaction between a user's segment and their own lagged movement
        # even when one is there. The stall is real (the marginal round does
        # not improve holdout MAE) but "no further improvement" and "the model
        # is finished" are different claims, and only the first is measured.
        #
        # The floor is the second claim, stated in the config rather than
        # discovered: fit at least this many rounds, then let regularisation
        # and subsampling handle the rest. Recorded on the model so the run log
        # can say the floor bound rather than quietly pretending the search
        # chose it.
        chosen = int(np.median(rounds))
        self.stopping_rounds_ = rounds
        self.stopping_floored_ = chosen < self.min_estimators
        self.best_iteration_ = max(chosen, self.min_estimators)

    def _predict(self, dataset: Dataset) -> npt.NDArray[np.float64]:
        """Predict, putting the level back together in change mode.

        Parameters
        ----------
        dataset : Dataset
            Rows to predict for.

        Returns
        -------
        numpy.ndarray of float
            One prediction per row, on the balance scale in both modes.
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
            Shape of the booster, target mode, and how many features it used.
        """
        if not self.is_fitted:
            return f"{self.name} (not fitted)"
        importance = self.feature_importance()
        used = int((importance > 0).sum()) if importance is not None else 0
        rounds = self.best_iteration_ or self.params["n_estimators"]
        if self.stopping_floored_:
            stopped = (
                f" (early stopping wanted {int(np.median(self.stopping_rounds_))}"
                f", floored to min_estimators={self.min_estimators})"
            )
        elif self.best_iteration_:
            stopped = " (early stopped)"
        else:
            stopped = ""
        return (
            f"{self.name}: {rounds} trees{stopped}, depth "
            f"{self.params['max_depth']}, lr {self.params['learning_rate']:g}, "
            f"target={self.target_mode}, split on {used} of "
            f"{len(self._feature_columns)} features"
        )
