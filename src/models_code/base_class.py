"""The contract every model in this project implements.

One interface covers the trivial baseline and the fitted estimators alike. That
is the point: the comparison is only honest if the 3-month average is scored by
exactly the same code path as XGBoost, on exactly the same rows, through the
same split. A baseline that gets its own bespoke scoring script is a baseline
nobody can trust.

Two decisions shape this class.

**Models take a ``Dataset``, not an ``(X, y)`` pair.** The baseline needs to
find three named columns and average them; a regression needs a matrix of
numbers with the gaps filled. If the interface handed everything a bare array,
the baseline could not locate its columns and would need a separate entry
point. Passing the labelled dataset lets each model ask for what it needs.

**Each model owns its own preprocessing.** Imputation and scaling happen inside
``_fit``, on the training rows the model was handed, so there is no way to fit a
scaler on rows a model should not have seen. It also means the saved model is
the whole transformation, and prediction time cannot drift from training time.
The baseline owns nothing, because it needs nothing.

**A model saves itself.** ``save`` and ``load`` live here rather than in a
separate persistence module, because the thing worth persisting is the whole
object: preprocessing, fitted statistics, the months it was trained on, and the
columns it expects. Splitting that across a pickle and a sidecar file only
creates the question of whether the two still describe each other.

The public ``fit`` and ``predict`` are deliberately not the methods a subclass
overrides. They run the guard rails (is it fitted, are the columns present, is
the output the right shape and finite) and then call ``_fit`` and ``_predict``,
so a new model cannot forget the checks by forgetting to call ``super()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd

from src.data.data import Dataset
from src.models_code.entity_scaler import (
    DEFAULT_FLOOR_FRACTION,
    DEFAULT_MIN_ROWS,
    EntityScaler,
)

# What a regression is asked to predict. Every mode returns predictions on the
# balance scale, so all four are scored by identical metric code and are
# directly comparable in one results table. Defined here rather than in one of
# the model modules, so the two do not have to import the alias from each other.
#
#   level              the closing balance itself. On this panel the level is
#                      almost entirely last month's balance, so a model fitted
#                      on it spends its capacity rediscovering that.
#   change             the movement from last month, in dollars. Honest about
#                      what is being predicted, but a dollar is not a dollar
#                      here: the panel spans accounts whose ordinary monthly
#                      movement differs by three orders of magnitude, so a
#                      squared-error fit on this target is fitted almost
#                      entirely to the largest few accounts.
#   scaled_change      the movement divided by that user's own typical movement
#                      (see entity_scaler). Scale-free, defined on every row
#                      including zero and negative balances, and exactly
#                      invertible. The intended default.
#   signed_log_change  the movement in signed-log space. The log framing, made
#                      usable on a target that goes negative. Compresses the
#                      extremes harder than scaling does, at the cost of an
#                      inverse that is sensitive near zero.
TargetMode = Literal["level", "change", "scaled_change", "signed_log_change"]

# Modes whose training target is a movement rather than a balance. Used in
# several places to decide whether the anchor is needed at all.
CHANGE_MODES: frozenset[str] = frozenset(
    {"change", "scaled_change", "signed_log_change"}
)


def signed_log(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return ``sign(x) * log1p(|x|)``, a log that is defined for every real x.

    Plain ``log1p`` is undefined at or below -1, and closing balances on this
    panel reach zero and go negative, so the textbook
    ``log(1 + spend_t) - log(1 + spend_t-1)`` cannot be used as written. This
    is the standard substitute: it is continuous, strictly increasing, odd
    about zero, and behaves like the identity for small values and like a
    logarithm for large ones -- which is the compression the log was wanted for
    in the first place.

    Parameters
    ----------
    values : numpy.ndarray of float
        Any real values, including negatives and zero.

    Returns
    -------
    numpy.ndarray of float
        The signed log. NaN in, NaN out.
    """
    array = np.asarray(values, dtype="float64")
    return np.sign(array) * np.log1p(np.abs(array))


def inverse_signed_log(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Invert :func:`signed_log`.

    Parameters
    ----------
    values : numpy.ndarray of float
        Values on the signed-log axis.

    Returns
    -------
    numpy.ndarray of float
        Values back on the original axis. Exact to floating point, so a
        round trip through both functions returns the input.
    """
    array = np.asarray(values, dtype="float64")
    return np.sign(array) * np.expm1(np.abs(array))


class NotFittedError(RuntimeError):
    """Raised when a model is asked to predict before it has been fitted.

    A distinct exception type rather than a bare ``RuntimeError`` so a caller
    can tell this apart from a genuine failure inside a fitted model.
    """


class Model(ABC):
    """Base class for every model, from the 3-month average to XGBoost.

    Subclasses implement ``_fit`` and ``_predict`` and inherit the checks. A
    fitted instance also records what it was trained on, which is what lets the
    evaluation assert afterwards that no test month preceded the end of
    training.

    Parameters
    ----------
    name : str
        Label used in result tables, log lines and saved filenames. Should be
        unique within a run.

    Attributes
    ----------
    name : str
        As passed in.
    """

    #: Run-wide config values this class will accept from ``ModelSpec.factory``
    #: (see RUN_DEFAULTS in train.py). Empty here: a model that learns nothing
    #: from the config beyond its own params declares nothing. Subclasses widen
    #: it. Declared rather than reflected so the answer is greppable.
    SHARED_ARGUMENTS: ClassVar[tuple[str, ...]] = ()

    def __init__(self, name: str) -> None:
        self.name = name
        # Populated by fit(). Kept private so a half-built model cannot be
        # mistaken for a fitted one by a caller poking at attributes.
        self._is_fitted = False
        self._training_months: pd.DatetimeIndex = pd.DatetimeIndex([])
        self._feature_columns: tuple[str, ...] = ()
        self._target_column = ""

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted.

        Returns
        -------
        bool
            True once ``fit`` has completed without raising.
        """
        return self._is_fitted

    @property
    def training_months(self) -> pd.DatetimeIndex:
        """Return the months the model was fitted on.

        Returns
        -------
        pandas.DatetimeIndex
            Ascending months, empty before fitting.
        """
        return self._training_months

    @property
    def trained_through(self) -> pd.Timestamp | None:
        """Return the last month in the training data.

        The value evaluation checks against: a test month at or before this one
        was already seen, and a score over it is in-sample.

        Returns
        -------
        pandas.Timestamp or None
            Last training month, or None before fitting.
        """
        if len(self._training_months) == 0:
            return None
        return pd.Timestamp(self._training_months.max())

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return the feature columns present when the model was fitted.

        Recorded so a saved model can state what it expects, and so a mismatch
        against a later feature table is visible without unpickling anything.

        Returns
        -------
        tuple of str
            Column names, empty before fitting.
        """
        return self._feature_columns

    @property
    def target_column(self) -> str:
        """Return the name of the column the model was fitted to predict.

        Returns
        -------
        str
            Target column name, empty before fitting.
        """
        return self._target_column

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return the columns this model needs at prediction time.

        Defaults to the feature columns seen during fitting. A model that reads
        specific named columns, such as the 3-month average, overrides this so
        that a missing column is reported by name instead of surfacing as a
        KeyError from inside ``_predict``.

        Returns
        -------
        tuple of str
            Column names that must be present in any dataset passed to
            ``predict``.
        """
        return self._feature_columns

    @abstractmethod
    def _fit(self, dataset: Dataset) -> None:
        """Fit the model. Implemented by each subclass.

        Called by ``fit`` after the guard rails have run. Everything that
        learns a statistic, including any imputation or scaling, belongs here,
        because ``dataset`` is guaranteed to hold training rows only.

        Parameters
        ----------
        dataset : Dataset
            Training rows.

        Returns
        -------
        None
        """

    @abstractmethod
    def _predict(self, dataset: Dataset) -> npt.NDArray[np.float64]:
        """Produce predictions. Implemented by each subclass.

        Parameters
        ----------
        dataset : Dataset
            Rows to predict for.

        Returns
        -------
        numpy.ndarray of float
            One prediction per row, in row order.
        """

    def fit(self, dataset: Dataset) -> Model:
        """Fit the model on a training dataset.

        Parameters
        ----------
        dataset : Dataset
            Training rows. Must be the training side of a split, never the
            whole panel.

        Returns
        -------
        Model
            ``self``, so a fit can be chained into a predict.

        Raises
        ------
        ValueError
            If the dataset has no rows, or if a required column is absent.
        """
        if dataset.n_rows == 0:
            raise ValueError(f"{self.name}: cannot fit on an empty dataset")

        # Recorded before _fit so that required_columns is populated for any
        # subclass that derives it from the training frame.
        self._feature_columns = dataset.feature_columns
        self._target_column = dataset.target_column
        self._training_months = dataset.months

        missing = [
            column
            for column in self.required_columns
            if column not in dataset.frame.columns
        ]
        if missing:
            raise ValueError(f"{self.name}: training data is missing {missing}")

        self._fit(dataset)
        self._is_fitted = True
        return self

    def predict(self, dataset: Dataset) -> npt.NDArray[np.float64]:
        """Predict for a dataset, checking the result before returning it.

        Parameters
        ----------
        dataset : Dataset
            Rows to predict for. Typically the test side of a split.

        Returns
        -------
        numpy.ndarray of float
            One prediction per row, in row order.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        ValueError
            If a required column is absent, or if the subclass returned the
            wrong number of predictions or a non-finite one.
        """
        if not self._is_fitted:
            raise NotFittedError(f"{self.name}: fit() before predict()")

        missing = [
            column
            for column in self.required_columns
            if column not in dataset.frame.columns
        ]
        if missing:
            raise ValueError(f"{self.name}: prediction data is missing {missing}")

        predictions = np.asarray(self._predict(dataset), dtype="float64")

        # A length mismatch would silently misalign every prediction against
        # its true value and score a model on the wrong rows.
        if predictions.shape != (dataset.n_rows,):
            raise ValueError(
                f"{self.name}: expected {dataset.n_rows} predictions, got "
                f"shape {predictions.shape}"
            )

        # A NaN prediction poisons the mean of any metric computed over it, and
        # it is easier to explain here than three steps later in a score table.
        non_finite = int((~np.isfinite(predictions)).sum())
        if non_finite:
            raise ValueError(
                f"{self.name}: produced {non_finite} non-finite predictions"
            )
        return predictions

    def save(self, path: str | Path) -> Path:
        """Write the fitted model to a single file.

        The whole object is pickled, not just the estimator inside it. Because
        each model owns its own preprocessing -- the imputer, the per-entity
        scaler, the anchor arithmetic that turns a predicted movement back into
        a balance -- this one file is the entire transformation from a raw
        feature row to a prediction. There is no second step to reconstruct,
        and so no way for prediction time to drift away from training time.

        It also carries what the model was trained on. ``trained_through`` and
        ``feature_columns`` are attributes of the pickled object, which is what
        lets ``test.py`` derive the test region from the model itself instead
        of recomputing a cut-off from the config and hoping the two agree.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file. Parent directories are created if absent.

        Returns
        -------
        pathlib.Path
            Where it was written.

        Raises
        ------
        NotFittedError
            If the model has not been fitted. An unfitted model on disk is a
            trap: it loads without complaint and refuses at predict time, three
            steps from the thing that actually went wrong.
        """
        if not self._is_fitted:
            raise NotFittedError(f"{self.name}: fit() before save()")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> Model:
        """Read a fitted model back from a file written by ``save``.

        Parameters
        ----------
        path : str or pathlib.Path
            The file to read.

        Returns
        -------
        Model
            The fitted model, ready to predict.

        Raises
        ------
        FileNotFoundError
            If the file is absent, with the path named, since the usual cause
            is that ``train.py`` has not been run yet.
        TypeError
            If the file does not hold a model of this project.
        """
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(
                f"No saved model at {source}. Run `python -m src.train` first."
            )
        loaded = joblib.load(source)
        if not isinstance(loaded, Model):
            raise TypeError(
                f"{source} holds a {type(loaded).__name__}, not a Model"
            )
        return loaded

    def feature_importance(self) -> pd.Series | None:
        """Return per-feature importance, when the model family has one.

        Defaults to None. The baselines have no features to rank, and a model
        that has them overrides this. Used by the feature work in step 4, which
        asks which columns actually carry the model.

        Returns
        -------
        pandas.Series or None
            Importance per feature name, descending, or None.
        """
        return None

    def describe(self) -> str:
        """Return a one-line description for the run log.

        Returns
        -------
        str
            The model name, and what it was trained on once it is fitted.
        """
        if not self._is_fitted:
            return f"{self.name} (not fitted)"
        through = self.trained_through
        return (
            f"{self.name}: fitted on {len(self._training_months)} months "
            f"through {through:%Y-%m}"
            if through is not None
            else f"{self.name}: fitted"
        )

    def __repr__(self) -> str:
        """Return an unambiguous representation.

        Returns
        -------
        str
            Class name, model name and fitted state.
        """
        state = "fitted" if self._is_fitted else "unfitted"
        return f"{type(self).__name__}(name={self.name!r}, {state})"


class AnchoredModel(Model):
    """A regression that can be fitted to the movement rather than the level.

    Everything the linear model and the booster share. Both of them face the
    same three problems -- deciding what to regress on, making sure the anchor
    column survives feature selection, and putting the level back together
    afterwards -- and before this class existed both solved them with the same
    code, copied.

    Subclasses implement ``_fit`` and ``_predict`` as usual. A ``_predict`` in
    this family ends with ``return self._restore_level(dataset, predicted)``,
    which is a no-op in level mode.

    Parameters
    ----------
    name : str
        Label used in result tables and saved filenames.
    target_mode : str, optional
        ``level`` or ``change``. Default ``change``.
    anchor_column : str, optional
        Column holding last month's balance. Read only in change mode.
    search : dict, optional
        The model spec's ``search`` block, or None for fixed parameters. Held
        here so every tunable model reports its result the same way; what to do
        with it is the subclass's business.

    Attributes
    ----------
    best_params_ : dict or None
        Parameters the search settled on, or None when no search ran.
    search_cv_score_ : float or None
        The search's best cross-validated score, on the scoring metric named in
        the search block.

    Raises
    ------
    ValueError
        If ``target_mode`` is not recognised.
    """

    # Both regressions need the seed and the anchor, and both should take them
    # from the run rather than restating them per model.
    SHARED_ARGUMENTS: ClassVar[tuple[str, ...]] = ("random_state", "anchor_column")

    def __init__(
        self,
        name: str,
        target_mode: TargetMode = "scaled_change",
        anchor_column: str = "prev_1m_closing_balance_usd",
        search: dict[str, Any] | None = None,
        scale_floor_fraction: float = DEFAULT_FLOOR_FRACTION,
        scale_min_rows: int = DEFAULT_MIN_ROWS,
    ) -> None:
        super().__init__(name)
        allowed = ("level", *sorted(CHANGE_MODES))
        if target_mode not in allowed:
            raise ValueError(
                f"Unknown target_mode {target_mode!r}; expected one of {allowed}"
            )
        self.target_mode: TargetMode = target_mode
        self.anchor_column = anchor_column
        self.search = search or {}
        self.scale_floor_fraction = float(scale_floor_fraction)
        self.scale_min_rows = int(scale_min_rows)

        self.best_params_: dict[str, Any] | None = None
        self.search_cv_score_: float | None = None

        # Fitted in _training_target, which is reached only from _fit, so this
        # can never see a row the model was not handed for training.
        self.scaler_: EntityScaler | None = None

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return the columns needed at prediction time.

        Change mode reads the anchor as well as the features, and the anchor is
        usually also a feature, so the two lists are merged rather than
        concatenated.

        Returns
        -------
        tuple of str
            Feature columns, plus the anchor column in change mode.
        """
        if (
            self.target_mode in CHANGE_MODES
            and self.anchor_column not in self._feature_columns
        ):
            return (*self._feature_columns, self.anchor_column)
        return self._feature_columns

    def _training_target(self, dataset: Dataset) -> pd.Series:
        """Return what the estimator should be fitted against.

        Parameters
        ----------
        dataset : Dataset
            Training rows.

        Returns
        -------
        pandas.Series
            The closing balance in level mode; otherwise the movement from
            last month, on the axis this mode fits. A user's first month has no
            anchor, so its movement is NaN; ``_fit`` drops those rows rather
            than inventing an anchor, which would invent a movement to learn
            from.

        Raises
        ------
        KeyError
            If a change mode is configured and the anchor column is absent.

        Notes
        -----
        ``scaled_change`` fits the per-entity scaler here, as a side effect.
        That is deliberate and safe: this method is called from ``_fit`` and
        nowhere else, so the scaler only ever sees training rows, and every
        cross-validation fold gets a freshly built model and therefore a
        freshly fitted scaler. Fitting it in ``build_dataset`` instead would
        estimate every user's scale over the whole panel, holdout included.
        """
        if self.target_mode == "level":
            return dataset.target
        if self.anchor_column not in dataset.frame.columns:
            raise KeyError(
                f"{self.name}: {self.target_mode} needs {self.anchor_column!r}, "
                f"which is not in the data"
            )

        anchor = dataset.frame[self.anchor_column]
        if self.target_mode == "signed_log_change":
            return pd.Series(
                signed_log(dataset.target.to_numpy(dtype="float64"))
                - signed_log(anchor.to_numpy(dtype="float64")),
                index=dataset.target.index,
            )

        change = dataset.target - anchor
        if self.target_mode == "change":
            return change

        entities = dataset.frame[dataset.id_column]
        self.scaler_ = EntityScaler(
            floor_fraction=self.scale_floor_fraction,
            min_rows=self.scale_min_rows,
        ).fit(entities, change)
        return pd.Series(self.scaler_.transform(entities, change), index=change.index)

    def _restore_level(
        self, dataset: Dataset, predicted: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Put a change-mode prediction back on the balance scale.

        Parameters
        ----------
        dataset : Dataset
            The rows that were predicted.
        predicted : numpy.ndarray of float
            Raw estimator output: a balance in level mode, a movement in change
            mode.

        Returns
        -------
        numpy.ndarray of float
            Predictions on the balance scale in every mode. A row with no
            anchor keeps the raw prediction rather than becoming NaN, which
            would fail the finiteness check in ``predict``.

        Raises
        ------
        RuntimeError
            If scaled mode reaches prediction with no fitted scaler, which
            would mean ``_predict`` ran without ``_fit``.
        """
        if self.target_mode == "level":
            return predicted

        anchor = dataset.frame[self.anchor_column].to_numpy(dtype="float64")

        if self.target_mode == "signed_log_change":
            # Reconstructed in log space and inverted once, rather than
            # inverting the movement and adding it: the sum is the quantity the
            # estimator was actually fitted to predict.
            level = inverse_signed_log(signed_log(anchor) + predicted)
            return np.where(np.isfinite(anchor), level, predicted)

        if self.target_mode == "scaled_change":
            if self.scaler_ is None:
                raise RuntimeError(
                    f"{self.name}: scaled_change predicted before the scaler was fitted"
                )
            movement = self.scaler_.inverse_transform(
                dataset.frame[dataset.id_column], predicted
            )
        else:
            movement = predicted

        return np.where(np.isfinite(anchor), anchor + movement, movement)

    def _usable_rows(self, target: pd.Series) -> npt.NDArray[np.bool_]:
        """Return the mask of rows with a target the estimator can learn from.

        Parameters
        ----------
        target : pandas.Series
            Output of ``_training_target``.

        Returns
        -------
        numpy.ndarray of bool
            True where the target is present.

        Raises
        ------
        ValueError
            If no row has a usable target.
        """
        usable = target.notna().to_numpy(dtype=bool)
        if not usable.any():
            raise ValueError(f"{self.name}: no rows with a usable target")
        return usable


# A model is rebuilt from scratch for every cross-validation fold, so what gets
# passed around is a zero-argument factory rather than an instance. Reusing one
# instance across folds would carry fold 1's fitted statistics into fold 2.
ModelFactory = Callable[[], Model]
