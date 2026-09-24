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
from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd

from src.data.data import Dataset
from src.models_code.exceptions import NotFittedError


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
