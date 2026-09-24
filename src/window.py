"""Time-respecting splits for a panel of monthly, per-entity observations.

The data is one row per user per month, and the question being asked is the
one you would ask standing on the first day of a month: where does each user
land by the end of it. A split that shuffles rows answers a different and much
easier question, because a model trained on a shuffled split has seen months
that come after the ones it is scored on, for the very same users.

So every split here cuts along the month axis and nothing else:

* ``holdout_split`` keeps the last ``test_share`` of months as the test set and
  everything before them as train.
* ``expanding_folds`` walks that same cut backwards through the training
  region, giving several (train, test) pairs where the training window grows
  and always sits entirely before its test window.
* ``MonthlyExpandingSplit`` is the same logic wearing a scikit-learn
  cross-validator interface, so it can be handed to ``cross_val_score`` or a
  search object as ``cv=``.

Everything here is used by the *training* side only -- ``train.py``,
``optuna_search.py`` and ``features_selection.py``, all of which live entirely inside
the training region. ``test.py`` does not import this module. It asks the saved
model which months it was fitted on and scores everything strictly after that,
so the boundary between train and test is a fact recorded by the fit rather
than a calculation repeated in two places that can drift apart.

Two deliberate choices worth stating, because both are the sort of thing that
is otherwise read as an oversight:

Every entity appears in both train and test. That is intentional: the task is
to forecast the next month for users who are already known, not to generalise
to users never seen before. The split protects against seeing the future, not
against seeing the user.

``gap_months`` defaults to zero. The features are lags and rolling windows over
months strictly before the target month, so a model predicting month t uses
information that genuinely existed on the first day of month t, even when month
t-1 is in the training set. If a feature is ever added that is computed from
the target month itself, this default becomes wrong and the gap is the dial
that fixes it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.data.data import MONTH_PERIOD, Dataset
from src.month_cut import cut_positions


class Split(BaseModel):
    """One (train, test) pair, described by rows and by months.

    Carries the month boundaries alongside the row positions so a fold can be
    printed, logged and checked without going back to the frame it came from.

    Attributes
    ----------
    name : str
        Short label for logs and result tables, for example ``fold_2``.
    train_positions : numpy.ndarray of numpy.intp
        Row positions of the training rows, ascending.
    test_positions : numpy.ndarray of numpy.intp
        Row positions of the test rows, ascending.
    train_months : pandas.DatetimeIndex
        The distinct months in the training rows, ascending.
    test_months : pandas.DatetimeIndex
        The distinct months in the test rows, ascending.
    """

    # numpy arrays and DatetimeIndex carry no pydantic schema, so the model
    # checks the types and leaves the contents to check_no_future_leak.
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    train_positions: npt.NDArray[np.intp]
    test_positions: npt.NDArray[np.intp]
    train_months: pd.DatetimeIndex
    test_months: pd.DatetimeIndex

    @property
    def n_train(self) -> int:
        """Return the number of training rows.

        Returns
        -------
        int
            Size of ``train_positions``.
        """
        return int(self.train_positions.size)

    @property
    def n_test(self) -> int:
        """Return the number of test rows.

        Returns
        -------
        int
            Size of ``test_positions``.
        """
        return int(self.test_positions.size)

    def summary(self) -> str:
        """Return a one-line description of the split.

        Returns
        -------
        str
            Row counts and the month span on each side of the cut.
        """
        return (
            f"{self.name}: train {self.n_train:,} rows "
            f"({_span(self.train_months)}) -> "
            f"test {self.n_test:,} rows ({_span(self.test_months)})"
        )

    def as_indices(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """Return the split as the plain tuple scikit-learn expects.

        Returns
        -------
        tuple of numpy.ndarray
            ``(train_positions, test_positions)``.
        """
        return self.train_positions, self.test_positions


def _span(months: pd.DatetimeIndex) -> str:
    """Return a compact description of a run of months.

    Parameters
    ----------
    months : pandas.DatetimeIndex
        Months to describe, assumed ascending.

    Returns
    -------
    str
        For example ``2022-01..2024-12 (36m)``, or ``empty`` when there are
        none.
    """
    if len(months) == 0:
        return "empty"
    return f"{months[0]:%Y-%m}..{months[-1]:%Y-%m} ({len(months)}m)"


def month_values(times: pd.Series) -> pd.Series:
    """Return the month each row belongs to, normalised to the month start.

    Applied defensively: ``Dataset`` already normalises, but this function is
    also reachable from the scikit-learn splitter, which receives whatever a
    caller put in ``groups``.

    Parameters
    ----------
    times : pandas.Series
        Dates or timestamps, one per row.

    Returns
    -------
    pandas.Series
        datetime64 series of month-start timestamps, same index as ``times``.
    """
    parsed = pd.to_datetime(times)
    # Wrapped: the period round trip is typed as returning an array rather than
    # a Series, and callers index the result by row position.
    return pd.Series(
        parsed.dt.to_period(MONTH_PERIOD).dt.to_timestamp(), index=times.index
    )


def positions_in_months(
    times: pd.Series, months: pd.DatetimeIndex
) -> npt.NDArray[np.intp]:
    """Return the positions of the rows falling in a given set of months.

    Parameters
    ----------
    times : pandas.Series
        The month of every row, as returned by ``month_values``.
    months : pandas.DatetimeIndex
        Months to select.

    Returns
    -------
    numpy.ndarray of numpy.intp
        Ascending row positions. Positional, not label based, so the result
        stays valid for a frame with any index.
    """
    mask = times.isin(months).to_numpy(dtype=bool)
    return np.flatnonzero(mask).astype(np.intp)


def plan_month_cut(
    months: pd.DatetimeIndex, test_share: float, gap_months: int = 0
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split an ordered list of months into a train block and a test block.

    The last ``floor(test_share * len(months))`` months become test, and the
    ``gap_months`` before them are dropped from both sides.

    Parameters
    ----------
    months : pandas.DatetimeIndex
        All distinct months, ascending.
    test_share : float
        Share of months held out at the end, between 0 and 1.
    gap_months : int, optional
        Months to discard between the two blocks. Default 0.

    Returns
    -------
    tuple of pandas.DatetimeIndex
        ``(train_months, test_months)``, both ascending.

    Raises
    ------
    ValueError
        If the share or gap is invalid, or either block would be empty.
    """
    # Same rule as the Spark feature build, so both cut at the same month.
    train_end, test_start = cut_positions(len(months), test_share, gap_months)
    return months[:train_end], months[test_start:]


def holdout_split(
    dataset: Dataset,
    test_share: float,
    gap_months: int = 0,
    name: str = "holdout",
) -> Split:
    """Hold out the last ``test_share`` of the panel's months as the test set.

    This is the split the headline number is reported on. It mimics the real
    situation exactly: fit on everything up to a date, predict the months that
    come after it, for the users already known.

    Parameters
    ----------
    dataset : Dataset
        The full panel.
    test_share : float
        Share of months at the end to hold out, between 0 and 1.
    gap_months : int, optional
        Months discarded between train and test. Default 0, which is correct
        while every feature is a lag over strictly earlier months.
    name : str, optional
        Label recorded on the returned split. Default ``"holdout"``.

    Returns
    -------
    Split
        Row positions and month spans for both sides of the cut.

    Raises
    ------
    ValueError
        If the panel is too short for the requested cut, or if either side
        ends up with no rows.
    """
    times = month_values(dataset.times)
    train_months, held_out = plan_month_cut(dataset.months, test_share, gap_months)

    split = Split(
        name=name,
        train_positions=positions_in_months(times, train_months),
        test_positions=positions_in_months(times, held_out),
        train_months=train_months,
        test_months=held_out,
    )

    # Months can be present in the list yet carry no surviving rows, for
    # instance after rows without a target were dropped upstream.
    if split.n_train == 0 or split.n_test == 0:
        raise ValueError(
            f"Split produced an empty side ({split.summary()}); check the "
            f"month coverage of the data"
        )
    return split


def expanding_folds(
    dataset: Dataset,
    n_folds: int,
    test_months: int,
    gap_months: int = 0,
    min_train_months: int = 6,
) -> list[Split]:
    """Build expanding-window folds, oldest training window first.

    Each fold tests on ``test_months`` consecutive months and trains on
    everything before them, so the training window grows with every fold and
    never contains a month later than its own test window. This is the shape
    scikit-learn calls ``TimeSeriesSplit``, done on the month axis rather than
    the row axis: on a panel, cutting by row would slice a single month across
    the boundary and put the same month on both sides.

    Folds are generated from the end of the data backwards and then returned in
    chronological order, so that the last fold always ends at the last month
    available and it is the earlier folds that lose history.

    Parameters
    ----------
    dataset : Dataset
        The panel to fold. Pass the training portion of a holdout split, not
        the full panel, or cross-validation will read the held-out months.
    n_folds : int
        How many folds to attempt. Fewer are returned when the history runs
        out, which is a fact about the data, not an error.
    test_months : int
        Length of each fold's test window, in months.
    gap_months : int, optional
        Months discarded between each fold's train and test windows. Default 0.
    min_train_months : int, optional
        A fold whose training window is shorter than this is not produced.
        Default 6, so the earliest folds are not scored on a model fitted to a
        couple of months of history.

    Returns
    -------
    list of Split
        Chronological folds. Possibly empty when the panel is too short.

    Raises
    ------
    ValueError
        If ``n_folds`` is not positive, or the window sizes are invalid.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be at least 1, got {n_folds}")
    if test_months < 1:
        raise ValueError(f"test_months must be at least 1, got {test_months}")
    if gap_months < 0:
        raise ValueError(f"gap_months cannot be negative, got {gap_months}")

    months = dataset.months
    times = month_values(dataset.times)

    folds: list[Split] = []
    # Walk backwards: fold 0 tests on the final months, fold 1 on the ones
    # before those, and so on. Reversed at the end so the returned list reads
    # in time order.
    for step in range(n_folds):
        test_end = len(months) - step * test_months
        test_start = test_end - test_months
        train_end = test_start - gap_months
        if test_start < 0 or train_end < min_train_months:
            # Out of history. Stop rather than emit a fold trained on less
            # than the caller said they would accept.
            break

        fold_train = months[:train_end]
        fold_test = months[test_start:test_end]
        fold = Split(
            name=f"fold_{n_folds - step}",
            train_positions=positions_in_months(times, fold_train),
            test_positions=positions_in_months(times, fold_test),
            train_months=fold_train,
            test_months=fold_test,
        )
        # A month with no surviving rows produces an empty side; skip such a
        # fold instead of letting a zero-row score into the average.
        if fold.n_train and fold.n_test:
            folds.append(fold)

    folds.reverse()
    # Renamed after reversing so fold_1 is the earliest, which is how the
    # results table reads top to bottom.
    return [
        Split(
            name=f"fold_{position}",
            train_positions=fold.train_positions,
            test_positions=fold.test_positions,
            train_months=fold.train_months,
            test_months=fold.test_months,
        )
        for position, fold in enumerate(folds, start=1)
    ]


def check_no_future_leak(split: Split) -> None:
    """Assert that no training month is at or after any test month.

    Cheap enough to call on every fold, and it is the one defect that quietly
    invalidates every number produced after it, so it is checked rather than
    assumed.

    Parameters
    ----------
    split : Split
        The split to check.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the training months overlap or post-date the test months.
    """
    if len(split.train_months) == 0 or len(split.test_months) == 0:
        raise ValueError(f"{split.name} has an empty side: {split.summary()}")

    last_train = split.train_months.max()
    first_test = split.test_months.min()
    if last_train >= first_test:
        raise ValueError(
            f"{split.name} leaks: training data runs to {last_train:%Y-%m}, "
            f"which is not before the first test month {first_test:%Y-%m}"
        )

    overlap = split.train_months.intersection(split.test_months)
    if len(overlap):
        raise ValueError(
            f"{split.name} has {len(overlap)} months on both sides of the cut"
        )


class MonthlyExpandingSplit:
    """Expanding-window month folds, as a scikit-learn cross-validator.

    Implements the ``split`` / ``get_n_splits`` pair, so it can be passed
    directly as ``cv=`` to ``cross_val_score``, ``cross_validate`` or a search
    object. It exists because the stock ``TimeSeriesSplit`` folds on row
    position: on a panel where 150 rows share a month, that cuts through the
    middle of a month and puts it on both sides of the boundary.

    The month of each row is not something the splitter can infer from ``X``,
    so it is supplied through ``groups`` in the usual scikit-learn way.

    Parameters
    ----------
    n_folds : int, optional
        Number of folds to attempt. Default 4.
    test_months : int, optional
        Length of each test window, in months. Default 3.
    gap_months : int, optional
        Months discarded between train and test in every fold. Default 0.
    min_train_months : int, optional
        Shortest acceptable training window. Default 6.

    Examples
    --------
    >>> cv = MonthlyExpandingSplit(n_folds=4, test_months=3)
    >>> scores = cross_val_score(  # doctest: +SKIP
    ...     pipeline, X, y, groups=months, cv=cv, scoring="neg_root_mean_squared_error"
    ... )
    """

    def __init__(
        self,
        n_folds: int = 4,
        test_months: int = 3,
        gap_months: int = 0,
        min_train_months: int = 6,
    ) -> None:
        self.n_folds = n_folds
        self.test_months = test_months
        self.gap_months = gap_months
        self.min_train_months = min_train_months

    def _folds(self, groups: Any) -> list[Split]:
        """Build the folds for one call, from the months in ``groups``.

        Wraps the months in a throwaway ``Dataset`` so that exactly the same
        code path produces the folds here as produces them for the holdout
        report. Two implementations of "which months are train" is one more
        than this project can afford.

        Parameters
        ----------
        groups : array-like
            The month of every row, in row order.

        Returns
        -------
        list of Split
            Chronological folds.

        Raises
        ------
        ValueError
            If ``groups`` is None.
        """
        if groups is None:
            raise ValueError(
                "MonthlyExpandingSplit needs the month of each row; pass it as "
                "groups=, for example groups=dataset.times"
            )
        months = month_values(pd.Series(groups).reset_index(drop=True))
        # Only the time column matters to the fold maths, so the frame carries
        # placeholders for the other three roles.
        frame = pd.DataFrame(
            {
                "_entity": np.zeros(len(months), dtype=np.int64),
                "_month": months,
                "_target": np.zeros(len(months), dtype="float64"),
                "_feature": np.zeros(len(months), dtype="float64"),
            }
        )
        dataset = Dataset(
            frame=frame,
            target_column="_target",
            id_column="_entity",
            time_column="_month",
            feature_columns=("_feature",),
        )
        return expanding_folds(
            dataset,
            n_folds=self.n_folds,
            test_months=self.test_months,
            gap_months=self.gap_months,
            min_train_months=self.min_train_months,
        )

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
        """Yield (train, test) row positions for each fold.

        Parameters
        ----------
        X : array-like
            Feature matrix. Used only for its length.
        y : array-like, optional
            Target. Unused, accepted for interface compatibility.
        groups : array-like
            The month of every row, in row order. Required.

        Yields
        ------
        tuple of numpy.ndarray
            ``(train_positions, test_positions)`` for one fold.

        Raises
        ------
        ValueError
            If ``groups`` is missing or its length does not match ``X``.
        """
        folds = self._folds(groups)
        n_rows = len(X)
        if n_rows != len(pd.Series(groups)):
            raise ValueError(
                f"groups has {len(pd.Series(groups))} entries but X has {n_rows} rows"
            )
        for fold in folds:
            # Checked on the way out: a splitter that silently emits a leaking
            # fold is the failure this whole module exists to prevent.
            check_no_future_leak(fold)
            yield fold.as_indices()

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return how many folds this splitter will actually produce.

        The answer depends on the data, since short history yields fewer folds
        than requested, so ``groups`` is needed to answer honestly.

        Parameters
        ----------
        X : array-like, optional
            Unused, accepted for interface compatibility.
        y : array-like, optional
            Unused, accepted for interface compatibility.
        groups : array-like
            The month of every row. Required.

        Returns
        -------
        int
            Number of folds.

        Raises
        ------
        ValueError
            If ``groups`` is None.
        """
        return len(self._folds(groups))


class SplitSettings(BaseModel):
    """The ``split`` block of the config, parsed and bounds-checked.

    A typed model rather than a dict of ints, so that a negative month count or
    a misspelled key fails here with a readable message instead of producing a
    split that is quietly the wrong shape.

    Attributes
    ----------
    test_share : float
        Share of months held out at the end of the panel for the headline number.
    gap_months : int
        Months discarded between train and test. Zero is correct while every
        feature is a lag over strictly earlier months.
    cv_folds : int
        Expanding-window folds to attempt inside the training region.
    cv_test_months : int
        Length of each fold's test window, in months.
    min_train_months : int
        Shortest training window a fold is allowed to have.
    """

    # extra="forbid" so `test_month: 6` is an error rather than a typo that
    # silently leaves the default in place and changes every number reported.
    model_config = ConfigDict(extra="forbid", frozen=True)

    test_share: float = Field(default=0.2, gt=0, lt=1)
    gap_months: int = Field(default=0, ge=0)
    cv_folds: int = Field(default=4, ge=1)
    cv_test_months: int = Field(default=3, ge=1)
    min_train_months: int = Field(default=6, ge=1)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SplitSettings:
        """Build settings from a parsed config.

        Parameters
        ----------
        config : dict
            Parsed config. A missing ``split`` block gives defaults.

        Returns
        -------
        SplitSettings
            Validated settings.

        Raises
        ------
        pydantic.ValidationError
            If a value is out of bounds or an unknown key is present.
        """
        return cls.model_validate(config.get("split") or {})

    def describe(self) -> str:
        """Return a one-line description for the run log.

        Returns
        -------
        str
            The split shape, in words.
        """
        gap = f", {self.gap_months}m gap" if self.gap_months else ""
        return (
            f"holdout: last {self.test_share:.0%} of months{gap}; "
            f"cv: {self.cv_folds} expanding folds of {self.cv_test_months}m, "
            f"min train {self.min_train_months}m"
        )
