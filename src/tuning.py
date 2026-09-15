"""Hyperparameter search, over folds that respect the month axis.

Two jobs, both small.

**Turning a config block into a search space.** The space lives in the YAML next
to the model it belongs to, not in the class body. That is the whole reason this
module exists: a search space written in Python ends up as one live grid and four
commented-out ones, and nobody can tell which was last run. A space in the config
is a git diff instead.

The config shape is deliberately narrow. A list is a discrete choice; a mapping
with ``low`` and ``high`` is a continuous range, log-uniform when ``log: true``::

    space:
      max_depth:     [3, 4, 5, 6]
      learning_rate: {low: 0.01, high: 0.2, log: true}

**Running the search on the right folds.** ``RandomizedSearchCV`` defaults to
``KFold``, which on a panel would put a user's later months in the training side
of a fold and their earlier months in the test side. Every search here runs on
folds from ``MonthlyExpandingSplit``, materialised as an explicit list of index
pairs. Passing the splitter object would work too, but then ``groups`` has to
reach it through scikit-learn's metadata routing; a precomputed list needs no
routing and is easier to assert on.

Why randomised rather than exhaustive: with six axes a grid that tried four
values each would be 4096 fits per fold. A random sample of 40 covers the space
better per fit, because most of these axes barely matter and a grid spends the
same effort on all of them regardless.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.model_selection import RandomizedSearchCV

from src.splitting import MonthlyExpandingSplit

# Enough draws to cover six axes without the search costing more than the run it
# is meant to improve. Overridden per model by `search.n_iter`.
DEFAULT_N_ITER = 40

# scikit-learn scoring name. Matches the project's headline metric: on a target
# this skewed, ranking on squared error ranks models by how well they fit a
# handful of very large accounts.
DEFAULT_SCORING = "neg_median_absolute_error"


def build_search_space(space: dict[str, Any]) -> dict[str, Any]:
    """Turn the config's ``space`` block into scikit-learn distributions.

    Parameters
    ----------
    space : dict
        Parameter name to either a list of discrete values or a mapping with
        ``low`` and ``high`` (and optionally ``log``).

    Returns
    -------
    dict
        Parameter name to a list or a frozen ``scipy.stats`` distribution,
        ready to pass as ``param_distributions``.

    Raises
    ------
    ValueError
        If an entry is neither a list nor a ``low``/``high`` mapping, or if the
        range is empty.
    """
    built: dict[str, Any] = {}
    for name, spec in space.items():
        if isinstance(spec, (list, tuple)):
            if not spec:
                raise ValueError(f"Search space for {name!r} is an empty list")
            built[name] = list(spec)
            continue

        if not isinstance(spec, dict) or "low" not in spec or "high" not in spec:
            raise ValueError(
                f"Search space for {name!r} must be a list of values or a "
                f"mapping with low and high; got {spec!r}"
            )

        low, high = float(spec["low"]), float(spec["high"])
        if not low < high:
            raise ValueError(
                f"Search space for {name!r} needs low < high; got {low} and {high}"
            )

        if spec.get("log"):
            if low <= 0:
                raise ValueError(
                    f"Search space for {name!r} is log scaled, so low must be "
                    f"positive; got {low}"
                )
            built[name] = loguniform(low, high)
        else:
            # scipy's uniform takes a location and a width, not two bounds.
            built[name] = uniform(loc=low, scale=high - low)

    return built


def month_folds(
    times: pd.Series,
    n_folds: int,
    test_months: int,
    gap_months: int = 0,
    min_train_months: int = 6,
) -> list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
    """Return expanding-window folds as an explicit list of index pairs.

    Parameters
    ----------
    times : pandas.Series
        The month of every row, in row order.
    n_folds : int
        Folds to attempt. Short history yields fewer.
    test_months : int
        Length of each fold's test window, in months.
    gap_months : int, optional
        Months discarded between train and test. Default 0.
    min_train_months : int, optional
        Shortest acceptable training window. Default 6.

    Returns
    -------
    list of (numpy.ndarray, numpy.ndarray)
        ``(train_positions, test_positions)`` per fold, oldest first. Ready to
        pass as ``cv=``.
    """
    splitter = MonthlyExpandingSplit(
        n_folds=n_folds,
        test_months=test_months,
        gap_months=gap_months,
        min_train_months=min_train_months,
    )
    # X is used only for its length, and split() re-checks every fold for leak
    # on the way out.
    return list(splitter.split(np.empty((len(times), 1)), groups=times))


def run_search(
    estimator: Any,
    features: pd.DataFrame,
    target: npt.NDArray[np.float64],
    folds: list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]],
    search: dict[str, Any],
    random_state: int = 42,
) -> tuple[dict[str, Any], float]:
    """Run a randomised search and return the winning parameters.

    Parameters
    ----------
    estimator : sklearn estimator
        Unfitted. Not modified; the search clones it.
    features : pandas.DataFrame
        Training features, already reduced to the rows with a usable target.
    target : numpy.ndarray of float
        Training target, aligned to ``features``.
    folds : list of (numpy.ndarray, numpy.ndarray)
        Output of ``month_folds``, indexing into ``features``.
    search : dict
        The spec's ``search`` block: ``space``, and optionally ``n_iter`` and
        ``scoring``.
    random_state : int, optional
        Seed for the draw, so two runs of one config sample the same
        candidates. Default 42.

    Returns
    -------
    tuple of (dict, float)
        Best parameters, and the cross-validated score they earned.

    Raises
    ------
    ValueError
        If the search block has no ``space``, or ``folds`` is empty.
    """
    space = search.get("space")
    if not space:
        raise ValueError("A search block needs a non-empty `space`")
    if not folds:
        raise ValueError(
            "No folds were produced, so there is nothing to search over. Check "
            "cv_folds and min_train_months against the length of the panel."
        )

    searcher = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=build_search_space(space),
        n_iter=int(search.get("n_iter", DEFAULT_N_ITER)),
        scoring=search.get("scoring", DEFAULT_SCORING),
        cv=folds,
        random_state=random_state,
        # refit=False: the caller refits on its own terms -- for the booster
        # that means refitting on every training month with the round count
        # early stopping found, which this object has no way to know about.
        refit=False,
        # Single process, measured rather than assumed. On roughly 5,000 rows by
        # 21 columns one fit takes about a second, which is far too little to
        # amortise spawning a worker and pickling the data to it: n_jobs=-1
        # measured *slower* than n_jobs=1 over six candidates (17.2s against
        # 15.1s). It also scored differently, because XGBoost's tree building is
        # order-dependent across threads, and on Windows it killed the loky
        # workers outright with a TerminatedWorkerError partway through the run.
        # The booster itself is single threaded for the same determinism reason.
        n_jobs=1,
        # A candidate that fails scores worst instead of killing the run. A
        # search that dies on candidate 31 of 40 has wasted the other 30.
        error_score=float("nan"),
    )
    searcher.fit(features, target)

    # numpy scalars out of the distributions repr as "np.float64(0.81...)" in
    # the run summary and are not JSON serialisable without a fallback. Convert
    # once here so every consumer downstream sees plain Python numbers.
    best = {name: _plain(value) for name, value in searcher.best_params_.items()}
    return best, float(searcher.best_score_)


def _plain(value: Any) -> Any:
    """Return a numpy scalar as its Python equivalent, other values unchanged.

    Parameters
    ----------
    value : Any
        A parameter value from a search result.

    Returns
    -------
    Any
        ``int``, ``float`` or the original object.
    """
    if isinstance(value, np.generic):
        return value.item()
    return value
