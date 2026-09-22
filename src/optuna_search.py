"""Optuna search over model settings *and* training-target treatment.

``tuning.py`` searches estimator hyperparameters with ``RandomizedSearchCV``,
which can only vary what sits inside an sklearn estimator. The settings worth
searching on this panel mostly do not: the cap on the training movement, the
market scale for the 2024-07 regime shift and the recency weighting all change
the *target a model is fitted to*, and have to be re-estimated inside every
fold. Here each trial builds a whole model from a parameter dict and scores it
on the same expanding month folds ``train.py`` cross-validates on, so anything
a model constructor accepts can be searched.

**The objective is a ratio against persistence, fold by fold.** Each fold's
MAE is divided by persistence's MAE on the same rows, and the ratios are
averaged. Below 1.0 beats persistence. A ratio rather than raw dollars because
the regime shift multiplies the typical movement by about five: in dollars the
post-shift folds would decide the search on size alone, and a ratio says how
much of the *achievable* error a trial removed in each period.

**Later folds can count for more.** ``fold_weights: linear`` weights fold k of
n by k, so the last fold -- the one nearest the holdout, and on this panel the
only one wholly after the shift -- carries the most. ``equal`` weights them
the same.

**The search overfits the folds.** It is choosing among many configurations on
five windows, and the best trial's score is optimistic for exactly that reason.
Two checks keep it honest: the winner is refitted on the ``seed_check`` seeds
to show how much of its lead is seed noise, and the holdout -- which no trial
reads -- is scored once, by ``src.test``, on the final model only.

The space lives in the config next to the model it tunes, in the same shape as
``tuning.py``'s: a list is a categorical choice (``null`` allowed), a mapping
with ``low`` and ``high`` is a range, integer when both bounds are integers,
log-uniform when ``log: true``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import optuna
from loguru import logger

from src.data.data import Dataset
from src.models_code.base_class import Model
from src.window import Split

DEFAULT_N_TRIALS = 50
DEFAULT_FOLD_WEIGHTS = "linear"
DEFAULT_STORAGE = "optuna_study.db"
DEFAULT_SEED_CHECK: tuple[int, ...] = (1, 2, 3)

# Builds an unfitted model from a complete parameter dict, and narrows a
# dataset to the columns that model's family reads. Passed in by train.py so
# this module does not import it back.
ModelBuilder = Callable[[dict[str, Any]], Model]
DatasetNarrower = Callable[[Dataset], Dataset]


def fold_weights(n_folds: int, scheme: str) -> npt.NDArray[np.float64]:
    """Return the weight of each fold, oldest first, summing to one.

    Parameters
    ----------
    n_folds : int
        Number of folds.
    scheme : str
        ``linear`` (fold k weighs k) or ``equal``.

    Returns
    -------
    numpy.ndarray of float
        Normalised weights.

    Raises
    ------
    ValueError
        If the scheme is unknown.
    """
    if scheme == "linear":
        raw = np.arange(1, n_folds + 1, dtype="float64")
    elif scheme == "equal":
        raw = np.ones(n_folds, dtype="float64")
    else:
        raise ValueError(f"optuna.fold_weights must be linear or equal, got {scheme!r}")
    return raw / raw.sum()


def suggest(trial: optuna.Trial, space: dict[str, Any]) -> dict[str, Any]:
    """Sample one value per entry of a config search space.

    Parameters
    ----------
    trial : optuna.Trial
        The trial to sample for.
    space : dict
        Parameter name to a list of choices or a ``low``/``high`` mapping.

    Returns
    -------
    dict
        Parameter name to the sampled value.

    Raises
    ------
    ValueError
        If an entry is neither shape, or a range is empty.
    """
    sampled: dict[str, Any] = {}
    for name, spec in space.items():
        if isinstance(spec, (list, tuple)):
            if not spec:
                raise ValueError(f"Optuna space for {name!r} is an empty list")
            sampled[name] = trial.suggest_categorical(name, list(spec))
            continue
        if not isinstance(spec, dict) or "low" not in spec or "high" not in spec:
            raise ValueError(
                f"Optuna space for {name!r} must be a list or a low/high mapping; "
                f"got {spec!r}"
            )
        low, high, log = spec["low"], spec["high"], bool(spec.get("log", False))
        if not low < high:
            raise ValueError(f"Optuna space for {name!r} needs low < high")
        if isinstance(low, int) and isinstance(high, int):
            sampled[name] = trial.suggest_int(name, low, high, log=log)
        else:
            sampled[name] = trial.suggest_float(name, float(low), float(high), log=log)
    return sampled


def fold_ratios(
    build: ModelBuilder,
    params: dict[str, Any],
    narrow: DatasetNarrower,
    train: Dataset,
    folds: list[Split],
    reference_mae: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fit a fresh model per fold and return its MAE and ratio to persistence.

    Parameters
    ----------
    build : Callable
        Builds an unfitted model from ``params``.
    params : dict
        Complete constructor arguments.
    narrow : Callable
        Narrows a dataset to this model family's columns.
    train : Dataset
        The training region the folds index into.
    folds : list of Split
        Expanding month folds.
    reference_mae : numpy.ndarray of float
        Persistence MAE per fold.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        MAE per fold, and MAE over ``reference_mae`` per fold.
    """
    maes = np.empty(len(folds), dtype="float64")
    for index, fold in enumerate(folds):
        fit_rows = narrow(train.take(fold.train_positions))
        validate_rows = train.take(fold.test_positions)
        model = build(params)
        model.fit(fit_rows)
        prediction = model.predict(validate_rows)
        truth = validate_rows.target.to_numpy(dtype="float64")
        maes[index] = float(np.mean(np.abs(truth - prediction)))
    return maes, maes / reference_mae


def reference_fold_mae(
    reference: Callable[[], Model], train: Dataset, folds: list[Split]
) -> npt.NDArray[np.float64]:
    """Return the reference model's MAE on every fold.

    Parameters
    ----------
    reference : Callable
        Zero-argument factory for the reference model (persistence).
    train : Dataset
        The training region.
    folds : list of Split
        Expanding month folds.

    Returns
    -------
    numpy.ndarray of float
        MAE per fold.

    Raises
    ------
    ValueError
        If the reference scores zero on a fold, which would make every ratio
        infinite.
    """
    maes = np.empty(len(folds), dtype="float64")
    for index, fold in enumerate(folds):
        model = reference()
        model.fit(train.take(fold.train_positions))
        validate_rows = train.take(fold.test_positions)
        truth = validate_rows.target.to_numpy(dtype="float64")
        maes[index] = float(np.mean(np.abs(truth - model.predict(validate_rows))))
    if np.any(maes <= 0):
        raise ValueError(f"Reference MAE is zero on a fold: {maes}")
    return maes


def run_study(
    name: str,
    space: dict[str, Any],
    base_params: dict[str, Any],
    build: ModelBuilder,
    narrow: DatasetNarrower,
    train: Dataset,
    folds: list[Split],
    reference_mae: npt.NDArray[np.float64],
    settings: dict[str, Any],
    output_dir: Path,
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one model's study and return its winning parameters and a record.

    Parameters
    ----------
    name : str
        Study name, unique per table and model. A previous study of this name
        in the same storage is replaced, so the file always holds the latest
        run of each.
    space : dict
        The model spec's ``optuna`` block.
    base_params : dict
        The spec's fixed ``params``; sampled values override them.
    build : Callable
        Builds an unfitted model from a complete parameter dict.
    narrow : Callable
        Narrows a dataset to this model family's columns.
    train : Dataset
        The training region. The holdout is not in it.
    folds : list of Split
        Expanding month folds inside ``train``.
    reference_mae : numpy.ndarray of float
        Persistence MAE per fold.
    settings : dict
        The config's ``optuna`` block.
    output_dir : pathlib.Path
        Where the study storage lives.
    random_state : int, optional
        Sampler seed, so two runs of one config draw the same trials.

    Returns
    -------
    tuple of (dict, dict)
        The winning complete parameter dict, and a JSON-ready record of the
        study: best trial, fold scores, seed check and trial counts.

    Raises
    ------
    RuntimeError
        If every trial failed.
    """
    n_trials = int(settings.get("n_trials", DEFAULT_N_TRIALS))
    scheme = str(settings.get("fold_weights", DEFAULT_FOLD_WEIGHTS))
    weights = fold_weights(len(folds), scheme)
    storage_path = output_dir / str(settings.get("storage", DEFAULT_STORAGE))
    storage = f"sqlite:///{storage_path.as_posix()}"

    # Optuna's own INFO lines (one per trial) would bury the run log; the
    # summary train.py prints and the storage file carry everything they said.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        optuna.delete_study(study_name=name, storage=storage)
    except KeyError:
        pass
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )

    def objective(trial: optuna.Trial) -> float:
        params = {**base_params, **suggest(trial, space)}
        try:
            maes, ratios = fold_ratios(build, params, narrow, train, folds, reference_mae)
        except ValueError as error:
            # An incompatible pair the space allows (clip on a level target,
            # say). Recorded as pruned so the study shows it, not fatal.
            trial.set_user_attr("error", str(error))
            raise optuna.TrialPruned(str(error)) from error
        trial.set_user_attr("fold_mae", maes.round(2).tolist())
        trial.set_user_attr("fold_ratio", ratios.round(4).tolist())
        value = float(np.dot(weights, ratios))
        logger.debug(f"  {name} trial {trial.number}: {value:.4f}  {trial.params}")
        return value

    study.optimize(objective, n_trials=n_trials)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f"{name}: every Optuna trial failed")
    best = study.best_trial
    best_params = {**base_params, **best.params}

    # The winner on other seeds, over the same folds. Ridge is deterministic
    # and repeats itself exactly; the booster's subsampling does not.
    seeds = [int(seed) for seed in settings.get("seed_check", DEFAULT_SEED_CHECK)]
    seed_scores: dict[str, float] = {}
    for seed in seeds:
        _, ratios = fold_ratios(
            build, {**best_params, "random_state": seed}, narrow, train, folds, reference_mae
        )
        seed_scores[str(seed)] = round(float(np.dot(weights, ratios)), 4)

    record = {
        "study": name,
        "storage": storage_path.name,
        "objective": f"{scheme}-weighted mean over folds of MAE / persistence MAE",
        "best_value": round(float(best.value or 0.0), 4),
        "best_trial": best.number,
        "best_params": best.params,
        "fold_ratio": best.user_attrs.get("fold_ratio"),
        "fold_mae": best.user_attrs.get("fold_mae"),
        "reference_fold_mae": reference_mae.round(2).tolist(),
        "fold_weights": weights.round(4).tolist(),
        "seed_check": seed_scores,
        "n_trials": len(study.trials),
        "n_complete": len(completed),
        "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
    }
    return best_params, record
