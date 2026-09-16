"""Choose the feature set the models are fitted on, by XGBoost gain.

The rule this module implements, and the reasoning behind each half of it:

**Gain decides, not correlation.** Pearson and Spearman measure a pairwise,
largely linear association with the target, one column at a time. A booster
does not use features one at a time -- a column can be worthless alone and
decisive in a split beneath another one -- so a ranking built from correlation
ranks the wrong thing. Gain is the total loss reduction a column actually
bought across every split it appears in, on this data, in this model. That is
the ranking. Correlation appears here only as ``redundant_pairs``, an optional
pre-filter for near-duplicate columns, and it removes columns rather than
choosing them.

**R squared validates, it does not select.** Keeping whatever feature nudges
the cross-validated score upwards is a wrapper method, and a wrapper method run
to convergence fits the folds themselves: the score improves, the model does
not. So the candidate sets come from the gain ranking, which never looks at a
score, and the score is used once per candidate set, to answer a yes or no
question -- is this set still good enough.

**Parsimony breaks the tie.** Every top-k set that stays within ``tolerance``
of the full-feature model is acceptable, so the smallest of them is taken. A
set that scores marginally better with six more columns is not better; it is
the same model carrying six columns that have to be built, stored and
monitored every month from here on.

Everything runs on the training region only. The holdout months are cut off
before the first gain is read, because choosing columns is fitting: let the
held-out months influence which features exist and the holdout has been spent
before a single model is scored on it.

Run it with::

    python -m src.features_selection

It writes a JSON summary next to the models and prints the sweep. It does not
edit the config -- the chosen set is printed as a ``drop_columns`` block to
paste, so that the decision stays a reviewed edit rather than a file this
script rewrites underneath you.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.config import load_config, resolve_output_dir
from src.data.data import Dataset, build_dataset
from src.metrics import anchor_values, score
from src.models_code.xgboost_model import XGBoostModel
from src.window import Split, SplitSettings, expanding_folds, holdout_split

# Metrics where a smaller number is the better one. Everything else is read as
# higher-is-better, which at present means the two R squareds.
LOWER_IS_BETTER = frozenset({"mae", "rmse", "smape", "median_ae"})

# Defaults for the `feature_selection` block, applied key by key so a config
# that sets one of them does not have to restate the rest.
DEFAULTS: dict[str, Any] = {
    "model": "xgboost_scaled",
    "redundancy_threshold": 0.95,
    "tolerance": 0.02,
    # r2_change, not mae. This default said "mae" while the config said
    # r2_change and the docstring explained at length why mae is the wrong
    # choice here -- so the module disagreed with itself in three places and
    # only the config was right. See select_features for the measurement.
    "tolerance_metric": "r2_change",
    # THE GATE. The full-feature model must beat a no-change prediction by at
    # least this much cross-validated r2_change before any feature set is
    # returned at all.
    #
    # Without it the sweep cannot fail. It returns the smallest k within
    # tolerance of the full set, and if the full set is worthless then every k
    # is equally worthless, every k is "within tolerance", and k=1 is returned
    # with a straight face. That is exactly what the noise test is designed to
    # catch, so without a gate the noise test can never fail and is not a test.
    #
    # r2_change is the right quantity to gate on because zero has an absolute
    # meaning for it: r2_change = 0 is precisely the accuracy of assuming the
    # balance does not move, which needs no features and no model. A set that
    # cannot clear that has found nothing.
    "min_signal": 0.01,
    # Stage 3. A feature is kept only if its out-of-fold permutation importance
    # is positive on average AND its coefficient of variation across folds is
    # below this. A column that is decisive in one quarter and irrelevant the
    # next is fitted to that quarter.
    "stability_max_cv": 1.5,
    "permutation_repeats": 5,
}


def redundant_pairs(dataset: Dataset, threshold: float = 0.95) -> pd.DataFrame:
    """List feature pairs that are near-duplicates of each other.

    The optional pre-filter, and the only place correlation is read. Both
    measures are computed because they disagree on this panel and the
    disagreement carries information: Pearson can be set by a handful of very
    large balances, Spearman cannot, so a pair that only Pearson calls
    redundant is usually two columns that share an outlier rather than a
    signal. A pair is reported when *either* measure is above the threshold,
    and the reader decides.

    This does not drop anything. Dropping one of a redundant pair is a
    judgement about which of the two is the more direct measurement, and the
    gain ranking below is perfectly capable of splitting the credit between
    them rather than dropping either -- which is why this is a pre-filter to
    consult, not a step in the pipeline.

    Parameters
    ----------
    dataset : Dataset
        Training rows only. Passing the full panel would let the held-out
        months influence which columns survive.
    threshold : float, optional
        Absolute correlation at or above which a pair is reported. Default
        0.95, high enough that a reported pair is a near-duplicate rather than
        two columns that merely move together.

    Returns
    -------
    pandas.DataFrame
        One row per flagged pair with ``pearson`` and ``spearman``, strongest
        first. Empty when nothing is above the threshold.
    """
    features = dataset.features
    pearson = features.corr().abs()
    spearman = features.corr(method="spearman").abs()

    # Upper triangle only: the full matrix holds every pair twice and a
    # diagonal of ones, and neither is a finding.
    upper = np.triu(np.ones(pearson.shape), k=1).astype(bool)
    pairs = pd.DataFrame(
        {
            "pearson": pearson.where(upper).stack(),
            "spearman": spearman.where(upper).stack(),
        }
    )
    flagged = pairs[(pairs["pearson"] >= threshold) | (pairs["spearman"] >= threshold)]
    return flagged.sort_values("spearman", ascending=False)


def redundancy_filter(
    dataset: Dataset,
    pairs: pd.DataFrame,
    anchor_column: str,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Stage 1. Drop all but one member of each near-duplicate group.

    ``redundant_pairs`` reports; this decides. Which member to keep is the one
    judgement in the stage, and it is made by Spearman against the movement
    being predicted -- rank correlation rather than Pearson, because Pearson on
    this panel can be set by a handful of very large balances, and against the
    MOVEMENT rather than the level, because the level is what every one of
    these columns already predicts and ranking on it would just re-elect the
    largest column in each group.

    Run before the gain ranking rather than after it. Gain splits the credit
    between two near-identical columns, so both land mid-table and neither
    looks decisive -- a genuine signal held by a duplicated pair can be ranked
    below a single mediocre column purely for being represented twice.

    Parameters
    ----------
    dataset : Dataset
        Training rows only.
    pairs : pandas.DataFrame
        Output of ``redundant_pairs``.
    anchor_column : str
        Last month's balance, used to form the movement.

    Returns
    -------
    tuple of (tuple of str, pandas.DataFrame)
        The surviving feature columns in the dataset's original order, and one
        row per dropped column naming what it was a duplicate of and the two
        correlations that said so. Both are empty of drops when nothing was
        flagged.
    """
    features = dataset.feature_columns
    if pairs.empty:
        return tuple(features), pd.DataFrame(
            columns=["dropped", "kept_instead", "pearson", "spearman"]
        )

    # Connected components over the flagged pairs: A~B and B~C puts all three
    # in one group, which is the honest reading of "these columns are the same
    # measurement" and avoids dropping A for B while keeping C.
    parent: dict[str, str] = {name: name for name in features}

    def find(name: str) -> str:
        """Return the group representative for a column."""
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for left, right in pairs.index:
        if left in parent and right in parent:
            parent[find(left)] = find(right)

    # Spearman of each feature against the movement, which is what decides the
    # survivor within a group.
    movement = dataset.target - dataset.frame[anchor_column]
    strength = (
        dataset.features.corrwith(movement, method="spearman").abs().fillna(0.0)
    )

    groups: dict[str, list[str]] = {}
    for name in features:
        groups.setdefault(find(name), []).append(name)

    keep: set[str] = set()
    records: list[dict[str, Any]] = []
    for members in groups.values():
        winner = max(members, key=lambda name: float(strength.get(name, 0.0)))
        keep.add(winner)
        for name in members:
            if name == winner:
                continue
            pair = _pair_row(pairs, name, winner)
            records.append(
                {
                    "dropped": name,
                    "kept_instead": winner,
                    "pearson": pair.get("pearson", float("nan")),
                    "spearman": pair.get("spearman", float("nan")),
                    "rho_vs_movement": float(strength.get(name, 0.0)),
                    "winner_rho_vs_movement": float(strength.get(winner, 0.0)),
                }
            )

    survivors = tuple(name for name in features if name in keep)
    return survivors, pd.DataFrame(records)


def _pair_row(pairs: pd.DataFrame, left: str, right: str) -> dict[str, float]:
    """Return the correlations for one pair, in whichever order it is stored.

    Parameters
    ----------
    pairs : pandas.DataFrame
        Output of ``redundant_pairs``, indexed by the ordered pair.
    left, right : str
        The two column names.

    Returns
    -------
    dict of str to float
        The row as a dict, or an empty dict when the two are only indirectly
        connected -- which happens in a group of three or more, where A and C
        can share a group without A~C itself having been flagged.
    """
    for key in ((left, right), (right, left)):
        if key in pairs.index:
            row = pairs.loc[[key]].iloc[0]
            return {str(name): float(row[name]) for name in pairs.columns}
    return {}


def permutation_stability(
    train: Dataset,
    folds: list[Split],
    feature_columns: tuple[str, ...],
    params: dict[str, Any],
    anchor_column: str,
    repeats: int = 5,
) -> pd.DataFrame:
    """Stage 3. Measure each feature's importance on the VALIDATION folds.

    Gain, which drives stage 2, is read off the training data: it says how much
    loss a column removed on the rows the booster was fitted to, which a column
    that is memorising noise does extremely well. Permutation importance
    measured on a fold's held-out months asks a different question -- shuffle
    this column and see whether out-of-sample accuracy actually falls -- and a
    noise column scores zero on it however much gain it had.

    The second output is the one that matters more. Importance is computed per
    fold, so its spread across folds is available, and a feature whose
    importance swings from decisive to irrelevant between consecutive quarters
    is fitted to a quarter rather than to the panel. On financial data that is
    the common failure, and an average importance hides it completely.

    Parameters
    ----------
    train : Dataset
        The training region.
    folds : list of Split
        Expanding-window folds inside it.
    feature_columns : tuple of str
        The set to measure, normally stage 2's output.
    params : dict
        Fixed constructor arguments, so every fold fits the same booster.
    anchor_column : str
        Last month's balance, for the change framing.
    repeats : int, optional
        Shuffles per feature per fold. Default 5.

    Returns
    -------
    pandas.DataFrame
        One row per feature with ``importance_mean`` and ``importance_std``
        across folds, ``importance_cv`` (the coefficient of variation, the
        stability number), ``folds_positive`` and ``n_folds``. Sorted by mean
        importance, largest first.

    Notes
    -----
    Scored on the SCALED training target rather than on dollars. The point of
    this stage is which columns carry signal, and a dollar-denominated score
    would hand that verdict back to the largest accounts -- the bias the whole
    exercise is removing.
    """
    from sklearn.inspection import permutation_importance

    reduced = train.model_copy(update={"feature_columns": feature_columns})
    seed = int(params.get("random_state", 42))

    per_fold: list[pd.Series] = []
    for fold in folds:
        model = XGBoostModel(name="stability", search=None, **params)
        fold_train = reduced.take(fold.train_positions)
        model.fit(fold_train)

        fold_test = reduced.take(fold.test_positions)
        # The same transformation the model was fitted through, so the
        # estimator is scored on the axis it actually predicts.
        target = model._training_target(fold_test)
        usable = target.notna().to_numpy(dtype=bool)
        if not usable.any() or model._estimator is None:
            continue

        result = permutation_importance(
            model._estimator,
            fold_test.features.loc[usable],
            target.to_numpy(dtype="float64")[usable],
            n_repeats=repeats,
            random_state=seed,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        per_fold.append(
            pd.Series(result.importances_mean, index=list(feature_columns))
        )

    if not per_fold:
        raise ValueError("No fold produced a permutation importance")

    matrix = pd.concat(per_fold, axis=1)
    mean = matrix.mean(axis=1)
    std = matrix.std(axis=1)
    summary = pd.DataFrame(
        {
            "importance_mean": mean,
            "importance_std": std,
            # Coefficient of variation on the absolute mean, so a feature whose
            # mean importance is near zero reads as unstable (a large number)
            # rather than as suspiciously steady.
            "importance_cv": std / mean.abs().replace(0.0, np.nan),
            "folds_positive": (matrix > 0).sum(axis=1),
            "n_folds": matrix.shape[1],
        }
    )
    return summary.sort_values("importance_mean", ascending=False)


def gain_ranking(
    train: Dataset,
    params: dict[str, Any],
    search: dict[str, Any] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Fit the full-feature baseline and rank every column by gain.

    Steps one and two of the recipe in one call, because they are one fit: the
    baseline the tolerance is measured against and the ranking the candidates
    come from have to be the same model, or the sweep below is comparing a
    reduced set to a booster it was never derived from.

    The search runs here and only here. Re-tuning inside every candidate set
    would take the run from minutes to hours, and worse, it would let the
    hyperparameters absorb the loss from a removed column -- which is how a
    sweep ends up reporting that twelve features are as good as twenty-one when
    what it measured was a deeper tree.

    Parameters
    ----------
    train : Dataset
        The training region. Never the full panel.
    params : dict
        Constructor arguments for ``XGBoostModel``, from the model's config
        entry.
    search : dict, optional
        The spec's ``search`` block. None runs fixed parameters.

    Returns
    -------
    tuple of (pandas.Series, dict)
        Gain per feature, largest first, covering every input column including
        the ones the booster never split on; and the parameters the fitted
        model ended up with, to be held fixed across the sweep.

    Raises
    ------
    RuntimeError
        If the fitted model reports no importance, which can only happen if the
        model class stops being a booster.
    """
    model = XGBoostModel(name="gain_baseline", search=search, **params)
    model.fit(train)

    ranking = model.feature_importance()
    if ranking is None:
        raise RuntimeError("The baseline reported no feature importance")

    # Everything the search settled on, with the round count early stopping
    # chose written in, so every candidate set below is the same booster.
    # `params` carries target_mode and the anchor, which live on the base class
    # rather than in model.params; the fitted values win where the two overlap.
    settled = {**params, **model.params}
    settled["n_estimators"] = model.best_iteration_ or model.params["n_estimators"]
    return ranking, settled


def cv_scores(
    train: Dataset,
    folds: list[Split],
    feature_columns: tuple[str, ...],
    params: dict[str, Any],
    anchor_column: str,
) -> dict[str, float]:
    """Cross-validate one candidate feature set over the expanding folds.

    A fresh model per fold. Reusing one would carry fold 1's fitted booster
    into fold 2, which is the same leak as fitting a scaler before the split
    and considerably harder to see in the output.

    Parameters
    ----------
    train : Dataset
        The training region.
    folds : list of Split
        Expanding-window folds inside that region.
    feature_columns : tuple of str
        The candidate set. The frame keeps every column either way -- only the
        model's view of it narrows -- so the anchor is still readable in change
        mode even when it is not among the features.
    params : dict
        Fixed constructor arguments, from ``gain_ranking``.
    anchor_column : str
        Last month's balance, for the change framing of R squared.

    Returns
    -------
    dict of str to float
        Mean across folds of ``r2_change``, ``mae``, ``rmse`` and ``smape``,
        plus ``r2_change_std`` and the fold count. The spread is returned
        alongside the mean because a set that wins on average while swinging
        between periods has not earned the tolerance it fits inside.
    """
    reduced = train.model_copy(update={"feature_columns": feature_columns})

    rows: list[dict[str, float]] = []
    for fold in folds:
        model = XGBoostModel(name="candidate", search=None, **params)
        model.fit(reduced.take(fold.train_positions))

        fold_test = reduced.take(fold.test_positions)
        scores = score(
            fold_test.target,
            model.predict(fold_test),
            anchor=anchor_values(fold_test, anchor_column),
        )
        rows.append(
            {
                "r2_change": scores.r2_change,
                "mae": scores.mae,
                "rmse": scores.rmse,
                "smape": scores.smape,
            }
        )

    scored = pd.DataFrame(rows)
    summary = {name: float(scored[name].mean()) for name in scored.columns}
    summary["r2_change_std"] = float(scored["r2_change"].std())
    summary["folds"] = float(len(scored))
    return summary


def select_features(
    train: Dataset,
    folds: list[Split],
    ranking: pd.Series,
    params: dict[str, Any],
    anchor_column: str,
    tolerance: float = 0.02,
    metric: str = "r2_change",
    min_signal: float = 0.01,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Sweep the gain ranking and return the smallest set inside the tolerance.

    Steps three and four: take the top k features by gain for every k, refit,
    and keep the smallest k whose cross-validated ``metric`` is still within
    ``tolerance`` of the full-feature model. The full set is always inside its
    own tolerance, so this cannot fail to return an answer.

    Parameters
    ----------
    train : Dataset
        The training region.
    folds : list of Split
        Expanding-window folds inside that region.
    ranking : pandas.Series
        Gain per feature, largest first, from ``gain_ranking``.
    params : dict
        Fixed constructor arguments, held across every candidate set.
    anchor_column : str
        Last month's balance.
    tolerance : float, optional
        Allowed relative degradation against the full-feature baseline.
        Default 0.02, the loose end of the 1-2% band.
    metric : str, optional
        Which score the tolerance applies to. Default ``r2_change``, and the
        default is a measurement rather than a preference: across the full
        sweep on this panel the dollar metrics move by about 4% end to end,
        so a 2% band on one of them is half the width of the entire signal and
        the rule collapses into "take the smallest set". The R squared moves by
        a factor of nine over the same sweep, which is what gives the band
        something to discriminate with. Also accepts ``mae``, ``rmse`` and
        ``smape``; all four are reported per candidate whichever one decides.
    min_signal : float, optional
        The absolute gate. The full-feature model's cross-validated
        ``r2_change`` must exceed this before any set is returned. Default
        0.01.

        This exists because the tolerance rule alone cannot say no. It returns
        the smallest k within ``tolerance`` of the full set -- a purely
        relative test -- so if the full set is worthless, every k matches it,
        every k passes, and k=1 comes back looking like a decision. Gating on
        r2_change works because zero is not an arbitrary point for it: it is
        exactly the accuracy of predicting no movement at all, which requires
        no features. A set that cannot beat that has found nothing, and the
        honest return is the empty set.

    Returns
    -------
    tuple of (tuple of str, pandas.DataFrame)
        The chosen feature set -- EMPTY when the gate is not cleared -- and the
        full sweep: one row per k with the cross-validated scores, the feature
        added at that k, and whether it passed. The sweep is the thing to plot.

    Raises
    ------
    KeyError
        If ``metric`` is not one of the scores ``cv_scores`` returns.
    """
    ordered = list(ranking.index)

    rows: list[dict[str, Any]] = []
    for k in range(1, len(ordered) + 1):
        scores = cv_scores(train, folds, tuple(ordered[:k]), params, anchor_column)
        if metric not in scores:
            raise KeyError(
                f"Unknown tolerance metric {metric!r}; available: {sorted(scores)}"
            )
        rows.append({"n_features": k, "added": ordered[k - 1], **scores})

    sweep = pd.DataFrame(rows).set_index("n_features")
    # The last row is the full-feature set, since k runs 1..len(ordered).
    baseline = float(sweep[metric].to_numpy(dtype="float64")[-1])

    # The gate, before the tolerance band is even computed. Read off the
    # full-feature row, because that is the best this feature table can do: if
    # it cannot beat a no-change prediction, no subset of it can either.
    full_signal = float(sweep["r2_change"].to_numpy(dtype="float64")[-1])
    sweep.attrs["full_r2_change"] = full_signal
    sweep.attrs["min_signal"] = min_signal
    if not np.isfinite(full_signal) or full_signal <= min_signal:
        sweep["within_tolerance"] = False
        return (), sweep

    if metric in LOWER_IS_BETTER:
        # Smaller is better, so the band opens upwards from the baseline.
        sweep["within_tolerance"] = sweep[metric] <= baseline * (1.0 + tolerance)
    else:
        # Higher is better. abs() on the baseline so the band keeps its width
        # rather than inverting when the baseline R squared is negative.
        sweep["within_tolerance"] = (baseline - sweep[metric]) <= tolerance * abs(baseline)

    chosen_k = int(sweep.index[sweep["within_tolerance"]][0])
    return tuple(ordered[:chosen_k]), sweep


def shuffle_target(
    train: Dataset, anchor_column: str, random_state: int = 42
) -> Dataset:
    """Return the training region with the MOVEMENT shuffled across rows.

    The movement, not the level. Each row keeps its own anchor and is given
    another row's month-on-month change::

        target' = anchor + shuffle(target - anchor)

    Parameters
    ----------
    train : Dataset
        The training region.
    anchor_column : str
        Last month's balance.
    random_state : int, optional
        Seed for the permutation. Default 42.

    Returns
    -------
    Dataset
        A copy whose targets carry a permuted set of movements. The marginal
        distribution of movements is exactly preserved, every anchor is
        untouched, and the link from a row's features to its own movement --
        the only thing feature selection is entitled to find -- is destroyed.

    Notes
    -----
    Shuffling the LEVEL instead, which is the obvious first implementation, does
    not work here and fails loudly enough to be worth recording. Permuting the
    target column outright also breaks the pairing between a row's target and
    its own anchor, so the movement becomes ``other_row_balance - my_anchor``,
    whose variance is dominated by the anchor. A model then scores well on
    r2_change simply by tracking the anchor -- and measured on this panel it
    scored r2_change +0.253 on shuffled data against +0.076 on the real thing,
    a noise run beating the genuine one. That is a broken test, not a leak:
    the shuffle had manufactured a target that was easier than the real one.

    Shuffling the movement leaves the anchor exactly where it was, so the only
    thing a model can still do is predict the average movement, which is what
    r2_change scores as zero by construction.
    """
    frame = train.frame.copy()
    rng = np.random.default_rng(random_state)

    anchor = frame[anchor_column].to_numpy(dtype="float64")
    movement = frame[train.target_column].to_numpy(dtype="float64") - anchor

    # Permute only among the rows that HAVE a movement. A user's first month has
    # no anchor and therefore no movement, and letting those NaNs circulate
    # would hand a missing target to a row that had a perfectly good one --
    # quietly shrinking the shuffled training set and making the noise run a
    # comparison against a different number of rows.
    finite = np.isfinite(movement)
    shuffled = movement.copy()
    shuffled[finite] = movement[finite][rng.permutation(int(finite.sum()))]

    frame[train.target_column] = anchor + shuffled
    return train.model_copy(update={"frame": frame})


def noise_test(
    train: Dataset,
    folds: list[Split],
    ranking: pd.Series,
    params: dict[str, Any],
    anchor_column: str,
    tolerance: float,
    metric: str,
    min_signal: float,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run the whole selection against a shuffled target and expect nothing.

    The sanity check the brief asks for, and the only part of this module that
    can tell you the rest of it is wrong. Everything else here measures which
    features are best; this measures whether the machinery is capable of
    reporting that there are none.

    Shuffle the target across time, re-run the sweep, and require that zero
    features are selected. A pipeline that still picks features out of noise is
    not selecting features -- it is fitting its own folds, or reading account
    size through a column that was supposed to be scale-free, and every set it
    has ever chosen is suspect.

    Parameters
    ----------
    train : Dataset
        The training region.
    folds : list of Split
        The same folds the real selection uses.
    ranking : pandas.Series
        The real gain ranking, reused only for its column order -- the order
        is irrelevant on shuffled data, and re-fitting a ranking would double
        the runtime of a check that is already the slow half of the module.
    params : dict
        Fixed constructor arguments.
    anchor_column : str
        Last month's balance.
    tolerance, metric, min_signal : float, str, float
        The same rule the real selection ran under. Loosening any of them for
        the noise test would be testing a different pipeline.
    random_state : int, optional
        Seed for the shuffle. Default 42.

    Returns
    -------
    dict
        ``passed``, the number of features ``selected``, and the full-feature
        ``r2_change`` the shuffled data reached -- which should be at or below
        zero, since nothing can beat a no-change prediction on noise.

        A shuffled r2_change ABOVE the real one is not a leak, it is a broken
        shuffle: it means the permutation made the target easier rather than
        impossible. See ``shuffle_target``.
    """
    shuffled = shuffle_target(train, anchor_column, random_state=random_state)
    chosen, sweep = select_features(
        shuffled,
        folds,
        ranking,
        params,
        anchor_column,
        tolerance=tolerance,
        metric=metric,
        min_signal=min_signal,
    )
    return {
        "passed": len(chosen) == 0,
        "selected": len(chosen),
        "selected_features": list(chosen),
        "full_r2_change": sweep.attrs.get("full_r2_change", float("nan")),
        "min_signal": min_signal,
    }


def run(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the selection end to end, print it, and write its summary.

    Parameters
    ----------
    config : dict, optional
        Parsed config. Loaded from ``config/ml_config.yaml`` when omitted.

    Returns
    -------
    dict
        The summary that was written: the ranking, the sweep, the chosen set
        and the columns it implies dropping.

    Raises
    ------
    KeyError
        If ``feature_selection.model`` does not name an entry in ``models``.
    ValueError
        If the training region is too short to fold.
    """
    settings = config if config is not None else load_config()
    options = {**DEFAULTS, **(settings.get("feature_selection") or {})}

    specs = {entry["name"]: entry for entry in settings["models"]}
    if options["model"] not in specs:
        raise KeyError(
            f"feature_selection.model names {options['model']!r}, which is not "
            f"one of {sorted(specs)}"
        )
    spec = specs[options["model"]]
    params = dict(spec.get("params") or {})
    params.setdefault("random_state", settings["data"].get("random_state", 42))
    anchor_column = params.get("anchor_column", "prev_1m_closing_balance_usd")

    # Selection ranks every candidate column, so a frozen `features` block
    # from a previous run is cleared first. Otherwise this would re-rank
    # only the features it chose last time and could never let a dropped
    # column back in -- a ratchet that narrows the model on every run.
    candidates = {**settings, "features": []}
    dataset = build_dataset(candidates)
    print(dataset.summary())
    frozen = settings.get("features") or []
    if frozen:
        print(
            f"  ({len(frozen)} features are frozen in the config; this run "
            f"ignores that block and re-ranks all "
            f"{len(dataset.feature_columns)} candidates)"
        )

    # The holdout is cut before anything is measured. Selection is fitting.
    split = SplitSettings.from_config(settings)
    holdout = holdout_split(dataset, split.test_months, split.gap_months)
    train = dataset.take(holdout.train_positions)

    folds = expanding_folds(
        train,
        n_folds=split.cv_folds,
        test_months=split.cv_test_months,
        gap_months=split.gap_months,
        min_train_months=split.min_train_months,
    )
    if not folds:
        raise ValueError(
            "No folds were produced, so there is nothing to validate against. "
            "Check cv_folds and min_train_months against the training months."
        )
    print(f"Selecting on {len(train.months)} training months, {len(folds)} folds\n")

    tolerance = float(options["tolerance"])
    metric = str(options["tolerance_metric"])
    min_signal = float(options["min_signal"])
    all_features = train.feature_columns

    # ---- Stage 1: collinearity and redundancy -------------------------------
    pairs = redundant_pairs(train, float(options["redundancy_threshold"]))
    print(f"=== Stage 1: redundancy (either measure >= {options['redundancy_threshold']})")
    print(pairs.to_string() if not pairs.empty else "  no pairs above the threshold")

    survivors, redundancy_drops = redundancy_filter(train, pairs, anchor_column)
    if not redundancy_drops.empty:
        print(f"\n  dropping {len(redundancy_drops)} of {len(all_features)}:")
        print(redundancy_drops.to_string(index=False))
    print(f"  -> {len(survivors)} features into stage 2")

    filtered = train.model_copy(update={"feature_columns": survivors})

    # ---- Stage 2: temporal cross-validation ---------------------------------
    ranking, settled = gain_ranking(filtered, params, spec.get("search"))
    print("\n=== Stage 2: gain ranking, then the parsimony sweep")
    print(ranking.to_string())

    chosen, sweep = select_features(
        filtered,
        folds,
        ranking,
        settled,
        anchor_column,
        tolerance=tolerance,
        metric=metric,
        min_signal=min_signal,
    )
    print(f"\n--- Sweep (tolerance {tolerance:.0%} on {metric})")
    print(sweep.to_string())

    full_signal = float(sweep.attrs.get("full_r2_change", float("nan")))
    print(
        f"\n  gate: full-set r2_change {full_signal:+.4f} vs "
        f"min_signal {min_signal:+.4f} -- "
        f"{'CLEARED' if len(chosen) else 'NOT CLEARED, selecting nothing'}"
    )

    # ---- Stage 3: out-of-fold permutation stability -------------------------
    stability = pd.DataFrame()
    stable: tuple[str, ...] = ()
    if chosen:
        stability = permutation_stability(
            filtered,
            folds,
            chosen,
            settled,
            anchor_column,
            repeats=int(options["permutation_repeats"]),
        )
        max_cv = float(options["stability_max_cv"])
        keep_mask = (stability["importance_mean"] > 0.0) & (
            stability["importance_cv"].fillna(np.inf) <= max_cv
        )
        stability["keep"] = keep_mask
        stable = tuple(name for name in chosen if bool(keep_mask.get(name, False)))

        print(
            f"\n=== Stage 3: permutation importance on the validation folds "
            f"(cv <= {max_cv})"
        )
        print(stability.to_string())
        print(f"  -> {len(stable)} of {len(chosen)} survive the stability check")
    else:
        print("\n=== Stage 3: skipped, stage 2 selected nothing")

    # ---- Stage 4: the noise test --------------------------------------------
    print("\n=== Stage 4: noise test (target shuffled across time)")
    noise = noise_test(
        train,
        folds,
        pd.Series(1.0, index=list(survivors)),
        settled,
        anchor_column,
        tolerance=tolerance,
        metric=metric,
        min_signal=min_signal,
        random_state=int(settings["data"].get("random_state", 42)),
    )
    verdict = "PASS" if noise["passed"] else "FAIL"
    print(
        f"  full-set r2_change on shuffled target {noise['full_r2_change']:+.4f}, "
        f"{noise['selected']} features selected"
    )
    print(f"  NOISE TEST: {verdict}")
    if not noise["passed"]:
        print(
            "  A pipeline that finds features in noise is not selecting "
            "features. Treat every set above as unproven until this passes: "
            "the likely causes are a leaked column or a surviving size proxy."
        )

    # ---- Result --------------------------------------------------------------
    dropped = [name for name in all_features if name not in stable]
    print(
        f"\n=== Chosen: {len(stable)} of {len(all_features)} features "
        f"(dropping {len(dropped)})"
    )
    for name in stable:
        print(f"  keep  {name:<45} gain {ranking.get(name, float('nan')):>12,.4f}")
    if stable:
        # The hand-off. Selection suggests; a person reads it and commits
        # the result. Printed ready to paste so the config edit is
        # mechanical, with the reason each column went printed underneath
        # rather than inside the block: the block is the answer, the
        # reasons are the working.
        print("\nPaste as the top-level `features:` block in config/ml_config.yaml:")
        print("\nfeatures:")
        for name in stable:
            print(f"  - {name:<45} # gain {ranking.get(name, float('nan')):,.4f}")
        if dropped:
            print(f"\n  # not selected ({len(dropped)}):")
            for name in dropped:
                print(f"  #   {name:<43} {_drop_reason(name, chosen, survivors)}")
    else:
        print(
            "\nNothing was selected, so there is no features block to "
            "paste. Leave the config as it is and read the gate above."
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": options["model"],
        "tolerance": tolerance,
        "tolerance_metric": metric,
        "min_signal": min_signal,
        "training_months": [f"{month:%Y-%m}" for month in train.months],
        "folds": len(folds),
        "fitted_params": settled,
        "stage1_redundant_pairs": pairs.reset_index(names=["left", "right"]).to_dict(
            orient="records"
        ),
        "stage1_dropped": redundancy_drops.to_dict(orient="records"),
        "stage1_survivors": list(survivors),
        "stage2_gain_ranking": {name: float(value) for name, value in ranking.items()},
        "stage2_sweep": sweep.reset_index().to_dict(orient="records"),
        "stage2_chosen": list(chosen),
        "stage2_gate_cleared": bool(chosen),
        "stage3_stability": stability.reset_index(names=["feature"]).to_dict(
            orient="records"
        )
        if not stability.empty
        else [],
        "stage4_noise_test": noise,
        "chosen_features": list(stable),
        "drop_columns": dropped,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(resolve_output_dir(settings)) / f"feature_selection_{stamp}.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSummary written to {path}")
    return summary


def _drop_reason(
    name: str, chosen: tuple[str, ...], survivors: tuple[str, ...]
) -> str:
    """Return the stage that dropped a column, for the pasted config block.

    Parameters
    ----------
    name : str
        The dropped column.
    chosen : tuple of str
        Stage 2's output.
    survivors : tuple of str
        Stage 1's output.

    Returns
    -------
    str
        A short phrase naming the stage, so the config comment says why the
        column went rather than only that it did.
    """
    if name not in survivors:
        return "stage 1: redundant with a column that stayed"
    if name not in chosen:
        return "stage 2: outside the parsimony band"
    return "stage 3: unstable across validation folds"


def main() -> None:
    """Run the selection.

    Returns
    -------
    None
    """
    run()


if __name__ == "__main__":
    main()
