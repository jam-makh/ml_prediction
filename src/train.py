"""Fit every configured model and save it. Scoring happens in ``test.py``.

This script never touches the holdout. It cuts the holdout off at the start,
hands the training region to everything below, and the held-out months are not
read again by any line in this file. That is the entire reason training and
testing are two scripts: a script that both fits and reports the headline
number can always be edited into one that reports a number it fitted on, and
the edit does not look like a mistake.

One pass, in this order:

1. Load the feature table and turn it into a ``Dataset``.
2. Cut the last months off as a holdout, and keep only the training region.
3. Cross-validate every model on expanding windows *inside* that region. This
   is the number used to choose between models, because it averages over
   several periods instead of betting the comparison on one.
4. Refit each model on the whole training region.
5. Save each fitted model as a single file.

Every model is rebuilt from scratch for each fold. Reusing a fitted instance
would carry fold 1's learned statistics into fold 2, which is the same leak as
fitting a scaler before the split, just harder to see.

**What the saved file carries.** Each model pickles itself whole, including the
months it was fitted on. ``test.py`` reads that and scores everything strictly
after it, so the train/test boundary is a fact produced by the fit rather than a
cut-off recomputed from config in two places. Change ``split.test_months``
between a train run and a test run and nothing silently moves: the models still
say where they stopped.

**Feature selection does not happen here.** The feature set comes from the
``features`` block of the config, which is where ``features_selection.py``'s
answer was pasted after someone read it. Re-selecting on every training run
would make two runs over the same data disagree, and would quietly give the
booster extra attempts to fit the folds that the baseline never got.

Before any of that, the run checks whether a month's targets simply repeat the
previous month's balances, which is what an unfinished month in the source
table looks like from here. That check is why 2025-07 is excluded in the config.

Run it with::

    docker compose run --rm train
    python -m src.train            # locally, same code path

then score what it saved::

    python -m src.test
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.log import setup_logging
from src.config.config import load_config, resolve_output_dir
from src.data.data import (
    Dataset,
    build_dataset,
    resolve_drop_columns,
    suspicious_months,
    unchanged_share_by_month,
)
from src.evaluate import (
    AMOUNT_COLUMNS,
    MOVEMENT_COLUMNS,
    EvaluationSettings,
    results_table,
)
from src.metrics import Scores, anchor_values, score
from src.models_code.base_class import AnchoredModel, Model, ModelFactory
from src.models_code.baseline import LagAverageBaseline
from src.models_code.ridge_reg import RidgeRegression
from src.models_code.xgboost_model import XGBoostModel
from src.optuna_search import reference_fold_mae, run_study
from src.tuning import DEFAULT_SCORING
from src.window import (
    Split,
    SplitSettings,
    expanding_folds,
    holdout_split,
)

# Every model kind a config may name, mapped to its constructor. Order here is
# the order of the work: the trivial predictor to set the bar, a linear model
# that can be explained to a stakeholder, and boosted trees only if they beat it
# by enough to justify losing that explanation.
#
# `lag_average` covers both baselines -- three_month_average and persistence are
# the same class with a different n_months, set in the config.
MODEL_REGISTRY: dict[str, Callable[..., Model]] = {
    "lag_average": LagAverageBaseline,
    "ridge": RidgeRegression,
    "xgboost": XGBoostModel,
}

# Run-wide values a model may accept from the config rather than from its own
# params. Which of them a given class actually takes is declared on the class as
# `SHARED_ARGUMENTS` -- the baseline takes neither, the two regressions take
# both. Declared rather than discovered by reflection: an explicit attribute is
# greppable, and the alternative was inspecting every constructor's signature at
# runtime to find out what it would tolerate.
RUN_DEFAULTS = ("random_state", "anchor_column")

# Which `data.drop_columns` list each model kind reads. Before this existed the
# dataset was built once with `default_drop_columns`, so every model got the
# same list: dropping a column "for xgboost" silently dropped it for ridge too,
# and the ridge list was never read at all.
FAMILY_BY_KIND: dict[str, str] = {
    "lag_average": "mean",
    "ridge": "ridge_regression",
    "xgboost": "xgboost",
}


def model_path(output_dir: Path, name: str) -> Path:
    """Return the file a model of this name is saved to.

    One file per model, overwritten on every run. Named from the model rather
    than from a timestamp, so ``test.py`` can find it without being told which
    run to look at, and so a stale artefact from a config that no longer exists
    cannot be picked up by accident.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory models are written to.
    name : str
        The model's name from the config.

    Returns
    -------
    pathlib.Path
        The destination path.
    """
    return output_dir / f"{name}.joblib"


class ModelSpec(BaseModel):
    """One entry from the ``models`` block of the config.

    Attributes
    ----------
    name : str
        Label used in result tables and the saved filename. Unique within a run.
    kind : str
        Which constructor to use, a key of ``MODEL_REGISTRY``.
    params : dict
        Keyword arguments passed straight to that constructor.
    search : dict
        Hyperparameter search space, or empty for fixed parameters. Passed to
        constructors that accept a ``search`` argument and ignored by the rest,
        so putting one on a baseline is a config error rather than a surprise.
    optuna : dict
        Optuna search space over any constructor argument, including the
        training-target treatment (``clip``, ``market_scale``,
        ``recency_half_life``). Run by ``train.py`` before cross-validation
        when the top-level ``optuna.enabled`` is true; the winner replaces
        ``params`` for the rest of the run. See ``src.optuna_search``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
    search: dict[str, Any] = Field(default_factory=dict)
    optuna: dict[str, Any] = Field(default_factory=dict)

    def factory(self, defaults: dict[str, Any] | None = None) -> ModelFactory:
        """Return a zero-argument callable that builds this model.

        A factory rather than an instance, because cross-validation needs a
        fresh, unfitted model for every fold. Reusing one instance across folds
        would carry fold 1's fitted statistics into fold 2.

        Parameters
        ----------
        defaults : dict, optional
            Run-wide values from the config, keyed by the names in
            ``RUN_DEFAULTS``. Only those the target class declares in its own
            ``SHARED_ARGUMENTS`` are passed on, so a baseline is never handed an
            anchor column it has no use for. A value set in this spec's
            ``params`` wins, since it sits next to the model it applies to.

        Returns
        -------
        Callable
            Builds a new unfitted ``Model`` each time it is called.

        Raises
        ------
        KeyError
            If ``kind`` is not a registered model kind.
        TypeError
            If ``params`` names an argument the constructor does not accept.
            Raised at build time rather than swallowed, so a misspelled
            parameter fails the run instead of silently leaving the default.
        """
        if self.kind not in MODEL_REGISTRY:
            raise KeyError(
                f"Unknown model kind {self.kind!r} for model {self.name!r}; "
                f"registered kinds are {sorted(MODEL_REGISTRY)}"
            )
        constructor = MODEL_REGISTRY[self.kind]

        accepted = getattr(constructor, "SHARED_ARGUMENTS", ())
        arguments: dict[str, Any] = dict(self.params)
        for key, value in (defaults or {}).items():
            if key in accepted:
                arguments.setdefault(key, value)
        if self.search:
            arguments["search"] = self.search

        def build() -> Model:
            try:
                return constructor(name=self.name, **arguments)
            except TypeError as error:
                raise TypeError(
                    f"Model {self.name!r} (kind {self.kind!r}): {error}"
                ) from error

        return build


def read_model_specs(config: dict[str, Any]) -> list[ModelSpec]:
    """Parse the ``models`` block into validated specs.

    Parameters
    ----------
    config : dict
        Parsed config.

    Returns
    -------
    list of ModelSpec
        One spec per configured model, in config order.

    Raises
    ------
    ValueError
        If the block is empty or two models share a name.
    """
    entries = config.get("models") or []
    specs = [ModelSpec.model_validate(entry) for entry in entries]
    if not specs:
        raise ValueError("The models block is empty; nothing to train")

    names = [spec.name for spec in specs]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        # Two models sharing a name would overwrite each other's saved file and
        # each other's row in every result table keyed by name.
        raise ValueError(f"Model names must be unique; repeated: {sorted(duplicates)}")
    return specs


def fit_and_predict(
    factory: ModelFactory, train: Dataset, test: Dataset
) -> tuple[Model, npt.NDArray[np.float64]]:
    """Build a fresh model, fit it on train, and predict test.

    Parameters
    ----------
    factory : Callable
        Builds an unfitted model.
    train : Dataset
        Training rows.
    test : Dataset
        Rows to predict.

    Returns
    -------
    tuple of (Model, numpy.ndarray)
        The fitted model and its predictions, in test row order.
    """
    model = factory()
    model.fit(train)
    return model, model.predict(test)


def dataset_for(
    spec: ModelSpec, dataset: Dataset, data_settings: dict[str, Any] | None
) -> Dataset:
    """Return ``dataset`` with the feature list this model's family should see.

    Only narrows: the columns come from the dataset's own feature list minus
    the family's ``data.drop_columns`` entry, so the base dataset has to be
    built with the widest list (``default_drop_columns`` pointing at an empty
    one). The frame itself is untouched, and a fitted model predicts on the
    columns it recorded at fit time, so scoring code can keep passing the
    base dataset.

    Parameters
    ----------
    spec : ModelSpec
        The model about to be fitted.
    dataset : Dataset
        The base dataset.
    data_settings : dict or None
        The config's ``data`` block. None returns ``dataset`` unchanged.

    Returns
    -------
    Dataset
        Same rows, narrowed ``feature_columns``.

    Raises
    ------
    ValueError
        If the drop list removes every feature.
    """
    if not data_settings:
        return dataset
    dropped = set(resolve_drop_columns(data_settings, FAMILY_BY_KIND.get(spec.kind)))
    columns = tuple(name for name in dataset.feature_columns if name not in dropped)
    if not columns:
        raise ValueError(f"{spec.name}: data.drop_columns removed every feature")
    return dataset.model_copy(update={"feature_columns": columns})


def score_predictions(
    predictions: dict[str, npt.NDArray[np.float64]],
    test: Dataset,
    settings: EvaluationSettings,
) -> dict[str, Scores]:
    """Score every model's predictions on one set of rows.

    Parameters
    ----------
    predictions : dict of str to numpy.ndarray
        Model name to its predictions over ``test``.
    test : Dataset
        The rows that were predicted, carrying the truth and the anchor.
    settings : EvaluationSettings
        Which model is the reference and which column is the anchor.

    Returns
    -------
    dict of str to Scores
        Model name to its scores.
    """
    truth = test.target
    anchor = anchor_values(test, settings.anchor_column)

    # Missing reference is not an error: cross-validation can be asked to score
    # a subset of models. The skill column is simply absent in that case.
    reference = predictions.get(settings.reference_model)

    scored: dict[str, Scores] = {}
    for name, prediction in predictions.items():
        # A model is not measured against itself, which would report a skill of
        # exactly zero and read as though it had been compared to something.
        against = reference if name != settings.reference_model else None
        scored[name] = score(
            truth,
            prediction,
            anchor=anchor,
            reference_pred=against,
            mape_floor=settings.mape_floor,
        )
    return scored


def fit_fold(
    specs: list[ModelSpec],
    dataset: Dataset,
    split: Split,
    defaults: dict[str, Any] | None = None,
    data_settings: dict[str, Any] | None = None,
) -> tuple[dict[str, Model], dict[str, npt.NDArray[np.float64]]]:
    """Fit every model on one fold and predict its validation rows.

    Parameters
    ----------
    specs : list of ModelSpec
        Models to run.
    dataset : Dataset
        The dataset the split indexes into. Always the training region.
    split : Split
        The fit and validate rows. Already checked for leak when it was built.
    defaults : dict, optional
        Run-wide constructor values, handed to each model as it is built.
    data_settings : dict, optional
        The config's ``data`` block, for each family's drop list. See
        ``dataset_for``.

    Returns
    -------
    tuple of (dict, dict)
        The fitted models and their predictions over the validation rows, both
        keyed by model name.
    """
    fit_rows = dataset.take(split.train_positions)
    validate_rows = dataset.take(split.test_positions)

    models: dict[str, Model] = {}
    predictions: dict[str, npt.NDArray[np.float64]] = {}
    for spec in specs:
        model, prediction = fit_and_predict(
            spec.factory(defaults),
            dataset_for(spec, fit_rows, data_settings),
            validate_rows,
        )
        models[spec.name] = model
        predictions[spec.name] = prediction

    return models, predictions


def cross_validate(
    specs: list[ModelSpec],
    train: Dataset,
    folds: list[Split],
    settings: EvaluationSettings,
    defaults: dict[str, Any] | None = None,
    data_settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Score every model on every expanding-window fold.

    Parameters
    ----------
    specs : list of ModelSpec
        Models to run.
    train : Dataset
        The training region. Never the full panel, or the folds would reach
        into the holdout.
    folds : list of Split
        Folds produced by ``expanding_folds``.
    settings : EvaluationSettings
        Reference model and anchor column.
    defaults : dict, optional
        Run-wide constructor values. A fresh model is built per fold, so
        anything a model learns -- an imputer, a scaler, a chosen alpha, a
        searched parameter set -- is refitted inside each fold rather than
        carried across them.
    data_settings : dict, optional
        The config's ``data`` block, for each family's drop list.

    Returns
    -------
    pandas.DataFrame
        One row per (fold, model), with the fields behind both results tables
        (``AMOUNT_COLUMNS`` and ``MOVEMENT_COLUMNS``). Empty when no folds
        were produced.
    """
    rows: list[dict[str, Any]] = []
    for fold in folds:
        _, predictions = fit_fold(specs, train, fold, defaults, data_settings)
        scored = score_predictions(
            predictions, train.take(fold.test_positions), settings
        )
        for model_name, scores in scored.items():
            rows.append(
                {
                    "fold": fold.name,
                    "test_months": f"{fold.test_months[0]:%Y-%m}"
                    f"..{fold.test_months[-1]:%Y-%m}",
                    "model": model_name,
                    **{
                        field: getattr(scores, field)
                        for field in (*AMOUNT_COLUMNS, *MOVEMENT_COLUMNS)
                    },
                }
            )
    return pd.DataFrame(rows)


def summarise_folds(
    fold_scores: pd.DataFrame, columns: dict[str, str], sort_by: str = "wape"
) -> pd.DataFrame:
    """Average each model's fold scores into one row per model.

    Parameters
    ----------
    fold_scores : pandas.DataFrame
        Output of ``cross_validate``.
    columns : dict of str to str
        ``AMOUNT_COLUMNS`` or ``MOVEMENT_COLUMNS``.
    sort_by : str, optional
        Column to rank on. Default ``wape``.

    Returns
    -------
    pandas.DataFrame
        One row per model: the fold mean of ``r2``, ``mae``, ``rmse`` and
        ``wape``, plus ``wape_std`` across folds. The spread is worth as much
        as the mean: a model that wins on average but swings wildly between
        periods is not the safer choice.
    """
    if fold_scores.empty:
        return fold_scores

    grouped = fold_scores.groupby("model")
    table = results_table(grouped[list(columns)].mean(), columns, sort_by=sort_by)
    wape_field = next(field for field, name in columns.items() if name == "wape")
    table["wape_std"] = grouped[wape_field].std()
    table["folds"] = grouped["fold"].count()
    return table


def save_best_params(model: AnchoredModel, output_dir: Path) -> Path:
    """Write a tuned model's winning settings to a readable JSON file.

    The same values are pickled inside the model; this copy is for reading and
    for pasting into the config as fixed ``params`` once the search is settled.
    Written for every model that was tuned -- by Optuna, by its own search, or
    by ridge's inner alpha CV -- not only the booster.

    Parameters
    ----------
    model : AnchoredModel
        A fitted model.
    output_dir : pathlib.Path
        Directory the models are saved to.

    Returns
    -------
    pathlib.Path
        ``<output_dir>/<model name>_best_params.json``, overwritten each run.
    """
    through = model.trained_through
    record: dict[str, Any] = {
        "model": model.name,
        "trained_through": f"{through:%Y-%m}" if through is not None else None,
        "saved_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "target_mode": model.target_mode,
        "clip": model.clip,
        "clip_cap_fitted": model.clip_cap_,
        "market_scale": model.market_scale,
        "recency_half_life": model.recency_half_life,
        "best_params": model.best_params_,
        "optuna": model.optuna_,
    }
    if isinstance(model, XGBoostModel):
        record["xgboost_params"] = model.params
        record["n_estimators"] = model.best_iteration_ or model.params["n_estimators"]
        if model.search:
            record["search_scoring"] = model.search.get("scoring", DEFAULT_SCORING)
            record["search_cv_score"] = model.search_cv_score_
    path = output_dir / f"{model.name}_best_params.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


def tune_with_optuna(
    specs: list[ModelSpec],
    train: Dataset,
    folds: list[Split],
    evaluation: EvaluationSettings,
    defaults: dict[str, Any],
    settings: dict[str, Any],
    output_dir: Path,
) -> tuple[list[ModelSpec], dict[str, dict[str, Any]]]:
    """Run an Optuna study for every spec with an ``optuna`` block.

    Parameters
    ----------
    specs : list of ModelSpec
        Every configured model.
    train : Dataset
        The training region. The holdout is not in it, so no trial can read it.
    folds : list of Split
        The same expanding folds cross-validation uses.
    evaluation : EvaluationSettings
        Names the reference model every trial is scored against.
    defaults : dict
        Run-wide constructor values.
    settings : dict
        Parsed config.
    output_dir : pathlib.Path
        Where the study storage is written.

    Returns
    -------
    tuple of (list of ModelSpec, dict)
        The specs with each tuned model's ``params`` replaced by its winner,
        and the study record per tuned model name.

    Raises
    ------
    ValueError
        If a model is to be tuned but the reference model is not configured,
        or there are no folds to tune on.
    """
    block = settings.get("optuna") or {}
    to_tune = [spec for spec in specs if spec.optuna]
    if not block.get("enabled", False) or not to_tune:
        return specs, {}
    if not folds:
        raise ValueError("Optuna needs cross-validation folds; none were produced")

    by_name = {spec.name: spec for spec in specs}
    reference = by_name.get(evaluation.reference_model)
    if reference is None:
        raise ValueError(
            f"Optuna scores against {evaluation.reference_model!r}, which is "
            f"not in the models block"
        )
    reference_mae = reference_fold_mae(reference.factory(defaults), train, folds)
    data_block = settings.get("data") or {}
    table = str(data_block.get("active_table", "table"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"\n--- Optuna ({int(block.get('n_trials', 50))} trials per model, "
        f"{len(folds)} folds, objective = MAE / persistence MAE)"
    )
    records: dict[str, dict[str, Any]] = {}
    tuned: list[ModelSpec] = []
    for spec in specs:
        if not spec.optuna:
            tuned.append(spec)
            continue

        base = dict(spec.params)
        # Optuna owns alpha once it searches it; ridge's own RidgeCV grid would
        # otherwise re-pick alpha inside every fit and override the trial.
        if spec.kind == "ridge" and "alpha" in spec.optuna:
            base["alphas"] = None

        def build(params: dict[str, Any], spec: ModelSpec = spec) -> Model:
            return spec.model_copy(update={"params": params}).factory(defaults)()

        def narrow(dataset: Dataset, spec: ModelSpec = spec) -> Dataset:
            return dataset_for(spec, dataset, data_block)

        best_params, record = run_study(
            name=f"{table}_{spec.name}",
            space=spec.optuna,
            base_params=base,
            build=build,
            narrow=narrow,
            train=train,
            folds=folds,
            reference_mae=reference_mae,
            settings=block,
            output_dir=output_dir,
            random_state=int(defaults.get("random_state", 42)),
        )
        records[spec.name] = record
        tuned.append(spec.model_copy(update={"params": best_params, "optuna": {}}))
        logger.info(
            f"  {spec.name}: best {record['best_value']:.4f} (trial "
            f"{record['best_trial']}, {record['n_complete']} complete, "
            f"{record['n_pruned']} pruned)"
        )
        logger.info(f"    params     : {record['best_params']}")
        logger.info(f"    fold ratio : {record['fold_ratio']}")
        logger.info(f"    seed check : {record['seed_check']}")

    logger.info(
        "  Below 1.0 beats persistence on the folds. The cross-validation "
        "table below reuses these folds, so it is optimistic for tuned models; "
        "the holdout in src.test is the number that counts."
    )
    return tuned, records


def run(config: dict[str, Any] | None = None) -> dict[str, Model]:
    """Fit every configured model on the training region and save it.

    Parameters
    ----------
    config : dict, optional
        Parsed config. Loaded from ``config/ml_config.yaml`` when omitted.

    Returns
    -------
    dict of str to Model
        The fitted models, keyed by name, as they were saved.
    """
    settings = config if config is not None else load_config()
    split_settings = SplitSettings.from_config(settings)
    evaluation = EvaluationSettings.from_config(settings)
    specs = read_model_specs(settings)

    # Run-wide constructor values. Each model takes only the ones its class
    # declares in SHARED_ARGUMENTS, and its own params still win.
    data_block = settings.get("data") or {}
    defaults: dict[str, Any] = {
        "random_state": int(data_block.get("random_state", 42)),
        "anchor_column": evaluation.anchor_column,
    }

    dataset = build_dataset(settings)
    logger.info(f"Data      : {dataset.summary()}")
    logger.info(f"Split     : {split_settings.describe()}")
    logger.info(f"Models    : {', '.join(spec.name for spec in specs)}")

    frozen = settings.get("features") or []
    if frozen:
        logger.info(
            f"Features  : {len(frozen)} columns, frozen in the config's "
            f"`features` block"
        )
    else:
        logger.info(
            f"Features  : {len(dataset.feature_columns)} columns, derived by "
            f"exclusion (no `features` block; run src.features_selection to "
            f"freeze a set)"
        )

    # Re-run on every run rather than trusting the config to stay correct. A
    # month whose targets simply repeat the previous month's balances is an
    # unfinished month, and it makes every persistence-flavoured model look
    # near perfect. Warned about rather than dropped automatically: silently
    # discarding data is how a pipeline starts lying about its own inputs.
    flagged = suspicious_months(dataset, evaluation.anchor_column)
    if len(flagged):
        shares = unchanged_share_by_month(dataset, evaluation.anchor_column)
        logger.warning("Months where the target barely moves from last month:")
        for month in flagged:
            logger.warning(
                f"  {month:%Y-%m}: {100 * shares[month]:.0f}% of rows unchanged. "
                f"Add it to data.exclude_months, or explain why it is real."
            )

    # The holdout is cut here and then discarded. `train` is the only dataset
    # anything below this line sees, so nothing in this file can read a
    # held-out month even by mistake.
    holdout = holdout_split(
        dataset, split_settings.test_months, split_settings.gap_months
    )
    train = dataset.take(holdout.train_positions)
    logger.info(f"Training  : {holdout.n_train:,} rows, {len(train.months)} months, "
          f"{train.months[0]:%Y-%m}..{train.months[-1]:%Y-%m}")
    logger.info(f"Held out  : {holdout.n_test:,} rows from "
          f"{holdout.test_months[0]:%Y-%m} -- not read again in this script")

    folds = expanding_folds(
        train,
        n_folds=split_settings.cv_folds,
        test_months=split_settings.cv_test_months,
        gap_months=split_settings.gap_months,
        min_train_months=split_settings.min_train_months,
    )

    output_dir = resolve_output_dir(settings)
    specs, studies = tune_with_optuna(
        specs, train, folds, evaluation, defaults, settings, output_dir
    )

    logger.info(
        f"\n--- Cross-validation ({len(folds)} expanding folds inside the "
        f"training region)"
    )
    fold_scores = cross_validate(
        specs, train, folds, evaluation, defaults, data_block
    )
    if fold_scores.empty:
        logger.info(
            "  No folds were produced. Check cv_folds and min_train_months "
            "against the number of training months."
        )
    else:
        for title, columns in (
            ("Amounts: the balance itself", AMOUNT_COLUMNS),
            ("Movements: change from last month's balance", MOVEMENT_COLUMNS),
        ):
            fold_summary = summarise_folds(
                fold_scores, columns, sort_by=evaluation.headline_metric
            )
            logger.info(f"\n  {title} (mean over folds, sorted by "
                  f"{evaluation.headline_metric})")
            logger.info(fold_summary.to_string(float_format=lambda value: f"{value:,.3f}"))
        logger.info(
            "\n  These are validation scores from inside the training region, "
            "for choosing between models.\n  The headline comparison is "
            "`python -m src.test`, on months no model here has seen."
        )

    # The final fit: every model, on the whole training region, once.
    logger.info("\n--- Fitting on the full training region")
    fitted: dict[str, Model] = {}
    for spec in specs:
        model = spec.factory(defaults)()
        model.fit(dataset_for(spec, train, data_block))
        if spec.name in studies and isinstance(model, AnchoredModel):
            model.optuna_ = studies[spec.name]
        fitted[spec.name] = model
        logger.info(f"  {model.describe()}")
        dropped = [name for name in train.feature_columns if name not in model.feature_columns]
        logger.info(
            f"    {len(model.feature_columns)} features"
            + (f", dropped {dropped}" if dropped else "")
        )

    # What the search settled on. Printed rather than written to a summary
    # file: it is also pickled with the model, so the artefact can be asked
    # directly rather than cross-referenced against a run log.
    tuned = {
        name: model
        for name, model in fitted.items()
        if getattr(model, "best_params_", None) is not None
    }
    if tuned:
        logger.info("\n--- Tuning")
        for name, model in tuned.items():
            logger.info(f"  {name}: {getattr(model, 'best_params_', None)}")
            rounds = getattr(model, "best_iteration_", None)
            if rounds:
                logger.info(
                    f"    {rounds} rounds, the median of "
                    f"{getattr(model, 'stopping_rounds_', [])} across the "
                    f"early-stopping folds"
                )

    # Which columns carried each fitted model. Gain is measured over splits the
    # tree actually made, so a column that earns its place only in combination
    # with another shows up here and nowhere in a correlation table.
    for name, model in fitted.items():
        importance = model.feature_importance()
        if importance is None:
            continue
        logger.info(f"\n--- {name}: feature importance ({importance.name}), top 12")
        logger.info(importance.head(12).to_string(float_format=lambda v: f"{v:,.4f}"))

    # Saved last, so nothing is written unless everything fitted. Each file is
    # the whole model: preprocessing, fitted statistics, the months it saw and
    # the columns it expects.
    logger.info(f"\n--- Saved to {output_dir}")
    for spec in specs:
        path = fitted[spec.name].save(model_path(output_dir, spec.name))
        logger.info(f"  {path.name}")
        model = fitted[spec.name]
        if isinstance(model, AnchoredModel) and (
            model.best_params_ is not None or model.optuna_ is not None
        ):
            logger.info(f"  {save_best_params(model, output_dir).name}")

    logger.info("\nNow score them:  python -m src.test")
    return fitted


def main() -> None:
    """Run the training pipeline. The container's entry point.

    Returns
    -------
    None
    """
    settings = load_config()
    logger.info(f"Logging to {setup_logging(settings, run_name='train')}")
    run(settings)


if __name__ == "__main__":
    main()
