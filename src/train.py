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

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.config.config import load_config, resolve_output_dir
from src.data.data import (
    Dataset,
    build_dataset,
    suspicious_months,
    unchanged_share_by_month,
)
from src.evaluate import EvaluationSettings
from src.metrics import Scores, anchor_values, score
from src.models_code.base_class import Model, ModelFactory
from src.models_code.baseline import LagAverageBaseline
from src.models_code.mlr import RidgeRegression
from src.models_code.xgboost_model import XGBoostModel
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
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
    search: dict[str, Any] = Field(default_factory=dict)

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
            spec.factory(defaults), fit_rows, validate_rows
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

    Returns
    -------
    pandas.DataFrame
        One row per (fold, model), with the RMSE, the honest R squared and the
        skill score. Empty when no folds were produced.
    """
    rows: list[dict[str, Any]] = []
    for fold in folds:
        _, predictions = fit_fold(specs, train, fold, defaults)
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
                    "rmse": scores.rmse,
                    "mae": scores.mae,
                    "smape": scores.smape,
                    "r2_change": scores.r2_change,
                    "skill": scores.skill,
                }
            )
    return pd.DataFrame(rows)


def summarise_folds(fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Average each model's fold scores into one row per model.

    Parameters
    ----------
    fold_scores : pandas.DataFrame
        Output of ``cross_validate``.

    Returns
    -------
    pandas.DataFrame
        One row per model, with the mean and the spread across folds. The
        spread is worth as much as the mean: a model that wins on average but
        swings wildly between periods is not the safer choice.
    """
    if fold_scores.empty:
        return fold_scores

    return (
        fold_scores.groupby("model")
        .agg(
            folds=("fold", "count"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            smape_mean=("smape", "mean"),
            r2_change_mean=("r2_change", "mean"),
            skill_mean=("skill", "mean"),
        )
        .sort_values("rmse_mean")
    )


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
    print(f"Data      : {dataset.summary()}")
    print(f"Split     : {split_settings.describe()}")
    print(f"Models    : {', '.join(spec.name for spec in specs)}")

    frozen = settings.get("features") or []
    if frozen:
        print(
            f"Features  : {len(frozen)} columns, frozen in the config's "
            f"`features` block"
        )
    else:
        print(
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
        print("\nWARNING: months where the target barely moves from last month:")
        for month in flagged:
            print(
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
    print(f"Training  : {holdout.n_train:,} rows, {len(train.months)} months, "
          f"{train.months[0]:%Y-%m}..{train.months[-1]:%Y-%m}")
    print(f"Held out  : {holdout.n_test:,} rows from "
          f"{holdout.test_months[0]:%Y-%m} -- not read again in this script")

    folds = expanding_folds(
        train,
        n_folds=split_settings.cv_folds,
        test_months=split_settings.cv_test_months,
        gap_months=split_settings.gap_months,
        min_train_months=split_settings.min_train_months,
    )

    print(
        f"\n--- Cross-validation ({len(folds)} expanding folds inside the "
        f"training region)"
    )
    fold_scores = cross_validate(specs, train, folds, evaluation, defaults)
    if fold_scores.empty:
        print(
            "  No folds were produced. Check cv_folds and min_train_months "
            "against the number of training months."
        )
    else:
        fold_summary = summarise_folds(fold_scores)
        print(fold_summary.to_string(float_format=lambda value: f"{value:,.2f}"))
        print(
            "\n  These are validation scores from inside the training region, "
            "for choosing between models.\n  The headline comparison is "
            "`python -m src.test`, on months no model here has seen."
        )

    # The final fit: every model, on the whole training region, once.
    print("\n--- Fitting on the full training region")
    fitted: dict[str, Model] = {}
    for spec in specs:
        model = spec.factory(defaults)()
        model.fit(train)
        fitted[spec.name] = model
        print(f"  {model.describe()}")

    # What the search settled on. Printed rather than written to a summary
    # file: it is also pickled with the model, so the artefact can be asked
    # directly rather than cross-referenced against a run log.
    tuned = {
        name: model
        for name, model in fitted.items()
        if getattr(model, "best_params_", None) is not None
    }
    if tuned:
        print("\n--- Tuning")
        for name, model in tuned.items():
            print(f"  {name}: {getattr(model, 'best_params_', None)}")
            rounds = getattr(model, "best_iteration_", None)
            if rounds:
                print(
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
        print(f"\n--- {name}: feature importance ({importance.name}), top 12")
        print(importance.head(12).to_string(float_format=lambda v: f"{v:,.4f}"))

    # Saved last, so nothing is written unless everything fitted. Each file is
    # the whole model: preprocessing, fitted statistics, the months it saw and
    # the columns it expects.
    output_dir = resolve_output_dir(settings)
    print(f"\n--- Saved to {output_dir}")
    for spec in specs:
        path = fitted[spec.name].save(model_path(output_dir, spec.name))
        print(f"  {path.name}")

    print("\nNow score them:  python -m src.test")
    return fitted


def main() -> None:
    """Run the training pipeline. The container's entry point.

    Returns
    -------
    None
    """
    run()


if __name__ == "__main__":
    main()
