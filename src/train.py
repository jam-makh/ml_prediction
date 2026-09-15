"""The entry point the container runs: fit every configured model and score it.

One pass, in this order:

1. Load the feature table and turn it into a ``Dataset``.
2. Cut the last months off as a holdout. Nothing below ever reads them until
   the final scoring step.
3. Cross-validate every model on expanding windows inside the training region.
   This is the number used to compare models, because it averages over several
   periods instead of betting the comparison on one.
4. Refit each model on the whole training region and score it once on the
   holdout. This is the number reported, and it is produced exactly once.
5. Seal each fitted model with ``save_model`` and write a JSON run summary.

Every model is rebuilt from scratch for each fold. Reusing a fitted instance
would carry fold 1's learned statistics into fold 2, which is the same leak as
fitting a scaler before the split, just harder to see.

Every model is reported with both R squareds, the flattering one against the
balance level and the honest one against the movement. See ``metrics.py`` for
why quoting only the first would make every model here look excellent, the
trivial baseline included.

Before any of that, the run checks whether a month's targets simply repeat the
previous month's balances, which is what an unfinished month in the source
table looks like from here. That check is why 2025-07 is excluded in the config.

The holdout scoring goes through ``evaluate.py`` rather than calling
``metrics.py`` directly, so the out-of-sample assertion and the per-month and
per-user breakdowns happen on the numbers that actually get quoted.

Run it with::

    docker compose run --rm train
    python -m src.train            # locally, same code path
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.config.config import load_config, resolve_output_dir
from src.data.data import Dataset, build_dataset, suspicious_months, unchanged_share_by_month
from src.evaluate import evaluate_models, report_table, worst_months_table, worst_users_table
from src.metrics import Scores, anchor_values, score
from src.models_code.base_class import Model, ModelFactory
from src.models_code.baseline import LagAverageBaseline
from src.models_code.mlr import RidgeRegression
from src.models_code.xgboost_model import XGBoostModel
from src.save_model import save_model
from src.splitting import (
    Split,
    SplitSettings,
    check_no_future_leak,
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


class ModelSpec(BaseModel):
    """One entry from the ``models`` block of the config.

    Attributes
    ----------
    name : str
        Label used in result tables and saved filenames. Unique within a run.
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


class EvaluationSettings(BaseModel):
    """The ``evaluation`` block of the config.

    Attributes
    ----------
    reference_model : str
        Name of the model every skill score is measured against. Normally the
        3-month average, since that is the bar the brief sets.
    anchor_column : str
        Column holding last month's balance, used by the change framing.
    mape_floor : float
        Rows with a smaller absolute true value are left out of the percentage
        error.
    headline_metric : str
        Which column the comparison table is ranked on. Defaults to the median
        absolute error, since on a target this skewed RMSE ranks models by how
        well they fit the largest handful of accounts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_model: str = "three_month_average"
    anchor_column: str = "prev_1m_closing_balance_usd"
    mape_floor: float = 1_000.0
    headline_metric: str = "median_ae"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> EvaluationSettings:
        """Build settings from a parsed config.

        Parameters
        ----------
        config : dict
            Parsed config. A missing ``evaluation`` block gives defaults.

        Returns
        -------
        EvaluationSettings
            Validated settings.
        """
        return cls.model_validate(config.get("evaluation") or {})


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
        # Two models sharing a name would overwrite each other in every result
        # dict keyed by name, and the run would silently report one of them.
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

    Notes
    -----
    No leak check here. The split was checked when it was built, and
    ``evaluate.check_out_of_sample`` re-checks the model's own recorded
    training months before any number is produced from them. A third copy in
    the middle asserted the same thing with a third error message.
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


def evaluate_split(
    specs: list[ModelSpec],
    dataset: Dataset,
    split: Split,
    settings: EvaluationSettings,
    defaults: dict[str, Any] | None = None,
) -> tuple[dict[str, Model], dict[str, npt.NDArray[np.float64]]]:
    """Fit every model on one split and predict its test rows.

    Deliberately does not score. The cross-validation path wants ``Scores``
    objects and the holdout path wants the richer ``EvaluationReport``, and
    having this function produce one of them meant the holdout was fitted here,
    scored, and then predicted a second time through the other path.

    Parameters
    ----------
    specs : list of ModelSpec
        Models to run.
    dataset : Dataset
        The dataset the split indexes into.
    split : Split
        The train and test rows. Already checked for leak when it was built.
    settings : EvaluationSettings
        Kept in the signature because every caller has it to hand and the next
        step always needs it.
    defaults : dict, optional
        Run-wide constructor values, handed to each model as it is built.

    Returns
    -------
    tuple of (dict, dict)
        The fitted models and their predictions over the test rows, both keyed
        by model name.
    """
    train = dataset.take(split.train_positions)
    test = dataset.take(split.test_positions)

    models: dict[str, Model] = {}
    predictions: dict[str, npt.NDArray[np.float64]] = {}
    for spec in specs:
        model, prediction = fit_and_predict(spec.factory(defaults), train, test)
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
        One row per (fold, model), with the RMSE, both R squareds and the skill
        score. Empty when no folds were produced.
    """
    rows: list[dict[str, Any]] = []
    for fold in folds:
        _, predictions = evaluate_split(specs, train, fold, settings, defaults)
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
            r2_change_mean=("r2_change", "mean"),
            skill_mean=("skill", "mean"),
        )
        .sort_values("rmse_mean")
    )


def run(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the whole pipeline once and return the summary it wrote.

    Parameters
    ----------
    config : dict, optional
        Parsed config. Loaded from ``config/ml_config.yaml`` when omitted.

    Returns
    -------
    dict
        The run summary, the same object written to the JSON file.
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

    dropped = data_block.get("drop_columns") or []
    if dropped:
        print(f"Dropped   : {len(dropped)} columns -- {', '.join(dropped)}")

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

    # The holdout is cut first and the training region is what everything else
    # sees, so cross-validation cannot reach into the held-out months.
    holdout = holdout_split(
        dataset, split_settings.test_months, split_settings.gap_months
    )
    print(f"Holdout   : {holdout.summary()}")

    train = dataset.take(holdout.train_positions)
    folds = expanding_folds(
        train,
        n_folds=split_settings.cv_folds,
        test_months=split_settings.cv_test_months,
        gap_months=split_settings.gap_months,
        min_train_months=split_settings.min_train_months,
    )

    print(f"\n--- Cross-validation ({len(folds)} expanding folds inside the training region)")
    fold_scores = cross_validate(specs, train, folds, evaluation, defaults)
    fold_summary = summarise_folds(fold_scores)
    print(fold_summary.to_string(float_format=lambda value: f"{value:,.2f}"))

    # The holdout, fitted and predicted exactly once. The predictions are handed
    # to evaluate.py rather than left for it to recompute; it still re-checks
    # that no scored month was trained on before it computes anything.
    fitted, holdout_predictions = evaluate_split(
        specs, dataset, holdout, evaluation, defaults
    )
    test = dataset.take(holdout.test_positions)
    reports = evaluate_models(
        fitted,
        test,
        anchor_column=evaluation.anchor_column,
        reference_model=evaluation.reference_model,
        mape_floor=evaluation.mape_floor,
        predictions=holdout_predictions,
    )

    print("\n--- Holdout (scored once, on months no model has seen)")
    for report in reports.values():
        print(f"  {report.describe()}")

    print(f"\n--- Side by side (sorted by {evaluation.headline_metric})")
    table = report_table(reports, sort_by=evaluation.headline_metric)
    print(table.to_string(float_format=lambda value: f"{value:,.3f}"))

    # Where the best model is worst, which is what the findings need.
    best_name = str(table.index[0])
    best_report = reports[best_name]
    print(f"\n--- {best_name}: holdout months, worst first")
    print(
        worst_months_table(best_report).to_string(
            float_format=lambda value: f"{value:,.0f}"
        )
    )
    print(f"\n--- {best_name}: worst users")
    print(
        worst_users_table(best_report).to_string(
            float_format=lambda value: f"{value:,.0f}"
        )
    )

    # Which columns carried each fitted model. Selection itself happens in the
    # notebook against Pearson and Spearman, which is univariate -- it scores
    # each column against the target one at a time, so a column that only
    # matters alongside another ranks low. This is the check that speaks to
    # that: gain for the booster is measured over splits the tree actually
    # made, so a column that earns its place only in combination shows up here
    # and nowhere in the correlation table.
    importances: dict[str, list[dict[str, Any]]] = {}
    for name in table.index:
        importance = fitted[str(name)].feature_importance()
        if importance is None:
            continue
        importances[str(name)] = (
            importance.rename("importance").rename_axis("feature")
            .reset_index().to_dict(orient="records")
        )
        print(f"\n--- {name}: feature importance ({importance.name}), top 12")
        print(
            importance.head(12).to_string(
                float_format=lambda value: f"{value:,.4f}"
            )
        )

    if not importances:
        print(
            "\nNo model in this run exposes feature importance; the table is "
            "all baselines."
        )

    # What the search settled on, for the run summary and for the write-up.
    tuning: dict[str, dict[str, Any]] = {}
    for name, model in fitted.items():
        chosen = getattr(model, "best_params_", None)
        if chosen is None:
            continue
        tuning[name] = {
            "best_params": chosen,
            "search_cv_score": getattr(model, "search_cv_score_", None),
            "best_iteration": getattr(model, "best_iteration_", None),
            "stopping_rounds": getattr(model, "stopping_rounds_", []),
        }
    if tuning:
        print("\n--- Tuning")
        for name, found in tuning.items():
            print(f"  {name}: {found['best_params']}")
            rounds = found["best_iteration"]
            if rounds:
                per_fold = found["stopping_rounds"]
                print(
                    f"    {rounds} rounds, the median of {per_fold} across the "
                    f"early-stopping folds"
                )

    # Seal the fitted models. Done after scoring so the metadata beside each
    # artefact carries the score it actually earned on the holdout.
    print("\n--- Sealed artefacts")
    output_dir = resolve_output_dir(settings)
    saved: list[dict[str, Any]] = []
    for spec in specs:
        path, artifact = save_model(
            fitted[spec.name],
            output_dir,
            kind=spec.kind,
            params=spec.params,
            scores=reports[spec.name].scores.model_dump(),
            data_summary=dataset.summary(),
            split_summary=holdout.summary(),
            n_training_rows=holdout.n_train,
        )
        saved.append({"path": str(path), "sha256": artifact.model_sha256})
        print(f"  {path.name}  sha256 {artifact.model_sha256[:12]}")

    summary: dict[str, Any] = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": dataset.summary(),
        # Recorded so a run's numbers can be read against the exact feature set
        # that produced them, without going back to the config's git history.
        "features": {
            "used": list(dataset.feature_columns),
            "dropped": list(dropped),
        },
        "split": split_settings.model_dump(),
        "evaluation": evaluation.model_dump(),
        "models": [spec.model_dump() for spec in specs],
        "cross_validation": {
            "folds": [fold.summary() for fold in folds],
            "per_fold": fold_scores.to_dict(orient="records"),
            "per_model": fold_summary.reset_index().to_dict(orient="records"),
        },
        "holdout": {
            "split": holdout.summary(),
            "reports": {
                name: report.model_dump() for name, report in reports.items()
            },
        },
        "best_model": best_name,
        "headline_metric": evaluation.headline_metric,
        "tuning": tuning,
        "feature_importance": importances,
        "artifacts": saved,
    }

    output_path = write_summary(summary, settings)
    print(f"\nSummary written to {output_path}")
    return summary


def write_summary(summary: dict[str, Any], config: dict[str, Any]) -> Path:
    """Write the run summary as JSON next to the saved models.

    One file per run, named by timestamp, so a run never overwrites the
    evidence of the one before it.

    Parameters
    ----------
    summary : dict
        The summary to write.
    config : dict
        Parsed config, used to locate the output directory.

    Returns
    -------
    pathlib.Path
        Where the file was written.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = resolve_output_dir(config) / f"run_{stamp}.json"
    # default=str so a stray Timestamp or numpy scalar is written rather than
    # failing a run that has already done all its work.
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return path


def main() -> None:
    """Run the pipeline. The container's entry point.

    Returns
    -------
    None
    """
    run()


if __name__ == "__main__":
    main()
