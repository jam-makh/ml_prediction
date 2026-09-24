"""One command: fit everything, score everything, print train against test.

``src/train.py`` and ``src/test.py`` are separate modules for a reason -- the
one that fits must not be able to read the months the other one scores -- but
that is an argument about *module boundaries*, not about how many things you
should have to type. This runs both in order and then prints the table neither
of them can print alone.

That table is the point. ``train.py`` reports cross-validation scores from
inside the training region; ``test.py`` reports the holdout. Neither shows the
**in-sample** score, and the gap between in-sample and holdout is the single
most useful diagnostic there is::

    model               train_mae    test_mae    gap
    xgboost_scaled          3,112      16,455    5.3x   <- memorising
    ridge_change           18,520      19,045    1.0x   <- honest, maybe underfit
    three_month_average    17,890      18,090    1.0x   <- cannot overfit

A booster that scores 30k on the rows it was fitted on and 160k on rows it has
not seen has not learned the problem, it has learned the training set. No
cross-validated average says that as directly as those two numbers side by
side.

The in-sample scores are computed here rather than in ``train.py`` deliberately.
Scoring a model on its own training rows is exactly the mistake the whole
structure is built to prevent, so it happens in one clearly labelled place,
under a column name that says what it is, and never anywhere a headline number
is produced.

Run it with::

    python pipeline.py
"""

from __future__ import annotations

import contextlib
from typing import Any

import pandas as pd
from loguru import logger

from src.config.config import load_config
from src.data.data import build_dataset
from src.evaluate import EvaluationReport, EvaluationSettings
from src.importance import collect_importance, save_importance, top_features
from src.log import console_level, setup_logging
from src.metrics import Scores, anchor_values, score
from src.models_code.anchored_model import AnchoredModel
from src.models_code.base_class import Model
from src.models_code.xgboost_model import XGBoostModel
from src.test import run as run_test
from src.test import split_on_boundary, training_boundary
from src.train import run as run_train


def in_sample_scores(
    models: dict[str, Model], config: dict[str, Any]
) -> dict[str, Scores]:
    """Score every model on the rows it was fitted on.

    Deliberately in-sample. These numbers are not a measure of how well a model
    works -- they are a measure of how hard it fitted, which is only meaningful
    next to the holdout column.

    The training region is derived the same way ``test.py`` derives the test
    region: from the months each model records, not from the config. So the
    rows scored here are exactly the rows the models were handed, even if the
    config has been edited since.

    Parameters
    ----------
    models : dict of str to Model
        Fitted models, keyed by name.
    config : dict
        Parsed config, for the anchor column and the data to rebuild.

    Returns
    -------
    dict of str to Scores
        In-sample scores, keyed by model name.
    """
    evaluation = EvaluationSettings.from_config(config)
    dataset = build_dataset(config)
    seen, _ = split_on_boundary(dataset, training_boundary(models))

    truth = seen.target
    anchor = anchor_values(seen, evaluation.anchor_column)

    return {
        name: score(
            truth,
            model.predict(seen),
            anchor=anchor,
            mape_floor=evaluation.mape_floor,
        )
        for name, model in models.items()
    }


def combined_table(
    train_scores: dict[str, Scores],
    reports: dict[str, EvaluationReport],
    reference_model: str = "persistence",
) -> pd.DataFrame:
    """Put the in-sample and holdout numbers on one row per model.

    Parameters
    ----------
    train_scores : dict of str to Scores
        In-sample scores from ``in_sample_scores``.
    reports : dict of str to EvaluationReport
        Holdout reports from ``src.test.run``.
    reference_model : str, optional
        The model ``skill_mae`` is measured against. Default ``persistence``.

    Returns
    -------
    pandas.DataFrame
        One row per model, best holdout WAPE first. ``gap`` is holdout MAE
        over in-sample MAE: 1.0 means the model does the same on rows it has
        never seen, and anything much above that is memorisation.
    """
    reference = reports.get(reference_model)
    reference_mae = reference.scores.mae if reference is not None else float("nan")

    rows = []
    for name, report in reports.items():
        fitted = train_scores.get(name)
        train_rmse = fitted.rmse if fitted is not None else float("nan")
        test_rmse = report.scores.rmse
        train_mae = fitted.mae if fitted is not None else float("nan")
        test_mae = report.scores.mae
        rows.append(
            {
                "model": name,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_mae": train_mae,
                "test_mae": test_mae,
                # Over MAE, not RMSE. RMSE on this panel is set by a handful of
                # very large accounts, so the RMSE ratio measures how calm those
                # accounts happened to be in the holdout window rather than how
                # hard the model fitted: persistence, which cannot overfit by
                # construction, scored the same 0.81 as the booster, while the
                # MAE ratio separated them at 2.3x and 2.5x. A diagnostic a
                # constant predictor passes is not a diagnostic.
                #
                # Guarded: a baseline can in principle score 0 in-sample, and a
                # divide-by-zero here would kill the run after all the work.
                "gap": test_mae / train_mae if train_mae else float("nan"),
                "test_wape": report.scores.wape,
                "test_wape_change": report.scores.wape_change,
                "test_r2_change": report.scores.r2_change,
                "skill": report.scores.skill,
                # The same comparison on MAE. `skill` is RMSE-based, and RMSE
                # here is set by a few very large accounts; the headline metric
                # and the booster's objective are both absolute error.
                "skill_mae": 1.0 - test_mae / reference_mae if reference_mae else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("test_wape")


def run_once(config: dict[str, Any] | None = None, quiet: bool = False) -> pd.DataFrame:
    """Fit, score, and return the combined table for ONE feature table.

    Parameters
    ----------
    config : dict, optional
        Parsed config. Loaded from ``config/ml_config.yaml`` when omitted.
    quiet : bool, optional
        Suppress the output of the two underlying scripts and print only the
        combined table. Default False.

    Returns
    -------
    pandas.DataFrame
        The combined table.
    """
    settings = config if config is not None else load_config()

    # Quiet raises the console threshold only. The log file still receives
    # every line the two scripts write, so a failure can be read back there.
    # A factory, since a generator-based context manager can be entered once.
    def context() -> Any:
        return console_level("WARNING") if quiet else contextlib.nullcontext()

    if quiet:
        logger.info("Training ...")
    with context():
        models = run_train(settings)

    if quiet:
        logger.info("Scoring ...")
    with context():
        reports = run_test(settings)
        train_scores = in_sample_scores(models, settings)

    evaluation = EvaluationSettings.from_config(settings)
    table = combined_table(train_scores, reports, evaluation.reference_model)

    logger.info("\n" + "=" * 78)
    logger.info("  TRAIN vs TEST")
    logger.info("=" * 78)
    logger.info(table.to_string(float_format=lambda value: f"{value:,.3f}"))
    logger.info(
        "\n  train_*  scored on the rows each model was FITTED on. In-sample."
        "\n           Not a measure of quality -- only of how hard it fitted."
        "\n  test_*   scored on months no model has seen. This is the answer."
        "\n  gap      test_mae / train_mae. 1.0 is honest; 3x+ is memorisation."
        "\n           Over MAE rather than RMSE: RMSE here is set by a few very"
        "\n           large accounts, and the RMSE ratio gave a constant"
        "\n           predictor the same 0.81 as the booster."
        "\n  skill    versus the reference baseline, on RMSE. Positive means it"
        "\n           earned its complexity; around zero means it did not."
        "\n  skill_mae  the same on MAE: 1 - test_mae / reference test_mae."
    )
    report_tuning_and_importance(models, settings)
    return table


def report_tuning_and_importance(models: dict[str, Model], config: dict[str, Any]) -> None:
    """Print what Optuna chose and which features carry the booster.

    Both are in the log file already, from ``train.py``; they are repeated here
    because the pipeline runs training quietly, and these are the two things a
    run is read for after the table. The importance table and figure are also
    written next to the models, so ``src.importance`` need not be run after.

    Parameters
    ----------
    models : dict of str to Model
        The fitted models from this run.
    config : dict
        Parsed config, for the output directory.

    Returns
    -------
    None
    """
    tuned = {
        name: model
        for name, model in models.items()
        if isinstance(model, AnchoredModel) and model.optuna_ is not None
    }
    if tuned:
        logger.info("\n" + "=" * 78)
        logger.info("  OPTUNA BEST PARAMS  (also in <model>_best_params.json)")
        logger.info("=" * 78)
        for name, model in tuned.items():
            record = model.optuna_ or {}
            logger.info(
                f"  {name}: CV ratio to persistence {record.get('best_value')} "
                f"(below 1.0 beats it), seeds {record.get('seed_check')}"
            )
            for key, value in (record.get("best_params") or {}).items():
                logger.info(f"    {key:<22} {value}")
            if model.clip_cap_ is not None:
                logger.info(f"    {'clip cap (fitted)':<22} {model.clip_cap_:,.0f}")

    table = collect_importance(
        {name: model for name, model in models.items() if isinstance(model, AnchoredModel)}
    )
    boosters = [name for name, model in models.items() if isinstance(model, XGBoostModel)]
    for name in boosters:
        top = top_features(table, name, "total_gain")
        logger.info("\n" + "=" * 78)
        logger.info(f"  {name.upper()}: TOP FEATURES  (share of total gain)")
        logger.info("=" * 78)
        for feature, share in top.items():
            logger.info(f"  {feature:<45} {share:>6.1f}%")

    csv_path, png_path = save_importance(table, config)
    logger.info(f"\n  Saved {csv_path.name} and {png_path.name} to {csv_path.parent}")


def run(config: dict[str, Any] | None = None, quiet: bool = True) -> pd.DataFrame:
    """Run the benchmark over every table in ``data.compare_tables``.

    Comparing feature tables is the question this project keeps asking, and
    answering it by editing ``data.active_table`` between two runs makes the
    two halves of the comparison depend on nobody having changed anything else
    in between. Listing the tables instead runs them back to back against one
    identical config, and prints them side by side.

    Each table gets its own ``output.model_dir`` (``<model_dir>_<table>``), so
    the second run cannot overwrite the first one's saved models -- which is
    exactly what would happen if both wrote to ``models/``.

    Parameters
    ----------
    config : dict, optional
        Parsed config. Loaded from ``config/ml_config.yaml`` when omitted.
    quiet : bool, optional
        Suppress the underlying scripts' output. Default True, because two
        full runs of it is a great deal of scrollback.

    Returns
    -------
    pandas.DataFrame
        One row per table per model, indexed by (table, model).
    """
    settings = config if config is not None else load_config()
    data_block = settings.get("data") or {}
    tables = list(data_block.get("compare_tables") or ())

    if not tables:
        # No comparison configured: behave exactly as before.
        return run_once(settings, quiet=quiet)

    known = data_block.get("tables") or {}
    unknown = [name for name in tables if name not in known]
    if unknown:
        raise KeyError(
            f"data.compare_tables names {unknown}, which are not in "
            f"data.tables ({sorted(known)})"
        )

    base_dir = str((settings.get("output") or {}).get("model_dir", "models"))
    collected: list[pd.DataFrame] = []
    for name in tables:
        logger.info("\n" + "#" * 78)
        logger.info(f"#  TABLE: {name}  ({known[name]})")
        logger.info("#" * 78)
        per_table = {
            **settings,
            "data": {**data_block, "active_table": name},
            "output": {**(settings.get("output") or {}), "model_dir": f"{base_dir}_{name}"},
        }
        table = run_once(per_table, quiet=quiet)
        collected.append(table.assign(table=name).set_index("table", append=True))

    combined = pd.concat(collected).reorder_levels(["table", "model"]).sort_index()

    logger.info("\n" + "=" * 78)
    logger.info("  ACROSS TABLES")
    logger.info("=" * 78)
    logger.info(
        combined[
            ["test_mae", "test_wape", "test_wape_change", "test_r2_change", "gap", "skill", "skill_mae"]
        ]
        .to_string(float_format=lambda value: f"{value:,.3f}")
    )
    logger.info(
        "\n  Compare DOWN a column within one table, not across tables: WAPE's"
        "\n  denominator moves with any change to the panel. `skill` is the"
        "\n  cross-table comparison, since it is a ratio against that table's"
        "\n  own persistence."
    )
    return combined


def main() -> None:
    settings = load_config()
    log_file = setup_logging(settings, run_name="pipeline")
    logger.info(f"Logging to {log_file}")
    run(settings)


if __name__ == "__main__":
    main()
