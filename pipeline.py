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
    ridge                  18,520      19,045    1.0x   <- honest, maybe underfit
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

from src.config.config import load_config, resolve_output_dir
from src.data.data import Dataset, build_dataset
from src.evaluate import EvaluationReport, EvaluationSettings, predictions_table
from src.feature_selection import (
    SelectionResult,
    coefficient_report_lines,
    save_shap,
    shap_importance,
)
from src.log import console_level, setup_logging
from src.metrics import Scores, anchor_values, grade, score
from src.models_code.anchored_model import AnchoredModel
from src.models_code.base_class import Model
from src.models_code.ridge_reg import RidgeRegression
from src.test import run as run_test
from src.test import split_on_boundary, training_boundary
from src.train import run as run_train


def training_rows(models: dict[str, Model], config: dict[str, Any]) -> Dataset:
    """Rebuild the rows the models were fitted on, from the months they record.

    Parameters
    ----------
    models : dict of str to Model
        Fitted models, keyed by name.
    config : dict
        Parsed config, for the data to rebuild.

    Returns
    -------
    Dataset
        Every row at or before the models' last training month.
    """
    # Same boundary test.py uses, so these are exactly the rows the models were handed.
    seen, _ = split_on_boundary(build_dataset(config), training_boundary(models))
    return seen


def in_sample_scores(
    models: dict[str, Model], seen: Dataset, config: dict[str, Any]
) -> dict[str, Scores]:
    """Score every model on the rows it was fitted on; a measure of fit, not of quality.

    Parameters
    ----------
    models : dict of str to Model
        Fitted models, keyed by name.
    seen : Dataset
        The training rows, from ``training_rows``.
    config : dict
        Parsed config, for the anchor column and MAPE floor.

    Returns
    -------
    dict of str to Scores
        In-sample scores, keyed by model name.
    """
    evaluation = EvaluationSettings.from_config(config)
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
                "train_mse": fitted.mse if fitted is not None else float("nan"),
                "test_mse": report.scores.mse,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_median_ae": fitted.median_ae if fitted is not None else float("nan"),
                "test_median_ae": report.scores.median_ae,
                "train_sme": fitted.sme if fitted is not None else float("nan"),
                "test_sme": report.scores.sme,
                "train_smape": fitted.smape if fitted is not None else float("nan"),
                "test_smape": report.scores.smape,
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


def percent_table(
    train_scores: dict[str, Scores],
    reports: dict[str, EvaluationReport],
    reference_model: str = "persistence",
) -> pd.DataFrame:
    """Express every model's errors as percentages, so they read without knowing the dollar scale.

    Parameters
    ----------
    train_scores : dict of str to Scores
        In-sample scores from ``in_sample_scores``.
    reports : dict of str to EvaluationReport
        Holdout reports from ``src.test.run``.
    reference_model : str, optional
        The naive baseline MASE divides by. Default ``persistence``.

    Returns
    -------
    pandas.DataFrame
        One row per model, best holdout WAPE first, with a ``grade`` band on test WAPE.
    """
    reference = reports.get(reference_model)
    reference_mae = reference.scores.mae if reference is not None else float("nan")

    rows = []
    for name, report in reports.items():
        test = report.scores
        fitted = train_scores.get(name)
        rows.append(
            {
                "model": name,
                "train_wape_%": fitted.wape if fitted is not None else float("nan"),
                "test_wape_%": test.wape,
                # WAPE is MAE over mean|truth|, so scaling it by RMSE/MAE gives RMSE over mean|truth|.
                "test_nrmse_%": test.wape * test.rmse / test.mae if test.mae else float("nan"),
                "test_smape_%": test.smape,
                # Well above ~1.5 means a few large misses dominate the error.
                "rmse/mae": test.rmse / test.mae if test.mae else float("nan"),
                "test_wape_change_%": test.wape_change,
                # MASE: below 1 beats the naive baseline, above 1 loses to it.
                "mase": test.mae / reference_mae if reference_mae else float("nan"),
                "r2_change": test.r2_change,
                "gap_%": 100.0 * (test.mae / fitted.mae - 1.0) if fitted is not None and fitted.mae else float("nan"),
                "grade": grade(test.wape),
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("test_wape_%")


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
        seen = training_rows(models, settings)
        train_scores = in_sample_scores(models, seen, settings)

    evaluation = EvaluationSettings.from_config(settings)

    # In-sample counterpart of predictions_test.csv, written next to it.
    predictions_path = resolve_output_dir(settings) / "predictions_train.csv"
    predictions_table(models, seen, evaluation.anchor_column).to_csv(predictions_path, index=False)
    logger.info(f"\n  Predictions: {predictions_path} and predictions_test.csv")
    table = combined_table(train_scores, reports, evaluation.reference_model)
    # Persist the train-vs-test comparison next to the prediction files.
    table_path = resolve_output_dir(settings) / "train_vs_test.csv"
    table.to_csv(table_path)
    logger.info(f"  Train vs test table: {table_path}")

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
        "\n  sme      signed mean error, truth - prediction. Positive = predicts too low."
        "\n  smape    symmetric % error, 0-200. Scale-free, so small accounts count equally."
    )

    percents = percent_table(train_scores, reports, evaluation.reference_model)
    # Saved beside the dollar table so both can be opened side by side.
    percents.to_csv(resolve_output_dir(settings) / "train_vs_test_percent.csv")

    logger.info("\n" + "=" * 78)
    logger.info("  TRAIN vs TEST  (percent)")
    logger.info("=" * 78)
    logger.info(percents.to_string(float_format=lambda value: f"{value:,.2f}"))
    logger.info(
        "\n  wape_%         MAE as % of the average balance. Graded: <10 highly accurate,"
        "\n                 10-20 good, 20-50 reasonable, 50+ poor."
        "\n  nrmse_%        RMSE as % of the average balance; punishes big misses more."
        "\n  rmse/mae       well above ~1.5 means a few outliers dominate the error."
        "\n  wape_change_%  error as % of the real month-to-month movement. The honest one."
        "\n  mase           test MAE / reference test MAE. Below 1 beats 'no change'."
        "\n  gap_%          how much worse test MAE is than train MAE. +100% = doubled."
        "\n  The grade is on the balance, which last month's value already explains,"
        "\n  so judge models on wape_change_% and mase, not on the grade alone."
    )
    report_tuning(models)
    report_selection(models, settings)
    report_shap(models, seen, settings)
    return table


def report_tuning(models: dict[str, Model]) -> None:
    """Print what Optuna chose, since the pipeline runs training quietly.

    Parameters
    ----------
    models : dict of str to Model
        The fitted models from this run.

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


def banner(title: str) -> None:
    """Print a section heading in the style of the rest of the report.

    Parameters
    ----------
    title : str
        Heading text.

    Returns
    -------
    None
    """
    logger.info("\n" + "=" * 78)
    logger.info(f"  {title}")
    logger.info("=" * 78)


def report_selection(models: dict[str, Model], config: dict[str, Any]) -> None:
    """Print what RFECV kept and dropped, and what the linear models zeroed or barely use.

    Parameters
    ----------
    models : dict of str to Model
        The fitted models from this run.
    config : dict
        Parsed config, for the selection settings and output directory.

    Returns
    -------
    None
    """
    block = config.get("feature_selection") or {}
    rfecv_path = resolve_output_dir(config) / f"{block.get('model')}_rfecv.json"
    # Only when this run's training ran RFECV, so a file left from an older run is not reported.
    if (block.get("rfecv") or {}).get("enabled", False) and rfecv_path.exists():
        result = SelectionResult.model_validate_json(rfecv_path.read_text(encoding="utf-8"))
        banner(f"RFECV: {result.model.upper()}  (also in {rfecv_path.name})")
        for line in result.report_lines():
            logger.info(line)

    weakest_n = int((block.get("linear") or {}).get("weakest_n", 10))
    for name, model in models.items():
        if isinstance(model, RidgeRegression):
            banner(f"{name.upper()}: COEFFICIENTS")
            for line in coefficient_report_lines(model, weakest_n):
                logger.info(line)


def report_shap(models: dict[str, Model], seen: Dataset, config: dict[str, Any]) -> None:
    """Rank each model's features by mean |SHAP| on the training rows, and save csv and chart.

    Parameters
    ----------
    models : dict of str to Model
        The fitted models from this run.
    seen : Dataset
        The training rows, from ``training_rows``.
    config : dict
        Parsed config, for the SHAP settings and output directory.

    Returns
    -------
    None
    """
    block = (config.get("feature_selection") or {}).get("shap") or {}
    if not block.get("enabled", False):
        return
    top_n = int(block.get("top_n", 15))
    output_dir = resolve_output_dir(config)

    for name, model in models.items():
        table = shap_importance(model, seen)
        # Baselines have no features to explain.
        if table is None:
            continue
        csv_path, png_path = save_shap(table, name, output_dir, top_n)
        banner(f"{name.upper()}: TOP {min(top_n, len(table))} FEATURES BY MEAN |SHAP|  (training rows)")
        logger.info(f"  {'feature':<45} {'mean |SHAP|':>12} {'mean SHAP':>12}")
        for feature, row in table.head(top_n).iterrows():
            logger.info(f"  {feature!s:<45} {row['mean_abs_shap']:>12,.0f} {row['mean_shap']:>+12,.0f}")
        logger.info(f"  Dollars of monthly change. Saved {csv_path.name} and {png_path.name}")


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
