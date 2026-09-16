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

    model              train_rmse   test_rmse    gap
    xgboost_scaled         30,112     160,455    5.3x   <- memorising
    ridge_change          185,220     190,455    1.0x   <- honest, maybe underfit
    three_month_average   178,900     180,900    1.0x   <- cannot overfit

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

    python pipeline.py              # train, test, and the combined table
    python pipeline.py --quiet      # the combined table only
"""

from __future__ import annotations

import argparse
import contextlib
import io
from typing import Any

import pandas as pd

from src.config.config import load_config
from src.data.data import build_dataset
from src.evaluate import EvaluationReport, EvaluationSettings
from src.metrics import Scores, anchor_values, score
from src.models_code.base_class import Model
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
    train_scores: dict[str, Scores], reports: dict[str, EvaluationReport]
) -> pd.DataFrame:
    """Put the in-sample and holdout numbers on one row per model.

    Parameters
    ----------
    train_scores : dict of str to Scores
        In-sample scores from ``in_sample_scores``.
    reports : dict of str to EvaluationReport
        Holdout reports from ``src.test.run``.

    Returns
    -------
    pandas.DataFrame
        One row per model, best holdout WAPE first. ``gap`` is holdout RMSE
        over in-sample RMSE: 1.0 means the model does the same on rows it has
        never seen, and anything much above that is memorisation.
    """
    rows = []
    for name, report in reports.items():
        fitted = train_scores.get(name)
        train_rmse = fitted.rmse if fitted is not None else float("nan")
        test_rmse = report.scores.rmse
        rows.append(
            {
                "model": name,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                # Guarded: a baseline can in principle score 0 in-sample, and a
                # divide-by-zero here would kill the run after all the work.
                "gap": test_rmse / train_rmse if train_rmse else float("nan"),
                "train_mae": fitted.mae if fitted is not None else float("nan"),
                "test_mae": report.scores.mae,
                "test_wape": report.scores.wape,
                "test_r2_change": report.scores.r2_change,
                "skill": report.scores.skill,
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("test_wape")


def run(config: dict[str, Any] | None = None, quiet: bool = False) -> pd.DataFrame:
    """Fit, score, and return the combined table.

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

    # Captured rather than silenced, so a traceback still carries whatever the
    # script had printed up to the point it failed.
    buffer = io.StringIO()
    context: Any = contextlib.redirect_stdout(buffer) if quiet else contextlib.nullcontext()

    if quiet:
        print("Training ...", flush=True)
    try:
        with context:
            models = run_train(settings)
    except Exception:
        if quiet:
            print(buffer.getvalue())
        raise

    if quiet:
        print("Scoring ...", flush=True)
    try:
        with context:
            reports = run_test(settings)
            train_scores = in_sample_scores(models, settings)
    except Exception:
        if quiet:
            print(buffer.getvalue())
        raise

    table = combined_table(train_scores, reports)

    print("\n" + "=" * 78)
    print("  TRAIN vs TEST")
    print("=" * 78)
    print(table.to_string(float_format=lambda value: f"{value:,.3f}"))
    print(
        "\n  train_*  scored on the rows each model was FITTED on. In-sample."
        "\n           Not a measure of quality -- only of how hard it fitted."
        "\n  test_*   scored on months no model has seen. This is the answer."
        "\n  gap      test_rmse / train_rmse. 1.0 is honest; 3x+ is memorisation."
        "\n  skill    versus the reference baseline. Positive means it earned its"
        "\n           complexity; around zero means it did not."
    )
    return table


def main() -> None:
    """Parse arguments and run the pipeline.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Fit every model, score it, and print train against test."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only the combined table",
    )
    arguments = parser.parse_args()
    run(quiet=arguments.quiet)


if __name__ == "__main__":
    main()
