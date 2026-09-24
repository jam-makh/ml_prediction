"""Score the saved models on months none of them was trained on.

The counterpart to ``train.py``, and deliberately a separate script. This one
never fits anything: it loads what training saved, works out which rows are
genuinely unseen, scores every model on exactly those rows, and prints one
table so the three can be read against each other.

**Where the boundary comes from.** Not from the config. Each saved model
carries the months it was fitted on, so this script asks the models and scores
everything strictly after the last of them. That matters because the obvious
alternative -- recompute "the last N months" from the config -- has two failure
modes, both silent:

* someone edits ``split.test_share`` between the two runs, the boundary moves,
  and the models are scored on months they were trained on. The numbers improve.
  Nothing warns.
* new rows land in the feature table between the two runs, the last N months
  become a different N months, and the boundary slides backwards into the
  training region.

Asking the artefact removes both. A model cannot be wrong about what it saw.
When new months arrive, the test set simply grows to include them, which is
correct: those months really are unseen.

**The comparison is the point.** The three models are scored through identical
code on identical rows: the 3-month average that any spreadsheet could produce,
a linear model, and the booster. A baseline that gets its own scoring path is a
baseline nobody can trust, and without a trustworthy baseline an RMSE is a
number with nothing to be compared to.

Run it with::

    python -m src.test
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.log import setup_logging
from src.config.config import load_config, resolve_output_dir
from src.data.data import Dataset, build_dataset
from src.evaluate import (
    AMOUNT_COLUMNS,
    MOVEMENT_COLUMNS,
    EvaluationReport,
    EvaluationSettings,
    evaluate_models,
    report_table,
    results_table,
    tier_table,
    worst_months_table,
    worst_users_table,
)
from src.metrics import assign_tiers
from src.models_code.base_class import Model


def load_models(config: dict[str, Any]) -> dict[str, Model]:
    """Load every model the config names, from the directory training wrote to.

    Parameters
    ----------
    config : dict
        Parsed config. The ``models`` block supplies the roster of names, which
        is also the set of filenames to look for.

    Returns
    -------
    dict of str to Model
        Fitted models, keyed by name, in config order.

    Raises
    ------
    FileNotFoundError
        If any configured model has no saved file, naming all of the missing
        ones at once rather than failing on the first. The usual cause is that
        ``train.py`` has not been run since the model was added to the config.
    ValueError
        If the ``models`` block is empty.
    """
    entries = config.get("models") or []
    names = [str(entry["name"]) for entry in entries]
    if not names:
        raise ValueError("The models block is empty; there is nothing to score")

    output_dir = resolve_output_dir(config)
    missing = [name for name in names if not (output_dir / f"{name}.joblib").exists()]
    if missing:
        raise FileNotFoundError(
            f"No saved model for {missing} in {output_dir}. "
            f"Run `python -m src.train` first."
        )

    return {name: Model.load(output_dir / f"{name}.joblib") for name in names}


def training_boundary(models: dict[str, Model]) -> pd.Timestamp:
    """Return the last month any of the models was trained on.

    Every model in a run is fitted on the same training region, so these should
    all agree. They are checked rather than assumed, because the one case where
    they disagree is the one that matters: a stale artefact left behind from an
    earlier run with a different split, which would otherwise be scored on
    months it had already seen while the rest were not.

    The maximum is taken rather than the minimum. If the artefacts do somehow
    differ, scoring after the *latest* of them is the choice that keeps every
    model out of sample; the minimum would keep the comparison wide at the cost
    of letting one model see its own training months.

    Parameters
    ----------
    models : dict of str to Model
        The loaded models.

    Returns
    -------
    pandas.Timestamp
        The last training month.

    Raises
    ------
    ValueError
        If a model records no training months, or if the models disagree about
        where training ended.
    """
    through: dict[str, pd.Timestamp] = {}
    for name, model in models.items():
        last = model.trained_through
        if last is None:
            raise ValueError(
                f"{name} carries no training months, so there is no way to "
                f"tell which rows it has seen. Re-run `python -m src.train`."
            )
        through[name] = last

    distinct = set(through.values())
    if len(distinct) > 1:
        detail = ", ".join(
            f"{name} through {month:%Y-%m}" for name, month in sorted(through.items())
        )
        raise ValueError(
            f"The saved models were not trained on the same region ({detail}). "
            f"One of them is stale. Re-run `python -m src.train` so all of "
            f"them are fitted on the same months."
        )
    return max(distinct)


def split_on_boundary(
    dataset: Dataset, boundary: pd.Timestamp
) -> tuple[Dataset, Dataset]:
    """Cut the panel at the month training ended.

    Parameters
    ----------
    dataset : Dataset
        The full panel.
    boundary : pandas.Timestamp
        The last training month, from ``training_boundary``.

    Returns
    -------
    tuple of (Dataset, Dataset)
        The rows at or before the boundary, and the rows strictly after it.
        The first is not scored; it is needed to cut the account-size tiers on
        training data only.

    Raises
    ------
    ValueError
        If no rows fall after the boundary, which means the models were trained
        through the end of the available data and there is nothing unseen left
        to score them on.
    """
    months = dataset.times
    train_positions = np.flatnonzero((months <= boundary).to_numpy(dtype=bool))
    test_positions = np.flatnonzero((months > boundary).to_numpy(dtype=bool))

    if test_positions.size == 0:
        raise ValueError(
            f"Every row in the data is at or before {boundary:%Y-%m}, the last "
            f"month the models were trained on. There is nothing out of sample "
            f"to score. Either new months have not landed yet, or the models "
            f"were trained on the whole panel."
        )
    return (
        dataset.take(train_positions.astype(np.intp)),
        dataset.take(test_positions.astype(np.intp)),
    )


def leaderboard(
    reports: dict[str, EvaluationReport],
    columns: dict[str, str],
    sort_by: str = "wape",
) -> pd.DataFrame:
    """Lay the models out as one short table, best first.

    Parameters
    ----------
    reports : dict of str to EvaluationReport
        One report per model.
    columns : dict of str to str
        ``AMOUNT_COLUMNS`` or ``MOVEMENT_COLUMNS``.
    sort_by : str, optional
        Column to rank on. Default ``wape``.

    Returns
    -------
    pandas.DataFrame
        One row per model: ``r2``, ``mae``, ``rmse`` and ``wape``.
    """
    return results_table(report_table(reports), columns, sort_by=sort_by)


def run(config: dict[str, Any] | None = None) -> dict[str, EvaluationReport]:
    """Score every saved model on the unseen months and print the comparison.

    Parameters
    ----------
    config : dict, optional
        Parsed config. Loaded from ``config/ml_config.yaml`` when omitted.

    Returns
    -------
    dict of str to EvaluationReport
        One report per model, keyed by name.
    """
    settings = config if config is not None else load_config()
    evaluation = EvaluationSettings.from_config(settings)

    models = load_models(settings)
    boundary = training_boundary(models)

    dataset = build_dataset(settings)
    seen, test = split_on_boundary(dataset, boundary)

    logger.info(f"Data      : {dataset.summary()}")
    logger.info(f"Models    : {', '.join(models)}")
    logger.info(f"Trained   : through {boundary:%Y-%m} (read from the saved models)")
    logger.info(
        f"Scoring   : {test.n_rows:,} rows over "
        f"{test.months[0]:%Y-%m}..{test.months[-1]:%Y-%m}, "
        f"{len(test.months)} month(s) no model has seen"
    )

    # Account-size tiers, cut on the rows the models actually trained on.
    # Tiering on the whole panel would let an account's unseen months help
    # decide which bucket its unseen months are then scored in. Computed once
    # and handed to every report, so the tier rows are comparable down the
    # table.
    tiers = assign_tiers(
        seen.entities,
        seen.frame[evaluation.anchor_column],
        quantiles=evaluation.tier_quantiles,
    )

    # evaluate_models re-checks, per model, that no scored month was trained
    # on. The boundary above already guarantees it; this is the assertion that
    # sits next to the number actually being quoted.
    reports = evaluate_models(
        models,
        test,
        anchor_column=evaluation.anchor_column,
        reference_model=evaluation.reference_model,
        mape_floor=evaluation.mape_floor,
        tiers=tiers,
    )

    fmt = lambda value: f"{value:,.3f}"  # noqa: E731

    logger.info(f"\n=== Amounts: the balance itself (sorted by {evaluation.headline_metric})")
    table = leaderboard(reports, AMOUNT_COLUMNS, sort_by=evaluation.headline_metric)
    logger.info(table.to_string(float_format=fmt))

    logger.info(
        f"\n=== Movements: change from last month's balance "
        f"(sorted by {evaluation.headline_metric})"
    )
    movements = leaderboard(
        reports, MOVEMENT_COLUMNS, sort_by=evaluation.headline_metric
    )
    logger.info(movements.to_string(float_format=fmt))
    logger.info(
        "\n  wape is a percentage. r2 below 0 on movements means worse than "
        "predicting no change.\n  mae and rmse are the same dollar errors in "
        "both tables, less the rows with no previous month."
    )

    logger.info("\n=== Per model")
    for name in table.index:
        logger.info(f"\n  {reports[str(name)].describe()}")

    # Where the winner is worst, which is the part a single number cannot say.
    best_name = str(table.index[0])
    best = reports[best_name]

    logger.info(f"\n=== {best_name}: months scored, worst first")
    logger.info(
        worst_months_table(best).to_string(float_format=lambda value: f"{value:,.0f}")
    )

    tiers_table = tier_table(best)
    if not tiers_table.empty:
        # The breakdown the global metrics cannot give: a model that looks
        # adequate overall can be adequate on the enterprise tier that
        # dominates the dollar totals and useless on the half of accounts that
        # are small.
        logger.info(f"\n=== {best_name}: by account-size tier")
        logger.info(tiers_table.to_string(float_format=lambda value: f"{value:,.1f}"))

    logger.info(f"\n=== {best_name}: worst users")
    logger.info(worst_users_table(best).to_string(float_format=lambda value: f"{value:,.0f}"))

    return reports


def main() -> None:
    """Score the saved models.

    Returns
    -------
    None
    """
    settings = load_config()
    logger.info(f"Logging to {setup_logging(settings, run_name='test')}")
    run(settings)


if __name__ == "__main__":
    main()
