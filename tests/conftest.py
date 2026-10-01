"""Shared fixtures: the saved models, the holdout split and both prediction files, loaded once."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.config.config import PROJECT_ROOT, load_config
from src.data.data import Dataset, build_dataset
from src.models_code.base_class import Model
from src.test import load_models, split_on_boundary, training_boundary

# Top-15 run in models/, full-feature run copied to models_full/.
PREDICTION_FILES = {
    "top15": PROJECT_ROOT / "models" / "predictions_test.csv",
    "full": PROJECT_ROOT / "models_full" / "predictions_test.csv",
}


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Return the active config.

    Returns
    -------
    dict
        Parsed ``config/ml_config.yaml``.
    """
    return load_config()


@pytest.fixture(scope="session")
def models(config: dict[str, Any]) -> dict[str, Model]:
    """Return every saved model in ``models/``.

    Parameters
    ----------
    config : dict
        Active config.

    Returns
    -------
    dict of str to Model
        Fitted models keyed by name.
    """
    return load_models(config)


@pytest.fixture(scope="session")
def split(config: dict[str, Any], models: dict[str, Model]) -> tuple[Dataset, Dataset]:
    """Return the training rows and the holdout, cut where the saved models stopped training.

    Parameters
    ----------
    config : dict
        Active config.
    models : dict of str to Model
        Saved models; their training months set the boundary.

    Returns
    -------
    tuple of (Dataset, Dataset)
        Training rows, holdout rows.
    """
    return split_on_boundary(build_dataset(config), training_boundary(models))


@pytest.fixture(scope="session")
def holdout(split: tuple[Dataset, Dataset]) -> pd.DataFrame:
    """Return holdout features next to persistence, xgboost top-15 and xgboost full.

    Parameters
    ----------
    split : tuple of (Dataset, Dataset)
        Training rows and holdout.

    Returns
    -------
    pandas.DataFrame
        Holdout feature rows plus ``actual``, ``actual_full``, ``persistence``,
        ``xgboost_top15`` and ``xgboost_full``.
    """
    test = split[1]
    keys = [test.id_column, test.time_column]
    top15 = pd.read_csv(PREDICTION_FILES["top15"], parse_dates=[test.time_column])
    full = pd.read_csv(PREDICTION_FILES["full"], parse_dates=[test.time_column])

    # Keep the actual from both runs so the alignment test can compare them.
    predictions = top15[keys + ["actual", "persistence", "xgboost"]].rename(
        columns={"xgboost": "xgboost_top15"}
    ).merge(
        full[keys + ["actual", "xgboost"]].rename(
            columns={"actual": "actual_full", "xgboost": "xgboost_full"}
        ),
        on=keys,
        how="outer",
    )
    return test.frame.merge(predictions, on=keys, how="inner")
