"""Retrain logic: load and hash the data, split it in time, refit the routed model, score it."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from dag_utils.artifacts import SplitScores
from dag_utils.routed_model import RoutedModel
from src.config.config import PROJECT_ROOT
from src.data.data import Dataset, build_dataset
from src.metrics import score
from src.models_code.baseline import LagAverageBaseline
from src.models_code.xgboost_model import XGBoostModel
from src.window import SplitSettings, holdout_split


def data_hash(frame: pd.DataFrame) -> str:
    """Return a SHA-256 of the frame's values and column names, independent of its index.

    Parameters
    ----------
    frame : pandas.DataFrame
        Training rows.

    Returns
    -------
    str
        64-character hex digest.
    """
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    # Column names are hashed too, so a renamed or added column changes the hash.
    digest.update("|".join(frame.columns).encode("utf-8"))
    return digest.hexdigest()


def load_dataset(config: dict[str, Any]) -> Dataset:
    """Load the feature table with xgboost's feature columns.

    Parameters
    ----------
    config : dict
        Parsed ``ml_config.yaml``.

    Returns
    -------
    Dataset
        Every row with a target, sorted by user then month.
    """
    return build_dataset(config, drop_columns_for="xgboost")


def split(dataset: Dataset, config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    """Cut the last ``split.test_share`` of months off as the holdout.

    Parameters
    ----------
    dataset : Dataset
        Full panel.
    config : dict
        Parsed ``ml_config.yaml``.

    Returns
    -------
    tuple of (Dataset, Dataset)
        Training rows, holdout rows.
    """
    settings = SplitSettings.from_config(config)
    cut = holdout_split(dataset, settings.test_share, settings.gap_months)
    return dataset.take(cut.train_positions), dataset.take(cut.test_positions)


def saved_params(config: dict[str, Any]) -> dict[str, Any]:
    """Read the tuned xgboost parameters, without the training-data-specific Huber slope.

    Parameters
    ----------
    config : dict
        Parsed ``ml_config.yaml``; ``serving.params_file`` names the JSON.

    Returns
    -------
    dict
        Keyword arguments for ``XGBoostModel``.
    """
    path = PROJECT_ROOT / str(config["serving"]["params_file"])
    params = dict(json.loads(path.read_text(encoding="utf-8"))["xgboost_params"])
    # The slope is the old training rows' median change; dropping it lets the fit recompute it.
    params.pop("huber_slope", None)
    return params


def build_model(config: dict[str, Any], params: dict[str, Any]) -> RoutedModel:
    """Return an unfitted routed model with the saved xgboost parameters.

    Parameters
    ----------
    config : dict
        Parsed ``ml_config.yaml``.
    params : dict
        Output of ``saved_params``.

    Returns
    -------
    RoutedModel
        Persistence plus xgboost, unfitted.
    """
    anchor = str(config["evaluation"]["anchor_column"])
    return RoutedModel(
        xgboost=XGBoostModel(name="xgboost", anchor_column=anchor, **params),
        persistence=LagAverageBaseline(n_months=1, name="persistence"),
    )


def xgboost_scores(model: RoutedModel, dataset: Dataset) -> SplitScores:
    """Score the model on the rows it routes to xgboost, the only rows two versions differ on.

    Parameters
    ----------
    model : RoutedModel
        Fitted model.
    dataset : Dataset
        Rows to score.

    Returns
    -------
    SplitScores
        MAE, RMSE and WAPE over the xgboost rows.

    Raises
    ------
    ValueError
        If no row is routed to xgboost.
    """
    rows = dataset.take(np.flatnonzero((model.route(dataset) == "xgboost").to_numpy()))
    if rows.n_rows == 0:
        raise ValueError("no rows are routed to xgboost; nothing to score")
    result = score(rows.target, model.predict(rows))
    return SplitScores(mae=result.mae, rmse=result.rmse, wape=result.wape, n_rows=result.n_rows)
