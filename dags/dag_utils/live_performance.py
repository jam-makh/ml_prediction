"""Live performance: resolve served predictions against real balances and score them."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from dag_utils.artifacts import SplitScores
from dag_utils.db import ServingRepository


def refresh(repository: ServingRepository) -> int:
    """Resolve every prediction whose actual balance is now known; safe to rerun.

    Parameters
    ----------
    repository : ServingRepository
        Serving tables.

    Returns
    -------
    int
        Predictions newly resolved.
    """
    resolved = repository.resolve_live()
    logger.info(f"resolved {resolved} predictions" if resolved else "nothing new resolved")
    return resolved


def live_scores(errors: pd.DataFrame) -> SplitScores:
    """Score resolved rows from their stored signed error and actual.

    Parameters
    ----------
    errors : pandas.DataFrame
        Rows of ``live_performance`` with ``error`` and ``actual``.

    Returns
    -------
    SplitScores
        MAE, RMSE and WAPE; NaN metrics when there are no rows.
    """
    if errors.empty:
        return SplitScores(mae=float("nan"), rmse=float("nan"), wape=float("nan"), n_rows=0)
    error = errors["error"].to_numpy(dtype="float64")
    actual = errors["actual"].to_numpy(dtype="float64")
    denominator = float(np.sum(np.abs(actual)))
    return SplitScores(
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(np.square(error)))),
        # Same definition as src.metrics.wape, so live and holdout WAPE compare directly.
        wape=100.0 * float(np.sum(np.abs(error))) / denominator if denominator else float("nan"),
        n_rows=int(error.size),
    )
