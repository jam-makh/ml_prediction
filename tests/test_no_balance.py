"""No Balance Data: with last month's balance removed, each model must fall back to the last known balance.

The fallback is prev_2m, then prev_3m, then the training median. A model passes if its
error stays within ``TOLERANCE`` times the error of that fallback on its own.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config.config import load_config
from src.data.data import Dataset
from src.models_code.base_class import Model
from src.models_code.baseline import BALANCE_LAGS

# Room for a model's predicted change to add some error on top of the fallback balance.
TOLERANCE = 1.5
MODEL_NAMES = [str(entry["name"]) for entry in load_config().get("models") or []]


def last_known_balance(test: Dataset, train: Dataset) -> np.ndarray:
    """Return the newest of prev_2m and prev_3m per row, else the training median.

    Parameters
    ----------
    test : Dataset
        Holdout rows.
    train : Dataset
        Training rows, for the median.

    Returns
    -------
    numpy.ndarray of float
        One fallback balance per holdout row.
    """
    # bfill along columns picks prev_2m first, prev_3m when prev_2m is missing.
    older = test.frame[list(BALANCE_LAGS[1:])].bfill(axis=1).iloc[:, 0]
    return older.fillna(float(train.target.median())).to_numpy(dtype="float64")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_falls_back_to_last_known_balance(
    name: str, models: dict[str, Model], split: tuple[Dataset, Dataset]
) -> None:
    """Blank last month's balance and check the model stays near the last-known-balance fallback.

    Parameters
    ----------
    name : str
        Model to check.
    models : dict of str to Model
        Saved models.
    split : tuple of (Dataset, Dataset)
        Training rows and holdout.
    """
    train, test = split
    # No holdout row lacks last month's balance, so the case is made by blanking it.
    blanked = test.model_copy(
        update={"frame": test.frame.assign(**{BALANCE_LAGS[0]: np.nan})}
    )
    truth = test.target.to_numpy(dtype="float64")

    model_mae = float(np.mean(np.abs(truth - models[name].predict(blanked))))
    fallback_mae = float(np.mean(np.abs(truth - last_known_balance(test, train))))
    assert model_mae <= TOLERANCE * fallback_mae, (
        f"{name}: MAE {model_mae:,.0f} with no last balance, against {fallback_mae:,.0f} "
        f"for the last known balance; it is not falling back to an older balance"
    )
