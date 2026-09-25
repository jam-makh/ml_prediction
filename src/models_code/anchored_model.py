"""Regressions fitted to the monthly change rather than the balance itself.

``AnchoredModel`` is what Ridge and XGBoost share: they learn target minus last
month's balance, and add last month's balance back when predicting.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.data.data import Dataset
from src.models_code.base_class import Model


class AnchoredModel(Model):
    """A regression fitted to the change from last month's balance.

    Subclasses implement ``_fit`` and ``_predict``; ``_predict`` ends with
    ``return self._restore_level(dataset, predicted)``.

    Parameters
    ----------
    name : str
        Label used in result tables and saved filenames.
    anchor_column : str, optional
        Column holding last month's balance.

    Attributes
    ----------
    best_params_ : dict or None
        Penalty or booster settings the model was fitted with, or None.
    optuna_ : dict or None
        The Optuna trial that chose this model's settings, set by ``train.py``.
    """

    # Both regressions take the seed and the anchor from the run, not per model.
    SHARED_ARGUMENTS: ClassVar[tuple[str, ...]] = ("random_state", "anchor_column")

    def __init__(self, name: str, anchor_column: str = "prev_1m_closing_balance_usd") -> None:
        super().__init__(name)
        self.anchor_column = anchor_column
        self.best_params_: dict[str, Any] | None = None
        self.optuna_: dict[str, Any] | None = None

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return the columns needed at prediction time.

        Returns
        -------
        tuple of str
            Feature columns plus the anchor column.
        """
        # dict.fromkeys drops duplicates, since the anchor is usually also a feature.
        return tuple(dict.fromkeys([*self._feature_columns, self.anchor_column]))

    def _training_target(self, dataset: Dataset) -> pd.Series:
        """Return the change the estimator is fitted against.

        Parameters
        ----------
        dataset : Dataset
            Training rows.

        Returns
        -------
        pandas.Series
            Target minus anchor; NaN on a user's first month, which has no anchor.

        Raises
        ------
        KeyError
            If the anchor column is absent.
        """
        if self.anchor_column not in dataset.frame.columns:
            raise KeyError(f"{self.name}: needs {self.anchor_column!r}, which is not in the data")
        return dataset.target - dataset.frame[self.anchor_column]

    def _restore_level(
        self, dataset: Dataset, predicted: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Turn a predicted change back into a balance.

        Parameters
        ----------
        dataset : Dataset
            The rows that were predicted.
        predicted : numpy.ndarray of float
            Raw estimator output, a change.

        Returns
        -------
        numpy.ndarray of float
            Anchor plus change; the raw change where the anchor is missing, so
            ``predict``'s finiteness check still passes.
        """
        anchor = dataset.frame[self.anchor_column].to_numpy(dtype="float64")
        return np.where(np.isfinite(anchor), anchor + predicted, predicted)

    def _usable_rows(self, target: pd.Series) -> npt.NDArray[np.bool_]:
        """Return the mask of rows with a target the estimator can learn from.

        Parameters
        ----------
        target : pandas.Series
            Output of ``_training_target``.

        Returns
        -------
        numpy.ndarray of bool
            True where the target is present.

        Raises
        ------
        ValueError
            If no row has a usable target.
        """
        usable = target.notna().to_numpy(dtype=bool)
        if not usable.any():
            raise ValueError(f"{self.name}: no rows with a usable target")
        return usable
