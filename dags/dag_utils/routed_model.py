"""The served model: persistence for short histories, xgboost for the rest, one saved file."""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.data.data import Dataset
from src.models_code.base_class import Model
from src.models_code.baseline import LagAverageBaseline
from src.models_code.xgboost_model import XGBoostModel

Route = Literal["skip", "persistence", "xgboost"]

# Id 0 of seg_history: fewer than segments.history.min_months of balances, or a missing value.
UNASSIGNED_HISTORY = 0


class RoutedModel(Model):
    """Route each row to persistence or xgboost by its history segment.

    Parameters
    ----------
    xgboost : XGBoostModel
        Unfitted booster used for rows with enough history.
    persistence : LagAverageBaseline
        Unfitted last-month baseline used for short histories.
    route_column : str, optional
        Segment column deciding the route. Default ``history_id``.
    name : str, optional
        Label for logs. Default ``routed``.
    """

    def __init__(
        self,
        xgboost: XGBoostModel,
        persistence: LagAverageBaseline,
        route_column: str = "history_id",
        name: str = "routed",
    ) -> None:
        super().__init__(name)
        self.xgboost = xgboost
        self.persistence = persistence
        self.route_column = route_column
        # Persistence reads exactly one lag, last month's balance.
        self.anchor_column = persistence.columns[0]
        # Set by the retrain before saving, so the file and its metadata.json can be matched.
        self.version = ""

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return the feature columns plus the routing and anchor columns.

        Returns
        -------
        tuple of str
            Columns any dataset passed to ``fit`` or ``predict`` must hold.
        """
        extra = (self.route_column, self.anchor_column, self.xgboost.anchor_column)
        return tuple(dict.fromkeys(self._feature_columns + extra))

    def route(self, dataset: Dataset) -> pd.Series:
        """Return the route of every row.

        Parameters
        ----------
        dataset : Dataset
            Rows to route.

        Returns
        -------
        pandas.Series
            ``skip`` without last month's balance, ``persistence`` for an unassigned
            history, ``xgboost`` otherwise; aligned to the dataset's rows.
        """
        frame = dataset.frame
        routes = pd.Series("xgboost", index=frame.index, dtype="object")
        routes[frame[self.route_column] == UNASSIGNED_HISTORY] = "persistence"
        # No balance last month means nothing to persist or to anchor a change on.
        routes[frame[self.anchor_column].isna()] = "skip"
        return routes

    def _fit(self, dataset: Dataset) -> None:
        """Fit both inner models on every training row, as the standalone models were.

        Parameters
        ----------
        dataset : Dataset
            Training rows, feature columns already narrowed for xgboost.

        Returns
        -------
        None
        """
        self.persistence.fit(dataset)
        self.xgboost.fit(dataset)

    def _predict(self, dataset: Dataset) -> npt.NDArray[np.float64]:
        """Predict each row with the model its route names.

        Parameters
        ----------
        dataset : Dataset
            Rows to predict; drop ``skip`` rows first, as they have no usable anchor.

        Returns
        -------
        numpy.ndarray of float
            One prediction per row, in row order.
        """
        use_xgboost = (self.route(dataset) == "xgboost").to_numpy()
        # Both models score every row; the route then picks one value per row.
        return np.where(
            use_xgboost, self.xgboost.predict(dataset), self.persistence.predict(dataset)
        )
