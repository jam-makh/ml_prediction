"""Regressions fitted to the monthly change rather than the balance itself.

``AnchoredModel`` is what Ridge and XGBoost share: the change target, its
optional clip, market scale and recency weights, and rebuilding the balance.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.data.data import Dataset
from src.models_code.base_class import Model

# The balance lags the market scale is read from; consecutive pairs give each row's last two movements.
DEFAULT_MARKET_LAG_COLUMNS: tuple[str, ...] = (
    "prev_1m_closing_balance_usd",
    "prev_2m_closing_balance_usd",
    "prev_3m_closing_balance_usd",
)

# A month with fewer lag movements than this falls back to the last training month's scale.
MARKET_SCALE_MIN_ROWS = 20


def market_scale_by_month(
    frame: pd.DataFrame, time_column: str, lag_columns: tuple[str, ...]
) -> pd.Series:
    """Return each month's median absolute lagged movement across the panel.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows carrying ``time_column`` and every column in ``lag_columns``.
    time_column : str
        The month column.
    lag_columns : tuple of str
        Balance lags, most recent first.

    Returns
    -------
    pandas.Series
        Scale per month; NaN where fewer than ``MARKET_SCALE_MIN_ROWS`` movements
        exist or the median is 0.
    """
    moves = pd.concat(
        [
            pd.DataFrame(
                {
                    "month": frame[time_column],
                    "move": (frame[newer] - frame[older]).abs(),
                }
            )
            for newer, older in zip(lag_columns, lag_columns[1:])
        ]
    ).dropna()
    grouped = moves.groupby("month")["move"]
    scale = grouped.median()
    scale[(grouped.count() < MARKET_SCALE_MIN_ROWS) | (scale <= 0)] = np.nan
    return scale


def parse_clip(clip: float | str | None) -> float | str | None:
    """Validate a ``clip`` setting and return it in canonical form.

    Parameters
    ----------
    clip : float, str or None
        A positive dollar cap, a quantile written ``"q0.995"``, or None. The
        strings ``"none"`` and ``""`` also mean None.

    Returns
    -------
    float, str or None
        A float dollar cap, the quantile string, or None.

    Raises
    ------
    ValueError
        If the cap is not positive or the quantile is not in (0.5, 1).
    """
    if clip is None or (isinstance(clip, str) and clip.strip().lower() in ("", "none")):
        return None
    if isinstance(clip, str):
        text = clip.strip().lower()
        if not text.startswith("q"):
            return parse_clip(float(text))
        level = float(text[1:])
        if not 0.5 < level < 1.0:
            raise ValueError(f"clip quantile must be in (0.5, 1), got {clip!r}")
        return f"q{level:g}"
    cap = float(clip)
    if cap <= 0:
        raise ValueError(f"clip must be positive, got {clip!r}")
    return cap


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
    clip : float or str, optional
        Cap on the training change, in dollars or as a quantile (``"q0.995"``).
        Training target only; predictions and scoring never see it. Default None.
    market_scale : bool, optional
        Divide the training change by the panel's typical movement that month,
        and multiply predictions back. Default False.
    recency_half_life : float, optional
        Weight training rows by ``0.5 ** (age_in_months / half_life)``. Default
        None, equal weights.
    market_lag_columns : tuple of str, optional
        The balance lags ``market_scale`` reads.

    Attributes
    ----------
    best_params_ : dict or None
        Parameters the fit chose itself (Ridge's alpha grid), or None.
    optuna_ : dict or None
        The Optuna trial that chose this model's settings, set by ``train.py``.
    """

    # Both regressions take the seed and the anchor from the run, not per model.
    SHARED_ARGUMENTS: ClassVar[tuple[str, ...]] = ("random_state", "anchor_column")

    def __init__(
        self,
        name: str,
        anchor_column: str = "prev_1m_closing_balance_usd",
        clip: float | str | None = None,
        market_scale: bool = False,
        recency_half_life: float | None = None,
        market_lag_columns: tuple[str, ...] = DEFAULT_MARKET_LAG_COLUMNS,
    ) -> None:
        super().__init__(name)
        self.anchor_column = anchor_column
        self.clip = parse_clip(clip)
        self.market_scale = bool(market_scale)
        self.recency_half_life = (
            float(recency_half_life) if recency_half_life is not None else None
        )
        self.market_lag_columns = tuple(market_lag_columns)

        self.best_params_: dict[str, Any] | None = None
        self.optuna_: dict[str, Any] | None = None

        # Fitted in _training_target, which only _fit reaches, so they never see held-out rows.
        self.clip_cap_: float | None = None
        self.fallback_market_scale_: float | None = None

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return the columns needed at prediction time.

        Returns
        -------
        tuple of str
            Feature columns, the anchor, and the balance lags when ``market_scale`` is on.
        """
        columns = [*self._feature_columns, self.anchor_column]
        if self.market_scale:
            columns.extend(self.market_lag_columns)
        # dict.fromkeys drops duplicates, since the anchor is usually also a feature.
        return tuple(dict.fromkeys(columns))

    def _training_target(self, dataset: Dataset) -> pd.Series:
        """Return the change the estimator is fitted against.

        Parameters
        ----------
        dataset : Dataset
            Training rows.

        Returns
        -------
        pandas.Series
            Target minus anchor, clipped and market-scaled when configured. NaN on
            a user's first month, which has no anchor.

        Raises
        ------
        KeyError
            If the anchor column is absent.
        """
        if self.anchor_column not in dataset.frame.columns:
            raise KeyError(f"{self.name}: needs {self.anchor_column!r}, which is not in the data")

        change = self._clip_change(dataset.target - dataset.frame[self.anchor_column])
        if self.market_scale:
            return change / self._market_scale(dataset, fit=True)
        return change

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
        movement = predicted
        if self.market_scale:
            movement = predicted * self._market_scale(dataset).to_numpy(dtype="float64")
        return np.where(np.isfinite(anchor), anchor + movement, movement)

    def _clip_change(self, change: pd.Series) -> pd.Series:
        """Cap the training change at ``clip``, fitting a quantile cap if needed.

        Parameters
        ----------
        change : pandas.Series
            Dollar change per training row.

        Returns
        -------
        pandas.Series
            Both tails capped at the same absolute value, or unchanged when ``clip`` is None.
        """
        if self.clip is None:
            return change
        if isinstance(self.clip, str):
            cap = float(change.abs().quantile(float(self.clip[1:])))
        else:
            cap = float(self.clip)
        self.clip_cap_ = cap
        return change.clip(lower=-cap, upper=cap)

    def _market_scale(self, dataset: Dataset, fit: bool = False) -> pd.Series:
        """Return the market scale for every row of ``dataset``.

        Parameters
        ----------
        dataset : Dataset
            Rows to scale; each month's scale comes from its own rows' lags.
        fit : bool, optional
            True during training: records the last training month's scale as the fallback.

        Returns
        -------
        pandas.Series
            One positive scale per row, aligned to ``dataset.frame``.

        Raises
        ------
        ValueError
            If no training month yields a scale.
        """
        by_month = market_scale_by_month(
            dataset.frame, dataset.time_column, self.market_lag_columns
        )
        if fit:
            measured = by_month.dropna()
            if measured.empty:
                raise ValueError(f"{self.name}: market_scale found no usable month")
            self.fallback_market_scale_ = float(measured.iloc[-1])
        per_row = dataset.frame[dataset.time_column].map(by_month)
        return per_row.fillna(self.fallback_market_scale_).astype("float64")

    def _training_weights(self, dataset: Dataset) -> npt.NDArray[np.float64] | None:
        """Return per-row recency weights for the training rows, or None.

        Parameters
        ----------
        dataset : Dataset
            Training rows.

        Returns
        -------
        numpy.ndarray of float or None
            ``0.5 ** (age / recency_half_life)``, age in months back from the
            latest training month; None when no half-life is set.
        """
        if self.recency_half_life is None:
            return None
        months = dataset.frame[dataset.time_column]
        latest = months.max()
        age = (latest.year - months.dt.year) * 12 + (latest.month - months.dt.month)
        return np.power(0.5, age.to_numpy(dtype="float64") / self.recency_half_life)

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
