"""Per-entity robust scale, so one whale cannot own the loss function.

The problem this exists to solve is visible in a single pair of numbers from
the last run before it: holdout RMSE 136,969 against a median absolute error of
5,309. A 26x gap like that is not a model that is broadly wrong, it is a model
that is fine for most users and catastrophic for a handful -- and because the
training target was a raw dollar movement, those few users wrote most of the
gradient. A 500,000 USD swing on an enterprise account and a 500 USD swing on a
small one are the same event in behavioural terms, and the model never got to
see them that way.

Dividing each user's movement by that user's own typical movement fixes the
weighting: after scaling, a row's contribution to the loss reflects how unusual
the movement is *for that account*, which is the thing worth predicting.

**Why MAD and not standard deviation.** The scale is estimated from the very
distribution whose outliers are the problem. A standard deviation is computed
from squared deviations, so a single 500k spike inflates the divisor for that
user and quietly shrinks every other month of theirs toward zero -- the outlier
would end up suppressing the signal instead of being normalised by it. The
median absolute deviation ignores the spike entirely. The 1.4826 factor rescales
MAD so that it equals the standard deviation for normally distributed data,
which keeps the scaled target on a familiar footing.

**Why this is not in the feature layer.** A per-entity statistic is fitted, not
derived: it summarises a set of rows. Computing it in ``build_dataset`` would
compute it over the whole panel, holdout months included, and every fold's
"training" scale would carry information from months that fold is about to be
scored on. So it lives here, is fitted inside ``_fit`` on exactly the rows the
model was handed, and is refitted from scratch for every cross-validation fold
because the runner builds a new model per fold.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

# Rescales the median absolute deviation to be comparable with a standard
# deviation on normally distributed data. The conventional constant, 1/Phi^-1(3/4).
MAD_TO_SIGMA = 1.4826

# A user whose movements are all identical has a MAD of exactly zero, and a user
# with one or two training rows can have one by accident. Dividing by that is an
# infinity, so every scale is floored at this fraction of the panel-wide median
# scale: small enough that a genuinely quiet account still reads as quiet,
# large enough that the division stays finite.
DEFAULT_FLOOR_FRACTION = 0.01

# Rows needed before a user's own MAD is trusted. Below this the estimate is
# noise, and the panel-wide scale is the better guess.
DEFAULT_MIN_ROWS = 6


class EntityScaler:
    """Divides each entity's movement by that entity's own typical movement.

    Fitted on training rows only, then applied unchanged to whatever rows are
    predicted. An entity absent at fit time -- a user whose first month falls in
    the test window -- gets the panel-wide fallback rather than an error, so a
    new account does not break a prediction.

    Parameters
    ----------
    floor_fraction : float, optional
        Minimum scale, as a fraction of the panel-wide median scale. Default
        0.01.
    min_rows : int, optional
        Rows an entity needs before its own scale is used instead of the
        panel-wide one. Default 6.

    Attributes
    ----------
    scales_ : pandas.Series or None
        Scale per entity id, after flooring. None until fitted.
    fallback_ : float
        Scale used for entities not seen during fitting.
    """

    def __init__(
        self,
        floor_fraction: float = DEFAULT_FLOOR_FRACTION,
        min_rows: int = DEFAULT_MIN_ROWS,
    ) -> None:
        self.floor_fraction = float(floor_fraction)
        self.min_rows = int(min_rows)
        self.scales_: pd.Series | None = None
        self.fallback_: float = 1.0

    @property
    def is_fitted(self) -> bool:
        """Return whether ``fit`` has run.

        Returns
        -------
        bool
            True once scales are available.
        """
        return self.scales_ is not None

    def fit(self, entities: pd.Series, changes: pd.Series) -> EntityScaler:
        """Estimate a robust scale for every entity.

        Parameters
        ----------
        entities : pandas.Series
            Entity id per row, aligned with ``changes``.
        changes : pandas.Series
            The movement being scaled, in the target's own units. Missing
            values are ignored rather than filled: a user's first month has no
            movement, and inventing one would invent a scale.

        Returns
        -------
        EntityScaler
            Self, fitted.

        Raises
        ------
        ValueError
            If no row has a usable movement, which means there is nothing to
            estimate a scale from.
        """
        frame = pd.DataFrame(
            {"entity": entities.to_numpy(), "change": changes.to_numpy(dtype="float64")}
        )
        frame = frame.loc[np.isfinite(frame["change"].to_numpy(dtype="float64"))]
        if frame.empty:
            raise ValueError("EntityScaler: no finite movements to fit a scale on")

        grouped = frame.groupby("entity")["change"]
        raw = MAD_TO_SIGMA * grouped.apply(_median_absolute_deviation)
        counts = grouped.size()

        # The panel-wide number, computed over the entities that had enough rows
        # to be believable. It is both the fallback for unseen entities and the
        # basis of the floor, so it is taken before any flooring happens.
        trusted = raw.loc[(counts >= self.min_rows) & (raw > 0.0)]
        panel = float(trusted.median()) if len(trusted) else float(raw[raw > 0.0].median())
        if not np.isfinite(panel) or panel <= 0.0:
            # Every entity is perfectly flat. Nothing is scaled; the mode then
            # behaves exactly like plain change mode, which is the honest
            # outcome rather than a division by an invented number.
            panel = 1.0

        floor = panel * self.floor_fraction
        scales = raw.where(counts >= self.min_rows, panel)
        self.scales_ = scales.clip(lower=floor).astype("float64")
        self.fallback_ = panel
        return self

    def scale_for(self, entities: pd.Series) -> npt.NDArray[np.float64]:
        """Return the scale to use for each row.

        Parameters
        ----------
        entities : pandas.Series
            Entity id per row.

        Returns
        -------
        numpy.ndarray of float
            One positive scale per row, the fallback where the entity was not
            seen during fitting.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        if self.scales_ is None:
            raise RuntimeError("EntityScaler: scale_for called before fit")
        mapped = entities.map(self.scales_).astype("float64")
        return np.asarray(
            mapped.fillna(self.fallback_).to_numpy(), dtype=np.float64
        )

    def transform(
        self, entities: pd.Series, changes: pd.Series
    ) -> npt.NDArray[np.float64]:
        """Divide movements by their entity's scale.

        Parameters
        ----------
        entities : pandas.Series
            Entity id per row.
        changes : pandas.Series
            Movements in the target's own units.

        Returns
        -------
        numpy.ndarray of float
            Scaled movements. NaN stays NaN, so the caller's usable-row mask
            still finds the rows with no movement to learn from.
        """
        return changes.to_numpy(dtype="float64") / self.scale_for(entities)

    def inverse_transform(
        self, entities: pd.Series, scaled: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Put scaled movements back into the target's own units.

        Parameters
        ----------
        entities : pandas.Series
            Entity id per row.
        scaled : numpy.ndarray of float
            Model output on the scaled axis.

        Returns
        -------
        numpy.ndarray of float
            Movements in dollars.
        """
        return np.asarray(scaled, dtype="float64") * self.scale_for(entities)

    def describe(self) -> str:
        """Return a one-line summary for the run log.

        Returns
        -------
        str
            Entity count and the quartiles of the fitted scales, which is what
            tells you whether the panel really does span orders of magnitude.
        """
        if self.scales_ is None:
            return "EntityScaler(unfitted)"
        quartiles = self.scales_.quantile([0.25, 0.5, 0.75])
        return (
            f"EntityScaler({len(self.scales_)} entities, "
            f"scale p25/p50/p75 {quartiles.iloc[0]:,.0f}/"
            f"{quartiles.iloc[1]:,.0f}/{quartiles.iloc[2]:,.0f}, "
            f"fallback {self.fallback_:,.0f})"
        )


def _median_absolute_deviation(values: pd.Series) -> float:
    """Return the median absolute deviation from the median.

    Parameters
    ----------
    values : pandas.Series
        Finite movements for one entity.

    Returns
    -------
    float
        The MAD, or 0.0 for an empty group. Unscaled -- the caller applies the
        1.4826 factor once rather than per group.
    """
    array = values.to_numpy(dtype="float64")
    if array.size == 0:
        return 0.0
    return float(np.median(np.abs(array - np.median(array))))
