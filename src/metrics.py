"""Scoring, and the two framings this problem has to be read in.

There is a way to score this task that makes everything look excellent,
including a prediction a person could do in their head. Scoring the balance
**level**, a user's balance next month is mostly their balance this month, and
most of the variance in the target is between users rather than within one. So
R squared over the level is measuring "can the model tell a large account from a
small one" rather than "can it forecast a month". It comes out at 1.000 for the
trivial baseline and stays there for everything after it.

The correction is smaller than it first looks, and it is worth being precise
about why, because the obvious version of it does nothing at all.

Subtracting last month's balance from both the truth and the prediction does
**not** change the error:

    (y - anchor) - (prediction - anchor) = y - prediction

RMSE, MAE and the median absolute error are identical in both framings, and so
is any skill score built from them. Reporting them twice under two headings
would be two columns of the same number wearing different hats.

What the subtraction changes is the **denominator**. R squared compares the
model's squared error against the variance of whatever it is asked to explain,
and there are two candidates:

``r2_level``
    Against the spread of balances across users. Enormous, so almost any error
    looks small against it. This is the flattering number.

``r2_change``
    Against the spread of month to month movements. That is the quantity
    actually being forecast, and it is where a model either beats "assume no
    change" or is exposed. A negative value means the model is doing worse than
    predicting that nothing moves.

So one set of dollar errors, reported once, and two R squareds that disagree.
The disagreement is the finding.

One metric is deliberately near-absent. MAPE and its relatives divide by the
true value, and these balances cross zero and sit near it for plenty of users,
so the percentage error explodes on exactly the rows where the dollar error is
smallest. A version restricted to rows above a floor is available, reported
alongside the share of rows it could be computed on, so it cannot be quoted
without its caveat attached.

The formulas, with e_i = y_i - yhat_i::

    RMSE       sqrt( mean( e_i^2 ) )
    MAE        mean( |e_i| )
    median AE  median( |e_i| )
    bias       mean( e_i )                    signed, in evaluate.py
    R2         1 - sum(e_i^2) / sum( (y_i - ybar)^2 )
    skill      1 - RMSE_model / RMSE_reference

``r2_level`` puts the balance in the denominator, ``r2_change`` puts the
movement ``y_i - anchor_i`` there. Skill compares two models directly and needs
no denominator of its own, which is why it survives a skewed target better than
either R squared.

Which of these to quote is a judgement, and it belongs with the data rather than
in a rule. On this panel the ten worst users carry roughly three quarters of the
total squared error, so RMSE ranks models mostly by how well they fit ten
accounts out of a hundred and fifty. The median absolute error is the default
ranking (``evaluation.headline_metric``) because it describes the typical user,
with RMSE kept beside it because the tail is somebody's problem too.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict

from src.data.data import Dataset

# Rows whose true balance is smaller than this in absolute terms are left out of
# the percentage error. At 1000 USD a 100 USD miss reads as 10 percent; at 1 USD
# the same miss reads as 10000 percent and swamps the average.
DEFAULT_MAPE_FLOOR = 1_000.0

FloatArray = npt.NDArray[np.float64]


class Scores(BaseModel):
    """One model's accuracy on one set of rows.

    Frozen, so a score cannot be edited after the fact into something more
    attractive than the run produced.

    Attributes
    ----------
    n_rows : int
        Rows the dollar errors were computed over.
    rmse : float
        Root mean squared error, in dollars. Squaring means the largest accounts
        dominate it, which matters on a target this skewed.
    mae : float
        Mean absolute error, in dollars. Less dominated by the extremes than
        RMSE, so the gap between the two is itself informative.
    median_ae : float
        Median absolute error. What a typical user experiences, as opposed to
        what the average is dragged to by a handful of very large accounts.
    wape : float
        Weighted absolute percentage error: total absolute error divided by
        total absolute truth, as a percentage. The headline number.

        It is the percentage error that survives this data. MAPE divides each
        row by its own truth, so a balance near zero produces an enormous
        percentage and the mean becomes meaningless -- which is why ``mape``
        below needs a 1,000 USD floor and a coverage figure to be quotable at
        all. WAPE shares one denominator across the whole set, so no single
        row can explode, and it stays interpretable: 0.35 means the errors add
        up to 35% of the money being predicted. Unlike RMSE it is not dominated
        by the squared tail, and unlike median_ae it does not ignore the tail
        entirely.

        Read it per tier as well as globally -- a good global WAPE on a panel
        this skewed can be a good number on the whales and a bad one on
        everybody else.
    smape : float
        Symmetric mean absolute percentage error, 0-200. The scale-free view of
        the same errors: every row contributes comparably regardless of how
        large the account is, which neither MAE nor RMSE does on a target this
        skewed. Read as reporting rather than as a decision rule -- the shared
        denominator keeps it finite where MAPE explodes, but it is still
        unstable when truth and prediction are both near zero. NaN when no row
        had a non-zero denominator.
    r2_level : float
        Variance of the balance level explained. The flattering number, near
        1.000 for anything at all, reported so the report can say so out loud.
    r2_change : float
        Variance of the month to month movement explained. The honest number.
        Negative means worse than assuming nothing moves.
    n_rows_change : int
        Rows ``r2_change`` used. Lower than ``n_rows`` because a user's first
        month has no previous balance to measure movement against.
    mape : float or None
        Mean absolute percentage error over rows above the floor, or None when
        no row qualified.
    mape_coverage : float
        Share of rows the MAPE could be computed on, between 0 and 1. A MAPE
        quoted without this number is not a number.
    skill : float or None
        Improvement in RMSE over a reference model, as ``1 - rmse / rmse_ref``.
        Positive is better than the reference, negative is worse. None when no
        reference was supplied.
    """

    model_config = ConfigDict(frozen=True)

    n_rows: int
    rmse: float
    mae: float
    median_ae: float
    wape: float = float("nan")
    smape: float = float("nan")
    r2_level: float
    r2_change: float = float("nan")
    n_rows_change: int = 0
    mape: float | None = None
    mape_coverage: float = 0.0
    skill: float | None = None

    def describe(self) -> str:
        """Return a one-line summary for the run log.

        Returns
        -------
        str
            The dollar errors once, then both R squareds side by side so the
            gap between them is impossible to miss.
        """
        line = (
            f"RMSE {self.rmse:>13,.0f} | MAE {self.mae:>12,.0f} | "
            f"MedAE {self.median_ae:>11,.0f} | WAPE {self.wape:>6.1f}% | "
            f"sMAPE {self.smape:>6.1f}% | "
            f"R2 level {self.r2_level:>6.3f} | R2 change {self.r2_change:>7.3f}"
        )
        if self.skill is not None:
            line += f" | skill {self.skill:+.3f}"
        return line


def _as_array(values: pd.Series | FloatArray | list[float]) -> FloatArray:
    """Return ``values`` as a flat float array.

    Parameters
    ----------
    values : pandas.Series or numpy.ndarray or list of float
        Values to convert.

    Returns
    -------
    numpy.ndarray of float
        One dimensional float64 array.
    """
    return np.asarray(values, dtype="float64").ravel()


def _finite_pair(y_true: FloatArray, y_pred: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Drop rows where either side is not finite.

    Parameters
    ----------
    y_true : numpy.ndarray of float
        True values.
    y_pred : numpy.ndarray of float
        Predicted values.

    Returns
    -------
    tuple of numpy.ndarray
        The finite pairs, in the original order.

    Raises
    ------
    ValueError
        If the two arrays differ in length, or if no finite pair survives.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Truth and predictions differ in shape: {y_true.shape} vs {y_pred.shape}"
        )
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    if not keep.any():
        raise ValueError("No rows left to score once non-finite values are dropped")
    return y_true[keep], y_pred[keep]


def r_squared(truth: FloatArray, prediction: FloatArray) -> float:
    """Return the share of the variance of ``truth`` the prediction explains.

    Written out rather than taken from sklearn so the zero-variance case is
    handled explicitly instead of returning something that looks like a score.
    A constant truth column has no variance to explain.

    Parameters
    ----------
    truth : numpy.ndarray of float
        True values, all finite.
    prediction : numpy.ndarray of float
        Predicted values, all finite.

    Returns
    -------
    float
        The coefficient of determination, or NaN when the truth is constant.
    """
    total = float(np.sum(np.square(truth - truth.mean())))
    if total <= 0.0:
        return float("nan")
    residual = float(np.sum(np.square(truth - prediction)))
    return float(1.0 - residual / total)


def _mape(
    y_true: FloatArray, y_pred: FloatArray, floor: float
) -> tuple[float | None, float]:
    """Compute a floored MAPE and the share of rows it used.

    Parameters
    ----------
    y_true : numpy.ndarray of float
        True values.
    y_pred : numpy.ndarray of float
        Predicted values.
    floor : float
        Rows with ``abs(y_true)`` below this are excluded.

    Returns
    -------
    tuple of (float or None, float)
        The percentage error and the share of rows above the floor. The error
        is None when no row qualified.
    """
    usable = np.abs(y_true) >= floor
    coverage = float(usable.mean())
    if not usable.any():
        return None, coverage
    errors = np.abs((y_true[usable] - y_pred[usable]) / y_true[usable])
    return float(100.0 * errors.mean()), coverage


def _smape(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Compute the symmetric mean absolute percentage error.

    ``mean(2 * |y - yhat| / (|y| + |yhat|)) * 100``, so the denominator moves
    with the prediction as well as the truth. That is what bounds the result at
    200 and stops one near-zero balance from owning the average the way it does
    in a plain MAPE -- but it does not rescue the case where truth and
    prediction are both near zero, and those rows are skipped rather than
    counted as a division by zero.

    Parameters
    ----------
    y_true : numpy.ndarray of float
        True values.
    y_pred : numpy.ndarray of float
        Predicted values, aligned to ``y_true``.

    Returns
    -------
    float
        Percentage error between 0 and 200, or NaN when every row had a zero
        denominator.
    """
    denominator = np.abs(y_true) + np.abs(y_pred)
    usable = denominator > 0.0
    if not usable.any():
        return float("nan")
    ratio = 2.0 * np.abs(y_true[usable] - y_pred[usable]) / denominator[usable]
    return float(100.0 * ratio.mean())


# Cut points for the account-size tiers, as quantiles of each entity's median
# absolute balance. 0.5 and 0.9 gives a half/40/10 split: an SMB half that the
# global dollar metrics currently say nothing about, a mid band, and the top
# decile that owns most of the squared error and needs to be looked at on its
# own rather than through an average it dominates.
DEFAULT_TIER_QUANTILES: tuple[float, float] = (0.5, 0.9)


def score(
    y_true: pd.Series | FloatArray,
    y_pred: pd.Series | FloatArray,
    anchor: pd.Series | FloatArray | None = None,
    reference_pred: pd.Series | FloatArray | None = None,
    mape_floor: float = DEFAULT_MAPE_FLOOR,
) -> Scores:
    """Score one set of predictions, in both framings at once.

    Parameters
    ----------
    y_true : pandas.Series or numpy.ndarray
        True closing balances.
    y_pred : pandas.Series or numpy.ndarray
        Predicted closing balances, aligned row for row with ``y_true``.
    anchor : pandas.Series or numpy.ndarray, optional
        Last month's closing balance, from ``baseline.anchor_values``. Supplies
        the change framing. Rows where it is missing are dropped from
        ``r2_change`` and from nothing else, since there is no movement to
        measure for a user's first month.
    reference_pred : pandas.Series or numpy.ndarray, optional
        Predictions from a reference model, usually the 3-month average. When
        given, the result carries a skill score against it.
    mape_floor : float, optional
        Rows with a smaller absolute true value are left out of the percentage
        error. Default 1000.

    Returns
    -------
    Scores
        All the metrics for these predictions.

    Raises
    ------
    ValueError
        If the inputs differ in length, or nothing finite is left to score.
    """
    truth, prediction = _finite_pair(_as_array(y_true), _as_array(y_pred))
    errors = truth - prediction

    rmse = float(np.sqrt(np.mean(np.square(errors))))
    mae = float(np.mean(np.abs(errors)))
    median_ae = float(np.median(np.abs(errors)))
    wape = _wape(truth, prediction)
    smape = _smape(truth, prediction)
    mape, coverage = _mape(truth, prediction, mape_floor)

    r2_change = float("nan")
    n_rows_change = 0
    if anchor is not None:
        # Same errors, different denominator. The subtraction cancels out of
        # the numerator entirely, which is exactly why the dollar metrics above
        # are not recomputed here.
        base = _as_array(anchor)
        movement, predicted_movement = _finite_pair(
            _as_array(y_true) - base, _as_array(y_pred) - base
        )
        r2_change = r_squared(movement, predicted_movement)
        n_rows_change = int(movement.size)

    skill: float | None = None
    if reference_pred is not None:
        reference_truth, reference = _finite_pair(
            _as_array(y_true), _as_array(reference_pred)
        )
        reference_rmse = float(np.sqrt(np.mean(np.square(reference_truth - reference))))
        # A reference that is already perfect leaves nothing to improve on, so
        # the skill score is undefined rather than infinite.
        skill = float(1.0 - rmse / reference_rmse) if reference_rmse > 0.0 else None

    return Scores(
        n_rows=int(truth.size),
        rmse=rmse,
        mae=mae,
        median_ae=median_ae,
        wape=wape,
        smape=smape,
        r2_level=r_squared(truth, prediction),
        r2_change=r2_change,
        n_rows_change=n_rows_change,
        mape=mape,
        mape_coverage=coverage,
        skill=skill,
    )


def _wape(
    truth: FloatArray, prediction: FloatArray
) -> float:
    """Return the weighted absolute percentage error, as a percentage.

    Parameters
    ----------
    truth : numpy.ndarray of float
        True values, already reduced to the finite rows.
    prediction : numpy.ndarray of float
        Predicted values, aligned row for row.

    Returns
    -------
    float
        ``100 * sum|truth - prediction| / sum|truth|``, or NaN when every true
        value is zero and there is no denominator to divide by.
    """
    denominator = float(np.sum(np.abs(truth)))
    if denominator <= 0.0:
        return float("nan")
    return 100.0 * float(np.sum(np.abs(truth - prediction))) / denominator


def assign_tiers(
    entities: pd.Series,
    balances: pd.Series,
    quantiles: tuple[float, float] = DEFAULT_TIER_QUANTILES,
) -> pd.Series:
    """Bucket entities into operational tiers by typical account size.

    The brief asks for metrics per account-size bucket rather than one global
    RMSE, and for good reason: on this panel the top handful of users own most
    of the squared error, so a single number describes them and nobody else.
    A tier breakdown is what shows whether a model that looks adequate overall
    is actually adequate for the 80% of accounts that are small.

    Parameters
    ----------
    entities : pandas.Series
        Entity id per row. Must come from TRAINING rows only -- see Notes.
    balances : pandas.Series
        The balance to size entities by, aligned row for row with
        ``entities``. Normally the anchor column.
    quantiles : tuple of float, optional
        The two cut points, as quantiles of the per-entity median balance.
        Default ``DEFAULT_TIER_QUANTILES``.

    Returns
    -------
    pandas.Series
        Tier name indexed by entity id: ``smb``, ``mid`` or ``enterprise``.

    Notes
    -----
    Sized on the MEDIAN of each entity's balances, not the mean or the latest:
    the median is the account's ordinary size, and it is not moved by the one
    spike that put the account in the news. Sized in ABSOLUTE value because
    three quarters of these balances are negative -- these are liability
    accounts, and a large debt is a large account.

    The caller must pass training rows only. Tiering on the whole panel would
    let an account's holdout months decide which bucket its holdout months are
    then scored in, which is a small leak but a real one and an easy one to
    avoid.
    """
    typical = (
        pd.DataFrame({"entity": entities.to_numpy(), "balance": balances.to_numpy()})
        .groupby("entity")["balance"]
        .apply(lambda values: float(np.nanmedian(np.abs(values.to_numpy(dtype="float64")))))
    )
    low, high = (float(typical.quantile(q)) for q in quantiles)
    tiers = pd.Series("mid", index=typical.index, dtype="object")
    tiers.loc[typical <= low] = "smb"
    tiers.loc[typical > high] = "enterprise"
    tiers.index.name = "entity"
    return tiers


def score_by_group(
    y_true: pd.Series | FloatArray,
    y_pred: pd.Series | FloatArray,
    groups: pd.Series,
    label: str = "group",
) -> pd.DataFrame:
    """Score separately within each group, for example per month or per user.

    The findings deliverable asks where the model is worst. A single number
    cannot answer that, and this is the breakdown that can.

    Parameters
    ----------
    y_true : pandas.Series or numpy.ndarray
        True values.
    y_pred : pandas.Series or numpy.ndarray
        Predicted values, aligned row for row.
    groups : pandas.Series
        Group label per row, such as the month or the user id.
    label : str, optional
        Name given to the index of the result. Default ``group``.

    Returns
    -------
    pandas.DataFrame
        One row per group, with ``n_rows``, ``rmse``, ``mae`` and ``median_ae``,
        sorted by RMSE descending so the worst group is first.
    """
    frame = pd.DataFrame(
        {
            "group": groups.reset_index(drop=True),
            "truth": _as_array(y_true),
            "prediction": _as_array(y_pred),
        }
    ).dropna(subset=["truth", "prediction"])

    frame["error"] = frame["truth"] - frame["prediction"]
    frame["absolute_error"] = frame["error"].abs()

    summary = frame.groupby("group", dropna=False).agg(
        n_rows=("error", "size"),
        rmse=("error", lambda errors: float(np.sqrt(np.mean(np.square(errors))))),
        mae=("absolute_error", "mean"),
        median_ae=("absolute_error", "median"),
    )
    # WAPE needs both sums from the same group, so it is computed alongside the
    # aggregation rather than inside it. This is the column to read across
    # tiers: the dollar errors are not comparable between an SMB bucket and an
    # enterprise one, and a percentage is.
    totals = frame.assign(absolute_truth=frame["truth"].abs()).groupby(
        "group", dropna=False
    )[["absolute_error", "absolute_truth"]].sum()
    summary["wape"] = 100.0 * (
        totals["absolute_error"] / totals["absolute_truth"].replace(0.0, np.nan)
    )
    summary.index.name = label
    return summary.sort_values("rmse", ascending=False)


def anchor_values(dataset: Dataset, column: str) -> pd.Series:
    """Return last month's balance for every row, for the change framing.

    Scoring a balance level is flattering to everybody, because most of a
    prediction is just "this user holds roughly what they held last month".
    Subtracting this anchor from both the truth and the prediction scores the
    part that is actually being forecast, which is the movement. That is what
    ``r2_change`` above is computed against.

    Lives here rather than beside the baselines because it is a scoring
    concern: no model reads it, every score does.

    Parameters
    ----------
    dataset : Dataset
        Rows to take the anchor from.
    column : str
        Column holding last month's balance. Normally
        ``evaluation.anchor_column`` from the config.

    Returns
    -------
    pandas.Series
        Float series aligned to the dataset rows. Rows with no previous month
        carry NaN, and the scorer drops them from the change framing.

    Raises
    ------
    KeyError
        If the anchor column is not in the dataset.
    """
    if column not in dataset.frame.columns:
        raise KeyError(
            f"Anchor column {column!r} is not in the data; available columns "
            f"starting with prev_ are "
            f"{[c for c in dataset.frame.columns if c.startswith('prev_')]}"
        )
    return dataset.frame[column].astype("float64")
