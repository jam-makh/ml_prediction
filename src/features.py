"""Derived features: ratios, momentum and calendar flags, built in Python.

Before this module the model saw twenty-one columns off ``feature_store_monthly``
and eighteen of them were absolute dollar amounts. That is the size-bias problem
stated as a feature table. Handed ``prev_1m_closing_balance_usd = 480,000``, a
tree learns "this is an enterprise account" long before it learns anything about
what the account is *doing*, because account size is the single most predictive
thing in the table and it is predictive of the level, which nobody needed a
model for. The lift a model can add is in the movement, and the movement is a
behavioural question: is this account accelerating, is its spending mix
shifting, is it transacting more often than it was.

So every feature here is a **ratio, a share, a difference of rates, or a
calendar flag** -- quantities that mean the same thing on a 5,000 USD account
and a 5,000,000 USD one.

**Why Python and not SQL.** The PostgreSQL instance belongs to someone else;
this project reads one table from it and owns no migrations. Beyond that,
``config/ml_config.yaml`` already states the rule: engineering that has to run
at prediction time lives in Python, where it cannot be forgotten by whoever runs
the job next.

**The leakage rule, which is what makes this module safe.** Every function here
is *row-wise over columns that are already lagged*. Nothing groups by entity and
looks along the time axis, nothing calls ``shift``, ``rolling`` or ``expanding``,
and nothing touches the target column. A feature for month t is therefore built
only from quantities that existed on day one of month t, which is exactly the
condition ``split.gap_months: 0`` relies on. Per-entity statistics that must be
*fitted* -- the scale in ``src/models_code/entity_scaler.py`` -- deliberately do
not live here, because fitting one over the whole panel would see the holdout.

**Why NaN and not zero on a bad denominator.** A ratio whose denominator is
about zero is unknown, not zero, and not huge. Filling it with 0.0 invents a
data point and puts it at one end of the distribution, where a tree will happily
split on it. XGBoost routes NaN natively and the Ridge pipeline's median imputer
already handles it, so NaN is the honest value and both models can take it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

# Denominators smaller than this in absolute value are treated as unknown. Not
# zero: a balance of 4 USD is a rounding artefact of an empty account, and
# dividing by it produces a ratio in the thousands that is pure noise and will
# out-rank every real feature in a gain ranking.
#
# A hundred dollars rather than one, because these balances cross zero rather
# than sitting well away from it -- 75.5% of rows are negative, the median is
# -13,523 USD, and the distribution passes straight through the origin on its
# way. Under 1% of rows (0.94%) have a balance inside +/-100, so this costs
# almost no coverage and removes the part of the distribution where a ratio is
# meaningless.
EPSILON = 100.0

# Ratios are clipped to +/-this. A ratio whose denominator squeaked past EPSILON
# can still be enormous, and measured over the real panel the last 1% of rows
# stretches growth_1m from about 7 out to 595 -- a decade and a half of range
# holding a handful of rows, all of it an artefact of a small denominator
# rather than an account that grew 59,000%.
#
# Clipped rather than dropped, because the row's other features are fine and
# the direction is still true: the account moved a lot. Clipped rather than
# winsorised at a quantile, because a quantile is a statistic fitted to the
# data, and fitting one over the whole panel would compute it from the holdout
# months too. A constant cannot leak. 25 sits well beyond the 1st and 99th
# percentiles of every ratio here (growth_1m: -6.6 to 7.5) so it touches only
# the tail it is aimed at.
RATIO_CLIP = 25.0

# The same guard for denominators that are counts rather than dollars. A count
# of transactions or merchants is a small integer -- the median month has 39
# transactions and the largest has 161 -- so EPSILON applied here would discard
# nearly every row rather than the unusable ones. The only count that cannot be
# divided by is zero, so the floor is one.
COUNT_EPSILON = 1.0

# The seven spending categories the feature table breaks the previous month's
# spend into. Their sum reproduces prev_1m_total_spend_usd to 2.3e-10, which is
# why that total is already in drop_columns -- and why the seven can be turned
# into shares of it without losing anything.
SPEND_CATEGORIES: tuple[str, ...] = (
    "groceries",
    "dining",
    "retail",
    "cash_atm",
    "transport",
    "bills",
    "other",
)

# Columns this module needs to find. A missing one is skipped rather than
# raised on: the feature table is allowed to grow a column without a code
# change, and it should be allowed to lose one without the run dying before it
# reports anything.
BALANCE_LAGS: tuple[str, str, str] = (
    "prev_1m_closing_balance_usd",
    "prev_2m_closing_balance_usd",
    "prev_3m_closing_balance_usd",
)


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    epsilon: float = EPSILON,
    clip: float | None = RATIO_CLIP,
) -> npt.NDArray[np.float64]:
    """Divide, returning NaN wherever the result would not be meaningful.

    Parameters
    ----------
    numerator : pandas.Series
        Top of the ratio.
    denominator : pandas.Series
        Bottom of the ratio.
    epsilon : float, optional
        Denominators below this in absolute value produce NaN. Default
        ``EPSILON``.
    clip : float or None, optional
        Symmetric bound applied to the result. Default ``RATIO_CLIP``; pass
        None for a ratio that is bounded by construction, such as a share of a
        total, where clipping would be a no-op that only obscures the intent.

    Returns
    -------
    numpy.ndarray of float
        The ratio, NaN where the denominator was too small to divide by or
        where either input was already missing. Never inf.
    """
    top = numerator.to_numpy(dtype="float64")
    bottom = denominator.to_numpy(dtype="float64")
    usable = np.isfinite(bottom) & (np.abs(bottom) >= epsilon)
    result = np.full(top.shape, np.nan, dtype="float64")
    np.divide(top, bottom, out=result, where=usable)
    if clip is not None:
        np.clip(result, -clip, clip, out=result)
    return result


def add_derived_features(
    frame: pd.DataFrame, time_column: str = "month"
) -> pd.DataFrame:
    """Return the frame with the derived feature columns appended.

    Called from ``build_dataset`` after sorting and before the feature list is
    resolved, so every derived column is picked up automatically by
    ``resolve_feature_columns`` and can be excluded through ``drop_columns``
    like any other.

    Parameters
    ----------
    frame : pandas.DataFrame
        The coerced, sorted feature table.
    time_column : str, optional
        Name of the month column. Default ``month``.

    Returns
    -------
    pandas.DataFrame
        A copy with the derived columns added. Source columns are left in
        place; which of them the model then sees is a ``drop_columns``
        decision, made in the config next to the measurement that justifies it.
    """
    out = frame.copy()
    for builder in (
        _add_balance_ratios,
        _add_flow_ratios,
        _add_spend_shares,
        _add_behaviour_ratios,
        _add_calendar_flags,
    ):
        builder(out, time_column)
    return out


def _add_balance_ratios(frame: pd.DataFrame, time_column: str) -> None:
    """Add momentum, acceleration and normalised volatility on the balance.

    Parameters
    ----------
    frame : pandas.DataFrame
        Modified in place.
    time_column : str
        Unused here; kept so every builder has one signature.

    Returns
    -------
    None
    """
    del time_column
    lag1, lag2, lag3 = BALANCE_LAGS
    if not all(column in frame.columns for column in BALANCE_LAGS):
        return

    # The three-lag mean, rebuilt rather than read: roll3_mean_closing_balance_usd
    # is in drop_columns because it is exactly this quantity (to 3.3e-05), and a
    # ratio against it is the whole point of dropping the level.
    roll3_mean = frame.loc[:, list(BALANCE_LAGS)].mean(axis=1)

    # Momentum. Above 1.0 means last month sat above the account's own recent
    # normal, below 1.0 means beneath it -- on any size of account.
    frame["ratio_balance_to_roll3"] = safe_divide(frame[lag1], roll3_mean.abs())

    # Growth rates, then the difference between them: acceleration. A single
    # growth rate says the account is rising; the gap between the short and the
    # long one says whether the rise is speeding up or running out, which is
    # what distinguishes a genuine regime change from a continuing trend.
    growth_1m = safe_divide(frame[lag1] - frame[lag2], frame[lag2].abs())
    growth_3m = safe_divide(frame[lag1] - frame[lag3], frame[lag3].abs()) / 2.0
    frame["growth_1m"] = growth_1m
    frame["growth_3m"] = growth_3m
    # Clipped again after the subtraction: both rates are already bounded, so
    # their difference can reach twice RATIO_CLIP, and the point of the bound is
    # that beyond it the number is denominator noise either way.
    frame["acceleration_1m_vs_3m"] = np.clip(
        growth_1m - growth_3m, -RATIO_CLIP, RATIO_CLIP
    )

    # Coefficient of variation: the standard deviation the table already
    # provides, divided by the level it was measured around. The raw std is a
    # dollar amount and therefore just another measure of account size; this is
    # the same information with the size taken out.
    if "roll3_std_closing_balance_usd" in frame.columns:
        frame["cv_balance_roll3"] = safe_divide(
            frame["roll3_std_closing_balance_usd"], roll3_mean.abs()
        )


def _add_flow_ratios(frame: pd.DataFrame, time_column: str) -> None:
    """Add credit/debit balance and net flow relative to the balance.

    Parameters
    ----------
    frame : pandas.DataFrame
        Modified in place.
    time_column : str
        Unused; see ``_add_balance_ratios``.

    Returns
    -------
    None
    """
    del time_column
    credited = "prev_1m_total_credited_usd"
    debited = "prev_1m_total_debited_usd"
    net = "prev_1m_net_flow_usd"
    balance = "prev_1m_closing_balance_usd"

    if credited in frame.columns and debited in frame.columns:
        # Above 1.0 is an account taking in more than it pays out. Scale-free by
        # construction: both sides are that account's own dollars.
        frame["ratio_credited_to_debited"] = safe_divide(
            frame[credited], frame[debited]
        )

    if net in frame.columns and balance in frame.columns:
        # Net flow as a fraction of the balance it is flowing into -- how much
        # the account moved relative to how much it holds. 0.5 on a small
        # account and 0.5 on a whale describe the same month.
        frame["net_flow_to_balance"] = safe_divide(frame[net], frame[balance].abs())

    if (
        "roll3_mean_net_flow_usd" in frame.columns
        and net in frame.columns
    ):
        # Is this month's flow unusual against the account's own recent flow.
        frame["ratio_net_flow_to_roll3"] = safe_divide(
            frame[net], frame["roll3_mean_net_flow_usd"].abs()
        )


def _add_spend_shares(frame: pd.DataFrame, time_column: str) -> None:
    """Turn the seven category spends into shares of the month's total.

    Parameters
    ----------
    frame : pandas.DataFrame
        Modified in place.
    time_column : str
        Unused; see ``_add_balance_ratios``.

    Returns
    -------
    None

    Notes
    -----
    Seven dollar columns become seven shares plus one ratio against the
    balance. The shares carry the account's spending *mix*, which is the
    behavioural content, while the dollar columns carried mix and size
    together and the size dominated. A shift from bills toward retail is the
    same shift whatever the account is worth.
    """
    del time_column
    columns = [
        f"prev_1m_spend_{category}_usd"
        for category in SPEND_CATEGORIES
        if f"prev_1m_spend_{category}_usd" in frame.columns
    ]
    if not columns:
        return

    # Summed rather than read from prev_1m_total_spend_usd, which is already in
    # drop_columns; the two agree to 2.3e-10 and this keeps the module working
    # whether or not that column is present.
    total = frame.loc[:, columns].sum(axis=1)
    for column in columns:
        category = column[len("prev_1m_spend_") : -len("_usd")]
        frame[f"share_spend_{category}"] = safe_divide(
            frame[column], total, clip=None
        )

    # Spend against the balance it was paid from: the one place total spend is
    # still worth having, once it is expressed relative to the account.
    if "prev_1m_closing_balance_usd" in frame.columns:
        frame["spend_to_balance"] = safe_divide(
            total, frame["prev_1m_closing_balance_usd"].abs()
        )


def _add_behaviour_ratios(frame: pd.DataFrame, time_column: str) -> None:
    """Add transaction-frequency features.

    Parameters
    ----------
    frame : pandas.DataFrame
        Modified in place.
    time_column : str
        Unused; see ``_add_balance_ratios``.

    Returns
    -------
    None

    Notes
    -----
    Counts are already scale-free in a way dollars are not -- a hundred
    transactions is a hundred transactions -- and frequency tends to move
    before value does: an account winding down transacts less before its
    balance reflects it. These are the features most likely to carry an early
    signal, which is why they get their own builder rather than being folded
    into the ratios above.
    """
    del time_column
    txns = "prev_1m_txn_count"
    merchants = "prev_1m_distinct_merchants"

    # Every denominator below is a count, so all three take COUNT_EPSILON, and
    # none takes the ratio clip: a count-over-count is bounded by the counts
    # themselves, and the per-transaction dollar amount is a magnitude in its
    # own right rather than a ratio that can run away.
    if txns in frame.columns and merchants in frame.columns:
        # Repeat rate: many transactions across few merchants is a
        # subscription-shaped account, the reverse is exploratory spending.
        frame["txns_per_merchant"] = safe_divide(
            frame[txns], frame[merchants], epsilon=COUNT_EPSILON, clip=None
        )

    if txns in frame.columns and "prev_1m_total_debited_usd" in frame.columns:
        # Average transaction size, which is a dollar amount -- but a per-event
        # one, so it separates "spending more often" from "spending bigger",
        # two different behaviours that the raw debit total merges.
        frame["avg_debit_per_txn"] = safe_divide(
            frame["prev_1m_total_debited_usd"],
            frame[txns],
            epsilon=COUNT_EPSILON,
            clip=None,
        )

    if txns in frame.columns and "accounts_held" in frame.columns:
        frame["txns_per_account"] = safe_divide(
            frame[txns], frame["accounts_held"], epsilon=COUNT_EPSILON, clip=None
        )


def _add_calendar_flags(frame: pd.DataFrame, time_column: str) -> None:
    """Add explicit seasonal indicators.

    Parameters
    ----------
    frame : pandas.DataFrame
        Modified in place.
    time_column : str
        Name of the month column.

    Returns
    -------
    None

    Notes
    -----
    ``month_of_year`` is already in the table as an integer 1-12, and that is
    the problem: a tree reads it as an ordinal, so isolating December costs it
    two splits and isolating "November or December" costs it a subtree. The
    flags below name the periods that actually behave differently -- the
    holiday months, quarter ends, the fiscal year end -- so one split reaches
    them. The sin/cos pair restores the wrap-around that the integer breaks,
    where December and January are eleven apart rather than adjacent.
    """
    if time_column not in frame.columns:
        return
    months = pd.to_datetime(frame[time_column])
    month_number = months.dt.month.to_numpy(dtype="float64")

    frame["is_november"] = (month_number == 11).astype("float64")
    frame["is_december"] = (month_number == 12).astype("float64")
    frame["is_q4"] = np.isin(month_number, (10.0, 11.0, 12.0)).astype("float64")
    frame["is_quarter_end"] = np.isin(month_number, (3.0, 6.0, 9.0, 12.0)).astype(
        "float64"
    )

    radians = 2.0 * np.pi * month_number / 12.0
    frame["month_sin"] = np.sin(radians)
    frame["month_cos"] = np.cos(radians)
