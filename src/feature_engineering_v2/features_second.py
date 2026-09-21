"""
Feature table v2: the derived columns, built in PySpark.
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.column import Column

if TYPE_CHECKING:  # pragma: no cover - import cost only paid by type checkers
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
# holding a handful of rows, all of it an artefact of a small denominator.
#
# Clipped rather than dropped, because the row's other features are fine and the
# direction is still true: the account moved a lot. Clipped rather than
# winsorised at a quantile, because a quantile is a statistic fitted to the
# data, and fitting one over the whole panel would compute it from the holdout
# months too. A constant cannot leak.
RATIO_CLIP = 25.0

# The same guard for denominators that are counts rather than dollars. A count
# of transactions or merchants is a small integer -- the median month has 39
# transactions and the largest has 161 -- so EPSILON applied here would discard
# nearly every row rather than the unusable ones. The only count that cannot be
# divided by is zero, so the floor is one.
COUNT_EPSILON = 1.0

# The seven spending categories the feature table breaks the previous month's
# spend into. Their sum reproduces prev_1m_total_spend_usd to 2.3e-10, which is
# why the seven can be turned into shares of it without losing anything.
SPEND_CATEGORIES: tuple[str, ...] = (
    "groceries",
    "dining",
    "retail",
    "cash_atm",
    "transport",
    "bills",
    "other",
)

# Columns this module needs to find. A missing one is skipped rather than raised
# on: the feature table is allowed to grow a column without a code change, and
# it should be allowed to lose one without the run dying before it reports
# anything.
BALANCE_LAGS: tuple[str, str, str] = (
    "prev_1m_closing_balance_usd",
    "prev_2m_closing_balance_usd",
    "prev_3m_closing_balance_usd",
)

# Raw source columns carried into v2 as features in their own right. v1 turned
# them into txns_per_merchant and txns_per_account; v2 hands a tree the counts
# and lets it make those comparisons itself (see _add_behaviour_ratios). The two
# activity counts are month M-1, so they are known before month M's balance is.
RAW_FEATURES: tuple[str, ...] = (
    "prev_1m_txn_count",
    "prev_1m_distinct_merchants",
    "accounts_held",
)

# Every column this module adds, in table order. Named here so a caller can
# assert on the set without running the job, and so the config's two
# drop_columns lists have something to be checked against.
DERIVED_COLUMNS: tuple[str, ...] = (
    "month_sin",
    "month_cos",
    "mom_growth_1_2",
    "mom_growth_2_3",
    "growth_3",
    "ratio_balance_to_roll3",
    "cv_balance_roll3",
    "ratio_credited_to_debited",
    "net_flow_to_balance",
    "ratio_net_flow_to_roll3",
    *(f"share_spend_{category}" for category in SPEND_CATEGORIES),
    "spend_to_balance",
    "avg_debit_per_txn",
)


def safe_divide(
    numerator: Column,
    denominator: Column,
    epsilon: float = EPSILON,
    clip: float | None = RATIO_CLIP,
) -> Column:
    """Divide, returning null wherever the result would not be meaningful.

    Parameters
    ----------
    numerator : pyspark.sql.Column
        Top of the ratio.
    denominator : pyspark.sql.Column
        Bottom of the ratio.
    epsilon : float, optional
        Denominators below this in absolute value produce null. Default
        ``EPSILON``.
    clip : float or None, optional
        Symmetric bound applied to the result. Default ``RATIO_CLIP``; pass
        None for a ratio bounded by construction, such as a share of a total,
        where clipping would be a no-op that only obscures the intent.

    Returns
    -------
    pyspark.sql.Column
        The ratio, null where the denominator was too small to divide by or
        where either input was already missing or NaN. Never infinite.

    Notes
    -----
    Spark already returns null for division by exact zero, so the guard here is
    not about avoiding an error -- it is about the band either side of zero
    where the division *succeeds* and produces a number that means nothing.

    NaN is tested for separately from null because a Spark double column
    distinguishes the two and ``isNull`` does not catch NaN. A NaN that reached
    this table is a broken input either way, and both should leave as null.
    """
    top = numerator.cast("double")
    bottom = denominator.cast("double")

    usable = (
        top.isNotNull()
        & bottom.isNotNull()
        & ~F.isnan(top)
        & ~F.isnan(bottom)
        & (F.abs(bottom) >= F.lit(epsilon))
    )
    ratio = top / bottom
    if clip is not None:
        # Inside the `when`, not around it: Spark's greatest/least skip nulls,
        # so greatest(null, -clip) is -clip, and clipping the guarded result
        # would turn every unusable row into the most extreme value there is.
        ratio = F.least(F.greatest(ratio, F.lit(-clip)), F.lit(clip))
    return F.when(usable, ratio).otherwise(F.lit(None).cast("double"))


def _row_mean(columns: tuple[str, ...]) -> Column:
    """Mean across columns for each row, skipping nulls.

    Parameters
    ----------
    columns : tuple of str
        Columns to average.

    Returns
    -------
    pyspark.sql.Column
        The mean, null for a row where every input was null.

    Notes
    -----
    Reproduces ``DataFrame.mean(axis=1)`` from the pandas module, which divides
    by the count of *present* values rather than by the number of columns. A
    user with two months of history should be described by the mean of the two
    months they have, not by a mean that has been quietly halved.
    """
    present = [F.col(name).cast("double") for name in columns]
    total = sum(
        (F.coalesce(column, F.lit(0.0)) for column in present),
        F.lit(0.0),
    )
    count = sum(
        (
            F.when(column.isNotNull(), F.lit(1)).otherwise(F.lit(0))
            for column in present
        ),
        F.lit(0),
    )
    return F.when(count > 0, total / count).otherwise(F.lit(None).cast("double"))


def _row_sum(columns: list[str]) -> Column:
    """Sum across columns for each row, treating null as zero.

    Parameters
    ----------
    columns : list of str
        Columns to add up.

    Returns
    -------
    pyspark.sql.Column
        The total.

    Notes
    -----
    Null-as-zero here, unlike :func:`_row_mean`, because this sums the seven
    category spends and a category with no spend in it is genuinely zero
    spend -- not an unknown amount. It is also the denominator the shares are
    taken against, so the seven shares sum to 1.0 by construction.
    """
    return sum(
        (F.coalesce(F.col(name).cast("double"), F.lit(0.0)) for name in columns),
        F.lit(0.0),
    )


def add_derived_features(frame: DataFrame, time_column: str = "date") -> DataFrame:
    """Return the frame with the v2 derived columns appended.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        The feature table, one row per user-month.
    time_column : str, optional
        Name of the date column. Default ``date``; pass the config's
        ``data.time_column`` rather than hard-coding it at the call site.

    Returns
    -------
    pyspark.sql.DataFrame
        The same rows with :data:`DERIVED_COLUMNS` added. Source columns are
        left in place -- including every raw dollar amount -- because which of
        them a given model sees is a ``drop_columns`` decision taken per model
        in the config, and because the baseline and the anchor arithmetic read
        the raw balance lags by name.

    Notes
    -----
    Row order is not touched. Unlike the pandas module this does not require
    the frame to be sorted, because nothing here looks along the time axis --
    which is the same property that makes it leak-free.
    """
    projected = frame
    for builder in (
        _add_calendar_terms,
        _add_balance_ratios,
        _add_flow_ratios,
        _add_spend_shares,
        _add_behaviour_ratios,
    ):
        projected = builder(projected, time_column)
    return projected


def _add_calendar_terms(frame: DataFrame, time_column: str) -> DataFrame:
    """Add the cyclical month encoding.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame to extend.
    time_column : str
        Name of the date column.

    Returns
    -------
    pyspark.sql.DataFrame
        With ``month_sin`` and ``month_cos`` added, or unchanged if the date
        column is absent.

    Notes
    -----
    The v1 module also emitted ``is_q4``, ``is_november``, ``is_december`` and
    ``is_quarter_end``. They are gone by decision, not by oversight: with 34
    training months there are only about three observations behind each
    single-month flag, which memorises dates rather than learning a season.

    That leaves sin/cos as the whole of the seasonal encoding. The pair exists
    because the integer month breaks the wrap-around -- December and January
    read as eleven apart rather than adjacent -- and one of the two alone is
    not enough, since sin maps March and September to the same value.
    """
    if time_column not in frame.columns:
        return frame

    month_number = F.month(F.col(time_column).cast("date")).cast("double")
    radians = F.lit(2.0) * F.lit(3.141592653589793) * month_number / F.lit(12.0)
    return frame.withColumns({"month_sin": F.sin(radians), "month_cos": F.cos(radians)})


def _add_balance_ratios(frame: DataFrame, time_column: str) -> DataFrame:
    """Add the growth family, momentum and normalised volatility.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame to extend.
    time_column : str
        Unused here; kept so every builder has one signature.

    Returns
    -------
    pyspark.sql.DataFrame
        With the balance-derived columns added, or unchanged if the three
        balance lags are not all present.

    Notes
    -----
    The v2 growth cut. v1 measured lag1 against lag2 and lag1 against lag3 --
    two rates sharing an anchor, so the second half-contains the first. v2
    measures the two adjacent steps instead, which are cleanly separable: the
    *difference* between them is acceleration, and a booster can form that
    difference in one split rather than being handed it as a column.

    ``growth_3`` is then the long-horizon rate over the same window, kept
    because a tree cannot multiply two adjacent rates together to recover it.
    It is arithmetically implied by the other two, so if stage 2 gain comes
    back flat across all three, this is the one to drop -- and it is already
    dropped for Ridge, where an implied third member of a correlated group is
    exactly what makes the coefficients unreadable.

    The division by two in ``growth_3`` puts it on the same per-month footing
    as the other two: lag1 against lag3 spans two months, so the raw ratio is
    twice the rate.
    """
    del time_column
    if not all(column in frame.columns for column in BALANCE_LAGS):
        return frame

    lag1, lag2, lag3 = (F.col(name) for name in BALANCE_LAGS)

    # The three-lag mean, rebuilt rather than read: roll3_mean_closing_balance_usd
    # is exactly this quantity (they agree to 3.3e-05), and a ratio against it
    # is the whole point of not using the level.
    roll3_mean = _row_mean(BALANCE_LAGS)

    added = {
        # Momentum. Above 1.0 means last month sat above the account's own
        # recent normal, below 1.0 means beneath it -- on any size of account.
        "ratio_balance_to_roll3": safe_divide(lag1, F.abs(roll3_mean)),
        # Adjacent month-on-month steps. Each is a plain rate of change over
        # one month, so the two are directly comparable to each other.
        "mom_growth_1_2": safe_divide(lag1 - lag2, F.abs(lag2)),
        "mom_growth_2_3": safe_divide(lag2 - lag3, F.abs(lag3)),
        # The same movement measured over the full two-month window, halved to
        # a per-month rate.
        "growth_3": safe_divide(lag1 - lag3, F.abs(lag3)) / F.lit(2.0),
    }

    # Coefficient of variation: the standard deviation the table already
    # provides, divided by the level it was measured around. The raw std is a
    # dollar amount and therefore just another measure of account size; this is
    # the same information with the size taken out.
    if "roll3_std_closing_balance_usd" in frame.columns:
        added["cv_balance_roll3"] = safe_divide(
            F.col("roll3_std_closing_balance_usd"), F.abs(roll3_mean)
        )

    return frame.withColumns(added)


def _add_flow_ratios(frame: DataFrame, time_column: str) -> DataFrame:
    """Add credit/debit balance and net flow relative to the balance.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame to extend.
    time_column : str
        Unused; see :func:`_add_balance_ratios`.

    Returns
    -------
    pyspark.sql.DataFrame
        With whichever flow ratios their inputs allow.

    Notes
    -----
    These three sit alongside the raw ``prev_1m_net_flow_usd``, which v2 admits
    as a feature in its own right. That is a deliberate redundancy rather than
    an oversight: the ratios go null below ``EPSILON``, and on an account whose
    balance is passing through zero the signed dollar flow is the only
    description of the month that survives.

    The two gross totals it is built from are *not* features. Both are read
    here, and ``prev_1m_total_debited_usd`` is read again by
    :func:`_add_behaviour_ratios`, but reading a column and letting a model
    learn from it are different decisions -- the first happens in this module,
    the second in ``data.drop_columns``.
    """
    del time_column
    credited = "prev_1m_total_credited_usd"
    debited = "prev_1m_total_debited_usd"
    net = "prev_1m_net_flow_usd"
    balance = "prev_1m_closing_balance_usd"

    added: dict[str, Column] = {}

    if credited in frame.columns and debited in frame.columns:
        # Above 1.0 is an account taking in more than it pays out. Scale-free
        # by construction: both sides are that account's own dollars.
        added["ratio_credited_to_debited"] = safe_divide(
            F.col(credited), F.col(debited)
        )

    if net in frame.columns and balance in frame.columns:
        # Net flow as a fraction of the balance it is flowing into -- how much
        # the account moved relative to how much it holds. 0.5 on a small
        # account and 0.5 on a whale describe the same month.
        added["net_flow_to_balance"] = safe_divide(F.col(net), F.abs(F.col(balance)))

    if net in frame.columns and "roll3_mean_net_flow_usd" in frame.columns:
        # Is this month's flow unusual against the account's own recent flow.
        added["ratio_net_flow_to_roll3"] = safe_divide(
            F.col(net), F.abs(F.col("roll3_mean_net_flow_usd"))
        )

    return frame.withColumns(added) if added else frame


def _add_spend_shares(frame: DataFrame, time_column: str) -> DataFrame:
    """Turn the seven category spends into shares of the month's total.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame to extend.
    time_column : str
        Unused; see :func:`_add_balance_ratios`.

    Returns
    -------
    pyspark.sql.DataFrame
        With one ``share_spend_*`` per category present, plus
        ``spend_to_balance``.

    Notes
    -----
    Seven dollar columns become seven shares plus one ratio against the
    balance. The shares carry the account's spending *mix*, which is the
    behavioural content; the dollar columns carried mix and size together and
    size dominated. A shift from bills toward retail is the same shift whatever
    the account is worth.

    The total is summed from the seven rather than read from
    ``prev_1m_total_spend_usd``. The two agree to 2.3e-10, and summing keeps
    this working whether or not that column is in the table.
    """
    del time_column
    columns = [
        f"prev_1m_spend_{category}_usd"
        for category in SPEND_CATEGORIES
        if f"prev_1m_spend_{category}_usd" in frame.columns
    ]
    if not columns:
        return frame

    total = _row_sum(columns)

    added = {
        # clip=None: a share of a total is bounded by construction, and a clip
        # here would be a no-op that only obscures the intent.
        f"share_spend_{column[len('prev_1m_spend_'):-len('_usd')]}": safe_divide(
            F.col(column), total, clip=None
        )
        for column in columns
    }

    # Spend against the balance it was paid from: the one place total spend is
    # still worth having, once it is expressed relative to the account.
    if "prev_1m_closing_balance_usd" in frame.columns:
        added["spend_to_balance"] = safe_divide(
            total, F.abs(F.col("prev_1m_closing_balance_usd"))
        )

    return frame.withColumns(added)


def _add_behaviour_ratios(frame: DataFrame, time_column: str) -> DataFrame:
    """Add the per-event spend magnitude.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame to extend.
    time_column : str
        Unused; see :func:`_add_balance_ratios`.

    Returns
    -------
    pyspark.sql.DataFrame
        With ``avg_debit_per_txn`` added where its two inputs are present.

    Notes
    -----
    A dollar amount, but a per-event one, so it separates "spending more often"
    from "spending bigger" -- two different behaviours that the raw debit total
    merges. It is also the likeliest surviving size proxy in the table, which
    is what stage 3's permutation-stability check exists to catch.

    ``txns_per_merchant`` and ``txns_per_account`` from v1 are not computed.
    Both were hypotheses rather than measurements, and the raw counts they were
    built from -- ``prev_1m_txn_count``, ``prev_1m_distinct_merchants``,
    ``accounts_held`` -- are all features in their own right, so a tree can
    reach the same comparisons without being handed them.

    The denominator here is a count, so it takes ``COUNT_EPSILON`` rather than
    ``EPSILON``, and no clip: this is a magnitude in its own right, not a ratio
    that can run away on a small denominator.
    """
    del time_column
    txns = "prev_1m_txn_count"
    debited = "prev_1m_total_debited_usd"

    if txns not in frame.columns or debited not in frame.columns:
        return frame

    return frame.withColumn(
        "avg_debit_per_txn",
        safe_divide(F.col(debited), F.col(txns), epsilon=COUNT_EPSILON, clip=None),
    )


def add_derived_features_pandas(
    frame: pd.DataFrame,
    time_column: str = "date",
    session: SparkSession | None = None,
) -> pd.DataFrame:
    """Run :func:`add_derived_features` over a pandas frame.

    The adapter that lets ``src/data/data.py`` call this module without the rest
    of the pipeline learning about Spark. pandas in, Spark in the middle, pandas
    out.

    Parameters
    ----------
    frame : pandas.DataFrame
        The coerced feature table.
    time_column : str, optional
        Name of the date column. Default ``date``.
    session : pyspark.sql.SparkSession, optional
        Session to run on. A local one is created and left running when
        omitted, so that repeated calls in a notebook share a JVM instead of
        paying to start one each time.

    Returns
    -------
    pandas.DataFrame
        A new frame with :data:`DERIVED_COLUMNS` added, row order preserved.

    Notes
    -----
    Row order is preserved by a position column carried through the round trip
    and sorted on at the end, not by trusting Spark to return rows in the order
    it received them. It will not: even a single-partition read makes no such
    promise, and ``build_dataset`` sorts by entity then time before calling
    this precisely so that downstream checks can reason about the order.

    This is the slow path by a wide margin at 6,300 rows -- the conversion
    dominates and the arithmetic is free. See the module docstring on why the
    Spark version exists anyway.
    """
    # Spark launches its Python workers as `python3` unless told otherwise, and
    # there is no such executable on Windows. Use the interpreter running this.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    spark = session or (
        SparkSession.builder.appName("features_second")
        .master("local[*]")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

    position = "__row_position__"
    staged = frame.reset_index(drop=True)
    staged[position] = range(len(staged))

    # Spark has no UUID type and cannot infer one, and psycopg2 hands user_id
    # back as uuid.UUID objects. Those columns cross as strings and the
    # originals are put back afterwards, so callers see the type they passed in.
    uuid_columns = [
        column
        for column in staged.columns
        if staged[column].dtype == object
        and staged[column].map(lambda value: isinstance(value, uuid.UUID)).any()
    ]
    originals = staged[uuid_columns].copy()
    staged[uuid_columns] = staged[uuid_columns].astype(str)

    derived = add_derived_features(
        spark.createDataFrame(staged), time_column=time_column
    )

    out = derived.toPandas().sort_values(position).reset_index(drop=True)
    out[uuid_columns] = originals
    return out.drop(columns=[position])


def build_v2_table(config: dict[str, Any], source: str = "v1") -> pd.DataFrame:
    """Return the v2 table: keys, target, balance anchors, raw counts and derived columns.

    Parameters
    ----------
    config : dict
        Parsed config.
    source : str, optional
        Entry of ``data.tables`` to derive from. Default ``v1`` -- the raw
        table, never the v2 one being replaced.

    Returns
    -------
    pandas.DataFrame
        One row per (user, month). The only raw source columns carried over
        are the three balance lags, which are not features but are what the
        anchor arithmetic and the 3-month baseline read, and
        :data:`RAW_FEATURES`.
    """
    from src.data.data import build_dataset

    dataset = build_dataset(config, table=source)
    frame = add_derived_features_pandas(dataset.frame, time_column=dataset.time_column)
    keys = [dataset.id_column, dataset.time_column, dataset.target_column]
    anchors = [column for column in BALANCE_LAGS if column in frame.columns]
    raw = [column for column in RAW_FEATURES if column in frame.columns]
    derived = [column for column in DERIVED_COLUMNS if column in frame.columns]
    return frame.loc[:, [*keys, *anchors, *raw, *derived]].copy()


def write_v2_table(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build the v2 table and replace ``data.tables.v2`` with it.

    Parameters
    ----------
    config : dict, optional
        Parsed config. Loaded from ``config/ml_config.yaml`` when omitted.

    Returns
    -------
    pandas.DataFrame
        The frame that was written.
    """
    from sqlalchemy import text

    from src.config.config import load_config
    from src.data.db_link import get_engine

    settings = config if config is not None else load_config()
    data_block = settings["data"]
    table = str(data_block["tables"]["v2"])
    id_column = str(data_block["id_column"])
    time_column = str(data_block["time_column"])

    frame = build_v2_table(settings)

    logger.info(f"Table   : {table}")
    logger.info(f"Rows    : {len(frame):,}")
    logger.info(f"Columns : {len(frame.columns)}")
    for column in frame.columns:
        null_share = float(frame[column].isna().mean())
        logger.info(f"  {column:<40} {100 * null_share:5.1f}% null")

    engine = get_engine()
    # Replace, not append: rows derived under two versions of this module must
    # never sit side by side in one table.
    frame.to_sql(table, engine, if_exists="replace", index=False, chunksize=5_000)
    with engine.begin() as connection:
        connection.execute(
            text(
                f'CREATE INDEX IF NOT EXISTS "{table}_user_month_idx" '
                f'ON "{table}" ("{id_column}", "{time_column}")'
            )
        )
    logger.info(f"\nWritten to {table} (replaced).")
    return frame


def main() -> None:
    """Write the v2 table.

    Returns
    -------
    None
    """
    from src.config.config import load_config
    from src.log import setup_logging

    settings = load_config()
    logger.info(f"Logging to {setup_logging(settings, run_name='features_v2')}")
    write_v2_table(settings)


if __name__ == "__main__":
    # Run as a file (`python src/.../features_second.py`) Python puts this
    # folder on the path, not the project root, so `src` cannot be imported.
    # `python -m src.feature_engineering_v2.features_second` does not need this.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
