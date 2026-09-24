"""

Third iteration of the Feature table (v3) and its segment tables.

Reads v1, adds trailing-window features, assigns per-row segments and writes
``fs_segment_types``, ``fs_segment_thresholds``, ``fs_segment_values`` and
``features_monthly_v3``.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from functools import reduce
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Run as a file, Python puts this folder on the path instead of the project root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from loguru import logger  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402
from pyspark.sql import Column, DataFrame, SparkSession, Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.window import WindowSpec  # noqa: E402

from src.config.config import load_config  # noqa: E402
from src.feature_engineering_v2.features_second import (  # noqa: E402
    BALANCE_LAGS,
    EPSILON,
    SPEND_CATEGORIES,
    safe_divide,
)
from src.log import setup_logging  # noqa: E402
from src.month_cut import cut_positions  # noqa: E402

SOURCE_TABLE = "feature_store_monthly"
TARGET_TABLE = "features_monthly_v3"
TYPES_TABLE = "fs_segment_types"
THRESHOLDS_TABLE = "fs_segment_thresholds"
VALUES_TABLE = "fs_segment_values"

# Maven coordinate of the Postgres JDBC driver; Spark downloads it on the first run.
POSTGRES_DRIVER = "org.postgresql:postgresql:42.7.7"

ID_COLUMN = "user_id"
TIME_COLUMN = "month"
TARGET_COLUMN = "target_closing_balance_usd"

BALANCE = "prev_1m_closing_balance_usd"
PREV_BALANCE = "prev_2m_closing_balance_usd"
NET_FLOW = "prev_1m_net_flow_usd"
CREDITED = "prev_1m_total_credited_usd"
DEBITED = "prev_1m_total_debited_usd"

# Helper columns used while building; none of them reaches the written table.
MONTH_INDEX = "_month_index"
HISTORY = "_history_months"
FLIP = "_sign_flip"

# v1 columns carried into v3 unchanged. prev_1m_total_spend_usd and the v1 delta column are left out.
V1_FEATURES: tuple[str, ...] = (
    *BALANCE_LAGS,
    NET_FLOW,
    CREDITED,
    DEBITED,
    *(f"prev_1m_spend_{category}_usd" for category in SPEND_CATEGORIES),
    "prev_1m_txn_count",
    "prev_1m_distinct_merchants",
    "accounts_held",
    "roll3_mean_closing_balance_usd",
    "roll3_std_closing_balance_usd",
    "roll3_mean_net_flow_usd",
    "roll3_mean_total_credited_usd",
    "roll3_mean_total_debited_usd",
)

DERIVED_COLUMNS: tuple[str, ...] = (
    "roll6_mean_balance",
    "roll6_std_balance",
    "roll6_slope_balance",
    "prev_1m_change",
    "roll3_mean_change",
    "roll6_std_net_flow",
    "net_flow_cv",
    "change_volatility_6m",
    "eom_cv_6m",
    "share_neg_last_6m",
    "months_since_sign_flip",
    "sign_flips_6m",
    "debt_depth_6m",
    "drawdown_6m",
    "income_cv_6m",
    "repayment_ratio",
    "turnover_roll6",
)

SEGMENT_COLUMNS: tuple[str, ...] = (
    "sign_regime_rank",
    "size_rank",
    "behaviour_rank",
)

WINDOW_MONTHS = 6
DRAWDOWN_CAP = 2.0

UNASSIGNED = "unassigned"
UNASSIGNED_RANK = 0

# Behaviour bands: calmest half, next 35%, most volatile 15% of training rows.
BEHAVIOUR_QUANTILES = (0.5, 0.85)
BEHAVIOUR_LABELS = ("stable", "moderate", "dynamic")

SIZE_QUANTILES = (0.5, 0.9)
SIZE_LABELS = ("smb", "mid", "enterprise")

# Smallest segment the router will trust; smaller ones are only warned about here.
MIN_ROWS_PER_SEGMENT = 100


class Band(BaseModel):
    """One labelled interval ``lower < value <= upper`` of a segment basis.

    Attributes
    ----------
    label : str
        Segment label.
    rank : int
        Ordinal position of the label, used as a numeric feature.
    lower : float or None
        Exclusive lower bound; None is unbounded.
    upper : float or None
        Inclusive upper bound; None is unbounded.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    rank: int
    lower: float | None = None
    upper: float | None = None


class SegmentDefinition(BaseModel):
    """A segmentation kind: the column it reads and the bands it cuts it into.

    Attributes
    ----------
    segment_type_id : int
        Key in ``fs_segment_types``.
    name : str
        Segmentation name, also the struct column it produces.
    basis_column : str
        Trailing-window column the bands are applied to.
    description : str
        Human-readable meaning, stored in ``fs_segment_types``.
    bands : tuple of Band
        Ordered intervals of the basis column.
    window_months : int
        Length of the trailing window the basis is computed over.
    min_months : int
        Rows with fewer months of history get ``short_history_label``.
    short_history_label : str or None
        Label used below ``min_months``; None means no override.
    """

    model_config = ConfigDict(frozen=True)

    segment_type_id: int
    name: str
    basis_column: str
    description: str
    bands: tuple[Band, ...]
    window_months: int = WINDOW_MONTHS
    min_months: int = 1
    short_history_label: str | None = None


SIGN_REGIME = SegmentDefinition(
    segment_type_id=1,
    name="sign_regime",
    basis_column="share_neg_last_6m",
    description="Share of negative month-end balances over M-6..M-1. Routes rows to a model.",
    bands=(
        Band(label="always_positive", rank=1, upper=0.1),
        Band(label="oscillating", rank=2, lower=0.1, upper=0.9),
        Band(label="always_negative", rank=3, lower=0.9),
    ),
)

def create_session() -> SparkSession:
    """Start a local Spark session with the Postgres JDBC driver on the classpath.

    Returns
    -------
    pyspark.sql.SparkSession
        The running session.
    """
    # Spark looks for `python3` by default, which does not exist on Windows.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder.appName("feature_eng_v3")
        .master("local[*]")
        .config("spark.jars.packages", POSTGRES_DRIVER)
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def jdbc_connection() -> tuple[str, dict[str, str]]:
    """Build the JDBC URL and credentials from the ``POSTGRES_*`` environment variables.

    Returns
    -------
    tuple of (str, dict of str to str)
        The JDBC URL and the connection properties.

    Raises
    ------
    KeyError
        If a ``POSTGRES_*`` variable is not set.
    """
    env = os.environ
    url = f"jdbc:postgresql://{env['POSTGRES_HOST']}:{env['POSTGRES_PORT']}/{env['POSTGRES_DB']}"
    properties = {
        "user": env["POSTGRES_USER"],
        "password": env["POSTGRES_PASSWORD"],
        "driver": "org.postgresql.Driver",
    }
    return url, properties


def read_source(spark: SparkSession) -> DataFrame:
    """Read v1, keep the columns v3 carries and make every value column a double.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Session to read with.

    Returns
    -------
    pyspark.sql.DataFrame
        One row per user-month with a month index for range windows.
    """
    url, properties = jdbc_connection()
    raw = spark.read.jdbc(url, SOURCE_TABLE, properties=properties)
    month = F.trunc(F.col(TIME_COLUMN).cast("date"), "month")
    return raw.select(
        F.col(ID_COLUMN).cast("string").alias(ID_COLUMN),
        month.alias(TIME_COLUMN),
        # Postgres numeric arrives as a decimal; doubles keep the arithmetic below simple.
        *(F.col(name).cast("double").alias(name) for name in (TARGET_COLUMN, *V1_FEATURES)),
        # Months counted from year 0, so a range window of k means k calendar months.
        (F.year(month) * 12 + F.month(month)).alias(MONTH_INDEX),
    )


def write_table(frame: DataFrame, table: str) -> None:
    """Replace a Postgres table with the frame.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Rows to write.
    table : str
        Destination table name.

    Returns
    -------
    None
    """
    url, properties = jdbc_connection()
    # Overwrite, so rows from two versions of this script never sit side by side.
    frame.write.jdbc(url, table, mode="overwrite", properties=properties)
    logger.info(f"Written  : {table}")


def _trailing(months: int) -> WindowSpec:
    """Window over the last ``months`` months up to M-1, read through the ``prev_1m`` columns.

    Parameters
    ----------
    months : int
        Number of months covered.

    Returns
    -------
    pyspark.sql.window.WindowSpec
        Range window over rows M-(months-1)..M of the same user.
    """
    # Row r holds month r-1 in its prev_1m columns, so rows M-5..M cover months M-6..M-1.
    return (
        Window.partitionBy(ID_COLUMN)
        .orderBy(MONTH_INDEX)
        .rangeBetween(-(months - 1), Window.currentRow)
    )


def _history() -> WindowSpec:
    """Window over every earlier row of the same user, up to and including M.

    Returns
    -------
    pyspark.sql.window.WindowSpec
        Unbounded-preceding range window.
    """
    return (
        Window.partitionBy(ID_COLUMN)
        .orderBy(MONTH_INDEX)
        .rangeBetween(Window.unboundedPreceding, Window.currentRow)
    )


def _add_trend(frame: DataFrame) -> DataFrame:
    """Add the 6-month balance level, spread, slope and coefficient of variation.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame to extend.

    Returns
    -------
    pyspark.sql.DataFrame
        With the months of history and the four balance trend columns.
    """
    window = _trailing(WINDOW_MONTHS)
    balance = F.col(BALANCE)
    # Month index only where the balance exists, so the slope never pairs with a missing value.
    x = F.when(balance.isNotNull(), F.col(MONTH_INDEX).cast("double"))
    trended = frame.withColumns(
        {
            HISTORY: F.count(balance).over(window),
            "roll6_mean_balance": F.avg(balance).over(window),
            "roll6_std_balance": F.stddev_samp(balance).over(window),
            # OLS slope in dollars per month; null with a single month, where the variance is zero.
            "roll6_slope_balance": F.covar_pop(x, balance).over(window)
            / F.nullif(F.var_pop(x).over(window), F.lit(0.0)),
        }
    )
    return trended.withColumn(
        "eom_cv_6m",
        safe_divide(F.col("roll6_std_balance"), F.abs(F.col("roll6_mean_balance"))),
    )


def _add_change(frame: DataFrame) -> DataFrame:
    """Add last month's balance change and its 3-month mean.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame to extend.

    Returns
    -------
    pyspark.sql.DataFrame
        With ``prev_1m_change`` and ``roll3_mean_change``.
    """
    changed = frame.withColumn("prev_1m_change", F.col(BALANCE) - F.col(PREV_BALANCE))
    # Rows M-2..M hold the changes of months M-3..M-1.
    return changed.withColumn("roll3_mean_change", F.avg("prev_1m_change").over(_trailing(3)))


def _add_flow_volatility(frame: DataFrame) -> DataFrame:
    """Add the 6-month spread of net flow, relative to its own mean and to the balance.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame to extend; needs ``roll6_mean_balance``.

    Returns
    -------
    pyspark.sql.DataFrame
        With ``roll6_std_net_flow``, ``net_flow_cv`` and ``change_volatility_6m``.
    """
    window = _trailing(WINDOW_MONTHS)
    flow = F.col(NET_FLOW)
    spread = F.stddev_samp(flow).over(window)
    return frame.withColumns(
        {
            "roll6_std_net_flow": spread,
            "net_flow_cv": safe_divide(spread, F.abs(F.avg(flow).over(window))),
            # Typical monthly move as a share of the balance: what persistence gets wrong.
            "change_volatility_6m": safe_divide(spread, F.abs(F.col("roll6_mean_balance"))),
        }
    )


def _add_income(frame: DataFrame) -> DataFrame:
    """Add turnover, income volatility and the repayment ratio over 6 months.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame to extend.

    Returns
    -------
    pyspark.sql.DataFrame
        With ``turnover_roll6``, ``income_cv_6m`` and ``repayment_ratio``.
    """
    window = _trailing(WINDOW_MONTHS)
    credited = F.col(CREDITED)
    # Absolute value, so the result holds whichever sign v1 stores debits with.
    debited = F.abs(F.col(DEBITED))
    mean_credited = F.avg(credited).over(window)
    return frame.withColumns(
        {
            "turnover_roll6": F.avg(credited + debited).over(window),
            "income_cv_6m": safe_divide(F.stddev_samp(credited).over(window), F.abs(mean_credited)),
            "repayment_ratio": safe_divide(mean_credited, F.avg(debited).over(window)),
        }
    )


def _max_drawdown(points: Column) -> Column:
    """Largest fall from a running peak, divided by the peak's absolute value.

    Parameters
    ----------
    points : pyspark.sql.Column
        Array of ``struct(t, b)`` sorted by month.

    Returns
    -------
    pyspark.sql.Column
        Drawdown capped at ``DRAWDOWN_CAP``; null for an empty array.
    """
    start = F.struct(F.lit(None).cast("double").alias("peak"), F.lit(0.0).alias("drawdown"))

    def step(state: Column, point: Column) -> Column:
        # greatest skips nulls, so the first point becomes the first peak.
        peak = F.greatest(state["peak"], point["b"])
        fall = F.when(F.abs(peak) >= EPSILON, (peak - point["b"]) / F.abs(peak)).otherwise(0.0)
        return F.struct(peak.alias("peak"), F.greatest(state["drawdown"], fall).alias("drawdown"))

    drawdown = F.aggregate(points, start, step)["drawdown"]
    return F.when(F.size(points) > 0, F.least(drawdown, F.lit(DRAWDOWN_CAP)))


def _add_sign_debt(frame: DataFrame) -> DataFrame:
    """Add the sign and debt columns: negative share, flips, debt depth and drawdown.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame to extend; needs ``turnover_roll6``.

    Returns
    -------
    pyspark.sql.DataFrame
        With the five sign and debt columns.
    """
    window = _trailing(WINDOW_MONTHS)
    balance = F.col(BALANCE)
    is_negative = balance < 0
    # 1 when the sign changed between M-2 and M-1, null when either balance is missing.
    flipped = frame.withColumn(FLIP, (is_negative != (F.col(PREV_BALANCE) < 0)).cast("int"))

    last_flip = F.max(F.when(F.col(FLIP) == 1, F.col(MONTH_INDEX))).over(_history())
    # With no flip on record, the sign has held for every observed month.
    months_since = F.coalesce(
        F.col(MONTH_INDEX) - last_flip + 1, F.count(balance).over(_history())
    )
    trough = F.min(balance).over(window)
    points = F.array_sort(
        F.collect_list(
            F.when(balance.isNotNull(), F.struct(F.col(MONTH_INDEX).alias("t"), balance.alias("b")))
        ).over(window)
    )
    return flipped.withColumns(
        {
            "share_neg_last_6m": F.avg(is_negative.cast("double")).over(window),
            "months_since_sign_flip": months_since.cast("double"),
            # Rows M-4..M hold the five sign changes between months M-6..M-1.
            "sign_flips_6m": F.sum(F.col(FLIP)).over(_trailing(WINDOW_MONTHS - 1)).cast("double"),
            # Deepest debt of the window in months of turnover, so it does not measure account size.
            "debt_depth_6m": safe_divide(
                F.when(trough < 0, -trough).otherwise(0.0), F.col("turnover_roll6")
            ),
            "drawdown_6m": _max_drawdown(points),
        }
    )


def add_window_features(frame: DataFrame) -> DataFrame:
    """Append every trailing-window column in ``DERIVED_COLUMNS``.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Output of :func:`read_source`.

    Returns
    -------
    pyspark.sql.DataFrame
        The same rows with the derived and helper columns added.
    """
    # Income runs before sign/debt because debt depth divides by turnover.
    builders = (_add_trend, _add_change, _add_flow_volatility, _add_income, _add_sign_debt)
    return reduce(lambda built, builder: builder(built), builders, frame)


def training_cutoff(frame: DataFrame, config: dict[str, Any]) -> date:
    """Return the last training month, using the same split the pipeline uses.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame holding the month column.
    config : dict
        Parsed config; reads ``split.test_share`` and ``split.gap_months``.

    Returns
    -------
    datetime.date
        Last month before the holdout and gap.
    """
    split = config["split"]
    months = sorted(row[0] for row in frame.select(TIME_COLUMN).distinct().collect())
    train_end, _ = cut_positions(
        len(months), float(split["test_share"]), int(split.get("gap_months", 0))
    )
    return months[train_end - 1]


def quantile_bands(
    training: DataFrame,
    column: str,
    labels: tuple[str, ...],
    ranks: tuple[int, ...],
    quantiles: tuple[float, ...],
) -> tuple[Band, ...]:
    """Cut a column into bands at exact quantiles of the training rows.

    Parameters
    ----------
    training : pyspark.sql.DataFrame
        Training-month rows only.
    column : str
        Column to cut.
    labels : tuple of str
        One label per band, lowest first.
    ranks : tuple of int
        One rank per band.
    quantiles : tuple of float
        Cut points, one fewer than labels.

    Returns
    -------
    tuple of Band
        The bands, lowest first.
    """
    # relativeError 0 gives exact quantiles, which is cheap at this table size.
    cuts = training.approxQuantile(column, list(quantiles), 0.0)
    edges: list[float | None] = [None, *cuts, None]
    return tuple(
        Band(label=label, rank=rank, lower=edges[i], upper=edges[i + 1])
        for i, (label, rank) in enumerate(zip(labels, ranks))
    )


def build_definitions(frame: DataFrame, cutoff: date) -> tuple[SegmentDefinition, ...]:
    """Return the three segmentations, with quantile cuts taken on training months only.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame with the derived columns.
    cutoff : datetime.date
        Last training month.

    Returns
    -------
    tuple of SegmentDefinition
        ``(sign_regime, size, behaviour)``.
    """
    training = frame.filter(F.col(TIME_COLUMN) <= F.lit(cutoff))
    size = SegmentDefinition(
        segment_type_id=2,
        name="size",
        basis_column="turnover_roll6",
        description="Mean monthly credited + |debited| over M-6..M-1; cuts at training q50/q90.",
        bands=quantile_bands(training, "turnover_roll6", SIZE_LABELS, (1, 2, 3), SIZE_QUANTILES),
    )
    behaviour = SegmentDefinition(
        segment_type_id=3,
        name="behaviour",
        basis_column="change_volatility_6m",
        description="Spread of monthly net flow over M-6..M-1 relative to the balance; cuts at training q50/q85.",
        bands=quantile_bands(
            training, "change_volatility_6m", BEHAVIOUR_LABELS, (1, 2, 3), BEHAVIOUR_QUANTILES
        ),
        min_months=3,
        short_history_label="stable",
    )
    return SIGN_REGIME, size, behaviour


def _segment(label: str, rank: int) -> Column:
    """Build a ``struct(label, rank)`` literal.

    Parameters
    ----------
    label : str
        Segment label.
    rank : int
        Segment rank.

    Returns
    -------
    pyspark.sql.Column
        The struct.
    """
    return F.struct(F.lit(label).alias("label"), F.lit(rank).alias("rank"))


def band_of(value: Column, bands: tuple[Band, ...]) -> Column:
    """Return the ``struct(label, rank)`` of the band that holds the value.

    Parameters
    ----------
    value : pyspark.sql.Column
        Basis value.
    bands : tuple of Band
        Bands to test, in order.

    Returns
    -------
    pyspark.sql.Column
        The matching struct, null when the value is null or outside every band.
    """
    chosen: Column = F.lit(None)
    # Built from the last band backwards, so the first matching band wins.
    for band in reversed(bands):
        inside = value.isNotNull()
        if band.lower is not None:
            inside &= value > band.lower
        if band.upper is not None:
            inside &= value <= band.upper
        chosen = F.when(inside, _segment(band.label, band.rank)).otherwise(chosen)
    return chosen


def assign_segments(frame: DataFrame, definition: SegmentDefinition) -> DataFrame:
    """Add a ``struct(label, rank)`` column named after the segmentation.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame with the basis column and the months of history.
    definition : SegmentDefinition
        Segmentation to apply.

    Returns
    -------
    pyspark.sql.DataFrame
        With the segment column; ``unassigned`` for rows with no prior month.
    """
    unassigned = _segment(UNASSIGNED, UNASSIGNED_RANK)
    segment = F.coalesce(band_of(F.col(definition.basis_column), definition.bands), unassigned)
    if definition.short_history_label is not None:
        fallback = next(band for band in definition.bands if band.label == definition.short_history_label)
        segment = F.when(
            F.col(HISTORY) < definition.min_months, _segment(fallback.label, fallback.rank)
        ).otherwise(segment)
    return frame.withColumn(definition.name, F.when(F.col(HISTORY) > 0, segment).otherwise(unassigned))


def segment_types(spark: SparkSession, definitions: tuple[SegmentDefinition, ...]) -> DataFrame:
    """Build the ``fs_segment_types`` rows.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Session to build with.
    definitions : tuple of SegmentDefinition
        The segmentations.

    Returns
    -------
    pyspark.sql.DataFrame
        One row per segmentation kind.
    """
    rows = [
        (d.segment_type_id, d.name, d.basis_column, d.window_months, d.description)
        for d in definitions
    ]
    schema = "segment_type_id INT, name STRING, basis_column STRING, window_months INT, description STRING"
    return spark.createDataFrame(rows, schema)


def segment_thresholds(spark: SparkSession, definitions: tuple[SegmentDefinition, ...]) -> DataFrame:
    """Build the ``fs_segment_thresholds`` rows, where each band is ``lower < x <= upper``.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Session to build with.
    definitions : tuple of SegmentDefinition
        The segmentations.

    Returns
    -------
    pyspark.sql.DataFrame
        One row per band.
    """
    rows = [
        (d.segment_type_id, band.label, band.rank, band.lower, band.upper)
        for d in definitions
        for band in d.bands
    ]
    schema = "segment_type_id INT, label STRING, rank INT, lower DOUBLE, upper DOUBLE"
    return spark.createDataFrame(rows, schema)


def segment_values(frame: DataFrame, definitions: tuple[SegmentDefinition, ...]) -> DataFrame:
    """Build the ``fs_segment_values`` rows: one per user, month and segmentation.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame with one struct column per segmentation.
    definitions : tuple of SegmentDefinition
        The segmentations to unpivot.

    Returns
    -------
    pyspark.sql.DataFrame
        Long table of ``user_id, month, segment_type_id, label, rank``.
    """
    parts = [
        frame.select(
            ID_COLUMN,
            TIME_COLUMN,
            F.lit(d.segment_type_id).alias("segment_type_id"),
            F.col(f"{d.name}.label").alias("label"),
            F.col(f"{d.name}.rank").alias("rank"),
        )
        for d in definitions
    ]
    return reduce(DataFrame.unionByName, parts)


def feature_table(frame: DataFrame) -> DataFrame:
    """Select the v3 columns in table order.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame with derived columns and segment structs.

    Returns
    -------
    pyspark.sql.DataFrame
        Keys, target, kept v1 columns, derived columns and segment refs.
    """
    return frame.select(
        ID_COLUMN,
        TIME_COLUMN,
        TARGET_COLUMN,
        *V1_FEATURES,
        *DERIVED_COLUMNS,
        F.col("sign_regime.rank").alias("sign_regime_rank"),
        F.col("size.rank").alias("size_rank"),
        F.col("behaviour.rank").alias("behaviour_rank"),
    )


def log_thresholds(definitions: tuple[SegmentDefinition, ...]) -> None:
    """Log every band so the fitted cuts are visible in the run log.

    Parameters
    ----------
    definitions : tuple of SegmentDefinition
        The segmentations.

    Returns
    -------
    None
    """
    logger.info("\nThresholds (lower < x <= upper):")
    for d in definitions:
        for band in d.bands:
            logger.info(f"  {d.name:<24} {band.label:<16} ({band.lower}, {band.upper}]")


def log_segment_counts(values: DataFrame, definitions: tuple[SegmentDefinition, ...], cutoff: date) -> None:
    """Log rows per segment in train and holdout, warning below ``MIN_ROWS_PER_SEGMENT``.

    Parameters
    ----------
    values : pyspark.sql.DataFrame
        The ``fs_segment_values`` rows.
    definitions : tuple of SegmentDefinition
        The segmentations, used for their names.
    cutoff : datetime.date
        Last training month.

    Returns
    -------
    None
    """
    names = {d.segment_type_id: d.name for d in definitions}
    region = F.when(F.col(TIME_COLUMN) <= F.lit(cutoff), "train").otherwise("holdout")
    counts = (
        values.withColumn("region", region)
        .groupBy("segment_type_id", "rank", "label")
        .pivot("region", ["train", "holdout"])
        .count()
        .orderBy("segment_type_id", "rank")
        .collect()
    )
    logger.info(f"\nRows per segment (train <= {cutoff}):")
    for row in counts:
        train, holdout = row["train"] or 0, row["holdout"] or 0
        line = f"  {names[row['segment_type_id']]:<12} {row['label']:<16} train {train:>5}  holdout {holdout:>5}"
        # Unassigned rows always go to persistence, so their size does not matter.
        if row["label"] != UNASSIGNED and min(train, holdout) < MIN_ROWS_PER_SEGMENT:
            logger.warning(f"{line}  < {MIN_ROWS_PER_SEGMENT}: router falls back to persistence")
        else:
            logger.info(line)


def log_null_shares(frame: DataFrame) -> None:
    """Log the share of nulls in every column.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Table to inspect.

    Returns
    -------
    None
    """
    shares = frame.select(
        [F.avg(F.col(name).isNull().cast("double")).alias(name) for name in frame.columns]
    ).collect()[0]
    logger.info("\nNull share per column:")
    for name in frame.columns:
        logger.info(f"  {name:<40} {100 * shares[name]:5.1f}%")


def run(config: dict[str, Any]) -> None:
    """Build the segment tables and ``features_monthly_v3`` and write them to Postgres.

    Parameters
    ----------
    config : dict
        Parsed config.

    Returns
    -------
    None
    """
    spark = create_session()
    try:
        # Cached, because every later step re-reads the windowed frame.
        featured = add_window_features(read_source(spark)).cache()
        cutoff = training_cutoff(featured, config)
        logger.info(f"Training months end at {cutoff}")

        segmentations = build_definitions(featured, cutoff)
        segmented = reduce(assign_segments, segmentations, featured).cache()

        values = segment_values(segmented, segmentations)
        v3 = feature_table(segmented)
        log_thresholds(segmentations)
        log_segment_counts(values, segmentations, cutoff)
        log_null_shares(v3)
        logger.info(f"\n{TARGET_TABLE}: {v3.count():,} rows, {len(v3.columns)} columns")

        write_table(segment_types(spark, segmentations), TYPES_TABLE)
        write_table(segment_thresholds(spark, segmentations), THRESHOLDS_TABLE)
        write_table(values, VALUES_TABLE)
        write_table(v3, TARGET_TABLE)
    finally:
        spark.stop()


def main() -> None:
    """
    Load the config, set up logging and run the build.

    Returns
    -------
    None
    """
    settings = load_config()
    logger.info(f"Logging to {setup_logging(settings, run_name='features_v3')}")
    run(settings)


if __name__ == "__main__":
    main()
