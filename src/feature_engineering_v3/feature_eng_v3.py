"""

Third iteration of the Feature table (v3) and its segment tables.

Reads v1, adds trailing-window features, assigns segments and writes one lookup
table per segmentation (``seg_sign_regime``, ``seg_size``, ``seg_behaviour``)
plus ``features_monthly_v3``, which references them through ``*_id`` columns.
The ``*_id`` columns are for routing and evaluation only, never model features.
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

import psycopg2  # noqa: E402
from loguru import logger  # noqa: E402
from psycopg2 import sql  # noqa: E402
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
from src.window import cut_positions  # noqa: E402

SOURCE_TABLE = "feature_store_monthly"
TARGET_TABLE = "features_monthly_v3"

# Maven coordinate of the Postgres JDBC driver; Spark downloads it on the first run.
POSTGRES_DRIVER = "org.postgresql:postgresql:42.7.7"

ID_COLUMN = "user_id"
TIME_COLUMN = "month"
# Names the source table's month column may carry; the first one present is read and written back as TIME_COLUMN.
SOURCE_TIME_COLUMNS = ("month", "date")
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
    "flow_volatility_6m",
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

WINDOW_MONTHS = 6
DRAWDOWN_CAP = 2.0

# Id 0 of every lookup table: rows with too little history or a missing basis value.
UNASSIGNED_ID = 0
UNASSIGNED = "unassigned"
UNASSIGNED_DESCRIPTION = "Too little history or a missing value to place the row; persistence is used."

SEGMENT_TABLE_SCHEMA = "id INT, category STRING, rule STRING, description STRING"

# Basis columns of the segmentations; their cut values live in the config under `segments`.
SHARE_NEG = "share_neg_last_6m"
DEBT_DEPTH = "debt_depth_6m"
TURNOVER = "turnover_roll6"
FLOW_VOLATILITY = "flow_volatility_6m"

SIZE_CATEGORIES = (
    ("low_turnover", "Little money moves through the account: mean monthly credited + |debited| over M-6..M-1."),
    ("mid_turnover", "Typical monthly credited + |debited| over M-6..M-1."),
    ("high_turnover", "Large monthly credited + |debited| over M-6..M-1."),
)

BEHAVIOUR_CATEGORIES = (
    ("stable", "Net flow barely moves from month to month, relative to turnover."),
    ("moderate", "Some month-to-month swing in net flow, relative to turnover."),
    ("dynamic", "Large month-to-month swings in net flow, relative to turnover."),
)


class Bound(BaseModel):
    """One interval condition ``lower < column <= upper``.

    Attributes
    ----------
    column : str
        Column the condition reads.
    lower : float or None
        Exclusive lower bound; None is unbounded.
    upper : float or None
        Inclusive upper bound; None is unbounded.
    or_null : bool
        Whether a null value also satisfies the condition.
    """

    model_config = ConfigDict(frozen=True)

    column: str
    lower: float | None = None
    upper: float | None = None
    or_null: bool = False

    def condition(self) -> Column:
        """Return the Spark condition; a null value fails it unless ``or_null`` is set.

        Returns
        -------
        pyspark.sql.Column
            Boolean column.
        """
        value = F.col(self.column)
        inside = value.isNotNull()
        if self.lower is not None:
            inside &= value > self.lower
        if self.upper is not None:
            inside &= value <= self.upper
        return inside | value.isNull() if self.or_null else inside

    def rule(self) -> str:
        """Return the condition as readable text, e.g. ``5000 < turnover_roll6 <= 20000``.

        Returns
        -------
        str
            The rule.
        """
        if self.lower is not None and self.upper is not None:
            text = f"{self.lower:g} < {self.column} <= {self.upper:g}"
        elif self.lower is not None:
            text = f"{self.column} > {self.lower:g}"
        else:
            text = f"{self.column} <= {self.upper:g}"
        return f"({text} or {self.column} is null)" if self.or_null else text


class Band(BaseModel):
    """One category of a segmentation: a row belongs to it when every bound holds.

    Attributes
    ----------
    id : int
        Primary key in the lookup table and the value of the ``*_id`` column in v3.
    category : str
        Category name.
    description : str
        Human-readable meaning.
    bounds : tuple of Bound
        Conditions a row must satisfy.
    """

    model_config = ConfigDict(frozen=True)

    id: int
    category: str
    description: str
    bounds: tuple[Bound, ...]

    def condition(self) -> Column:
        """Return the AND of every bound.

        Returns
        -------
        pyspark.sql.Column
            Boolean column.
        """
        return reduce(lambda left, right: left & right, (bound.condition() for bound in self.bounds))

    def rule(self) -> str:
        """Return the bounds as readable text joined by ``and``.

        Returns
        -------
        str
            The rule.
        """
        return " and ".join(bound.rule() for bound in self.bounds)


class SegmentDefinition(BaseModel):
    """A segmentation: its lookup table and the bands it places rows into.

    Attributes
    ----------
    name : str
        Segmentation name; v3 gets a ``{name}_id`` column.
    table : str
        Lookup table it is written to.
    bands : tuple of Band
        Categories, checked in order; the first match wins.
    min_months : int
        Rows with fewer months of history are unassigned.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    table: str
    bands: tuple[Band, ...]
    min_months: int = 1

    @property
    def id_column(self) -> str:
        """Name of the column in v3 that references this table.

        Returns
        -------
        str
            ``{name}_id``.
        """
        return f"{self.name}_id"

    def rows(self) -> list[tuple[int, str, str, str]]:
        """Return the lookup-table rows, unassigned first.

        Returns
        -------
        list of tuple of (int, str, str, str)
            ``(id, category, rule, description)`` per category.
        """
        history = "no prior month" if self.min_months == 1 else f"fewer than {self.min_months} months of history"
        unassigned = (UNASSIGNED_ID, UNASSIGNED, f"{history}, or a missing basis value", UNASSIGNED_DESCRIPTION)
        return [unassigned, *((band.id, band.category, band.rule(), band.description) for band in self.bands)]

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

    Raises
    ------
    ValueError
        If the source table has none of ``SOURCE_TIME_COLUMNS``.
    """
    url, properties = jdbc_connection()
    raw = spark.read.jdbc(url, SOURCE_TABLE, properties=properties)
    # The source has carried the month as both `month` and `date`, so read whichever exists.
    source_time = next((name for name in SOURCE_TIME_COLUMNS if name in raw.columns), None)
    if source_time is None:
        raise ValueError(f"{SOURCE_TABLE} has none of the month columns {SOURCE_TIME_COLUMNS}")
    month = F.trunc(F.col(source_time).cast("date"), "month")
    return raw.select(
        F.col(ID_COLUMN).cast("string").alias(ID_COLUMN),
        month.alias(TIME_COLUMN),
        # Postgres numeric arrives as a decimal; doubles keep the arithmetic below simple.
        *(F.col(name).cast("double").alias(name) for name in (TARGET_COLUMN, *V1_FEATURES)),
        # Months counted from year 0, so a range window of k means k calendar months.
        (F.year(month) * 12 + F.month(month)).alias(MONTH_INDEX),
    )


def write_table(frame: DataFrame, table: str, primary_key: str | None = None) -> None:
    """Replace a Postgres table with the frame, optionally adding a primary key.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Rows to write.
    table : str
        Destination table name.
    primary_key : str or None, optional
        Column to make the primary key. Default None, no key.

    Returns
    -------
    None
    """
    url, properties = jdbc_connection()
    # Overwrite, so rows from two versions of this script never sit side by side.
    frame.write.jdbc(url, table, mode="overwrite", properties=properties)
    if primary_key is not None:
        add_primary_key(table, primary_key)
    logger.info(f"Written  : {table}")


def add_primary_key(table: str, column: str) -> None:
    """Make a column the primary key of a table, which Spark's JDBC writer cannot do.

    Parameters
    ----------
    table : str
        Table to alter.
    column : str
        Key column.

    Returns
    -------
    None
    """
    env = os.environ
    connection = psycopg2.connect(
        host=env["POSTGRES_HOST"],
        port=env["POSTGRES_PORT"],
        dbname=env["POSTGRES_DB"],
        user=env["POSTGRES_USER"],
        password=env["POSTGRES_PASSWORD"],
    )
    statement = sql.SQL("ALTER TABLE {} ADD PRIMARY KEY ({})").format(
        sql.Identifier(table), sql.Identifier(column)
    )
    try:
        # The `with connection` block commits on success and rolls back on error.
        with connection, connection.cursor() as cursor:
            cursor.execute(statement)
    finally:
        connection.close()


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
    """Add turnover, flow volatility, income volatility and the repayment ratio over 6 months.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame to extend; needs ``roll6_std_net_flow``.

    Returns
    -------
    pyspark.sql.DataFrame
        With ``turnover_roll6``, ``flow_volatility_6m``, ``income_cv_6m`` and ``repayment_ratio``.
    """
    window = _trailing(WINDOW_MONTHS)
    credited = F.col(CREDITED)
    # Absolute value, so the result holds whichever sign v1 stores debits with.
    debited = F.abs(F.col(DEBITED))
    mean_credited = F.avg(credited).over(window)
    turnover = F.avg(credited + debited).over(window)
    return frame.withColumns(
        {
            "turnover_roll6": turnover,
            # Net-flow spread per dollar of turnover; unlike change_volatility_6m it stays finite near a zero balance.
            "flow_volatility_6m": safe_divide(F.col("roll6_std_net_flow"), turnover),
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
    # Flow volatility runs before income, and income before sign/debt, since each divides by the previous one's output.
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


def threshold_bands(
    column: str,
    cuts: list[float],
    categories: tuple[tuple[str, str], ...],
) -> tuple[Band, ...]:
    """Cut a column into bands at fixed values.

    Parameters
    ----------
    column : str
        Column to cut.
    cuts : list of float
        Ascending cut values, one fewer than categories.
    categories : tuple of (str, str)
        ``(category, description)`` per band, lowest first; ids run from 1.

    Returns
    -------
    tuple of Band
        The bands, lowest first.

    Raises
    ------
    ValueError
        If the cuts are not ascending or do not match the number of categories.
    """
    if len(cuts) != len(categories) - 1 or list(cuts) != sorted(cuts):
        raise ValueError(f"{column}: need {len(categories) - 1} ascending cuts, got {cuts}")
    edges: list[float | None] = [None, *cuts, None]
    return tuple(
        Band(
            id=i + 1,
            category=category,
            description=description,
            bounds=(Bound(column=column, lower=edges[i], upper=edges[i + 1]),),
        )
        for i, (category, description) in enumerate(categories)
    )


def sign_regime_bands(negative_share: float, debt_depth_months: float) -> tuple[Band, ...]:
    """Return the sign regime bands: not always negative, then always negative split by debt depth.

    Parameters
    ----------
    negative_share : float
        Share of negative months above which a row counts as always negative.
    debt_depth_months : float
        Deepest debt, in months of turnover, splitting shallow from deep.

    Returns
    -------
    tuple of Band
        Not always negative, shallow negative and deep negative.
    """
    negative = Bound(column=SHARE_NEG, lower=negative_share)
    return (
        Band(
            id=1,
            category="not_always_negative",
            description="Balance is at or above zero in at least some of M-6..M-1.",
            bounds=(Bound(column=SHARE_NEG, upper=negative_share),),
        ),
        Band(
            id=2,
            category="shallow_negative",
            description="Almost always negative; deepest debt is small relative to monthly turnover.",
            bounds=(negative, Bound(column=DEBT_DEPTH, upper=debt_depth_months)),
        ),
        Band(
            id=3,
            category="deep_negative",
            description="Almost always negative; deepest debt is large relative to monthly turnover, or turnover is too small to measure it.",
            # Null depth means turnover is near zero, so any debt is deep relative to it.
            bounds=(negative, Bound(column=DEBT_DEPTH, lower=debt_depth_months, or_null=True)),
        ),
    )


def build_definitions(settings: dict[str, Any]) -> tuple[SegmentDefinition, ...]:
    """Return the three segmentations with the fixed cuts from the config.

    Parameters
    ----------
    settings : dict
        The config's ``segments`` block.

    Returns
    -------
    tuple of SegmentDefinition
        ``(sign_regime, size, behaviour)``.

    Raises
    ------
    KeyError
        If a cut is missing from the config.
    ValueError
        If the size or behaviour cuts are malformed.
    """
    sign = settings["sign_regime"]
    sign_regime = SegmentDefinition(
        name="sign_regime",
        table="seg_sign_regime",
        bands=sign_regime_bands(float(sign["negative_share"]), float(sign["debt_depth_months"])),
    )
    size = SegmentDefinition(
        name="size",
        table="seg_size",
        bands=threshold_bands(TURNOVER, settings["size"]["turnover"], SIZE_CATEGORIES),
    )
    behaviour = SegmentDefinition(
        name="behaviour",
        table="seg_behaviour",
        bands=threshold_bands(
            FLOW_VOLATILITY, settings["behaviour"]["flow_volatility"], BEHAVIOUR_CATEGORIES
        ),
        min_months=int(settings["behaviour"]["min_months"]),
    )
    return sign_regime, size, behaviour


def assign_segments(frame: DataFrame, definition: SegmentDefinition) -> DataFrame:
    """Add the ``{name}_id`` column holding the id of the first band each row satisfies.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame with the basis columns and the months of history.
    definition : SegmentDefinition
        Segmentation to apply.

    Returns
    -------
    pyspark.sql.DataFrame
        With the id column; ``UNASSIGNED_ID`` for short history or no matching band.
    """
    chosen: Column = F.lit(UNASSIGNED_ID)
    # Built from the last band backwards, so the first matching band wins.
    for band in reversed(definition.bands):
        chosen = F.when(band.condition(), F.lit(band.id)).otherwise(chosen)
    enough_history = F.col(HISTORY) >= definition.min_months
    return frame.withColumn(
        definition.id_column, F.when(enough_history, chosen).otherwise(F.lit(UNASSIGNED_ID))
    )


def segment_table(spark: SparkSession, definition: SegmentDefinition) -> DataFrame:
    """Build a segmentation's lookup table: ``id, category, rule, description``.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Session to build with.
    definition : SegmentDefinition
        The segmentation.

    Returns
    -------
    pyspark.sql.DataFrame
        One row per category, unassigned included.
    """
    return spark.createDataFrame(definition.rows(), SEGMENT_TABLE_SCHEMA)


def feature_table(frame: DataFrame, definitions: tuple[SegmentDefinition, ...]) -> DataFrame:
    """Select the v3 columns in table order.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame with derived columns and segment ids.
    definitions : tuple of SegmentDefinition
        The segmentations whose id columns are kept.

    Returns
    -------
    pyspark.sql.DataFrame
        Keys, target, kept v1 columns, derived columns and segment ids.
    """
    return frame.select(
        ID_COLUMN,
        TIME_COLUMN,
        TARGET_COLUMN,
        *V1_FEATURES,
        *DERIVED_COLUMNS,
        *(d.id_column for d in definitions),
    )


def log_segment_tables(definitions: tuple[SegmentDefinition, ...]) -> None:
    """Log every lookup-table row so the cuts are visible in the run log.

    Parameters
    ----------
    definitions : tuple of SegmentDefinition
        The segmentations.

    Returns
    -------
    None
    """
    for d in definitions:
        logger.info(f"\n{d.table}:")
        for row_id, category, rule, _ in d.rows():
            logger.info(f"  {row_id}  {category:<18} {rule}")


def log_segment_counts(
    frame: DataFrame, definitions: tuple[SegmentDefinition, ...], cutoff: date, min_rows: int
) -> None:
    """Log rows and shares per category in train and holdout, warning below ``min_rows``.

    Parameters
    ----------
    frame : pyspark.sql.DataFrame
        Frame with the segment id columns.
    definitions : tuple of SegmentDefinition
        The segmentations.
    cutoff : datetime.date
        Last training month.
    min_rows : int
        Smallest category the router will trust.

    Returns
    -------
    None
    """
    regions = frame.withColumn(
        "region", F.when(F.col(TIME_COLUMN) <= F.lit(cutoff), "train").otherwise("holdout")
    )
    logger.info(f"\nRows per segment (train <= {cutoff}):")
    for d in definitions:
        categories = {row_id: category for row_id, category, _, _ in d.rows()}
        counts = (
            regions.groupBy(d.id_column)
            .pivot("region", ["train", "holdout"])
            .count()
            .fillna(0)
            .orderBy(d.id_column)
            .collect()
        )
        # Totals per region, so each share shows how the cuts land on train versus holdout.
        train_total = sum(row["train"] for row in counts) or 1
        holdout_total = sum(row["holdout"] for row in counts) or 1
        for row in counts:
            row_id, train, holdout = row[d.id_column], row["train"], row["holdout"]
            line = (
                f"  {d.name:<12} {categories[row_id]:<20}"
                f" train {train:>5} ({100 * train / train_total:5.1f}%)"
                f"  holdout {holdout:>5} ({100 * holdout / holdout_total:5.1f}%)"
            )
            # Unassigned rows always go to persistence, so their size does not matter.
            if row_id != UNASSIGNED_ID and min(train, holdout) < min_rows:
                logger.warning(f"{line}  < {min_rows}: router falls back to persistence")
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
    """Build the segment lookup tables and ``features_monthly_v3`` and write them to Postgres.

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
        # The cutoff only splits the logged counts; the segment cuts are fixed in the config.
        cutoff = training_cutoff(featured, config)
        logger.info(f"Training months end at {cutoff}")

        segment_settings = config["segments"]
        segmentations = build_definitions(segment_settings)
        segmented = reduce(assign_segments, segmentations, featured).cache()

        v3 = feature_table(segmented, segmentations)
        log_segment_tables(segmentations)
        log_segment_counts(segmented, segmentations, cutoff, int(segment_settings["min_rows"]))
        log_null_shares(v3)
        logger.info(f"\n{TARGET_TABLE}: {v3.count():,} rows, {len(v3.columns)} columns")

        for definition in segmentations:
            write_table(segment_table(spark, definition), definition.table, primary_key="id")
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
