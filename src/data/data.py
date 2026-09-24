"""Loading the monthly feature table and handing it on as a typed dataset.

This is the first link in the chain: everything downstream (preprocessing,
splitting, training, evaluation) receives a ``Dataset`` from here rather than a
bare DataFrame, so the four column roles -- entity, time, features, target --
are named once and never re-guessed further down.

Three things are enforced here rather than left to the caller:

* Types. Postgres ``numeric`` arrives through psycopg2 as ``Decimal`` objects
  in an ``object`` column. They compare and print like numbers, so the problem
  does not surface until an estimator raises several steps later.
* Ordering. Rows are sorted by entity then time. A per-user series that is not
  in time order makes any lag-shaped check downstream meaningless.
* Shape. One row per (user, month) pair, no duplicates, target present. A
  duplicated pair would put the same month on both sides of a time split.

The feature list is derived by exclusion: everything that is not the target,
the entity id or the time column. Adding a column to the feature table
therefore adds it to the model without a code change, which is the behaviour
we want while the table is still being iterated on.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict
from sqlalchemy.engine import Engine

from src.config.config import load_config
from src.data.db_link import load_dataframe

# Pandas period alias for a calendar month. Month values are normalised through
# it to the first of the month: the feature table already stores them that way,
# but normalising means a timestamp arriving with a time component cannot open a
# second, near-duplicate month.
MONTH_PERIOD = "M"


class Dataset(BaseModel):
    """A feature table with its four column roles named.

    Wraps a single DataFrame and records which column is the target, which
    identifies the entity (a user), which carries time, and which of the rest
    are model features. Frozen because a split or a subset should produce a new
    ``Dataset`` rather than quietly mutating the one every other part of the
    run is holding.

    Pydantic guards the container only: it checks that ``frame`` is a DataFrame
    and that the four role fields are strings. It cannot know whether the frame
    holds one row per user per month, which is why ``validate_frame`` exists
    further down and runs on the contents.

    Attributes
    ----------
    frame : pandas.DataFrame
        The rows, sorted by entity then time, with a clean 0..n-1 index.
    target_column : str
        Name of the column being predicted.
    id_column : str
        Name of the entity identifier column (one user, many months).
    time_column : str
        Name of the month column, dtype datetime64.
    feature_columns : tuple of str
        Names of the model input columns, in table order.
    """

    # arbitrary_types_allowed because pandas objects carry no pydantic schema.
    # frozen so a split or a subset has to produce a new Dataset.
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    frame: pd.DataFrame
    target_column: str
    id_column: str
    time_column: str
    feature_columns: tuple[str, ...]

    @property
    def features(self) -> pd.DataFrame:
        """Return the feature columns only.

        Returns
        -------
        pandas.DataFrame
            Shape (n_rows, n_features), columns in ``feature_columns`` order.
        """
        return self.frame.loc[:, list(self.feature_columns)]

    @property
    def target(self) -> pd.Series:
        """Return the target column.

        Returns
        -------
        pandas.Series
            Float series of length n_rows, aligned to ``features``.
        """
        return self.frame[self.target_column]

    @property
    def entities(self) -> pd.Series:
        """Return the entity identifier of every row.

        Useful as the ``groups`` argument of any grouped cross-validator.

        Returns
        -------
        pandas.Series
            One entity id per row, aligned to ``features``.
        """
        return self.frame[self.id_column]

    @property
    def times(self) -> pd.Series:
        """Return the month of every row.

        Returns
        -------
        pandas.Series
            datetime64 series, one month per row, aligned to ``features``.
        """
        return self.frame[self.time_column]

    @property
    def months(self) -> pd.DatetimeIndex:
        """Return the distinct months present, in ascending order.

        This is the axis a time-based split cuts along, so the splitter asks
        for it here rather than re-deriving it.

        Returns
        -------
        pandas.DatetimeIndex
            Sorted unique months.
        """
        return pd.DatetimeIndex(self.times.drop_duplicates().sort_values())

    @property
    def n_rows(self) -> int:
        """Return the number of rows.

        Returns
        -------
        int
            Row count of the wrapped frame.
        """
        return int(len(self.frame))

    def take(self, positions: npt.NDArray[np.intp]) -> Dataset:
        """Return the rows at the given positions as a new ``Dataset``.

        Positional rather than label based, because the splitters produce
        positions and a label lookup would go wrong the moment a caller hands
        in a frame with a non-unique index.

        Parameters
        ----------
        positions : numpy.ndarray of numpy.intp
            Row positions to keep, in the order they should appear.

        Returns
        -------
        Dataset
            A new dataset over the selected rows, index reset to 0..k-1, with
            the same column roles.
        """
        # copy() because a positional take can return a view, and a view handed
        # to a transformer is where a partial write happens silently.
        subset = self.frame.iloc[positions].copy().reset_index(drop=True)
        return self.model_copy(update={"frame": subset})

    def summary(self) -> str:
        """Return a one-line description of the dataset.

        Intended for the run log, so a run records what it was fitted on rather
        than leaving that to be reconstructed afterwards.

        Returns
        -------
        str
            Rows, entities, month span and feature count.
        """
        months = self.months
        span = f"{months[0]:%Y-%m} to {months[-1]:%Y-%m}" if len(months) else "none"
        return (
            f"{self.n_rows:,} rows, {self.entities.nunique():,} entities, "
            f"{len(months)} months ({span}), {len(self.feature_columns)} features"
        )


def coerce_types(
    frame: pd.DataFrame, id_column: str, time_column: str
) -> pd.DataFrame:
    """Return ``frame`` with the month parsed and every other column numeric.

    Postgres ``numeric`` columns arrive as ``Decimal`` objects inside an
    ``object`` column. Converting them here, once, at the point of entry, is
    what keeps the rest of the pipeline free of dtype defence.

    Parameters
    ----------
    frame : pandas.DataFrame
        The frame as returned by the database.
    id_column : str
        Entity identifier column, left untouched (it is a uuid, not a number).
    time_column : str
        Month column, parsed to datetime64 and normalised to the month start.

    Returns
    -------
    pandas.DataFrame
        A new frame with the same columns and converted dtypes.

    Raises
    ------
    ValueError
        If a column that should be numeric holds values that cannot be
        converted, or if the time column cannot be parsed as a date.
    """
    converted = frame.copy()

    # Normalising to the month start means two rows for the same month can
    # never land on opposite sides of a month boundary because one of them
    # carried a time component.
    parsed_time = pd.to_datetime(converted[time_column], errors="coerce")
    unparsed = int((parsed_time.isna() & converted[time_column].notna()).sum())
    if unparsed:
        raise ValueError(
            f"Column {time_column!r} holds values that are not dates; "
            f"{unparsed} of {len(parsed_time)} failed to parse"
        )
    converted[time_column] = parsed_time.dt.to_period(MONTH_PERIOD).dt.to_timestamp()

    # Everything except the id and the month is model input or the target, so
    # all of it has to be numeric.
    for column in converted.columns:
        if column in (id_column, time_column):
            continue
        try:
            converted[column] = pd.to_numeric(converted[column]).astype("float64")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Column {column!r} could not be converted to a number: {error}"
            ) from error

    return converted


def resolve_feature_columns(
    frame: pd.DataFrame,
    target_column: str,
    id_column: str,
    time_column: str,
    drop_columns: tuple[str, ...] = (),
    keep_columns: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return the model feature columns, from an explicit list or by exclusion.

    Two modes, and which one is in force is a deliberate decision recorded in
    the config.

    **Explicit** -- ``keep_columns`` names the features. This is what a run uses
    once ``features_selection.py`` has been run and its answer pasted into the
    ``features`` block: the set is then frozen, visible in a git diff, and
    identical between two runs over the same data. Selection is an occasional,
    committed decision; training reads the decision rather than retaking it.

    **By exclusion** -- no ``keep_columns``, so a feature is any column that is
    not the target, not the entity id, not the time column and not explicitly
    dropped. This is the mode to be in while the feature table is still being
    iterated on, since a new column reaches the model without a matching edit
    here. It is also the mode ``features_selection.py`` itself runs in: it has
    to see every candidate in order to rank them.

    Note what neither mode does: remove a column from the frame. A column that
    is not a feature is still there to be read by name. That is what lets the
    baseline average the raw balance lags, and the anchor arithmetic rebuild a
    dollar prediction, while no model is allowed to *learn* from account size.

    Parameters
    ----------
    frame : pandas.DataFrame
        The coerced frame.
    target_column : str
        Name of the target column.
    id_column : str
        Name of the entity identifier column.
    time_column : str
        Name of the month column.
    drop_columns : tuple of str, optional
        Columns to exclude by hand, for example one half of a pair that says
        the same thing twice. Default is no exclusions. Ignored when
        ``keep_columns`` is given, since an explicit list already says what is
        in.
    keep_columns : tuple of str, optional
        The exact feature set, in this order. Default empty, which selects the
        exclusion mode above.

    Returns
    -------
    tuple of str
        Feature column names. In the given order when explicit, otherwise in
        the order they appear in the frame.

    Raises
    ------
    KeyError
        If ``keep_columns`` names a column the frame does not have. Raised
        rather than skipped: a frozen feature set that silently shrinks because
        a column was renamed upstream would change every number in the run
        without changing anything visible in the config.
    ValueError
        If no feature columns survive the exclusions, or ``keep_columns``
        names one of the three reserved roles.
    """
    reserved = {target_column, id_column, time_column}

    if keep_columns:
        wanted = tuple(str(name) for name in keep_columns)

        clashes = [name for name in wanted if name in reserved]
        if clashes:
            raise ValueError(
                f"features lists {clashes}, which are the target, id or time "
                f"column and can never be model inputs"
            )

        missing = [name for name in wanted if name not in frame.columns]
        if missing:
            raise KeyError(
                f"The features block names {len(missing)} column(s) that are "
                f"not in the data: {missing}. Either the feature table changed "
                f"or the block is stale -- re-run "
                f"`python -m src.features_selection`."
            )
        return wanted

    excluded = {*reserved, *drop_columns}
    features = tuple(str(column) for column in frame.columns if column not in excluded)
    if not features:
        raise ValueError(
            "No feature columns left after exclusions; check data.drop_columns "
            "in the config"
        )
    return features


def validate_frame(
    frame: pd.DataFrame, target_column: str, id_column: str, time_column: str
) -> None:
    """Check the structural assumptions the rest of the pipeline rests on.

    Raises rather than warns. Each of these is a condition under which a later
    number would still be produced but would not mean what it appears to mean,
    and a wrong number that looks fine is worse than a stopped run.

    Parameters
    ----------
    frame : pandas.DataFrame
        The frame as returned by the database.
    target_column : str
        Name of the target column.
    id_column : str
        Name of the entity identifier column.
    time_column : str
        Name of the month column.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If any of the three named columns is absent.
    ValueError
        If the frame is empty, if the time column is entirely missing, or if an
        (entity, month) pair appears more than once.
    """
    missing = [
        name
        for name in (target_column, id_column, time_column)
        if name not in frame.columns
    ]
    if missing:
        raise KeyError(
            f"Configured columns are not in the query result: {missing}. "
            f"Available: {list(frame.columns)}"
        )

    if frame.empty:
        raise ValueError("The query returned no rows")

    if bool(frame[time_column].isna().all()):
        raise ValueError(f"Column {time_column!r} is entirely missing")

    # One row per user per month is the assumption a time split rests on: a
    # duplicated pair would place the same month in train and in test.
    duplicated = frame.duplicated(subset=[id_column, time_column])
    if bool(duplicated.any()):
        example = frame.loc[duplicated, [id_column, time_column]].head(3)
        raise ValueError(
            f"{int(duplicated.sum())} duplicate ({id_column}, {time_column}) rows; "
            f"the table must hold one row per entity per month. "
            f"First few:\n{example}"
        )


def load_feature_frame(
    config: dict[str, Any] | None = None, engine: Engine | None = None
) -> pd.DataFrame:
    """Run the configured query and return the raw result.

    Kept separate from ``build_dataset`` so the notebook can look at exactly
    what the database returned, before any coercion has had a chance to hide
    something.

    Parameters
    ----------
    config : dict, optional
        Parsed config. Loaded from ``config/ml_config.yaml`` when omitted.
    engine : sqlalchemy.engine.Engine, optional
        Engine to read through. A new one is built when omitted.

    Returns
    -------
    pandas.DataFrame
        The query result, untouched.
    """
    settings = config if config is not None else load_config()
    return load_dataframe(resolve_query(settings["data"]), engine=engine)


def resolve_table(data_settings: dict[str, Any], table: str | None = None) -> str:
    """Return the source table name a run should read.

    ``data.tables`` names each candidate table and ``data.active_table`` picks
    one, so switching which panel the models are trained and tested on is a
    one-word edit and both table names stay in git rather than one of them
    living in an editor's undo history.

    Parameters
    ----------
    data_settings : dict
        The ``data`` block of the parsed config.
    table : str, optional
        A key of ``data.tables``, or a raw table name. Defaults to
        ``data.active_table``.

    Returns
    -------
    str
        The table name to query.

    Raises
    ------
    KeyError
        If neither ``data.tables`` nor ``data.table`` is configured, or if the
        requested key is not among them.
    """
    tables = data_settings.get("tables") or {}
    wanted = table or data_settings.get("active_table") or data_settings.get("table")

    if not tables:
        if wanted:
            return str(wanted)
        raise KeyError("Config names no source table; set data.tables and data.active_table")

    if wanted is None:
        raise KeyError(
            f"data.tables defines {sorted(tables)} but data.active_table does "
            f"not say which to use"
        )

    wanted = str(wanted)
    if wanted in tables:
        return str(tables[wanted])
    if wanted in set(map(str, tables.values())):
        return wanted
    raise KeyError(
        f"No table named {wanted!r}; data.tables defines {sorted(tables)}"
    )


def resolve_query(data_settings: dict[str, Any], table: str | None = None) -> str:
    """Return the SQL to run against the source database.

    Parameters
    ----------
    data_settings : dict
        The ``data`` block of the parsed config.
    table : str, optional
        Overrides ``data.active_table``. See :func:`resolve_table`.

    Returns
    -------
    str
        ``data.query`` verbatim when one is configured, otherwise a select over
        the active table.

    Notes
    -----
    ``SELECT *`` rather than a column list, deliberately: the feature set is
    decided in ``data.drop_columns`` and the table is allowed to grow a column
    without a code change. The name is quoted because it comes from a config
    file the project owns, not from user input -- the quoting is here so a
    table name with a capital letter survives PostgreSQL's case folding, not as
    an injection guard.
    """
    explicit = data_settings.get("query")
    if explicit and table is None:
        return str(explicit)
    return f'SELECT * FROM "{resolve_table(data_settings, table)}"'


def resolve_drop_columns(
    data_settings: dict[str, Any], drop_columns_for: str | None = None
) -> tuple[str, ...]:
    """Return the drop list one model family should use.

    ``data.drop_columns`` in the config is a mapping of family name to list --
    ``regression`` and ``xgboost`` today -- because the two fail differently on
    the same column: a booster ignores a redundant feature, while Ridge splits
    the weight arbitrarily across a collinear group and its coefficients stop
    meaning anything. The lists are the disagreement written down.

    A flat list is still accepted and used for every model, so an older config
    keeps working.

    Parameters
    ----------
    data_settings : dict
        The ``data`` block of the parsed config.
    drop_columns_for : str, optional
        Which named list to read. Defaults to ``data.default_drop_columns``,
        and then to the only list present when there is exactly one.

    Returns
    -------
    tuple of str
        Column names to exclude from the feature set. Empty when the config
        names no drop columns at all.

    Raises
    ------
    KeyError
        If a name is asked for that the mapping does not have. Raised rather
        than falling back to an empty list, because a typo that silently
        selected *every* column -- including the raw balance lags the size
        filter exists to hide -- would train a model on account size and still
        report a number.
    """
    configured = data_settings.get("drop_columns") or ()

    if not isinstance(configured, dict):
        return tuple(str(name) for name in configured)

    wanted = drop_columns_for or data_settings.get("default_drop_columns")
    if wanted is None:
        if len(configured) != 1:
            raise KeyError(
                f"data.drop_columns has {len(configured)} lists "
                f"({sorted(configured)}) and no data.default_drop_columns to "
                f"choose between them; pass drop_columns_for explicitly"
            )
        (wanted,) = configured

    wanted = str(wanted)
    if wanted not in configured:
        raise KeyError(
            f"No drop_columns list named {wanted!r}; the config defines "
            f"{sorted(configured)}"
        )
    return tuple(str(name) for name in configured[wanted] or ())


def trim_whale_entities(
    frame: pd.DataFrame,
    id_column: str,
    time_column: str,
    size_column: str,
    share: float,
    test_share: float,
    gap_months: int = 0,
) -> pd.DataFrame:
    """Drop the largest ``share`` of entities, ranked by account size.

    The panel's error is not spread evenly across users: on the holdout the
    top decile of accounts carries a multiple of the median account's dollar
    error, and both the squared loss and RMSE are decided almost entirely by
    them. Dropping them is a statement about which population is being served,
    not a cleaning step -- the numbers that come out afterwards describe a
    different and smaller problem, and are not comparable with the numbers from
    the full panel.

    Two choices here matter.

    **Entities, not rows.** Trimming the largest *rows* would cut particular
    months out of a user's history, which puts a hole in exactly the lag and
    rolling features the whole table is built from, and -- worse -- selects
    those rows using the target. A user is either in the population or is not.

    **Ranked on training months only.** The ranking reads the anchor column
    over the months before the holdout cut, so which users are dropped cannot
    depend on anything inside the held-out window. Account size is stable
    enough here that ranking over the full panel would pick nearly the same
    users, which is precisely why it would be easy to leak by accident and not
    notice.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows, after the target and month filters.
    id_column, time_column : str
        Entity and month column names.
    size_column : str
        Column standing in for account size, normally the anchor
        (last month's closing balance). Ranked by each entity's median
        absolute value.
    share : float
        Share of entities to drop, between 0 and 1. 0.05 drops the largest 5%.
    test_share : float
        Share of months in the holdout, so the ranking can avoid it.
    gap_months : int, optional
        Months discarded between train and test. Default 0.

    Returns
    -------
    pandas.DataFrame
        ``frame`` without the dropped entities' rows. Returned unchanged when
        ``share`` is 0.

    Raises
    ------
    ValueError
        If ``share`` is not in [0, 1), or if the trim would empty the panel.
    """
    if not share:
        return frame
    if not 0 <= share < 1:
        raise ValueError(
            f"data.trim_top_entities must be in [0, 1), got {share!r}"
        )
    if size_column not in frame.columns:
        raise KeyError(
            f"data.trim_top_entities ranks on {size_column!r}, which is not in "
            f"the table"
        )

    # Imported here rather than at module scope: src.window imports Dataset
    # from this module, so a top-level import would be circular.
    from src.window import plan_month_cut

    months = pd.DatetimeIndex(sorted(frame[time_column].unique()))
    train_months, _ = plan_month_cut(months, test_share, gap_months)
    seen = frame.loc[frame[time_column].isin(train_months)]

    size = seen.groupby(id_column)[size_column].apply(lambda col: col.abs().median())
    # Entities with no usable size in the training region sort last and are
    # kept: a missing measurement is not evidence of being a whale.
    size = size.dropna().sort_values(ascending=False)

    n_drop = int(np.floor(len(size) * share))
    if n_drop == 0:
        return frame

    dropped = set(size.index[:n_drop])
    kept = frame.loc[~frame[id_column].isin(dropped)]
    if kept.empty:
        raise ValueError(
            f"data.trim_top_entities of {share} removed every row"
        )
    logger.info(
        f"Trimmed   : {n_drop} of {len(size)} users "
        f"({100 * n_drop / len(size):.1f}%) as whales, ranked on median "
        f"|{size_column}| over {len(train_months)} training months; "
        f"{len(frame) - len(kept):,} rows dropped"
    )
    return kept


def build_dataset(
    config: dict[str, Any] | None = None,
    engine: Engine | None = None,
    frame: pd.DataFrame | None = None,
    drop_columns_for: str | None = None,
    table: str | None = None,
) -> Dataset:
    """Load, coerce, validate and sort the feature table into a ``Dataset``.

    The single entry point used by both the notebook and the container job, so
    that an experiment and a scheduled run start from identical inputs.

    Parameters
    ----------
    config : dict, optional
        Parsed config. Loaded from ``config/ml_config.yaml`` when omitted.
    engine : sqlalchemy.engine.Engine, optional
        Engine to read through. Ignored when ``frame`` is supplied.
    frame : pandas.DataFrame, optional
        An already-loaded frame to use instead of querying. Lets the notebook
        re-run the checks on a subset without a second round trip.
    drop_columns_for : str, optional
        Which named list under ``data.drop_columns`` decides the feature set --
        ``regression`` or ``xgboost``. Defaults to
        ``data.default_drop_columns``. See :func:`resolve_drop_columns`.
    table : str, optional
        Which entry of ``data.tables`` to read. Defaults to
        ``data.active_table``. Ignored when ``frame`` is supplied.

    Returns
    -------
    Dataset
        Rows sorted by entity then month, index reset, roles assigned.

    Raises
    ------
    KeyError
        If a configured column is not present in the data.
    ValueError
        If the data fails one of the checks in ``validate_frame``, or if every
        row is missing the target.
    """
    settings = config if config is not None else load_config()
    data_settings = settings["data"]

    target_column = str(data_settings["target_column"])
    id_column = str(data_settings["id_column"])
    time_column = str(data_settings["time_column"])
    drop_columns = resolve_drop_columns(data_settings, drop_columns_for)
    renames = {
        str(old): str(new)
        for old, new in (data_settings.get("rename_columns") or {}).items()
    }
    # The frozen feature set, top level in the config rather than under `data`
    # because it is an output of feature selection rather than a property of
    # the source table. Empty or absent means "derive by exclusion".
    keep_columns = tuple(str(name) for name in (settings.get("features") or ()))

    if frame is not None:
        raw = frame
    else:
        raw = load_dataframe(resolve_query(data_settings, table), engine=engine)

    # Source renames, applied before anything else reads the frame so that every
    # later error message names the column the config names. Only renames whose
    # source column is actually present, so re-running over an already-renamed
    # frame -- which the notebook does -- is a no-op rather than an error.
    active = {old: new for old, new in renames.items() if old in raw.columns}
    if active:
        collisions = [new for new in active.values() if new in raw.columns]
        if collisions:
            raise ValueError(
                f"data.rename_columns would create duplicate column(s) "
                f"{collisions}; the target name is already in the table"
            )
        raw = raw.rename(columns=active)

    # Validate before coercing, so a missing column is reported as a missing
    # column rather than as a conversion failure on a column that is not there.
    validate_frame(raw, target_column, id_column, time_column)
    typed = coerce_types(raw, id_column=id_column, time_column=time_column)

    # A row without a target cannot be trained on and cannot be scored, and
    # keeping it would let it count towards the size of a fold.
    typed = typed.loc[typed[target_column].notna()]
    if typed.empty:
        raise ValueError(f"Every row is missing {target_column!r}")

    # The whale trim. After the target filter so the ranking sees the same
    # months everything else does, and before the sort so the sort is done once
    # on the surviving rows. Reads `split` as well as `data`, because which
    # months count as training is a property of the split and duplicating the
    # cut here is how the two drift apart.
    trim_share = float(data_settings.get("trim_top_entities") or 0.0)
    if trim_share:
        split_settings = settings.get("split") or {}
        typed = trim_whale_entities(
            typed,
            id_column=id_column,
            time_column=time_column,
            size_column=str(
                (settings.get("evaluation") or {}).get(
                    "anchor_column", "prev_1m_closing_balance_usd"
                )
            ),
            share=trim_share,
            test_share=float(split_settings.get("test_share", 0.2)),
            gap_months=int(split_settings.get("gap_months", 0)),
        )

    # Sorted by entity then time: every downstream check that reasons about
    # "the previous month" assumes this order, and sorting once here is cheaper
    # than defending against it in each of them.
    ordered = typed.sort_values([id_column, time_column]).reset_index(drop=True)

    features = resolve_feature_columns(
        ordered,
        target_column=target_column,
        id_column=id_column,
        time_column=time_column,
        drop_columns=drop_columns,
        keep_columns=keep_columns,
    )
    return Dataset(
        frame=ordered,
        target_column=target_column,
        id_column=id_column,
        time_column=time_column,
        feature_columns=features,
    )
