"""Materialise the engineered feature table into the database, to look at.

``feature_store_monthly`` holds the raw columns. What the models are actually
fitted on is that table plus the 25 derived columns from ``src/features.py``,
minus everything in ``data.drop_columns`` -- a set that only exists in memory,
halfway through a training run, where nobody can query it. This script writes
that set out as a real table so it can be read with SQL, eyeballed in a client,
or joined against something else.

**It is the same code path, not a reimplementation.** The rows come from
``build_dataset``, which is what ``train.py`` and ``test.py`` call. Writing the
derivations a second time in SQL would create two definitions of ``growth_1m``
that drift apart the first time one is edited, and the version in the table
would be the one nobody is training on.

**What is in the table beyond the features.** The three keys, and four columns
that are deliberately not features:

``prev_1m_closing_balance_usd``
    The anchor. Models predict a movement; this is what the movement is added
    to. Without it the table can be read but never trained from, because there
    would be no way to turn a predicted movement back into dollars.
``prev_2m_/prev_3m_closing_balance_usd``
    What the 3-month baseline averages. Without them there is no floor to
    measure the fitted models against.

They sit in the frame at training time for exactly these reasons and are
excluded from the feature list rather than from the data, so the same holds
here.

**This is a snapshot, not a source of truth.** It is written once, when you run
this, from whatever ``feature_store_monthly`` held at that moment. Nothing
refreshes it. If the source table gains rows, this one is stale until you run
the script again -- which is also why ``train.py`` does not read it by default.

Run it with::

    python -m src.export_features                 # replace the table
    python -m src.export_features --dry-run       # build it, print it, write nothing

To then train from it instead of the raw table, point ``data.query`` at it::

    query: |
      SELECT * FROM features_table_two

That works: every builder in ``src/features.py`` is guarded on the presence of
its inputs, so the ones whose raw dollar columns are no longer there simply do
not run, and the derived columns already in the table pass straight through.
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.config.config import load_config
from src.data.data import build_dataset
from src.data.db_link import get_engine
from src.models_code.baseline import BALANCE_LAGS

# Where the table lands when the config does not say. Overridable with
# `output.feature_table` so a second experiment can be written beside the first
# rather than over it.
DEFAULT_TABLE = "features_table_two"

# Columns kept alongside the features. Not features themselves -- every one of
# them is in `data.drop_columns` -- but the pipeline needs them, so a table
# without them is an inspection artefact rather than something you could train
# from. See the module docstring.
KEPT_NON_FEATURES: tuple[str, ...] = BALANCE_LAGS


def build_export(config: dict[str, Any]) -> pd.DataFrame:
    """Return the frame to write: keys, target, features, and the anchors.

    Parameters
    ----------
    config : dict
        Parsed config.

    Returns
    -------
    pandas.DataFrame
        One row per (user, month), ordered by user then month. Column order is
        keys first, then the non-feature columns the pipeline needs, then the
        features in the order the pipeline resolved them -- so the table reads
        left to right the way the pipeline does.
    """
    dataset = build_dataset(config)

    keys = [dataset.id_column, dataset.time_column, dataset.target_column]
    # Only the ones actually present: a source table that loses a lag column
    # should produce a smaller export, not a KeyError.
    kept = [
        column
        for column in KEPT_NON_FEATURES
        if column in dataset.frame.columns and column not in keys
    ]
    features = [
        column for column in dataset.feature_columns if column not in (*keys, *kept)
    ]

    return dataset.frame.loc[:, [*keys, *kept, *features]].copy()


def write_table(
    frame: pd.DataFrame,
    table: str,
    id_column: str,
    time_column: str,
    engine: Engine | None = None,
) -> None:
    """Replace ``table`` with ``frame`` and index it for panel queries.

    Parameters
    ----------
    frame : pandas.DataFrame
        What to write.
    table : str
        Destination table name.
    id_column : str
        Entity column, used in the index.
    time_column : str
        Month column, used in the index.
    engine : sqlalchemy.engine.Engine, optional
        Engine to write through. Built from the environment when omitted.

    Returns
    -------
    None

    Notes
    -----
    ``if_exists="replace"`` drops and recreates. That is the intended
    behaviour -- the table is a snapshot of a derivation, so a partial update
    would leave rows computed under two different versions of
    ``src/features.py`` sitting side by side with no way to tell them apart.
    """
    engine = engine or get_engine()
    frame.to_sql(table, engine, if_exists="replace", index=False, chunksize=5_000)

    # The query every reader will write is "this user, over time", and the
    # export is one row per (user, month). Created after the load rather than
    # before, because maintaining an index during the insert is slower than
    # building it once at the end.
    with engine.begin() as connection:
        connection.execute(
            text(
                f'CREATE INDEX IF NOT EXISTS "{table}_user_month_idx" '
                f'ON "{table}" ("{id_column}", "{time_column}")'
            )
        )


def run(config: dict[str, Any] | None = None, dry_run: bool = False) -> pd.DataFrame:
    """Build the export and, unless asked not to, write it.

    Parameters
    ----------
    config : dict, optional
        Parsed config. Loaded from ``config/ml_config.yaml`` when omitted.
    dry_run : bool, optional
        Build and describe the table but write nothing. Default False.

    Returns
    -------
    pandas.DataFrame
        The frame that was written, or would have been.
    """
    settings = config if config is not None else load_config()
    data_block = settings.get("data") or {}
    table = str((settings.get("output") or {}).get("feature_table", DEFAULT_TABLE))

    frame = build_export(settings)

    id_column = str(data_block["id_column"])
    time_column = str(data_block["time_column"])
    target_column = str(data_block["target_column"])
    keys = {id_column, time_column, target_column}
    kept = [column for column in KEPT_NON_FEATURES if column in frame.columns]
    features = [
        column for column in frame.columns if column not in keys and column not in kept
    ]

    print(f"Table     : {table}")
    print(
        f"Rows      : {len(frame):,}  "
        f"({frame[id_column].nunique():,} users x "
        f"{frame[time_column].nunique()} months)"
    )
    print(f"Months    : {frame[time_column].min():%Y-%m} .. "
          f"{frame[time_column].max():%Y-%m}")
    print(f"Columns   : {len(frame.columns)} = 3 keys + {len(kept)} anchors "
          f"+ {len(features)} features")

    print(f"\n--- Keys\n  {id_column}\n  {time_column}\n  {target_column}  (target)")
    print("\n--- Kept for the pipeline, not features")
    for column in kept:
        role = "anchor" if column == BALANCE_LAGS[0] else "baseline input"
        print(f"  {column:<40} {role}")

    print(f"\n--- Features ({len(features)})")
    for column in features:
        # Null share is the one thing worth seeing per column here: a ratio
        # whose denominator was below the epsilon guard is NaN by design, and a
        # column that is mostly NaN is one the models can barely use.
        null_share = float(frame[column].isna().mean())
        print(f"  {column:<40} {100 * null_share:5.1f}% null")

    if dry_run:
        print("\n--dry-run: nothing written.")
        return frame

    write_table(frame, table, id_column, time_column)
    print(f"\nWritten to {table} (replaced) and indexed on "
          f"({id_column}, {time_column}).")
    return frame


def main() -> None:
    """Parse arguments and run the export.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Write the engineered feature table to the database."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and describe the table, write nothing",
    )
    arguments = parser.parse_args()
    run(dry_run=arguments.dry_run)


if __name__ == "__main__":
    main()
