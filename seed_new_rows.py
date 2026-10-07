"""Append one fake month of feature rows so the DAGs have new data to predict.

Run ``python seed_new_rows.py``: each run adds the month after the latest one in the feature
table, so the tick of "a month passes" is one run. Defaults live in ``SeedSettings`` and can be
overridden by a ``seeding:`` block in ``ml_config.yaml``.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.engine import Engine

from dags.dag_utils.db import FeatureTable
from src.config.config import load_config
from src.data.db_link import get_engine

# A user needs this many prior balances to count as full history (history_id 2).
FULL_HISTORY_MONTHS = 6
# Id 0 of seg_history; the routed model serves it with persistence when a last-month balance exists.
UNASSIGNED_HISTORY = 0


class SeedSettings(BaseModel):
    """What one seeding run adds.

    Parameters
    ----------
    new_users : int
        Brand-new users appearing this month, each with an opening balance but no history, so
        the routed model serves them with persistence.
    shock : float
        Multiplier on each user's usual monthly swing; above 1 forces live error to decay.
    seed : int
        Base of the random generator; the month is mixed in so each run draws new values.
    """

    new_users: int = 1
    shock: float = 1.0
    seed: int = 42

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SeedSettings:
        """Build from the optional ``seeding`` block, with ``data.random_state`` as the seed.

        Parameters
        ----------
        config : dict
            Parsed ``ml_config.yaml``.

        Returns
        -------
        SeedSettings
            Defaults overridden by whatever the block sets.
        """
        block = {"seed": int(config["data"]["random_state"]), **(config.get("seeding") or {})}
        return cls.model_validate(block)


def _lag(balances: npt.NDArray[np.float64], months: int) -> float:
    """Return the balance ``months`` back, or NaN when the history is shorter.

    Parameters
    ----------
    balances : numpy.ndarray of float
        Closing balances, oldest first.
    months : int
        How far back, 1 being the latest.

    Returns
    -------
    float
        The balance, or NaN.
    """
    return float(balances[-months]) if len(balances) >= months else np.nan


def _window(balances: npt.NDArray[np.float64], size: int, stat: str) -> float:
    """Return a statistic over the last ``size`` balances, or NaN until the window is full.

    Parameters
    ----------
    balances : numpy.ndarray of float
        Closing balances, oldest first.
    size : int
        Window length in months.
    stat : {"mean", "std", "share_negative"}
        Statistic to compute; ``std`` uses ``ddof=1`` like the real feature table.

    Returns
    -------
    float
        The statistic, or NaN.
    """
    if len(balances) < size:
        return np.nan
    recent = balances[-size:]
    if stat == "mean":
        return float(np.mean(recent))
    if stat == "std":
        return float(np.std(recent, ddof=1))
    return float(np.mean(recent < 0))


class NextMonthBuilder:
    """Turn the feature table's history into the rows of the month after it.

    Parameters
    ----------
    table : FeatureTable
        Names of the id, month and target columns.
    settings : SeedSettings
        What to add this run.
    """

    def __init__(self, table: FeatureTable, settings: SeedSettings) -> None:
        self.table = table
        self.settings = settings

    def build(self, history: pd.DataFrame) -> pd.DataFrame:
        """Roll every active user forward one month and add the new users.

        Parameters
        ----------
        history : pandas.DataFrame
            The whole feature table as stored.

        Returns
        -------
        pandas.DataFrame
            One row per active user plus the newcomers, with the table's columns.
        """
        t = self.table
        frame = history.assign(**{t.month_column: pd.to_datetime(history[t.month_column])})
        frame = frame.sort_values([t.id_column, t.month_column]).reset_index(drop=True)
        last_month = frame[t.month_column].max()
        next_month = last_month + pd.offsets.MonthBegin(1)
        # Mixing the month into the seed gives each run fresh draws and fresh new-user ids.
        rng = np.random.default_rng(self.settings.seed + next_month.year * 12 + next_month.month)

        # Users with no row in the latest month have left, so they are not rolled forward.
        latest = frame.groupby(t.id_column).tail(1)
        active = latest.loc[latest[t.month_column] == last_month]
        steps = frame.groupby(t.id_column)[t.target_column].diff()
        # The typical swing stands in for users with too little history to measure their own.
        typical_step = float(steps.groupby(frame[t.id_column]).std().median())

        rows = [
            self._roll(user_rows, next_month, typical_step, rng)
            for _, user_rows in frame[frame[t.id_column].isin(active[t.id_column])].groupby(
                t.id_column
            )
        ]
        templates = active.iloc[rng.integers(0, len(active), size=self.settings.new_users)]
        rows += [self._newcomer(row, next_month, rng) for _, row in templates.iterrows()]

        built = pd.DataFrame(rows, columns=frame.columns)
        # Integer columns (segment ids) must stay integers for the insert.
        ints = frame.select_dtypes("integer").columns
        built[ints] = built[ints].astype(int)
        return built

    def _roll(
        self,
        user_rows: pd.DataFrame,
        next_month: pd.Timestamp,
        typical_step: float,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Build one existing user's next-month row from their own history.

        Parameters
        ----------
        user_rows : pandas.DataFrame
            All rows of one user, oldest first.
        next_month : pandas.Timestamp
            Month to create.
        typical_step : float
            Monthly swing used when the user has too few months to measure their own.
        rng : numpy.random.Generator
            Source of the target's random step.

        Returns
        -------
        dict
            The new row, keyed by column name.
        """
        t = self.table
        balances = user_rows[t.target_column].to_numpy(dtype=float)
        diffs = np.diff(balances)
        step = float(np.std(diffs, ddof=1)) if len(diffs) >= 2 else typical_step
        # Balances are often negative, so the target moves by an additive step, not a percentage.
        target = balances[-1] + rng.normal(0.0, step * self.settings.shock)
        return self._row(self._as_dict(user_rows.iloc[-1]), balances, next_month, target)

    def _newcomer(
        self, template: pd.Series, next_month: pd.Timestamp, rng: np.random.Generator
    ) -> dict[str, Any]:
        """Build a brand-new user's first row: an opening balance, no history, template flows.

        Parameters
        ----------
        template : pandas.Series
            An active user's latest row.
        next_month : pandas.Timestamp
            Month the newcomer first appears.
        rng : numpy.random.Generator
            Source of the id and the balance scale.

        Returns
        -------
        dict
            The new row, keyed by column name.
        """
        t = self.table
        carry = self._as_dict(template)
        carry[t.id_column] = str(uuid.UUID(bytes=rng.bytes(16), version=4))
        opening = float(template[t.target_column]) * rng.uniform(0.5, 1.5)
        target = opening * rng.uniform(0.95, 1.05)
        row = self._row(carry, np.array([opening]), next_month, target)
        # An opening balance but no history is history_id 0 with an anchor: persistence, not skip.
        row["history_id"] = UNASSIGNED_HISTORY
        return row

    def _row(
        self,
        carry: dict[str, Any],
        balances: npt.NDArray[np.float64],
        next_month: pd.Timestamp,
        target: float,
    ) -> dict[str, Any]:
        """Rebuild the balance-derived columns and keep every other column as it was.

        Parameters
        ----------
        carry : dict
            The previous row, whose flow, spend and segment columns are carried forward.
        balances : numpy.ndarray of float
            The user's closing balances so far, oldest first.
        next_month : pandas.Timestamp
            Month of the new row.
        target : float
            Closing balance of the new month.

        Returns
        -------
        dict
            The new row, keyed by column name.
        """
        t = self.table
        change = _lag(balances, 1) - _lag(balances, 2)
        row = dict(carry)
        row.update(
            {
                t.month_column: next_month,
                t.target_column: target,
                "prev_1m_closing_balance_usd": _lag(balances, 1),
                "prev_2m_closing_balance_usd": _lag(balances, 2),
                "prev_3m_closing_balance_usd": _lag(balances, 3),
                "prev_1m_change": change,
                # Net flow is the month's balance change, which is how the real table behaves.
                "prev_1m_net_flow_usd": change,
                "roll3_mean_closing_balance_usd": _window(balances, 3, "mean"),
                "roll3_std_closing_balance_usd": _window(balances, 3, "std"),
                "roll6_mean_balance": _window(balances, 6, "mean"),
                "roll6_std_balance": _window(balances, 6, "std"),
                "share_neg_last_6m": _window(balances, 6, "share_negative"),
                "history_id": self._history_id(len(balances)),
            }
        )
        return row

    @staticmethod
    def _as_dict(row: pd.Series) -> dict[str, Any]:
        """Return a row as a plain dict keyed by column name.

        Parameters
        ----------
        row : pandas.Series
            One row of the feature table.

        Returns
        -------
        dict
            Column name to value.
        """
        return {str(name): value for name, value in row.items()}

    @staticmethod
    def _history_id(n_balances: int) -> int:
        """Return the history segment: 0 none, 1 thin, 2 full.

        Parameters
        ----------
        n_balances : int
            Prior closing balances the user has.

        Returns
        -------
        int
            Id matching ``seg_history``.
        """
        if n_balances == 0:
            return UNASSIGNED_HISTORY
        return 2 if n_balances >= FULL_HISTORY_MONTHS else 1


class FeatureWriter:
    """Read the feature table and append rows to it.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        Connection to the business database.
    table : FeatureTable
        The feature table and its column names.
    """

    def __init__(self, engine: Engine, table: FeatureTable) -> None:
        self.engine = engine
        self.table = table

    def read(self) -> pd.DataFrame:
        """Return the whole feature table as stored.

        Returns
        -------
        pandas.DataFrame
            Every row and column.
        """
        with self.engine.connect() as conn:
            return pd.read_sql_query(text(f'SELECT * FROM "{self.table.name}"'), conn)

    def append(self, rows: pd.DataFrame) -> int:
        """Insert rows at the end of the feature table.

        Parameters
        ----------
        rows : pandas.DataFrame
            Rows with the table's columns.

        Returns
        -------
        int
            Rows inserted.
        """
        month = self.table.month_column
        # The column is a date, so the timestamp's time part is dropped before the insert.
        out = rows.assign(**{month: rows[month].dt.date})
        out.to_sql(
            self.table.name, self.engine, if_exists="append", index=False, method="multi",
            chunksize=500,
        )
        return len(out)


def main() -> None:
    """Seed one new month and log what was added.

    Returns
    -------
    None
    """
    config = load_config()
    settings = SeedSettings.from_config(config)
    table = FeatureTable.from_config(config)
    writer = FeatureWriter(get_engine(), table)

    rows = NextMonthBuilder(table, settings).build(writer.read())
    inserted = writer.append(rows)
    month = rows[table.month_column].iloc[0]
    logger.info(
        f"seeded {inserted} rows for {month:%Y-%m} into {table.name}: "
        f"{inserted - settings.new_users} existing users, {settings.new_users} new, "
        f"shock x{settings.shock}"
    )


if __name__ == "__main__":
    main()
