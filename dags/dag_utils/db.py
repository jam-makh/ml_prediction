"""
Repository for the ``serving`` schema: the four serving tables and every query against them.
Create the tables with ``python -m dag_utils.db``; the project root and dags/ go on PYTHONPATH.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

import pandas as pd
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Double,
    Integer,
    MetaData,
    Select,
    String,
    Table,
    Text,
    func,
    literal_column,
    select,
    tuple_,
    update,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateSchema
from sqlalchemy.sql.expression import TableClause, column, table

from src.config.config import load_config
from src.data.data import resolve_table
from src.data.db_link import get_engine


class PredictionRow(BaseModel):
    """One served prediction, as written to ``serving.predictions``.

    Parameters
    ----------
    user_id : str
        User the prediction is for.
    month : datetime.date
        Month whose closing balance is predicted.
    prediction : float
        Predicted closing balance in USD.
    route : {"persistence", "xgboost"}
        Which inner model produced the value.
    model_version : str
        Version of the champion that served it.
    """

    user_id: str
    month: date
    prediction: float
    route: Literal["persistence", "xgboost"]
    model_version: str


class PerformanceRow(BaseModel):
    """Scores of one model version on one split, as written to ``serving.model_performance``.

    Parameters
    ----------
    model_version : str
        Version the scores belong to.
    split : {"train", "holdout"}
        Rows the scores were computed on.
    mae, rmse, wape : float
        Error metrics in USD (WAPE in percent).
    n_rows : int
        Rows scored.
    data_hash : str
        Hash of the training data the version was fitted on.
    trained_at : datetime.datetime
        When the version was fitted.
    """

    model_version: str
    split: Literal["train", "holdout"]
    mae: float
    rmse: float
    wape: float
    n_rows: int
    data_hash: str
    trained_at: datetime


class FeatureTable(BaseModel):
    """Where the feature rows live and which raw columns key them and hold the target.

    Parameters
    ----------
    name : str
        Feature table name.
    id_column : str
        User id column.
    month_column : str
        Month column as named in the database, before any config rename.
    target_column : str
        Closing balance column, the actual a prediction is resolved against.
    """

    name: str
    id_column: str
    month_column: str
    target_column: str

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FeatureTable:
        """Build from the ``data`` block of the config.

        Parameters
        ----------
        config : dict
            Parsed ``ml_config.yaml``.

        Returns
        -------
        FeatureTable
            The active feature table and its raw column names.
        """
        data = config["data"]
        time_column = str(data["time_column"])
        renames = data.get("rename_columns") or {}
        # The config renames month -> date on load; the database still calls it month.
        month_column = next(
            (old for old, new in renames.items() if new == time_column), time_column
        )
        return cls(
            name=resolve_table(data),
            id_column=str(data["id_column"]),
            month_column=str(month_column),
            target_column=str(data["target_column"]),
        )

    def clause(self) -> TableClause:
        """Return a lightweight table clause so queries are built without reflecting the table.

        Returns
        -------
        sqlalchemy.sql.expression.TableClause
            The table with its id, month and target columns.
        """
        return table(
            self.name, column(self.id_column), column(self.month_column), column(self.target_column)
        )


class ServingRepository:
    """Owns the ``serving`` tables and every read and write the DAGs make against them.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        Connection to the business database.
    schema : str
        Schema holding the serving tables.
    features : FeatureTable
        Source of the feature rows and of the actuals.
    """

    def __init__(self, engine: Engine, schema: str, features: FeatureTable) -> None:
        self.engine = engine
        self.schema = schema
        self.features = features
        self._features = features.clause()
        self.metadata = MetaData(schema=schema)

        # One bookmark per user: every month up to last_month has been handled.
        self.checkpoints = Table(
            "checkpoints",
            self.metadata,
            Column("user_id", Text, primary_key=True),
            Column("last_month", Date, nullable=False),
            Column(
                "updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()
            ),
        )
        # Keyed on (user, month) so each row is predicted exactly once, by the model then serving.
        self.predictions = Table(
            "predictions",
            self.metadata,
            Column("user_id", Text, primary_key=True),
            Column("month", Date, primary_key=True),
            Column("prediction", Double, nullable=False),
            Column("route", String(16), nullable=False),
            Column("model_version", String(64), nullable=False),
            Column(
                "created_at", DateTime(timezone=True), nullable=False, server_default=func.now()
            ),
        )
        # One row per version and split; promoted_at stays NULL until the version goes live.
        self.model_performance = Table(
            "model_performance",
            self.metadata,
            Column("model_version", String(64), primary_key=True),
            Column("split", String(16), primary_key=True),
            Column("mae", Double, nullable=False),
            Column("rmse", Double, nullable=False),
            Column("wape", Double, nullable=False),
            Column("n_rows", Integer, nullable=False),
            Column("data_hash", String(64), nullable=False),
            Column("trained_at", DateTime(timezone=True), nullable=False),
            Column("promoted_at", DateTime(timezone=True), nullable=True),
        )
        # Signed error (prediction - actual) is kept so MAE, RMSE and WAPE all derive from it.
        self.live_performance = Table(
            "live_performance",
            self.metadata,
            Column("user_id", Text, primary_key=True),
            Column("month", Date, primary_key=True),
            Column("model_version", String(64), nullable=False),
            Column("prediction", Double, nullable=False),
            Column("actual", Double, nullable=False),
            Column("error", Double, nullable=False),
            Column(
                "resolved_at", DateTime(timezone=True), nullable=False, server_default=func.now()
            ),
        )

    @classmethod
    def from_config(
        cls, config: dict[str, Any], engine: Engine | None = None
    ) -> ServingRepository:
        """Build the repository from the config and the environment's database.

        Parameters
        ----------
        config : dict
            Parsed ``ml_config.yaml``; needs ``data`` and ``serving.schema``.
        engine : sqlalchemy.engine.Engine, optional
            Defaults to ``get_engine()``.

        Returns
        -------
        ServingRepository
            Ready to use; call ``ensure_tables`` before the first write.
        """
        return cls(
            engine=engine or get_engine(),
            schema=str(config["serving"]["schema"]),
            features=FeatureTable.from_config(config),
        )

    def ensure_tables(self) -> None:
        """Create the schema and any missing serving table; existing tables are left untouched.

        Returns
        -------
        None
        """
        with self.engine.begin() as conn:
            conn.execute(CreateSchema(self.schema, if_not_exists=True))
            self.metadata.create_all(conn, checkfirst=True)
        tables = sorted(self.metadata.tables)
        logger.info(f"serving tables ready in schema '{self.schema}': {tables}")

    def pending_rows(self, floor: date) -> list[tuple[str, date]]:
        """Return feature rows newer than each user's checkpoint, oldest month first per user.

        Parameters
        ----------
        floor : datetime.date
            Bookmark for users with no checkpoint yet, normally the champion's last training month.

        Returns
        -------
        list of (str, datetime.date)
            ``(user_id, month)`` pairs still to predict.
        """
        f = self._features
        user, month = f.c[self.features.id_column], f.c[self.features.month_column]
        query = (
            select(user, month)
            .select_from(f.outerjoin(self.checkpoints, self.checkpoints.c.user_id == user))
            # A user without a checkpoint starts from the floor, so training rows are never served.
            .where(month > func.coalesce(self.checkpoints.c.last_month, floor))
            .order_by(user, month)
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query).all()
        return [(str(user_id), row_month) for user_id, row_month in rows]

    def load_features(self, keys: list[tuple[str, date]]) -> pd.DataFrame:
        """Fetch every feature column for the given ``(user_id, month)`` pairs.

        Parameters
        ----------
        keys : list of (str, datetime.date)
            Rows to fetch, usually the output of ``pending_rows``.

        Returns
        -------
        pandas.DataFrame
            Raw feature rows with the database's column names.
        """
        f = self._features
        query: Select[Any] = (
            select(literal_column("*"))
            .select_from(f)
            .where(tuple_(f.c[self.features.id_column], f.c[self.features.month_column]).in_(keys))
        )
        with self.engine.connect() as conn:
            return pd.read_sql_query(query, conn)

    def record_user(self, user_id: str, rows: list[PredictionRow], last_month: date) -> None:
        """Write one user's predictions and move their checkpoint, in a single transaction.

        Parameters
        ----------
        user_id : str
            User being recorded.
        rows : list of PredictionRow
            Predictions to insert; empty when every pending month was skipped.
        last_month : datetime.date
            Latest month handled for this user.

        Returns
        -------
        None
        """
        checkpoint = insert(self.checkpoints).values(user_id=user_id, last_month=last_month)
        # GREATEST keeps a late retry from moving the bookmark backwards.
        checkpoint = checkpoint.on_conflict_do_update(
            index_elements=[self.checkpoints.c.user_id],
            set_={
                "last_month": func.greatest(
                    self.checkpoints.c.last_month, checkpoint.excluded.last_month
                ),
                "updated_at": func.now(),
            },
        )
        with self.engine.begin() as conn:
            if rows:
                conn.execute(
                    insert(self.predictions).on_conflict_do_nothing(),
                    [row.model_dump() for row in rows],
                )
            conn.execute(checkpoint)

    def save_performance(self, rows: list[PerformanceRow]) -> int:
        """Insert model scores; a version and split already saved is left as it is.

        Parameters
        ----------
        rows : list of PerformanceRow
            Scores to save.

        Returns
        -------
        int
            Rows actually inserted.
        """
        if not rows:
            return 0
        query = insert(self.model_performance).on_conflict_do_nothing()
        with self.engine.begin() as conn:
            result = conn.execute(query, [row.model_dump() for row in rows])
        return int(result.rowcount)

    def mark_promoted(self, model_version: str) -> None:
        """Stamp ``promoted_at`` on a version's rows, once.

        Parameters
        ----------
        model_version : str
            Version that just went live.

        Returns
        -------
        None
        """
        table_ = self.model_performance
        query = (
            update(table_)
            .where(table_.c.model_version == model_version, table_.c.promoted_at.is_(None))
            .values(promoted_at=func.now())
        )
        with self.engine.begin() as conn:
            conn.execute(query)

    def resolve_live(self) -> int:
        """Join predictions to their real closing balance and store the error, skipping rows done.

        Returns
        -------
        int
            Predictions newly resolved.
        """
        f, p = self._features, self.predictions
        actual = f.c[self.features.target_column]
        resolved = (
            select(
                p.c.user_id, p.c.month, p.c.model_version, p.c.prediction, actual,
                p.c.prediction - actual,
            )
            .select_from(
                p.join(
                    f,
                    (f.c[self.features.id_column] == p.c.user_id)
                    & (f.c[self.features.month_column] == p.c.month),
                )
            )
            # Only months whose closing balance is known can be resolved.
            .where(actual.is_not(None))
        )
        query = insert(self.live_performance).from_select(
            ["user_id", "month", "model_version", "prediction", "actual", "error"], resolved
        )
        with self.engine.begin() as conn:
            result = conn.execute(query.on_conflict_do_nothing())
        return int(result.rowcount)

    def live_errors(
        self, model_version: str, route: str | None = None, since: date | None = None
    ) -> pd.DataFrame:
        """Return the resolved rows of one model version, with the route that served each.

        Parameters
        ----------
        model_version : str
            Version whose live rows are wanted.
        route : str, optional
            Keep only rows served by this route, e.g. ``"xgboost"``; all routes when omitted.
        since : datetime.date, optional
            First month to include; all months when omitted.

        Returns
        -------
        pandas.DataFrame
            Columns of ``serving.live_performance`` plus ``route``.
        """
        live, p = self.live_performance, self.predictions
        query = (
            select(live, p.c.route)
            # The route lives on the prediction, so live rows are joined back to it.
            .join(p, (p.c.user_id == live.c.user_id) & (p.c.month == live.c.month))
            .where(live.c.model_version == model_version)
        )
        if route is not None:
            query = query.where(p.c.route == route)
        if since is not None:
            query = query.where(live.c.month >= since)
        with self.engine.connect() as conn:
            return pd.read_sql_query(query, conn)

    def last_trained_at(self) -> datetime | None:
        """Return when the most recent model version was fitted, for the retrain cooldown.

        Returns
        -------
        datetime.datetime or None
            Latest ``trained_at``; None before the first retrain.
        """
        with self.engine.connect() as conn:
            value = conn.execute(select(func.max(self.model_performance.c.trained_at))).scalar_one()
        return value if isinstance(value, datetime) else None


if __name__ == "__main__":
    ServingRepository.from_config(load_config()).ensure_tables()
