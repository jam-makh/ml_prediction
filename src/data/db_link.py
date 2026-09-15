"""Database access: engine construction and loading the training data."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config.config import database_url


def get_engine() -> Engine:
    """Create a SQLAlchemy engine for the configured database.

    pool_pre_ping avoids handing out a connection the server has already
    dropped -- easy to hit when the container sits idle between runs.
    """
    return create_engine(database_url(), pool_pre_ping=True)


def check_connection() -> bool:
    """Run a trivial query. Useful as the first cell of a notebook."""
    with get_engine().connect() as conn:
        return conn.execute(text("SELECT 1")).scalar_one() == 1


def load_dataframe(query: str, engine: Engine | None = None) -> pd.DataFrame:
    """Run `query` and return the result as a DataFrame."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        return pd.read_sql_query(text(query), conn)
