"""Loads settings from config/ml_config.yaml and the environment.

Kept separate from the code that uses it so the notebook and the container
read configuration the same way -- the notebook picks up .env via dotenv, the
container gets the same variables injected by compose.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# /app in the container, the repo root locally. Resolved from this file rather
# than the working directory, so `python -m src.train` behaves the same no
# matter where it is launched from. Three levels up: this file sits at
# src/config/config.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "ml_config.yaml"

# Reads .env when running locally. Inside the container there is no .env file
# and compose has already set the variables, so this is a no-op there --
# override=False makes sure it can never clobber them.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the parsed YAML config."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def database_url() -> str:
    """Build the SQLAlchemy URL from environment variables.

    Deliberately reads host and port from the environment instead of taking
    them from the YAML. Postgres has two valid addresses -- localhost:5433
    from Windows, postgres:5432 from inside the compose network -- and which
    is correct depends on where this code happens to be running. Compose sets
    them for the container; .env sets them for the notebook.
    """
    user = os.environ.get("POSTGRES_USER", "pipeline")
    password = os.environ.get("POSTGRES_PASSWORD", "pipeline")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5433")
    database = os.environ.get("POSTGRES_DB", "ml_prediction")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def resolve_output_dir(config: dict[str, Any]) -> Path:
    """Absolute path for saved models, creating it if needed."""
    model_dir = PROJECT_ROOT / config.get("output", {}).get("model_dir", "models")
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir
