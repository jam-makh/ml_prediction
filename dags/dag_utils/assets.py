"""Assets shared by the two DAGs, defined once so both files use the same name."""

from __future__ import annotations

from airflow.sdk import Asset

# Emitted by the prediction DAG's health check on decay; the retrain DAG is scheduled on it.
RETRAIN_REQUEST = Asset(name="joseph_retrain_request")
