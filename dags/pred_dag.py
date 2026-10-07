"""Daily prediction DAG: predict pending rows, resolve live errors, request a retrain on decay."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG
from airflow.sdk.exceptions import AirflowSkipException

from dag_utils import tasks
from dag_utils.assets import RETRAIN_REQUEST
from dag_utils.logs import route_to_airflow
from dag_utils.orchestration import DagSettings
from src.config.config import load_config

# Airflow re-imports this file inside every task process, so this covers every task's logs.
route_to_airflow()
SETTINGS = DagSettings.from_config(load_config())

DEFAULT_ARGS = {
    "owner": "joseph",
    "retries": 5,
    "retry_delay": timedelta(seconds=30),
}


def check_health() -> None:
    """Skip when the model is healthy, so the retrain asset is only emitted on decay.

    Returns
    -------
    None

    Raises
    ------
    AirflowSkipException
        If no retrain is needed; a skipped task emits no asset event.
    """
    if not tasks.check_health_and_maybe_trigger_retrain():
        raise AirflowSkipException("model healthy; no retrain requested")


with DAG(
    dag_id="joseph_prediction",
    schedule="@daily",
    start_date=datetime(2026, 10, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["joseph", "serving"],
):
    t1 = PythonOperator(
        task_id="check_pending_rows",
        python_callable=tasks.check_pending_rows,
        execution_timeout=SETTINGS.timeout("check_pending_rows"),
    )
    # `.output` hands over the list as a Python object; a Jinja template would render it as text.
    t2 = PythonOperator(
        task_id="run_predictions",
        python_callable=tasks.run_predictions,
        op_kwargs={"pending": t1.output},
        execution_timeout=SETTINGS.timeout("run_predictions"),
    )
    t3 = PythonOperator(
        task_id="refresh_live_performance",
        python_callable=tasks.refresh_live_performance,
        execution_timeout=SETTINGS.timeout("refresh_live_performance"),
    )
    t4 = PythonOperator(
        task_id="check_health_and_maybe_trigger_retrain",
        python_callable=check_health,
        outlets=[RETRAIN_REQUEST],
        execution_timeout=SETTINGS.timeout("check_health_and_maybe_trigger_retrain"),
    )

    t1 >> t2 >> t3 >> t4
