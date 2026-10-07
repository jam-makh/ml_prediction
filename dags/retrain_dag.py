"""Weekly retrain DAG, also started early by the prediction DAG's health check on MAE decay."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG
from airflow.timetables.assets import AssetOrTimeSchedule
from airflow.timetables.trigger import CronTriggerTimetable

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

# The weekly cron is a floor; a health-check asset event starts a run sooner.
SCHEDULE = AssetOrTimeSchedule(
    timetable=CronTriggerTimetable(SETTINGS.retrain_schedule, timezone="UTC"),
    assets=[RETRAIN_REQUEST],
)

with DAG(
    dag_id="joseph_retrain",
    schedule=SCHEDULE,
    start_date=datetime(2026, 10, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["joseph", "serving"],
):
    t1 = PythonOperator(
        task_id="load_data",
        python_callable=tasks.load_data,
        op_kwargs={"trigger": "{{ dag_run.run_type }}"},
        execution_timeout=SETTINGS.timeout("load_data"),
    )
    t2 = PythonOperator(
        task_id="train_test_split",
        python_callable=tasks.train_test_split,
        op_kwargs={"expected_hash": t1.output},
        execution_timeout=SETTINGS.timeout("train_test_split"),
    )
    # No retries: a failed hour-long fit should reach a human, not silently cost another hour.
    t3 = PythonOperator(
        task_id="train_and_save_model",
        python_callable=tasks.train_and_save_model,
        op_kwargs={"expected_hash": t1.output},
        retries=0,
        execution_timeout=SETTINGS.timeout("train_and_save_model"),
    )
    t4 = PythonOperator(
        task_id="save_model_performance",
        python_callable=tasks.save_model_performance,
        op_kwargs={"expected_version": t3.output},
        execution_timeout=SETTINGS.timeout("save_model_performance"),
    )
    t5 = PythonOperator(
        task_id="compare_challenger_to_champion",
        python_callable=tasks.compare_challenger_to_champion,
        op_kwargs={"expected_version": t3.output},
        execution_timeout=SETTINGS.timeout("compare_challenger_to_champion"),
    )
    t6 = PythonOperator(
        task_id="promote_if_approved",
        python_callable=tasks.promote_if_approved,
        op_kwargs={"decision": t5.output},
        execution_timeout=SETTINGS.timeout("promote_if_approved"),
    )

    t1 >> t2 >> t3 >> t4 >> t5 >> t6
