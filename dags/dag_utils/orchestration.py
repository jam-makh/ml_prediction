"""DAG-level settings: the retrain schedule and every task's timeout, with two env-var overrides."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

PREDICT_TIMEOUT_ENV = "PREDICT_TIMEOUT_MINUTES"
RETRAIN_SCHEDULE_ENV = "RETRAIN_SCHEDULE"


class DagSettings(BaseModel):
    """The ``serving.orchestration`` block of the config.

    Parameters
    ----------
    retrain_schedule : str
        Cron expression for the retrain DAG's weekly floor.
    timeouts : dict of str to int
        Minutes each task may run, keyed by task id.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    retrain_schedule: str = Field(min_length=1)
    timeouts: dict[str, PositiveInt]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DagSettings:
        """Validate ``serving.orchestration`` and apply the env-var overrides.

        Parameters
        ----------
        config : dict
            Parsed ``ml_config.yaml``.

        Returns
        -------
        DagSettings
            Validated settings.

        Raises
        ------
        pydantic.ValidationError
            If a value is missing, misspelled or a timeout is below one minute.
        ValueError
            If ``PREDICT_TIMEOUT_MINUTES`` is not an integer.
        """
        block = dict(config["serving"]["orchestration"])
        timeouts = dict(block.get("timeouts") or {})
        # Empty strings come from compose's `${VAR:-}` when the variable is unset in .env.
        if minutes := os.environ.get(PREDICT_TIMEOUT_ENV):
            timeouts["run_predictions"] = int(minutes)
        if schedule := os.environ.get(RETRAIN_SCHEDULE_ENV):
            block["retrain_schedule"] = schedule
        return cls.model_validate({**block, "timeouts": timeouts})

    def timeout(self, task_id: str) -> timedelta:
        """Return a task's ``execution_timeout``.

        Parameters
        ----------
        task_id : str
            Task id as listed under ``timeouts``.

        Returns
        -------
        datetime.timedelta
            The task's time limit.

        Raises
        ------
        KeyError
            If the task has no timeout configured.
        """
        if task_id not in self.timeouts:
            raise KeyError(f"no timeout configured for task {task_id!r}")
        return timedelta(minutes=self.timeouts[task_id])
