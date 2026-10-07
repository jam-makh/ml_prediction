"""Route loguru into Airflow's task logger, so every task's lines show in the UI at their level."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from loguru import logger

# Message exists only in loguru's type stubs, so it is imported for mypy alone.
if TYPE_CHECKING:
    from loguru import Message

AIRFLOW_LOGGER = "airflow.task"


def _forward(message: Message) -> None:
    """Send one loguru record to Airflow's task logger at its own level.

    Parameters
    ----------
    message : loguru.Message
        Formatted message carrying the record.

    Returns
    -------
    None
    """
    # SUCCESS and TRACE have no stdlib name, so the numeric level is passed instead.
    logging.getLogger(AIRFLOW_LOGGER).log(message.record["level"].no, message.rstrip())


def route_to_airflow() -> None:
    """Replace loguru's sinks with one that forwards to Airflow's task logger.

    Returns
    -------
    None
    """
    logger.remove()
    # Loguru appends the traceback to this format by itself when a record carries one.
    logger.add(_forward, level="DEBUG", format="{name}:{line} | {message}")
