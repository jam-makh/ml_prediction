"""Monitoring: live MAE decay against the champion's holdout MAE, with a retrain cooldown."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from dag_utils.artifacts import SplitScores
from dag_utils.live_performance import live_scores


class MonitoringRules(BaseModel):
    """The ``serving.monitoring`` block of the config.

    Parameters
    ----------
    mae_decay : float
        Relative rise of live MAE over the holdout MAE that triggers a retrain.
    window_months : int
        Most recent resolved months the live MAE is measured on.
    min_rows : int
        Resolved rows needed in the window before the signal is trusted.
    cooldown_days : int
        Days after a retrain during which no new one is triggered.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mae_decay: float = Field(gt=0)
    window_months: int = Field(ge=1)
    min_rows: int = Field(ge=1)
    cooldown_days: int = Field(ge=0)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MonitoringRules:
        """Validate ``serving.monitoring``.

        Parameters
        ----------
        config : dict
            Parsed ``ml_config.yaml``.

        Returns
        -------
        MonitoringRules
            Validated thresholds.

        Raises
        ------
        pydantic.ValidationError
            If a value is missing, out of range or misspelled.
        """
        return cls.model_validate(config["serving"]["monitoring"])


class HealthCheck(BaseModel):
    """One monitoring reading and the retrain decision it led to.

    Parameters
    ----------
    live : SplitScores
        Live scores over the window.
    baseline_mae : float
        Champion's holdout MAE.
    decay : float
        ``live.mae / baseline_mae - 1``.
    threshold : float
        ``mae_decay`` from the config.
    retrain : bool
        Whether the retrain DAG should be triggered.
    reason : str
        One line for the task log.
    """

    live: SplitScores
    baseline_mae: float
    decay: float
    threshold: float
    retrain: bool
    reason: str


def recent_window(errors: pd.DataFrame, months: int) -> pd.DataFrame:
    """Keep the last ``months`` resolved months.

    Parameters
    ----------
    errors : pandas.DataFrame
        Resolved rows with a ``month`` column.
    months : int
        Window length.

    Returns
    -------
    pandas.DataFrame
        Rows in the window; empty input gives empty output.
    """
    if errors.empty:
        return errors
    month = pd.to_datetime(errors["month"])
    start = month.max() - pd.DateOffset(months=months - 1)
    return errors.loc[month >= start]


def in_cooldown(last_trained: datetime | None, now: datetime, days: int) -> bool:
    """Return whether the last retrain is too recent for another.

    Parameters
    ----------
    last_trained : datetime.datetime or None
        Latest ``trained_at``; None before any retrain.
    now : datetime.datetime
        Current time, timezone-aware like ``last_trained``.
    days : int
        Cooldown length.

    Returns
    -------
    bool
        True while inside the cooldown.
    """
    return last_trained is not None and now - last_trained < timedelta(days=days)


def check(
    errors: pd.DataFrame,
    baseline_mae: float,
    last_trained: datetime | None,
    now: datetime,
    rules: MonitoringRules,
) -> HealthCheck:
    """Measure live MAE decay and decide whether to trigger a retrain.

    Parameters
    ----------
    errors : pandas.DataFrame
        The champion's resolved xgboost rows.
    baseline_mae : float
        Champion's holdout MAE from its metadata.
    last_trained : datetime.datetime or None
        Latest retrain time, for the cooldown.
    now : datetime.datetime
        Current time.
    rules : MonitoringRules
        Thresholds.

    Returns
    -------
    HealthCheck
        The reading and the decision.
    """
    live = live_scores(recent_window(errors, rules.window_months))
    decay = live.mae / baseline_mae - 1 if live.n_rows and baseline_mae > 0 else float("nan")
    reading = f"live MAE {live.mae:,.0f} vs holdout {baseline_mae:,.0f} ({decay:+.1%})"

    if live.n_rows < rules.min_rows:
        retrain, reason = False, f"only {live.n_rows} resolved rows (< {rules.min_rows})"
    elif decay <= rules.mae_decay:
        retrain, reason = False, f"{reading}: within {rules.mae_decay:.0%}"
    elif in_cooldown(last_trained, now, rules.cooldown_days):
        retrain, reason = False, f"{reading}: decayed, but last retrain was {last_trained}"
    else:
        retrain, reason = True, f"{reading}: decayed past {rules.mae_decay:.0%}"
    return HealthCheck(
        live=live,
        baseline_mae=baseline_mae,
        decay=decay,
        threshold=rules.mae_decay,
        retrain=retrain,
        reason=reason,
    )
