"""Scoring a fitted model honestly, and saying where it fails.

``metrics.py`` computes numbers from two arrays. This module decides which rows
those arrays may contain, refuses the ones that would produce a flattering
score, and turns a fitted model plus a split into a report a person can argue
with.

Three things happen here that do not happen in ``metrics.py``.

**The out-of-sample check.** Every evaluation asserts that the first month being
scored is strictly after the last month the model was trained on. The splitter
already guarantees this and the trainer already checks it; it is checked a third
time here because this is the function that produces the number that gets
quoted, and a quoted number should not depend on two other modules having
behaved. It costs a comparison.

**The breakdowns.** A single RMSE cannot say where a model is worst, and "where
is it worst, and is that a data problem, a feature problem or a real limit" is
the question the findings have to answer. So the report carries error by month,
error by user, and the quantiles of the absolute error, because on a target this
skewed the mean and the median tell different stories and both are true.

**The bias check.** Mean error, signed. A model can have a respectable RMSE and
still be systematically low by a few thousand dollars on every row, which no
absolute-error metric will ever show. On a balance forecast that is the
difference between noisy and wrong.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.models_code.base_class import Model
from src.data.data import Dataset
from src.metrics import Scores, anchor_values, score, score_by_group

# How many of the worst months and users a report keeps. Enough to see a
# pattern, few enough to read in a terminal.
WORST_N = 10


class OutOfSampleError(ValueError):
    """Raised when a model is about to be scored on months it was trained on.

    A distinct type because this is the one failure that invalidates every
    number downstream of it, and a caller may reasonably want to catch it
    specifically rather than swallowing it with other value errors.
    """


class EvaluationReport(BaseModel):
    """One model's performance on one set of rows, with the diagnostics.

    Attributes
    ----------
    model_name : str
        Which model this describes.
    trained_through : str or None
        Last training month, ``YYYY-MM``, carried so the report states its own
        provenance rather than relying on the reader to remember it.
    scored_months : list of str
        The months scored, ascending.
    n_rows : int
        Rows scored.
    scores : Scores
        The headline metrics.
    bias : float
        Mean signed error, truth minus prediction, in dollars. Positive means
        the model predicts too low on average.
    absolute_error_quantiles : dict of str to float
        Absolute error at p50, p75, p90, p95 and p99. The shape of the tail,
        which a single average hides.
    worst_months : list of dict
        Per-month errors, worst first.
    worst_users : list of dict
        Per-user errors, worst first, truncated.
    share_of_error_top_users : float
        Share of total squared error contributed by the worst users listed.
        High values mean the headline metric is describing a handful of rows.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_name: str
    trained_through: str | None
    scored_months: list[str]
    n_rows: int
    scores: Scores
    bias: float
    absolute_error_quantiles: dict[str, float] = Field(default_factory=dict)
    worst_months: list[dict[str, Any]] = Field(default_factory=list)
    worst_users: list[dict[str, Any]] = Field(default_factory=list)
    share_of_error_top_users: float = 0.0

    def describe(self) -> str:
        """Return a short readable block for the run log.

        Returns
        -------
        str
            Two lines: the headline metrics, then the diagnostics that decide
            whether those metrics mean anything.
        """
        span = (
            f"{self.scored_months[0]}..{self.scored_months[-1]}"
            if self.scored_months
            else "no months"
        )
        worst = self.worst_months[0]["month"] if self.worst_months else "n/a"
        return (
            f"{self.model_name} on {span} ({self.n_rows:,} rows)\n"
            f"  {self.scores.describe()}\n"
            f"  bias {self.bias:+,.0f} | p90 abs error "
            f"{self.absolute_error_quantiles.get('p90', float('nan')):,.0f} | "
            f"worst month {worst} | top {WORST_N} users hold "
            f"{100 * self.share_of_error_top_users:.0f}% of squared error"
        )


def check_out_of_sample(model: Model, test: Dataset) -> None:
    """Assert the model is not about to be scored on months it was trained on.

    Parameters
    ----------
    model : Model
        A fitted model.
    test : Dataset
        The rows about to be scored.

    Returns
    -------
    None

    Raises
    ------
    OutOfSampleError
        If any test month is at or before the last training month.
    """
    trained_through = model.trained_through
    if trained_through is None:
        # An unfitted model has no training months to overlap with, and predict()
        # will refuse it anyway. Nothing to check.
        return

    first_scored = test.months.min()
    if trained_through >= first_scored:
        raise OutOfSampleError(
            f"{model.name} was trained through {trained_through:%Y-%m} and is "
            f"being scored from {first_scored:%Y-%m}. Any number produced from "
            f"this is in-sample and means nothing."
        )


def error_quantiles(errors: npt.NDArray[np.float64]) -> dict[str, float]:
    """Return the absolute error at the quantiles worth reporting.

    Parameters
    ----------
    errors : numpy.ndarray of float
        Signed errors.

    Returns
    -------
    dict of str to float
        Absolute error at p50, p75, p90, p95 and p99.
    """
    absolute = np.abs(errors)
    wanted = (50, 75, 90, 95, 99)
    values = np.percentile(absolute, wanted)
    return {f"p{point}": float(value) for point, value in zip(wanted, values)}


def evaluate_predictions(
    model_name: str,
    predictions: npt.NDArray[np.float64],
    test: Dataset,
    anchor_column: str,
    reference_pred: npt.NDArray[np.float64] | None = None,
    trained_through: str | None = None,
    mape_floor: float = 1_000.0,
) -> EvaluationReport:
    """Build a full report from predictions that have already been made.

    Split out from ``evaluate_model`` so the same reporting can be applied to
    predictions loaded from a file, or produced by something that is not a
    ``Model`` at all.

    Parameters
    ----------
    model_name : str
        Label for the report.
    predictions : numpy.ndarray of float
        One prediction per row of ``test``.
    test : Dataset
        The rows being scored.
    anchor_column : str
        Column holding last month's balance, for the change framing.
    reference_pred : numpy.ndarray of float, optional
        Reference model's predictions, for a skill score.
    trained_through : str, optional
        Last training month, recorded on the report.
    mape_floor : float, optional
        Rows below this absolute true value are left out of the percentage
        error. Default 1000.

    Returns
    -------
    EvaluationReport
        Metrics and diagnostics.
    """
    truth = test.target
    anchor = anchor_values(test, anchor_column)
    scores = score(
        truth,
        predictions,
        anchor=anchor,
        reference_pred=reference_pred,
        mape_floor=mape_floor,
    )

    errors = truth.to_numpy(dtype="float64") - predictions

    by_month = score_by_group(truth, predictions, test.times, label="month")
    by_user = score_by_group(truth, predictions, test.entities, label="user")

    # How concentrated the error is. When a handful of users carry most of the
    # squared error, the headline RMSE is a statement about those users and not
    # about the model, and the report should say so rather than leave it to be
    # discovered.
    squared = np.square(errors)
    total_squared = float(squared.sum())
    top_users = by_user.head(WORST_N).index
    top_mask = test.entities.isin(top_users).to_numpy(dtype=bool)
    share = float(squared[top_mask].sum() / total_squared) if total_squared > 0 else 0.0

    months = [f"{month:%Y-%m}" for month in test.months]

    return EvaluationReport(
        model_name=model_name,
        trained_through=trained_through,
        scored_months=months,
        n_rows=test.n_rows,
        scores=scores,
        bias=float(errors.mean()),
        absolute_error_quantiles=error_quantiles(errors),
        worst_months=[
            {"month": f"{month:%Y-%m}", **row}
            for month, row in by_month.to_dict(orient="index").items()
        ],
        worst_users=[
            {"user": str(user), **row}
            for user, row in by_user.head(WORST_N).to_dict(orient="index").items()
        ],
        share_of_error_top_users=share,
    )


def evaluate_model(
    model: Model,
    test: Dataset,
    anchor_column: str,
    reference_pred: npt.NDArray[np.float64] | None = None,
    mape_floor: float = 1_000.0,
) -> tuple[EvaluationReport, npt.NDArray[np.float64]]:
    """Score a fitted model on held-out rows.

    Parameters
    ----------
    model : Model
        A fitted model.
    test : Dataset
        Rows to score on. Must be entirely after the model's training months.
    anchor_column : str
        Column holding last month's balance, for the change framing.
    reference_pred : numpy.ndarray of float, optional
        Reference model's predictions over the same rows, for a skill score.
    mape_floor : float, optional
        Rows below this absolute true value are left out of the percentage
        error. Default 1000.

    Returns
    -------
    tuple of (EvaluationReport, numpy.ndarray)
        The report, and the predictions it was built from, so a caller can
        reuse them as a reference for another model without predicting twice.

    Raises
    ------
    OutOfSampleError
        If the model was trained on any of the months being scored.
    """
    # First, before anything is computed. A number produced from overlapping
    # months is worse than no number, because it looks like a result.
    check_out_of_sample(model, test)

    predictions = model.predict(test)
    through = model.trained_through
    report = evaluate_predictions(
        model_name=model.name,
        predictions=predictions,
        test=test,
        anchor_column=anchor_column,
        reference_pred=reference_pred,
        trained_through=f"{through:%Y-%m}" if through is not None else None,
        mape_floor=mape_floor,
    )
    return report, predictions


def evaluate_models(
    models: dict[str, Model],
    test: Dataset,
    anchor_column: str,
    reference_model: str | None = None,
    mape_floor: float = 1_000.0,
    predictions: dict[str, npt.NDArray[np.float64]] | None = None,
) -> dict[str, EvaluationReport]:
    """Score several fitted models on the same held-out rows.

    Parameters
    ----------
    models : dict of str to Model
        Fitted models, keyed by name.
    test : Dataset
        Rows to score on.
    anchor_column : str
        Column holding last month's balance.
    reference_model : str, optional
        Name of the model every skill score is measured against. Predicted
        first so its predictions can be reused, and never scored against
        itself.
    mape_floor : float, optional
        Rows below this absolute true value are left out of the percentage
        error. Default 1000.
    predictions : dict of str to numpy.ndarray, optional
        Predictions already computed over exactly these rows, keyed by model
        name. Supplied by ``train.py``, which fits and predicts the holdout
        once and passes the result here rather than having every model predict
        a second time. A model missing from the dict is predicted normally, so
        a partial dict is safe.

    Returns
    -------
    dict of str to EvaluationReport
        One report per model.
    """
    supplied = predictions or {}

    def predict(name: str, model: Model) -> npt.NDArray[np.float64]:
        """Return this model's predictions, reusing a supplied array if given."""
        if name in supplied:
            # Still checked: reusing an array must not skip the assertion that
            # the model never saw these months.
            check_out_of_sample(model, test)
            return supplied[name]
        return model.predict(test)

    reference_pred: npt.NDArray[np.float64] | None = None
    if reference_model is not None and reference_model in models:
        check_out_of_sample(models[reference_model], test)
        reference_pred = predict(reference_model, models[reference_model])

    reports: dict[str, EvaluationReport] = {}
    for name, model in models.items():
        # The reference is not measured against itself, which would report a
        # skill of exactly zero and read as though it had been compared.
        against = reference_pred if name != reference_model else None

        if name in supplied:
            check_out_of_sample(model, test)
            through = model.trained_through
            reports[name] = evaluate_predictions(
                model_name=model.name,
                predictions=supplied[name],
                test=test,
                anchor_column=anchor_column,
                reference_pred=against,
                trained_through=f"{through:%Y-%m}" if through is not None else None,
                mape_floor=mape_floor,
            )
            continue

        report, _ = evaluate_model(
            model, test, anchor_column, reference_pred=against, mape_floor=mape_floor
        )
        reports[name] = report
    return reports


def report_table(reports: dict[str, EvaluationReport], sort_by: str = "median_ae") -> pd.DataFrame:
    """Lay several reports out as one table.

    Parameters
    ----------
    reports : dict of str to EvaluationReport
        Reports to compare.
    sort_by : str, optional
        Column to sort ascending by. Default ``median_ae``, because on a target
        this skewed the median absolute error is the statistic that describes
        the typical user rather than the largest handful. Pass ``rmse`` for the
        conventional ordering.

    Returns
    -------
    pandas.DataFrame
        One row per model, best first.
    """
    rows = [
        {
            "model": name,
            "rmse": report.scores.rmse,
            "mae": report.scores.mae,
            "median_ae": report.scores.median_ae,
            "p90_ae": report.absolute_error_quantiles.get("p90", float("nan")),
            "bias": report.bias,
            "r2_level": report.scores.r2_level,
            "r2_change": report.scores.r2_change,
            "skill": report.scores.skill,
        }
        for name, report in reports.items()
    ]
    table = pd.DataFrame(rows).set_index("model")
    column = sort_by if sort_by in table.columns else "median_ae"
    return table.sort_values(column)


def worst_months_table(report: EvaluationReport) -> pd.DataFrame:
    """Return a report's per-month errors as a frame, worst first.

    Parameters
    ----------
    report : EvaluationReport
        The report to unpack.

    Returns
    -------
    pandas.DataFrame
        Indexed by month.
    """
    return pd.DataFrame(report.worst_months).set_index("month")


def worst_users_table(report: EvaluationReport) -> pd.DataFrame:
    """Return a report's per-user errors as a frame, worst first.

    Parameters
    ----------
    report : EvaluationReport
        The report to unpack.

    Returns
    -------
    pandas.DataFrame
        Indexed by user id.
    """
    return pd.DataFrame(report.worst_users).set_index("user")
