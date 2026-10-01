"""Paired champion-challenger report over the holdout and its scenario slices.

The report is the result, so a challenger that loses does not fail the test.
Only alignment of the two prediction files is asserted.
"""

from __future__ import annotations

import pandas as pd

from src.config.config import PROJECT_ROOT
from src.scenarios import SCENARIOS, scenario_report

# (champion, challenger): each xgboost against persistence, then top-15 against full.
PAIRS = [
    ("persistence", "xgboost_top15"),
    ("persistence", "xgboost_full"),
    ("xgboost_full", "xgboost_top15"),
]
REPORT_PATH = PROJECT_ROOT / "results" / "scenario_tests.csv"


def test_prediction_files_align(holdout: pd.DataFrame) -> None:
    """Both runs must score the same holdout rows with the same truth.

    Parameters
    ----------
    holdout : pandas.DataFrame
        Holdout features joined to both prediction files.
    """
    predictions = holdout[["actual", "actual_full", "persistence", "xgboost_top15", "xgboost_full"]]
    # A missing value here means one file has a row the other lacks.
    assert predictions.notna().all().all(), "the two prediction files do not cover the same rows"
    assert (holdout["actual"] == holdout["actual_full"]).all(), "the two runs scored different truths"


def test_scenario_report(holdout: pd.DataFrame) -> None:
    """Write the paired comparison for every scenario and pair to ``results/scenario_tests.csv``.

    Parameters
    ----------
    holdout : pandas.DataFrame
        Holdout features joined to both prediction files.
    """
    report = scenario_report(holdout, SCENARIOS, PAIRS, id_column="user_id")
    report.to_csv(REPORT_PATH, index=False)

    # Shown with `pytest -s`; the CSV keeps full precision.
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"\n{report.round(3).to_string(index=False)}\n  saved {REPORT_PATH}")
    assert (report["n_rows"] > 0).all(), "a scenario selected no holdout rows"
