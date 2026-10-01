"""Holdout slices for the scenario tests and the paired champion-challenger comparison.

Each scenario is a named row mask over the holdout. Thresholds are fixed numbers, not
data cuts, so the test months never decide their own groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.stats import wilcoxon

# A user is a dynamic overdraft case after this many zero crossings in six months.
OVERDRAFT_MIN_FLIPS = 2
# Gig income swings by at least its own average; a training quantile drifted (median cv 0.18 in 2022, 0.79 in the holdout).
GIG_MIN_CV = 1.0


@dataclass(frozen=True)
class Scenario:
    """A named slice of the holdout.

    Attributes
    ----------
    name : str
        Short label used in the report.
    rule : str
        The condition in words, for the report.
    mask : Callable[[pandas.DataFrame], pandas.Series]
        Returns True for the rows that belong to the slice.
    """

    name: str
    rule: str
    mask: Callable[[pd.DataFrame], pd.Series]


class PairedResult(BaseModel):
    """One champion-challenger comparison on one slice.

    Attributes
    ----------
    scenario, champion, challenger : str
        What was compared, and on which slice.
    n_rows, n_users : int
        Size of the slice.
    champion_mae, challenger_mae : float
        Mean absolute error of each model on the slice, in dollars.
    mae_change_pct : float
        Challenger MAE relative to champion MAE; negative means the challenger is better.
    ci_low, ci_high : float
        95% user-bootstrap interval of the MAE difference, challenger minus champion.
    p_value : float
        Wilcoxon signed-rank on the per-user mean difference.
    verdict : str
        ``challenger better``, ``champion better`` or ``no difference``.
    """

    scenario: str
    champion: str
    challenger: str
    n_rows: int
    n_users: int
    champion_mae: float
    challenger_mae: float
    mae_change_pct: float
    ci_low: float
    ci_high: float
    p_value: float
    verdict: str


# All rows first, then each case.
SCENARIOS = (
    Scenario("all_rows", "every holdout row", lambda frame: frame["actual"].notna()),
    Scenario(
        "dynamic_overdraft",
        f"sign_flips_6m >= {OVERDRAFT_MIN_FLIPS}",
        lambda frame: frame["sign_flips_6m"] >= OVERDRAFT_MIN_FLIPS,
    ),
    Scenario(
        "zero_income",
        "prev_1m_total_credited_usd == 0",
        lambda frame: frame["prev_1m_total_credited_usd"] == 0,
    ),
    Scenario(
        "gig_income",
        f"income_cv_6m >= {GIG_MIN_CV}",
        lambda frame: frame["income_cv_6m"] >= GIG_MIN_CV,
    ),
)


def paired_test(
    frame: pd.DataFrame,
    champion: str,
    challenger: str,
    id_column: str,
    scenario: str = "all_rows",
    n_boot: int = 2000,
    seed: int = 42,
) -> PairedResult:
    """Compare two models on the same rows, treating each user as one unit.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows with ``actual``, both prediction columns and ``id_column``.
    champion, challenger : str
        Prediction columns to compare.
    id_column : str
        User column; rows of one user are resampled together.
    scenario : str, optional
        Slice label for the result. Default ``all_rows``.
    n_boot : int, optional
        Bootstrap draws. Default 2000.
    seed : int, optional
        Bootstrap seed. Default 42.

    Returns
    -------
    PairedResult
        Sizes, both MAEs, the bootstrap interval, the p-value and the verdict.
    """
    champion_error = (frame["actual"] - frame[champion]).abs()
    challenger_error = (frame["actual"] - frame[challenger]).abs()
    # Positive difference means the challenger missed by more on that row.
    per_user = (challenger_error - champion_error).groupby(frame[id_column]).agg(["sum", "size", "mean"])

    # Resample whole users, since one user's months are not independent.
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_user), size=(n_boot, len(per_user)))
    sums = per_user["sum"].to_numpy()[draws].sum(axis=1)
    sizes = per_user["size"].to_numpy()[draws].sum(axis=1)
    ci_low, ci_high = np.quantile(sums / sizes, [0.025, 0.975])

    champion_mae = float(champion_error.mean())
    challenger_mae = float(challenger_error.mean())
    # The interval decides the verdict, so it agrees with the dollars in the table.
    verdict = (
        "challenger better" if ci_high < 0 else "champion better" if ci_low > 0 else "no difference"
    )
    return PairedResult(
        scenario=scenario,
        champion=champion,
        challenger=challenger,
        n_rows=len(frame),
        n_users=len(per_user),
        champion_mae=champion_mae,
        challenger_mae=challenger_mae,
        mae_change_pct=100.0 * (challenger_mae - champion_mae) / champion_mae,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(wilcoxon(per_user["mean"]).pvalue),
        verdict=verdict,
    )


def scenario_report(
    frame: pd.DataFrame,
    scenarios: tuple[Scenario, ...],
    pairs: list[tuple[str, str]],
    id_column: str,
) -> pd.DataFrame:
    """Run every pair on every scenario and lay the results out as one table.

    Parameters
    ----------
    frame : pandas.DataFrame
        Holdout rows with features, ``actual`` and every prediction column.
    scenarios : tuple of Scenario
        Slices to score, usually ``SCENARIOS``.
    pairs : list of tuple of str
        ``(champion, challenger)`` prediction columns.
    id_column : str
        User column.

    Returns
    -------
    pandas.DataFrame
        One row per scenario and pair, columns as in ``PairedResult``.
    """
    results = [
        paired_test(frame.loc[scenario.mask(frame)], champion, challenger, id_column, scenario.name)
        for scenario in scenarios
        for champion, challenger in pairs
    ]
    return pd.DataFrame([result.model_dump() for result in results])
