"""The promotion rule: a challenger replaces the champion only if it is clearly better."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dag_utils.artifacts import SplitScores

Outcome = Literal["promote", "reject", "insufficient_data"]
Basis = Literal["first_run", "live", "holdout"]


class PromotionRules(BaseModel):
    """The ``serving.promotion`` block of the config.

    Parameters
    ----------
    mae_min_gain : float
        Smallest relative MAE drop that promotes, e.g. 0.02 for 2%.
    rmse_max_loss, wape_max_loss : float
        Largest relative rise in RMSE and WAPE tolerated alongside it.
    min_live_rows : int
        Rows the champion must be scored on: live rows first, unseen holdout rows otherwise.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mae_min_gain: float = Field(ge=0)
    rmse_max_loss: float = Field(ge=0)
    wape_max_loss: float = Field(ge=0)
    min_live_rows: int = Field(ge=1)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PromotionRules:
        """Validate ``serving.promotion``.

        Parameters
        ----------
        config : dict
            Parsed ``ml_config.yaml``.

        Returns
        -------
        PromotionRules
            Validated thresholds.

        Raises
        ------
        pydantic.ValidationError
            If a value is missing, negative or misspelled.
        """
        return cls.model_validate(config["serving"]["promotion"])


class Decision(BaseModel):
    """The outcome of one comparison and why.

    Parameters
    ----------
    outcome : {"promote", "reject", "insufficient_data"}
        What to do with the challenger.
    basis : {"first_run", "live", "holdout"}
        Which champion error the challenger was compared with.
    reason : str
        One line for the task log.
    """

    outcome: Outcome
    basis: Basis
    reason: str


def relative_change(new: float, old: float) -> float:
    """Return ``(new - old) / old``; negative means the error went down.

    Parameters
    ----------
    new, old : float
        Challenger and champion values of one metric.

    Returns
    -------
    float
        Relative change, ``inf`` when the champion's error is zero.
    """
    return (new - old) / old if old > 0 else float("inf")


def decide(
    challenger: SplitScores, champion: SplitScores, basis: Basis, rules: PromotionRules
) -> Decision:
    """Apply the promotion rule to two sets of scores.

    Parameters
    ----------
    challenger : SplitScores
        Challenger's holdout scores on xgboost rows.
    champion : SplitScores
        Champion's live or holdout scores on xgboost rows.
    basis : {"live", "holdout"}
        Where the champion's scores came from.
    rules : PromotionRules
        Thresholds.

    Returns
    -------
    Decision
        ``promote`` only when MAE falls enough and neither guard rail is broken.
    """
    if champion.n_rows < rules.min_live_rows:
        return Decision(
            outcome="insufficient_data",
            basis=basis,
            reason=f"champion scored on {champion.n_rows} rows (< {rules.min_live_rows})",
        )
    mae = relative_change(challenger.mae, champion.mae)
    rmse = relative_change(challenger.rmse, champion.rmse)
    wape = relative_change(challenger.wape, champion.wape)
    summary = f"MAE {mae:+.1%}, RMSE {rmse:+.1%}, WAPE {wape:+.1%} vs champion ({basis})"

    if -mae < rules.mae_min_gain:
        return Decision(outcome="reject", basis=basis, reason=f"{summary}: MAE gain too small")
    if rmse > rules.rmse_max_loss or wape > rules.wape_max_loss:
        return Decision(outcome="reject", basis=basis, reason=f"{summary}: guard rail broken")
    return Decision(outcome="promote", basis=basis, reason=summary)


def first_run() -> Decision:
    """Return the decision used when no champion exists yet.

    Returns
    -------
    Decision
        Always ``promote``.
    """
    return Decision(outcome="promote", basis="first_run", reason="no champion yet")
