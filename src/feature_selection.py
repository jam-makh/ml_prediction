"""Feature selection and the reports that explain what the models use.

RFECV runs on the booster after Optuna, on the same expanding month folds,
scored by WAPE on the change target -- per fold that equals Optuna's objective,
model MAE over persistence MAE. The holdout is never passed in. SHAP and the
linear coefficient lists only explain the fitted models; they select nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer

from src.data.data import Dataset
from src.metrics import wape
from src.models_code.base_class import Model
from src.models_code.ridge_reg import RidgeRegression
from src.models_code.xgboost_model import XGBoostModel
from src.window import Split

# A fold as scikit-learn takes it: training row positions, validation row positions.
FoldIndices = tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]

# Chart colours: one series, so one hue; text stays in ink, never the bar colour.
BAR_COLOUR = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"


class SelectionResult(BaseModel):
    """What RFECV kept and dropped for one model, and the CV curve behind it.

    Attributes
    ----------
    model : str
        Name of the model the selection was run for.
    n_features_in : int
        Number of candidate features.
    selected : tuple of str
        Features kept, in their original column order.
    dropped_in_order : tuple of str
        Features removed, the first removed (weakest) first.
    ranking : dict of str to int
        RFECV rank per feature; 1 is kept, higher was removed earlier.
    cv_wape_by_n_features : dict of int to float
        Mean fold WAPE (%) on the change target for each feature count tried.
    best_cv_wape : float
        The lowest value in ``cv_wape_by_n_features``.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    n_features_in: int
    selected: tuple[str, ...]
    dropped_in_order: tuple[str, ...]
    ranking: dict[str, int]
    cv_wape_by_n_features: dict[int, float]
    best_cv_wape: float

    def report_lines(self) -> list[str]:
        """Return the kept and dropped lists as printable lines.

        Returns
        -------
        list of str
            One summary line, the kept features, then the dropped ones in removal order.
        """
        lines = [
            f"  Kept {len(self.selected)} of {self.n_features_in} features "
            f"(best CV WAPE {self.best_cv_wape:.2f}% on the change target)",
            "",
            f"  Kept    : {', '.join(self.selected)}",
        ]
        if self.dropped_in_order:
            lines.append("  Dropped (first removed -> last removed):")
            lines.extend(
                f"    {index:>3}  {name}" for index, name in enumerate(self.dropped_in_order, start=1)
            )
        return lines


def remap_folds(folds: list[Split], usable: npt.NDArray[np.bool_]) -> list[FoldIndices]:
    """Translate fold positions to the rows left after dropping unusable ones.

    Parameters
    ----------
    folds : list of Split
        Expanding month folds, positions into the full training region.
    usable : numpy.ndarray of bool
        True for rows kept, one per row of the training region.

    Returns
    -------
    list of tuple of (numpy.ndarray, numpy.ndarray)
        Training and validation positions into the kept rows only.
    """
    # Position of each kept row once the others are gone; dropped rows are never looked up.
    new_position = np.cumsum(usable) - 1
    return [
        (
            new_position[fold.train_positions[usable[fold.train_positions]]],
            new_position[fold.test_positions[usable[fold.test_positions]]],
        )
        for fold in folds
    ]


def rfecv_select(
    model: XGBoostModel,
    train: Dataset,
    folds: list[Split],
    step: int = 1,
    min_features: int = 1,
) -> SelectionResult:
    """Run RFECV for the booster on the training-region folds.

    Parameters
    ----------
    model : XGBoostModel
        Unfitted booster carrying the tuned parameters.
    train : Dataset
        The training region, already narrowed to the model family's columns.
    folds : list of Split
        Expanding month folds inside ``train``.
    step : int, optional
        Features removed per elimination round. Default 1.
    min_features : int, optional
        Fewest features RFECV may keep. Default 1.

    Returns
    -------
    SelectionResult
        Kept and dropped features and the CV curve.

    Raises
    ------
    ValueError
        If no row has a change target to learn from.
    """
    # The change target the booster learns; a user's first month has none and is left out.
    target = model.training_target(train)
    usable = target.notna().to_numpy(dtype=bool)
    if not usable.any():
        raise ValueError(f"{model.name}: no rows with a usable change target")
    features = train.features.loc[usable]
    values = target.to_numpy(dtype="float64")[usable]

    selector = RFECV(
        estimator=model.make_estimator(values),
        step=step,
        min_features_to_select=min_features,
        # Time-ordered folds from window.py; a shuffled KFold would train on the future.
        cv=remap_folds(folds, usable),
        # WAPE on the change equals model MAE / persistence MAE, Optuna's per-fold objective.
        scoring=make_scorer(wape, greater_is_better=False),
        # The booster already uses every core, so parallel folds would oversubscribe.
        n_jobs=1,
    )
    selector.fit(features, values)

    columns = [str(name) for name in features.columns]
    ranking = {name: int(rank) for name, rank in zip(columns, selector.ranking_)}
    # The scorer is negated so higher is better; flip it back to a WAPE.
    curve = {
        int(n): round(float(-score), 4)
        for n, score in zip(
            selector.cv_results_["n_features"], selector.cv_results_["mean_test_score"]
        )
    }
    return SelectionResult(
        model=model.name,
        n_features_in=len(columns),
        selected=tuple(name for name in columns if ranking[name] == 1),
        # Highest rank was removed first, so descending rank reads weakest first.
        dropped_in_order=tuple(
            sorted((name for name in columns if ranking[name] > 1), key=lambda n: -ranking[n])
        ),
        ranking=ranking,
        cv_wape_by_n_features=curve,
        best_cv_wape=min(curve.values()),
    )


def save_selection(result: SelectionResult, output_dir: Path) -> Path:
    """Write a selection result to ``<model>_rfecv.json``.

    Parameters
    ----------
    result : SelectionResult
        The result to save.
    output_dir : pathlib.Path
        Directory the models are saved to.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    path = output_dir / f"{result.model}_rfecv.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def shap_importance(model: Model, dataset: Dataset) -> pd.DataFrame | None:
    """Rank a model's features by mean absolute SHAP value.

    Parameters
    ----------
    model : Model
        A fitted model.
    dataset : Dataset
        Rows to explain, normally the training region.

    Returns
    -------
    pandas.DataFrame or None
        ``mean_abs_shap`` (size of effect) and ``mean_shap`` (average push),
        indexed by feature, largest first; None for models without SHAP values.
    """
    values = model.shap_values(dataset)
    if values is None:
        return None
    table = pd.DataFrame({"mean_abs_shap": values.abs().mean(), "mean_shap": values.mean()})
    # Features the model never uses score exactly zero, so they are left off.
    table = table[table["mean_abs_shap"] > 0]
    return table.sort_values("mean_abs_shap", ascending=False)


def save_shap(
    table: pd.DataFrame, model_name: str, output_dir: Path, top_n: int = 15
) -> tuple[Path, Path]:
    """Write the full SHAP table to csv and the top features as a bar chart.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of ``shap_importance``.
    model_name : str
        Used in the file names and chart title.
    output_dir : pathlib.Path
        Directory the models are saved to.
    top_n : int, optional
        Bars on the chart. Default 15.

    Returns
    -------
    tuple of (pathlib.Path, pathlib.Path)
        The csv and png paths.
    """
    csv_path = output_dir / f"{model_name}_shap.csv"
    png_path = output_dir / f"{model_name}_shap.png"
    table.to_csv(csv_path, index_label="feature")

    # Reversed so the largest bar sits at the top.
    top = table.head(top_n).iloc[::-1]
    figure = Figure(figsize=(9, 0.42 * len(top) + 1.4), facecolor=SURFACE)
    axes = figure.subplots()
    axes.set_facecolor(SURFACE)
    bars = axes.barh(top.index, top["mean_abs_shap"], height=0.6, color=BAR_COLOUR)
    # Every bar carries its value, at the tip.
    axes.bar_label(
        bars, labels=[f"${value:,.0f}" for value in top["mean_abs_shap"]],
        padding=4, fontsize=9, color=INK,
    )
    axes.set_title(
        f"{model_name}: top {len(top)} features by mean |SHAP|, training rows",
        loc="left", fontsize=11, color=INK,
    )
    axes.set_xlabel("Mean |SHAP|, dollars of monthly change", color=INK_SECONDARY)
    axes.tick_params(colors=INK_SECONDARY, labelsize=9)
    axes.grid(False)
    # Room past the longest bar for its label.
    axes.margins(x=0.15)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    figure.tight_layout()
    figure.savefig(png_path, dpi=150)
    return csv_path, png_path


def coefficient_report_lines(model: RidgeRegression, weakest_n: int = 10) -> list[str]:
    """List the features a linear model zeroed, or, if none, the ones it uses least.

    Parameters
    ----------
    model : RidgeRegression
        A fitted ridge or elastic net.
    weakest_n : int, optional
        How many of the smallest coefficients to list when none are zero. Default 10.

    Returns
    -------
    list of str
        Printable lines; empty before fitting.
    """
    importance = model.feature_importance()
    if importance is None:
        return []
    zeroed = importance[importance == 0]
    if not zeroed.empty:
        return [f"  Zeroed by L1 ({len(zeroed)} of {len(importance)}): {', '.join(zeroed.index)}"]
    # Ridge never reaches exactly zero, so the smallest weights stand in, weakest first.
    weakest = importance.tail(weakest_n).iloc[::-1]
    lines = [f"  Weakest by |coef| (all {len(importance)} still used; dollars per standard deviation):"]
    lines.extend(f"    {name:<45} {value:>10,.1f}" for name, value in weakest.items())
    return lines
