"""Which columns each saved model leans on, as one diagnostic figure.

Separate from training, testing and the metrics on purpose. This never fits
anything and never reads the held-out months: it loads the models ``train.py``
saved and asks each one what it relied on to fit the training region.

Every XGBoost model gets a row of three panels, because the three measures
answer different questions and the disagreements between them are the point:

``total_gain``
    Loss reduction summed over every split on the column. How much of the fit
    the column carries. The main ranking.
``weight``
    How many splits used the column. A continuous column with many distinct
    values offers many places to split, so a high count with a low total gain
    is a column taking splits without earning them.
``cover``
    Average number of rows each split on the column reaches. A high-gain column
    with a low cover is a rule fitted to a handful of accounts -- on this panel,
    usually the very large ones.

Ridge models get one panel: the absolute standardised coefficient ``|w|``.

``total_gain`` and ``|w|`` are shown as a share of the model's total, so both
sum to 100% and sit on one shared axis. They still measure different things --
a split's loss reduction against a linear weight -- so the share says "this
column carries X% of this model", not that the two models agree.

All of these describe the fit on the training rows, not predictive value on
unseen months.

Run it with::

    python -m src.importance
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

from src.log import setup_logging
from src.config.config import load_config, resolve_output_dir
from src.models_code.base_class import Model
from src.models_code.ridge_reg import RidgeRegression
from src.models_code.xgboost_model import XGBoostModel
from src.test import load_models

# Bars per panel.
TOP_N = 10

# XGBoost measures, in panel order: (importance_type, axis label, as a share).
XGBOOST_MEASURES: tuple[tuple[str, str, bool], ...] = (
    ("total_gain", "share of total gain (%)", True),
    ("weight", "number of splits", False),
    ("cover", "average rows per split", False),
)

RIDGE_MEASURE = "abs_coef"
RIDGE_LABEL = "share of total |w| (%)"

BAR_COLOUR = "#4C72B0"


def to_share(values: pd.Series) -> pd.Series:
    """Return ``values`` as a percentage of their total.

    Parameters
    ----------
    values : pandas.Series
        Non-negative importance per feature.

    Returns
    -------
    pandas.Series
        Same index, summing to 100. All zeros when the total is zero.
    """
    total = float(values.sum())
    if total <= 0.0:
        return values * 0.0
    return 100.0 * values / total


def collect_importance(models: dict[str, Model]) -> pd.DataFrame:
    """Ask every model for its importance, in long form.

    Parameters
    ----------
    models : dict of str to Model
        Fitted models, keyed by name.

    Returns
    -------
    pandas.DataFrame
        One row per (model, measure, feature) with ``value`` (the raw number)
        and ``share`` (percent of the model's total, for the share measures
        only; NaN otherwise). Models with no features, such as the 3-month
        average, are skipped with a note.
    """
    frames: list[pd.DataFrame] = []
    for name, model in models.items():
        if isinstance(model, XGBoostModel):
            for measure, _, as_share in XGBOOST_MEASURES:
                values = model.feature_importance(importance_type=measure)
                if values is not None:
                    frames.append(_long(name, measure, values, as_share))
        elif isinstance(model, RidgeRegression):
            values = model.feature_importance()
            if values is not None:
                frames.append(_long(name, RIDGE_MEASURE, values, True))
        else:
            logger.info(f"  {name}: no feature importance (not a feature model), skipped")

    if not frames:
        raise ValueError("None of the configured models reports feature importance")
    return pd.concat(frames, ignore_index=True)


def _long(name: str, measure: str, values: pd.Series, as_share: bool) -> pd.DataFrame:
    """Lay one model's importance out as long-form rows.

    Parameters
    ----------
    name : str
        Model name.
    measure : str
        Which importance this is.
    values : pandas.Series
        Importance per feature.
    as_share : bool
        Whether to fill the ``share`` column.

    Returns
    -------
    pandas.DataFrame
        Columns ``model``, ``measure``, ``feature``, ``value``, ``share``.
    """
    share = to_share(values) if as_share else values * float("nan")
    return pd.DataFrame(
        {
            "model": name,
            "measure": measure,
            "feature": values.index.astype(str),
            "value": values.to_numpy(dtype="float64"),
            "share": share.to_numpy(dtype="float64"),
        }
    )


def top_features(table: pd.DataFrame, model: str, measure: str) -> pd.Series:
    """Return one panel's top features, the value it plots, largest first.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of ``collect_importance``.
    model : str
        Model name.
    measure : str
        Measure name.

    Returns
    -------
    pandas.Series
        Up to ``TOP_N`` values indexed by feature: the share where there is
        one, the raw value otherwise.
    """
    rows = table[(table["model"] == model) & (table["measure"] == measure)]
    column = "share" if rows["share"].notna().any() else "value"
    return rows.set_index("feature")[column].sort_values(ascending=False).head(TOP_N)


def label_bars(ax: Axes, bars: BarContainer, labels: list[str]) -> None:
    """Write each horizontal bar's value at its end.

    Outside the end when the label fits inside the axis, otherwise just inside
    the end in white, so the longest bar never pushes its label off the panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel.
    bars : matplotlib.container.BarContainer
        Bars returned by ``barh``.
    labels : list of str
        One label per bar.

    Returns
    -------
    None
    """
    low, high = ax.get_xlim()
    span = high - low
    room = 0.16 * span
    offset = 0.012 * span
    for bar, text in zip(bars, labels):
        value = bar.get_width()
        across = bar.get_y() + bar.get_height() / 2
        if value + room <= high:
            ax.text(value + offset, across, text, va="center", ha="left", fontsize=8)
        else:
            ax.text(value - offset, across, text, va="center", ha="right",
                    fontsize=8, color="white")


def draw_panel(ax: Axes, values: pd.Series, title: str, xlabel: str,
               as_share: bool, xmax: float | None = None) -> None:
    """Draw one top-N panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Where to draw.
    values : pandas.Series
        Values indexed by feature, largest first.
    title : str
        Panel title.
    xlabel : str
        Axis label.
    as_share : bool
        Label bars as percentages rather than plain numbers.
    xmax : float, optional
        Fixed right edge, so share panels line up on one scale.

    Returns
    -------
    None
    """
    # Reversed so the largest bar sits at the top.
    ordered = values.iloc[::-1]
    bars = ax.barh(ordered.index, ordered.to_numpy(), color=BAR_COLOUR)
    ax.set_xlim(0.0, (xmax if xmax is not None else float(values.max())) * 1.2)
    fmt = "{:.1f}%" if as_share else "{:,.0f}"
    label_bars(ax, bars, [fmt.format(value) for value in ordered.to_numpy()])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(axis="y", length=0, labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    sns.despine(ax=ax, left=True)


def plot_importance(table: pd.DataFrame) -> Figure:
    """Build the diagnostic figure.

    One row per XGBoost model (total gain, weight, cover), then one row of
    Ridge panels. Every share panel uses the same x-axis.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of ``collect_importance``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, not yet saved.
    """
    sns.set_theme(style="white", rc={"axes.grid": False})

    xgboost_models = [m for m in table["model"].unique()
                      if (table.loc[table["model"] == m, "measure"] == "total_gain").any()]
    ridge_models = [m for m in table["model"].unique()
                    if (table.loc[table["model"] == m, "measure"] == RIDGE_MEASURE).any()]

    # One axis for every share panel, so 20% looks the same everywhere.
    share_max = float(
        max(
            [top_features(table, m, "total_gain").max() for m in xgboost_models]
            + [top_features(table, m, RIDGE_MEASURE).max() for m in ridge_models]
        )
    )

    n_cols = len(XGBOOST_MEASURES)
    n_rows = len(xgboost_models) + (1 if ridge_models else 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.2 * n_rows),
                             squeeze=False)

    for row, model in enumerate(xgboost_models):
        for col, (measure, xlabel, as_share) in enumerate(XGBOOST_MEASURES):
            draw_panel(
                axes[row][col],
                top_features(table, model, measure),
                title=f"{model} — {measure}, top {TOP_N}",
                xlabel=xlabel,
                as_share=as_share,
                xmax=share_max if as_share else None,
            )

    if ridge_models:
        row = len(xgboost_models)
        for col in range(n_cols):
            if col < len(ridge_models):
                model = ridge_models[col]
                draw_panel(
                    axes[row][col],
                    top_features(table, model, RIDGE_MEASURE),
                    title=f"{model} — |w| standardised, top {TOP_N}",
                    xlabel=RIDGE_LABEL,
                    as_share=True,
                    xmax=share_max,
                )
            else:
                axes[row][col].set_visible(False)

    fig.suptitle("Feature importance by model", fontsize=13)
    fig.tight_layout()
    return fig


def run(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load the saved models, print their top features, save table and figure.

    Parameters
    ----------
    config : dict, optional
        Parsed config. Loaded from ``config/ml_config.yaml`` when omitted.

    Returns
    -------
    pandas.DataFrame
        The long-form importance table that was saved.
    """
    settings = config if config is not None else load_config()
    models = load_models(settings)
    logger.info(f"Models    : {', '.join(models)}")

    table = collect_importance(models)

    for (model, measure), rows in table.groupby(["model", "measure"], sort=False):
        top = top_features(table, str(model), str(measure))
        unit = "%" if rows["share"].notna().any() else ""
        logger.info(f"\n--- {model}: {measure}, top {TOP_N}")
        for feature, value in top.items():
            logger.info(f"  {feature:<45} {value:>14,.2f}{unit}")

    csv_path, png_path = save_importance(table, settings)
    logger.info(f"\nSaved {csv_path}\nSaved {png_path}")
    return table


def save_importance(table: pd.DataFrame, settings: dict[str, Any]) -> tuple[Path, Path]:
    """Write the importance table and its figure next to the saved models.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of ``collect_importance``.
    settings : dict
        Parsed config, for the output directory.

    Returns
    -------
    tuple of (pathlib.Path, pathlib.Path)
        The CSV and PNG paths.
    """
    output_dir = resolve_output_dir(settings)
    csv_path = output_dir / "feature_importance.csv"
    png_path = output_dir / "feature_importance.png"
    table.to_csv(csv_path, index=False)
    figure = plot_importance(table)
    figure.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return csv_path, png_path


def main() -> None:
    """Run the importance analysis.

    Returns
    -------
    None
    """
    settings = load_config()
    logger.info(f"Logging to {setup_logging(settings, run_name='importance')}")
    run(settings)


if __name__ == "__main__":
    main()
