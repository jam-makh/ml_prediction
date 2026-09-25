"""
Elastic net: ridge's L2 penalty plus lasso's L1, over the same pipeline.

The L1 part sets coefficients to exactly zero, so the fit states which columns
carry nothing. The L2 part keeps correlated balance lags together instead of
letting lasso pick one at random per fold. Everything but the estimator is
inherited from ``RidgeRegression``.

**No search inside the model.** ``alpha`` and ``l1_ratio`` come from the
config: Optuna fills them per trial, and once settled they are pasted into the
model's ``params``. Left empty with Optuna off, the build fails rather than
guessing.

**Alpha is not on ridge's scale.** scikit-learn divides elastic net's squared
loss by 2n, and the L1 term is in the target's units (dollars here).
"""

from __future__ import annotations

from sklearn.linear_model import ElasticNet

from src.models_code.ridge_reg import RidgeRegression

# Coordinate descent on correlated money columns needs more than sklearn's 1,000 passes.
DEFAULT_MAX_ITER = 10_000


class ElasticNetRegression(RidgeRegression):
    """Elastic net over the feature table, with ridge's preprocessing.

    Parameters
    ----------
    name : str, optional
        Label for result tables. Default ``elastic_net``.
    anchor_column : str, optional
        Column holding last month's balance.
    alpha : float
        Penalty strength, from Optuna or the config's ``params``.
    l1_ratio : float
        L1 share of the penalty, in (0, 1], from Optuna or the config's ``params``.
    max_iter : int, optional
        Coordinate-descent pass limit. Default :data:`DEFAULT_MAX_ITER`.
    impute_strategy : str, optional
        Passed to ``SimpleImputer``. Default ``median``.
    random_state : int, optional
        Accepted for interface symmetry. The cyclic solver is deterministic.

    Attributes
    ----------
    best_params_ : dict or None
        ``{"alpha": ..., "l1_ratio": ...}`` once fitted.

    Raises
    ------
    ValueError
        If ``alpha`` or ``l1_ratio`` is not set.
    """

    def __init__(
        self,
        name: str = "elastic_net",
        anchor_column: str = "prev_1m_closing_balance_usd",
        alpha: float | None = None,
        l1_ratio: float | None = None,
        max_iter: int = DEFAULT_MAX_ITER,
        impute_strategy: str = "median",
        random_state: int = 42,
    ) -> None:
        # Checked here as a pair, so one message names both settings.
        if alpha is None or l1_ratio is None:
            raise ValueError(self._unset_message(name, "alpha / l1_ratio"))
        super().__init__(
            name,
            anchor_column=anchor_column,
            alpha=alpha,
            impute_strategy=impute_strategy,
            random_state=random_state,
        )
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter

    def _build_estimator(self) -> ElasticNet:
        """Return the unfitted elastic net with the configured penalty.

        Returns
        -------
        sklearn.linear_model.ElasticNet
            Unfitted.
        """
        return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter)

    def _penalty_params(self) -> dict[str, float]:
        """Return the penalty settings this model fits with.

        Returns
        -------
        dict
            ``{"alpha": ..., "l1_ratio": ...}``.
        """
        return {"alpha": float(self.alpha), "l1_ratio": float(self.l1_ratio)}

    def describe(self) -> str:
        """Return a one-line description for the run log.

        Returns
        -------
        str
            Alpha, L1 share, and how many features kept a non-zero weight.
        """
        if not self.is_fitted or self._pipeline is None:
            return f"{self.name} (not fitted)"
        coefficients = self._pipeline.named_steps["regressor"].coef_
        # Kept features are the point of elastic net, so the count goes in the one-liner.
        kept = int((coefficients != 0).sum())
        return (
            f"{self.name}: elastic net alpha={self.alpha:g}, "
            f"l1_ratio={self.l1_ratio:g}, "
            f"{kept}/{len(self._feature_columns)} features non-zero"
        )
