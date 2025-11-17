"""Model factory utilities for online learning pipelines."""
from __future__ import annotations

from typing import Literal

from river import compose, ensemble, linear_model, metrics, optim, preprocessing, tree

from trading_bot.models.regime import RegimeAwareWrapper


OptimizerName = Literal["adam", "sgd", "rmsprop"]


def adaptive_regressor(
    *,
    optimizer_name: OptimizerName = "sgd",
    learning_rate: float = 0.001,
    intercept_lr: float = 0.001,
    l2: float = 1e-3,
    clip_gradient: float = 0.5,
    regime_aware: bool = True,
    ensemble_lr: float = 0.5,
    forest_size: int = 5,
    random_seed: int = 1337,
):
    """Create an adaptive, optionally regime-aware ensemble regressor."""

    def factory() -> ensemble.HedgeRegressor:
        linear_pipe = compose.Pipeline(
            preprocessing.StandardScaler(with_std=True),
            linear_model.LinearRegression(
                optimizer=_build_optimizer(optimizer_name, learning_rate),
                intercept_init=0.0,
                intercept_lr=intercept_lr,
                l2=l2,
                clip_gradient=clip_gradient,
            ),
        )

        h_tree = tree.HoeffdingTreeRegressor(
            grace_period=50,
            delta=1e-5,
            tau=1e-3,
            max_depth=10,
            leaf_prediction="adaptive",
        )

        forest_base = tree.HoeffdingTreeRegressor(
            grace_period=75,
            delta=1e-5,
            tau=1e-3,
            max_depth=12,
            leaf_prediction="adaptive",
        )

        adaptive_forest = ensemble.SRPRegressor(
            model=forest_base,
            n_models=forest_size,
            seed=random_seed,
            disable_weighted_vote=False,
        )

        # Blend heterogeneous experts via exponential weighting.
        models = [
            linear_pipe,
            h_tree,
            adaptive_forest,
        ]

        return ensemble.EWARegressor(models=models, learning_rate=ensemble_lr)

    base_model = factory()
    if not regime_aware:
        return base_model

    return RegimeAwareWrapper(factory, fallback_label=0)
def default_metrics() -> metrics.RegressionMetric:
    """Provide regression metrics tracked during online evaluation."""

    return metrics.MAE()


def _build_optimizer(name: OptimizerName, learning_rate: float):
    lowered = name.lower()
    if lowered == "adam":
        return optim.Adam(lr=learning_rate)
    if lowered == "sgd":
        return optim.SGD(lr=learning_rate)
    if lowered == "rmsprop":
        return optim.RMSProp(lr=learning_rate)
    raise ValueError(f"Unsupported optimizer '{name}'")
