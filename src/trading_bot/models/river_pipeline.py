"""Model factory utilities for online learning pipelines."""
from __future__ import annotations

from typing import Literal

from river import compose, metrics, optim, preprocessing
from river.linear_model import LinearRegression


OptimizerName = Literal["adam", "sgd", "rmsprop"]


def adaptive_regressor(
    *,
    optimizer_name: OptimizerName = "sgd",
    learning_rate: float = 0.001,
    intercept_lr: float = 0.001,
    l2: float = 1e-3,
    clip_gradient: float = 0.5,
) -> LinearRegression:
    """Create the adaptive regressor for forward edge estimation."""

    optimizer = _build_optimizer(optimizer_name, learning_rate)
    regressor = LinearRegression(
        optimizer=optimizer,
        intercept_init=0.0,
        intercept_lr=intercept_lr,
        l2=l2,
        clip_gradient=clip_gradient,
    )
    scaler = preprocessing.StandardScaler(with_std=True)
    return compose.Pipeline(scaler, regressor)
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
