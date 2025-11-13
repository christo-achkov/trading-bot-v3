"""Walk-forward backtesting utilities for streaming data."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from river import metrics as river_metrics


class OnlineModel(Protocol):
    """Protocol describing the River estimator interface we rely on."""

    def learn_one(self, x, y):
        ...

    def predict_one(self, x):
        ...


@dataclass
class BacktestResult:
    """Container for the core results of a simulation."""

    pnl: float
    hit_rate: float
    trades: int
    metric_summary: dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """Walk-forward simulator that updates the model per observation."""

    def __init__(self, model: OnlineModel) -> None:
        self._model = model

    def run(
        self,
        stream: Iterable[tuple[dict, int]],
        *,
        metrics: Sequence["river_metrics.base.Metric"] | None = None,
    ) -> BacktestResult:
        """Run the simulation over an `(features, label)` stream."""

        trades = 0
        wins = 0
        equity = 0.0
        metric_objects = list(metrics or [])

        for features, label in stream:
            prediction = self._model.predict_one(features)
            if prediction is None:
                self._model.learn_one(features, label)
                continue

            for metric in metric_objects:
                metric.update(label, prediction)

            if prediction != 0:
                trades += 1
                if prediction == label:
                    wins += 1
                    equity += 1
                else:
                    equity -= 1

            self._model.learn_one(features, label)
        hit_rate = wins / trades if trades else 0.0

        metric_summary = {metric.__class__.__name__: metric.get() for metric in metric_objects}
        logger.info(
            "Completed backtest with equity=%s, trades=%s, hit_rate=%s",
            equity,
            trades,
            hit_rate,
        )
        return BacktestResult(pnl=equity, hit_rate=hit_rate, trades=trades, metric_summary=metric_summary)
