"""Walk-forward backtesting utilities for streaming data."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, TYPE_CHECKING, Tuple

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from river import metrics as river_metrics
    from trading_bot.features.engineer import OnlineFeatureBuilder


class OnlineModel(Protocol):
    """Protocol describing the River estimator interface we rely on."""

    def learn_one(self, x, y):
        ...

    def predict_one(self, x):
        ...


class OnlineCalibrator(Protocol):
    """Protocol describing calibration models applied to raw predictions."""

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
    total_return: float = 0.0
    buy_hold_return: float = 0.0
    sharpe_ratio: float = 0.0
    total_costs: float = 0.0


class BacktestEngine:
    """Walk-forward simulator that updates the model per observation."""

    def __init__(self, model: OnlineModel, *, calibrator: OnlineCalibrator | None = None) -> None:
        self._model = model
        self._calibrator = calibrator

    def run(
        self,
        candles: Iterable[Dict[str, float]],
        *,
        builder: "OnlineFeatureBuilder",
        edge_threshold: float,
        transaction_cost: float = 0.0,
        slippage_cost: float = 0.0,
        volatility_threshold: float | None = None,
        trend_bias_threshold: float | None = None,
        metrics: Sequence["river_metrics.base.Metric"] | None = None,
        step_callback: Optional[Callable[[], None]] = None,
        diagnostics: Optional[List[Tuple[float, float]]] = None,
        edge_clip: float | None = None,
    ) -> BacktestResult:
        """Run the simulation over a candle stream using edge-based decisions."""

        trades = 0
        wins = 0
        equity = 0.0
        returns_count = 0
        returns_sum = 0.0
        returns_sum_sq = 0.0
        total_costs = 0.0
        metric_objects = list(metrics or [])

        previous_features: dict[str, float] | None = None
        previous_close: float | None = None
        first_close: float | None = None
        last_close: float | None = None
        trade_cost = max(transaction_cost, 0.0) + max(slippage_cost, 0.0)
        threshold = max(edge_threshold, 0.0)

        for candle in candles:
            features = builder.process(candle)
            close = float(candle["close"])

            if first_close is None:
                first_close = close

            if previous_features is not None and previous_close is not None:
                predicted_edge_raw = self._model.predict_one(previous_features)
                predicted_edge = 0.0 if predicted_edge_raw is None else float(predicted_edge_raw)

                calibrator_features = None
                if self._calibrator is not None:
                    calibrator_features = {"prediction": predicted_edge}
                    calibrated = self._calibrator.predict_one(calibrator_features)
                    if calibrated is not None:
                        predicted_edge = float(calibrated)

                if edge_clip is not None and edge_clip > 0.0:
                    predicted_edge = max(min(predicted_edge, edge_clip), -edge_clip)

                log_return = self._log_return(previous_close, close)

                decision = self._decision_from_edge(predicted_edge, threshold, trade_cost)
                decision = self._apply_risk_filters(
                    decision,
                    previous_features,
                    volatility_threshold,
                    trend_bias_threshold,
                )

                for metric in metric_objects:
                    metric.update(log_return, predicted_edge)

                if diagnostics is not None:
                    diagnostics.append((predicted_edge, log_return))

                cost_penalty = trade_cost if decision != 0 else 0.0
                strategy_return = decision * log_return - cost_penalty
                equity += strategy_return

                returns_count += 1
                returns_sum += strategy_return
                returns_sum_sq += strategy_return * strategy_return

                if decision != 0:
                    trades += 1
                    total_costs += cost_penalty
                    if strategy_return > 0:
                        wins += 1

                if step_callback is not None:
                    step_callback()

                self._model.learn_one(previous_features, log_return)
                if self._calibrator is not None and calibrator_features is not None:
                    self._calibrator.learn_one(calibrator_features, log_return)

            previous_features = features
            previous_close = close
            last_close = close

        hit_rate = wins / trades if trades else 0.0

        metric_summary = {metric.__class__.__name__: metric.get() for metric in metric_objects}
        buy_hold_return = 0.0
        if first_close is not None and last_close is not None and first_close > 0:
            buy_hold_return = self._log_return(first_close, last_close)

        sharpe_ratio = 0.0
        if returns_count > 1:
            mean_return = returns_sum / returns_count
            variance = max((returns_sum_sq / returns_count) - mean_return * mean_return, 0.0)
            std_dev = math.sqrt(variance)
            if std_dev > 0:
                minutes_per_year = 365 * 24 * 60
                sharpe_ratio = (mean_return / std_dev) * math.sqrt(minutes_per_year)

        logger.info(
            "Completed backtest with equity={:.6f}, trades={}, hit_rate={:.2%}",
            equity,
            trades,
            hit_rate,
        )
        return BacktestResult(
            pnl=equity,
            hit_rate=hit_rate,
            trades=trades,
            metric_summary=metric_summary,
            total_return=returns_sum,
            buy_hold_return=buy_hold_return,
            sharpe_ratio=sharpe_ratio,
            total_costs=total_costs,
        )

    @staticmethod
    def _decision_from_edge(predicted_edge: float, threshold: float, trading_cost: float) -> int:
        """Map a predicted log return into a directional decision."""

        cushion = threshold + trading_cost
        if abs(predicted_edge) <= cushion:
            return 0
        return 1 if predicted_edge > 0 else -1

    @staticmethod
    def _apply_risk_filters(
        decision: int,
        features: dict[str, float] | None,
        volatility_threshold: float | None,
        trend_bias_threshold: float | None,
    ) -> int:
        """Apply simple volatility and trend-aware gating before acting."""

        if decision == 0 or features is None:
            return decision

        volatility_regime = features.get("volatility_regime")
        if (
            volatility_threshold is not None
            and volatility_regime is not None
            and volatility_regime > volatility_threshold
        ):
            return 0

        trend_bias = features.get("trend_bias")
        if trend_bias_threshold is not None and trend_bias is not None:
            if abs(trend_bias) < trend_bias_threshold:
                return 0
            if decision * trend_bias <= 0:
                return 0

        return decision

    @staticmethod
    def _log_return(previous_close: float, current_close: float) -> float:
        return math.log(max(current_close, 1e-12) / max(previous_close, 1e-12))
