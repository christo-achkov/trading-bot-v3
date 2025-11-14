"""Walk-forward backtesting utilities for streaming data."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, Optional, Protocol, Sequence, TYPE_CHECKING

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

    def __init__(self, model: OnlineModel) -> None:
        self._model = model

    def run(
        self,
        candles: Iterable[Dict[str, float]],
        *,
        builder: "OnlineFeatureBuilder",
        label_threshold: float,
        prediction_horizon: int = 1,
        aggregation: str = "majority",
        signal_threshold: float = 0.85,
        transaction_cost: float = 0.0,
        slippage_cost: float = 0.0,
        volatility_threshold: float | None = None,
        trend_bias_threshold: float | None = None,
        metrics: Sequence["river_metrics.base.Metric"] | None = None,
        step_callback: Optional[Callable[[], None]] = None,
    ) -> BacktestResult:
        """Run the simulation over a candle stream using buffered predictions."""

        if prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")

        trades = 0
        wins = 0
        equity = 0.0
        returns_count = 0
        returns_sum = 0.0
        returns_sum_sq = 0.0
        total_costs = 0.0
        metric_objects = list(metrics or [])

        pending_features: dict[str, float] | None = None
        pending_close: float | None = None
        first_close: float | None = None
        last_close: float | None = None
        prediction_buffer: Deque[int] = deque(maxlen=prediction_horizon)

        for candle in candles:
            features = builder.process(candle)
            close = float(candle["close"])

            if first_close is None:
                first_close = close

            if pending_features is not None and pending_close is not None:
                raw_prediction = self._model.predict_one(pending_features)
                prediction = self._normalize_prediction(raw_prediction)
                prediction_buffer.append(prediction)

                log_return = self._log_return(pending_close, close)
                actual_label = self._label_from_log_return(log_return, label_threshold)

                decision = 0
                if len(prediction_buffer) == prediction_horizon:
                    decision = self._aggregate_predictions(
                        prediction_buffer,
                        aggregation,
                        signal_threshold,
                    )

                decision = self._apply_risk_filters(
                    decision,
                    pending_features,
                    volatility_threshold,
                    trend_bias_threshold,
                )

                for metric in metric_objects:
                    metric.update(actual_label, decision)

                cost_penalty = 0.0
                if decision != 0:
                    cost_penalty = transaction_cost + slippage_cost
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

                self._model.learn_one(pending_features, actual_label)

            pending_features = features
            pending_close = close
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
    def _normalize_prediction(raw_prediction) -> int:
        if raw_prediction is None:
            return 0
        if raw_prediction > 0:
            return 1
        if raw_prediction < 0:
            return -1
        return 0

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
    def _aggregate_predictions(
        predictions: Deque[int],
        mode: str,
        signal_threshold: float,
    ) -> int:
        mode = mode.lower()
        window = len(predictions)
        if window == 0:
            return 0

        active_predictions = [p for p in predictions if p != 0]
        active_count = len(active_predictions)

        if active_count == 0:
            return 0

        threshold_votes = max(1, int(math.ceil(signal_threshold * active_count)))

        if mode == "unanimous":
            if active_count < threshold_votes:
                return 0
            positives = all(p == 1 for p in active_predictions)
            negatives = all(p == -1 for p in active_predictions)
            if positives:
                return 1
            if negatives:
                return -1
            return 0

        if mode == "weighted":
            weighted_score = 0.0
            active_weight = 0.0
            for idx, vote in enumerate(predictions):
                if vote == 0:
                    continue
                weight = float(idx + 1)
                weighted_score += weight * vote
                active_weight += weight

            if active_weight == 0:
                return 0

            if abs(weighted_score) < signal_threshold * active_weight:
                return 0
            return 1 if weighted_score > 0 else -1

        # default majority logic
        positives = sum(1 for p in active_predictions if p > 0)
        negatives = sum(1 for p in active_predictions if p < 0)

        if positives >= threshold_votes and positives > negatives:
            return 1
        if negatives >= threshold_votes and negatives > positives:
            return -1
        return 0

    @staticmethod
    def _log_return(previous_close: float, current_close: float) -> float:
        return math.log(max(current_close, 1e-12) / max(previous_close, 1e-12))

    @staticmethod
    def _label_from_log_return(log_return: float, threshold: float) -> int:
        if log_return > threshold:
            return 1
        if log_return < -threshold:
            return -1
        return 0
