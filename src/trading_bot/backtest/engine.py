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
        trade_log: Optional[List[dict]] = None,
        long_threshold: float | None = None,
        short_threshold: float | None = None,
        adaptive_threshold: bool = False,
        volatility_feature: str = "volatility_regime",
        volatility_scale: float = 0.0,
        volatility_offset: float = 0.0,
        minimum_cushion: float | None = None,
        bull_cushion_offset: float = 0.0,
        bear_cushion_offset: float = 0.0,
        use_position_sizing: bool = False,
        position_scale: float = 1000.0,
        edge_scale: float = 1.0,
        hysteresis: float = 0.0,
        use_dynamic_cost: bool = False,
        spread_cost_feature: str = "micro_spread_bps",
        spread_cost_scale: float = 0.0,
        volatility_cost_feature: str = "return_std_short",
        volatility_cost_scale: float = 0.0,
        liquidity_cost_feature: str | None = "liquidity_density_50bps",
        liquidity_cost_scale: float = 0.0,
        dynamic_cost_floor: float = 0.0,
        dynamic_cost_cap: float | None = None,
        skip_low_edge_trades: bool = True,
        turnover_penalty: float = 0.0,
        cost_adjust_training: bool = True,
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
        current_position = 0.0

        static_trade_cost = max(transaction_cost, 0.0) + max(slippage_cost, 0.0)
        base_threshold = max(edge_threshold, 0.0)
        long_threshold_value = base_threshold if long_threshold is None else max(long_threshold, 0.0)
        short_threshold_value = base_threshold if short_threshold is None else max(short_threshold, 0.0)

        for candle in candles:
            features = builder.process(candle)
            close = float(candle["close"])

            if first_close is None:
                first_close = close

            if previous_features is not None and previous_close is not None:
                predicted_edge_raw = self._model.predict_one(previous_features)
                predicted_edge = float(predicted_edge_raw) if predicted_edge_raw is not None else 0.0

                calibrator_features: dict[str, float] | None = None
                if self._calibrator is not None:
                    calibrator_features = {
                        "prediction": predicted_edge,
                        "regime_label": previous_features.get("regime_label"),
                    }
                    calibrated = self._calibrator.predict_one(calibrator_features)
                    if calibrated is not None:
                        predicted_edge = float(calibrated)

                predicted_edge *= edge_scale

                if edge_clip is not None and edge_clip > 0.0:
                    predicted_edge = max(min(predicted_edge, edge_clip), -edge_clip)

                log_return = self._log_return(previous_close, close)

                cost_estimate = static_trade_cost
                if (
                    use_dynamic_cost
                    or spread_cost_scale > 0.0
                    or volatility_cost_scale > 0.0
                    or liquidity_cost_scale != 0.0
                ):
                    cost_estimate = self._estimate_dynamic_cost(
                        base_cost=static_trade_cost,
                        features=previous_features,
                        spread_feature=spread_cost_feature,
                        spread_scale=spread_cost_scale,
                        volatility_feature=volatility_cost_feature,
                        volatility_scale=volatility_cost_scale,
                        liquidity_feature=liquidity_cost_feature,
                        liquidity_scale=liquidity_cost_scale,
                        floor_value=dynamic_cost_floor,
                        cap_value=dynamic_cost_cap,
                    )

                if cost_adjust_training:
                    cost_basis = cost_estimate
                    raw_training_target = log_return - cost_basis
                else:
                    cost_basis = cost_estimate
                    raw_training_target = log_return

                long_cushion = long_threshold_value - math.log(1 - cost_estimate) if cost_estimate < 1 else long_threshold_value - math.log(1e-10)
                short_cushion = short_threshold_value - math.log(1 - cost_estimate) if cost_estimate < 1 else short_threshold_value - math.log(1e-10)

                if adaptive_threshold and volatility_scale > 0.0:
                    raw_vol = features.get(volatility_feature)
                    try:
                        vol_value = float(raw_vol) if raw_vol is not None else 0.0
                    except (TypeError, ValueError):
                        vol_value = 0.0
                    if math.isfinite(vol_value):
                        adjustment = (vol_value - volatility_offset) * volatility_scale
                        long_cushion += adjustment
                        short_cushion += adjustment

                trend_bias_value = 0.0
                if previous_features is not None:
                    raw_trend = previous_features.get("trend_bias")
                    try:
                        trend_bias_value = float(raw_trend) if raw_trend is not None else 0.0
                    except (TypeError, ValueError):
                        trend_bias_value = 0.0

                if bull_cushion_offset != 0.0 and trend_bias_value >= 0.0:
                    long_cushion += bull_cushion_offset
                if bear_cushion_offset != 0.0 and trend_bias_value <= 0.0:
                    short_cushion += bear_cushion_offset

                if minimum_cushion is not None:
                    floor_value = max(minimum_cushion, -math.log(1 - cost_estimate) if cost_estimate < 1 else -math.log(1e-10))
                    long_cushion = max(long_cushion, floor_value)
                    short_cushion = max(short_cushion, floor_value)

                long_cushion = max(long_cushion, -math.log(1 - cost_estimate) if cost_estimate < 1 else -math.log(1e-10))
                short_cushion = max(short_cushion, -math.log(1 - cost_estimate) if cost_estimate < 1 else -math.log(1e-10))

                should_skip = skip_low_edge_trades and abs(predicted_edge) <= (-math.log(1 - cost_estimate) if cost_estimate < 1 else -math.log(1e-10))
                if should_skip:
                    position = 0.0
                else:
                    position = self._position_from_edge(
                        predicted_edge,
                        long_cushion,
                        short_cushion,
                        use_position_sizing,
                        position_scale,
                    )

                position = self._apply_risk_filters(
                    position,
                    previous_features,
                    volatility_threshold,
                    trend_bias_threshold,
                )

                if hysteresis > 0.0 and current_position != 0.0 and position != 0.0:
                    if current_position > 0.0:
                        exit_threshold = long_cushion - hysteresis
                        if predicted_edge <= exit_threshold:
                            position = 0.0
                        elif position <= 0.0:
                            position = max(current_position, 0.0)
                        else:
                            position = max(position, current_position)
                    else:
                        exit_threshold = -short_cushion + hysteresis
                        if predicted_edge >= exit_threshold:
                            position = 0.0
                        elif position >= 0.0:
                            position = min(current_position, 0.0)
                        else:
                            position = min(position, current_position)

                executed = position != 0.0
                cost_penalty = cost_estimate * abs(position) if executed else 0.0
                cost_penalty_log = abs(position) * math.log(1 - cost_estimate) if executed and cost_estimate < 1 else 0.0
                gross_return = position * log_return
                strategy_return = gross_return + cost_penalty_log

                turnover_change = abs(position - current_position)
                penalty = turnover_penalty * turnover_change if turnover_penalty > 0.0 else 0.0
                learning_target = raw_training_target - penalty

                for metric in metric_objects:
                    metric.update(learning_target, predicted_edge)

                if diagnostics is not None:
                    diagnostics.append((predicted_edge, learning_target))

                equity += strategy_return

                returns_count += 1
                returns_sum += strategy_return
                returns_sum_sq += strategy_return * strategy_return

                if executed:
                    trades += 1
                    total_costs += cost_penalty
                    if strategy_return > 0.0:
                        wins += 1

                    if trade_log is not None:
                        timestamp = candle.get("close_time") or candle.get("open_time")
                        if isinstance(timestamp, str):
                            timestamp_str = timestamp
                        elif timestamp is None:
                            timestamp_str = ""
                        else:
                            timestamp_str = str(timestamp)

                        def _safe_feature(key: str) -> float:
                            raw_value = previous_features.get(key) if previous_features else None
                            try:
                                return float(raw_value) if raw_value is not None else 0.0
                            except (TypeError, ValueError):
                                return 0.0

                        trade_log.append(
                            {
                                "trade_id": trades,
                                "timestamp": timestamp_str,
                                "side": "long" if position > 0 else "short",
                                "predicted_edge": predicted_edge,
                                "log_return": log_return,
                                "gross_return": gross_return,
                                "fees": cost_penalty_log,
                                "net_return": strategy_return,
                                "equity_after": equity,
                                "position_size": position,
                                "long_cushion": long_cushion,
                                "short_cushion": short_cushion,
                                "cost_estimate": cost_estimate,
                                "micro_spread_bps": _safe_feature("micro_spread_bps"),
                                "microstructure_spread_available": _safe_feature("microstructure_spread_available"),
                                "volatility_regime": _safe_feature("volatility_regime"),
                                "volatility_slope": _safe_feature("volatility_slope"),
                                "regime_label": _safe_feature("regime_label"),
                                "regime_state_bull": _safe_feature("regime_state_bull"),
                                "regime_state_bear": _safe_feature("regime_state_bear"),
                                "regime_state_sideways": _safe_feature("regime_state_sideways"),
                                "funding_rate": _safe_feature("funding_rate"),
                                "funding_rate_diff": _safe_feature("funding_rate_diff"),
                                "funding_rate_alt": _safe_feature("funding_rate_alt"),
                                "funding_rate_diff_alt": _safe_feature("funding_rate_diff_alt"),
                                "funding_rate_basis": _safe_feature("funding_rate_basis"),
                                "funding_rate_diff_basis": _safe_feature("funding_rate_diff_basis"),
                                "orderbook_imbalance_top": _safe_feature("orderbook_imbalance_top"),
                                "orderbook_imbalance_25bps": _safe_feature("orderbook_imbalance_25bps"),
                                "orderbook_imbalance_50bps": _safe_feature("orderbook_imbalance_50bps"),
                                "orderbook_imbalance_100bps": _safe_feature("orderbook_imbalance_100bps"),
                                "liquidity_bid_depth_25bps": _safe_feature("liquidity_bid_depth_25bps"),
                                "liquidity_ask_depth_25bps": _safe_feature("liquidity_ask_depth_25bps"),
                                "liquidity_bid_depth_50bps": _safe_feature("liquidity_bid_depth_50bps"),
                                "liquidity_ask_depth_50bps": _safe_feature("liquidity_ask_depth_50bps"),
                                "liquidity_bid_depth_100bps": _safe_feature("liquidity_bid_depth_100bps"),
                                "liquidity_ask_depth_100bps": _safe_feature("liquidity_ask_depth_100bps"),
                                "liquidity_density_25bps": _safe_feature("liquidity_density_25bps"),
                                "liquidity_depth_25_available": _safe_feature("liquidity_depth_25_available"),
                                "liquidity_depth_50_available": _safe_feature("liquidity_depth_50_available"),
                                "liquidity_depth_100_available": _safe_feature("liquidity_depth_100_available"),
                                "liquidity_density_50bps": _safe_feature("liquidity_density_50bps"),
                                "liquidity_density_100bps": _safe_feature("liquidity_density_100bps"),
                                "liquidity_tier_ratio_50_25": _safe_feature("liquidity_tier_ratio_50_25"),
                                "liquidity_tier_ratio_100_25": _safe_feature("liquidity_tier_ratio_100_25"),
                                "liquidity_tier_ratio_100_50": _safe_feature("liquidity_tier_ratio_100_50"),
                                "turnover_change": turnover_change,
                                "turnover_penalty": penalty,
                                "realized_vol_slope_short": _safe_feature("realized_vol_slope_short"),
                                "realized_vol_slope_medium": _safe_feature("realized_vol_slope_medium"),
                                "realized_vol_trend": _safe_feature("realized_vol_trend"),
                                "funding_available": _safe_feature("funding_available"),
                            }
                        )

                if step_callback is not None:
                    step_callback()

                self._model.learn_one(previous_features, learning_target)
                if self._calibrator is not None and calibrator_features is not None:
                    self._calibrator.learn_one(calibrator_features, learning_target)

                current_position = position

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
    def _estimate_dynamic_cost(
        *,
        base_cost: float,
        features: dict[str, float] | None,
        spread_feature: str,
        spread_scale: float,
        volatility_feature: str,
        volatility_scale: float,
        liquidity_feature: str | None,
        liquidity_scale: float,
        floor_value: float,
        cap_value: float | None,
    ) -> float:
        """Blend spread, volatility, and liquidity cues into a dynamic cost estimate."""

        cost = max(base_cost, 0.0)
        if features is None:
            return BacktestEngine._bound_cost(cost, floor_value, cap_value)

        if spread_feature and spread_scale != 0.0:
            spread_raw = features.get(spread_feature)
            try:
                spread_value = float(spread_raw) if spread_raw is not None else 0.0
            except (TypeError, ValueError):
                spread_value = 0.0
            spread_fraction = spread_value
            if "bps" in spread_feature.lower():
                spread_fraction = spread_value / 10_000.0
            cost += spread_scale * max(spread_fraction, 0.0)

        if volatility_feature and volatility_scale != 0.0:
            volatility_raw = features.get(volatility_feature)
            try:
                volatility_value = float(volatility_raw) if volatility_raw is not None else 0.0
            except (TypeError, ValueError):
                volatility_value = 0.0
            cost += volatility_scale * max(volatility_value, 0.0)

        if liquidity_feature and liquidity_scale != 0.0:
            liquidity_raw = features.get(liquidity_feature)
            try:
                liquidity_value = float(liquidity_raw) if liquidity_raw is not None else 0.0
            except (TypeError, ValueError):
                liquidity_value = 0.0
            cost -= liquidity_scale * max(liquidity_value, 0.0)

        return BacktestEngine._bound_cost(cost, floor_value, cap_value)

    @staticmethod
    def _bound_cost(cost: float, floor_value: float, cap_value: float | None) -> float:
        bounded = max(cost, max(floor_value, 0.0))
        if cap_value is not None:
            bounded = min(bounded, max(cap_value, 0.0))
        return max(bounded, 0.0)

    @staticmethod
    def _position_from_edge(
        predicted_edge: float,
        long_cushion: float,
        short_cushion: float,
        use_position_sizing: bool,
        position_scale: float,
    ) -> float:
        """Convert an edge estimate into a position size in [-1, 1]."""

        long_cushion = max(long_cushion, 0.0)
        short_cushion = max(short_cushion, 0.0)

        if predicted_edge >= long_cushion:
            if use_position_sizing:
                excess = predicted_edge - long_cushion
                scaled = max(0.0, excess) * max(position_scale, 0.0)
                return min(1.0, scaled)
            return 1.0

        if predicted_edge <= -short_cushion:
            if use_position_sizing:
                excess = (-predicted_edge) - short_cushion
                scaled = max(0.0, excess) * max(position_scale, 0.0)
                return -min(1.0, scaled)
            return -1.0

        return 0.0

    @staticmethod
    def _apply_risk_filters(
        position: float,
        features: dict[str, float] | None,
        volatility_threshold: float | None,
        trend_bias_threshold: float | None,
    ) -> float:
        """Apply volatility and trend gates before taking a position."""

        if position == 0.0 or features is None:
            return position

        volatility_regime = features.get("volatility_regime")
        if (
            volatility_threshold is not None
            and volatility_regime is not None
            and volatility_regime > volatility_threshold
        ):
            return 0.0

        trend_bias = features.get("trend_bias")
        if trend_bias_threshold is not None and trend_bias is not None:
            if abs(trend_bias) < trend_bias_threshold:
                return 0.0
            if position * trend_bias <= 0:
                return 0.0

        return position

    @staticmethod
    def _log_return(previous_close: float, current_close: float) -> float:
        return math.log(max(current_close, 1e-12) / max(previous_close, 1e-12))
