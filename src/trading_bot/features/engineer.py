"""Feature engineering utilities operating on streaming candle data."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Sequence

try:  # pragma: no cover - optional dependency
    from river import cluster as river_cluster
except Exception:  # river may be unavailable during lightweight testing
    river_cluster = None


class RollingMean:
    """Simple rolling mean with bounded memory."""

    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window: Deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        self.window.append(value)
        return sum(self.window) / len(self.window)


class RollingStd:
    """Simple rolling population standard deviation."""

    def __init__(self, window_size: int) -> None:
        if window_size <= 1:
            raise ValueError("window_size must be greater than one")
        self.window: Deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        self.window.append(value)
        if len(self.window) < 2:
            return 0.0
        mean = sum(self.window) / len(self.window)
        variance = sum((x - mean) ** 2 for x in self.window) / len(self.window)
        return math.sqrt(variance)


class ExponentialMovingAverage:
    """Exponentially weighted moving average."""

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be positive")
        self._alpha = 2.0 / (period + 1.0)
        self._value: float | None = None

    def update(self, value: float) -> float:
        if self._value is None:
            self._value = value
        else:
            self._value += self._alpha * (value - self._value)
        return self._value


class AverageTrueRange:
    """Smoothed average true range tracker."""

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be positive")
        self._alpha = 1.0 / period
        self._atr: float | None = None

    def update(
        self,
        high: float,
        low: float,
        close: float,
        previous_close: float | None,
    ) -> float:
        true_range = high - low
        if previous_close is not None:
            true_range = max(
                true_range,
                abs(high - previous_close),
                abs(low - previous_close),
            )

        if self._atr is None:
            self._atr = true_range
        else:
            self._atr += self._alpha * (true_range - self._atr)
        return self._atr


class StreamingRSI:
    """Smoothed relative strength index following Wilder's method."""

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be positive")
        self._alpha = 1.0 / period
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._initialized = False

    def update(self, delta: float) -> float:
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)

        if not self._initialized:
            self._avg_gain = gain
            self._avg_loss = loss
            self._initialized = True
        else:
            self._avg_gain += self._alpha * (gain - self._avg_gain)
            self._avg_loss += self._alpha * (loss - self._avg_loss)

        if self._avg_loss < 1e-12 and self._avg_gain < 1e-12:
            return 50.0
        if self._avg_loss < 1e-12:
            return 100.0
        if self._avg_gain < 1e-12:
            return 0.0

        rs = self._avg_gain / (self._avg_loss + 1e-12)
        return 100.0 - (100.0 / (1.0 + rs))


@dataclass
class OnlineFeatureBuilder:
    """Construct derived features incrementally from streaming candles."""

    exclude_microstructure: bool = False

    return_fast: RollingMean = field(default_factory=lambda: RollingMean(5))
    return_medium: RollingMean = field(default_factory=lambda: RollingMean(30))
    return_slow: RollingMean = field(default_factory=lambda: RollingMean(120))
    return_long: RollingMean = field(default_factory=lambda: RollingMean(360))
    return_std_short: RollingStd = field(default_factory=lambda: RollingStd(30))
    return_std_medium: RollingStd = field(default_factory=lambda: RollingStd(120))
    return_std_long: RollingStd = field(default_factory=lambda: RollingStd(360))
    price_fast: RollingMean = field(default_factory=lambda: RollingMean(20))
    price_medium: RollingMean = field(default_factory=lambda: RollingMean(60))
    price_slow: RollingMean = field(default_factory=lambda: RollingMean(180))
    price_std_fast: RollingStd = field(default_factory=lambda: RollingStd(20))
    volume_fast: RollingMean = field(default_factory=lambda: RollingMean(20))
    volume_medium: RollingMean = field(default_factory=lambda: RollingMean(60))
    volume_slow: RollingMean = field(default_factory=lambda: RollingMean(240))
    volume_std_fast: RollingStd = field(default_factory=lambda: RollingStd(20))
    volume_std_slow: RollingStd = field(default_factory=lambda: RollingStd(120))
    price_ema_fast: ExponentialMovingAverage = field(default_factory=lambda: ExponentialMovingAverage(12))
    price_ema_slow: ExponentialMovingAverage = field(default_factory=lambda: ExponentialMovingAverage(26))
    atr_tracker: AverageTrueRange = field(default_factory=lambda: AverageTrueRange(5))
    rsi_tracker: StreamingRSI = field(default_factory=lambda: StreamingRSI(5))
    volatility_mean_long: RollingMean = field(default_factory=lambda: RollingMean(60))
    volatility_std_tracker: RollingStd = field(default_factory=lambda: RollingStd(60))
    funding_mean_long: RollingMean = field(default_factory=lambda: RollingMean(720))
    funding_std_tracker: RollingStd = field(default_factory=lambda: RollingStd(720))
    funding_ema_fast: ExponentialMovingAverage = field(default_factory=lambda: ExponentialMovingAverage(48))
    funding_alt_mean_long: RollingMean = field(default_factory=lambda: RollingMean(720))
    funding_basis_mean_long: RollingMean = field(default_factory=lambda: RollingMean(720))
    regime_clusterer: object | None = field(
        default_factory=lambda: river_cluster.KMeans(n_clusters=4, halflife=256, seed=1337) if river_cluster else None
    )

    _close_history: Deque[float] = field(default_factory=lambda: deque(maxlen=60))
    _volume_history: Deque[float] = field(default_factory=lambda: deque(maxlen=60))
    _funding_history: Deque[float] = field(default_factory=lambda: deque(maxlen=720))
    _prev_close: float | None = None
    _prev_funding_rate: float | None = None
    _prev_funding_rate_alt: float | None = None
    _prev_funding_rate_basis: float | None = None
    _prev_return_std_short: float | None = None
    _prev_return_std_medium: float | None = None
    _regime_stats: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def process(self, candle: Dict[str, float]) -> Dict[str, float]:
        """Update internal state and return the latest feature vector."""

        close = float(candle["close"])
        high = float(candle.get("high", close))
        low = float(candle.get("low", close))
        volume = float(candle.get("volume", 0.0))

        best_bid = candle.get("best_bid", candle.get("bid_price", candle.get("bid", close)))
        best_ask = candle.get("best_ask", candle.get("ask_price", candle.get("ask", close)))
        try:
            best_bid = float(best_bid) if best_bid is not None else close
        except (TypeError, ValueError):
            best_bid = close
        try:
            best_ask = float(best_ask) if best_ask is not None else close
        except (TypeError, ValueError):
            best_ask = close

        if best_bid <= 0 or not math.isfinite(best_bid):
            best_bid = close
        if best_ask <= 0 or not math.isfinite(best_ask):
            best_ask = close

        mid_price = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else close
        spread_abs = max(best_ask - best_bid, 0.0)
        spread_pct = spread_abs / max(mid_price, 1e-12)
        spread_bps = spread_pct * 10_000.0

        best_bid_size = candle.get("best_bid_size", candle.get("bid_size", candle.get("bid_qty", 0.0)))
        best_ask_size = candle.get("best_ask_size", candle.get("ask_size", candle.get("ask_qty", 0.0)))
        try:
            best_bid_size = float(best_bid_size) if best_bid_size is not None else 0.0
        except (TypeError, ValueError):
            best_bid_size = 0.0
        try:
            best_ask_size = float(best_ask_size) if best_ask_size is not None else 0.0
        except (TypeError, ValueError):
            best_ask_size = 0.0

        def _extract_float(keys: Sequence[str], default: float = 0.0) -> float:
            for key in keys:
                raw_value = candle.get(key)
                if raw_value is None:
                    continue
                try:
                    return float(raw_value)
                except (TypeError, ValueError):
                    continue
            return default

        def _normalise_levels(levels_data) -> list[tuple[float, float]]:
            levels: list[tuple[float, float]] = []
            if levels_data is None:
                return levels
            if hasattr(levels_data, "tolist"):
                try:
                    levels_data = levels_data.tolist()
                except TypeError:
                    pass
            if isinstance(levels_data, dict):
                levels_data = levels_data.get("levels", [])
            if isinstance(levels_data, tuple):
                levels_data = list(levels_data)
            if not isinstance(levels_data, list):
                return levels
            for level in levels_data:
                if hasattr(level, "tolist"):
                    try:
                        level = level.tolist()
                    except TypeError:
                        pass
                price = None
                qty = None
                if isinstance(level, dict):
                    price = level.get("price") or level.get("px") or level.get("rate")
                    qty = level.get("size") or level.get("qty") or level.get("volume") or level.get("amount")
                elif isinstance(level, (list, tuple)) and len(level) >= 2:
                    price = level[0]
                    qty = level[1]
                elif not isinstance(level, (str, bytes)) and hasattr(level, "__len__"):
                    try:
                        if len(level) >= 2:
                            price = level[0]
                            qty = level[1]
                    except TypeError:
                        price = None
                        qty = None
                if price is None or qty is None:
                    continue
                try:
                    price_val = float(price)
                    qty_val = float(qty)
                except (TypeError, ValueError):
                    continue
                if qty_val < 0:
                    qty_val = -qty_val
                levels.append((price_val, qty_val))
            return levels

        bids_source = candle.get("orderbook_bids")
        if bids_source is None:
            for key in ("bids", "bid_levels", "depth_bids"):
                candidate = candle.get(key)
                if candidate is not None:
                    bids_source = candidate
                    break
        asks_source = candle.get("orderbook_asks")
        if asks_source is None:
            for key in ("asks", "ask_levels", "depth_asks"):
                candidate = candle.get(key)
                if candidate is not None:
                    asks_source = candidate
                    break

        bids_levels = _normalise_levels(bids_source)
        asks_levels = _normalise_levels(asks_source)

        if best_bid_size <= 0.0 and bids_levels:
            best_bid_size = bids_levels[0][1]
        if best_ask_size <= 0.0 and asks_levels:
            best_ask_size = asks_levels[0][1]

        def _depth_within(levels: list[tuple[float, float]], *, side: str, midpoint: float, pct: float) -> float:
            if midpoint <= 0.0 or not levels:
                return 0.0
            limit = midpoint * pct
            total = 0.0
            for price, qty in levels:
                if qty <= 0.0 or price <= 0.0:
                    continue
                distance = midpoint - price if side == "bid" else price - midpoint
                if distance < 0:
                    continue
                if distance <= limit:
                    total += qty
            return total

        bid_depth_25_raw = _depth_within(bids_levels, side="bid", midpoint=mid_price, pct=0.0025)
        ask_depth_25_raw = _depth_within(asks_levels, side="ask", midpoint=mid_price, pct=0.0025)
        bid_depth_50_raw = _depth_within(bids_levels, side="bid", midpoint=mid_price, pct=0.005)
        ask_depth_50_raw = _depth_within(asks_levels, side="ask", midpoint=mid_price, pct=0.005)
        bid_depth_100_raw = _depth_within(bids_levels, side="bid", midpoint=mid_price, pct=0.01)
        ask_depth_100_raw = _depth_within(asks_levels, side="ask", midpoint=mid_price, pct=0.01)

        bid_depth_25 = math.log1p(max(bid_depth_25_raw, 0.0))
        ask_depth_25 = math.log1p(max(ask_depth_25_raw, 0.0))
        bid_depth_50 = math.log1p(max(bid_depth_50_raw, 0.0))
        ask_depth_50 = math.log1p(max(ask_depth_50_raw, 0.0))
        bid_depth_100 = math.log1p(max(bid_depth_100_raw, 0.0))
        ask_depth_100 = math.log1p(max(ask_depth_100_raw, 0.0))

        depth_total_25 = bid_depth_25_raw + ask_depth_25_raw
        depth_total_50 = bid_depth_50_raw + ask_depth_50_raw
        depth_total_100 = bid_depth_100_raw + ask_depth_100_raw
        has_spread = 1.0 if spread_abs > 0.0 else 0.0
        has_depth_25 = 1.0 if depth_total_25 > 0.0 else 0.0
        has_depth_50 = 1.0 if depth_total_50 > 0.0 else 0.0
        has_depth_100 = 1.0 if depth_total_100 > 0.0 else 0.0

        imbalance_top = 0.0
        top_sum = best_bid_size + best_ask_size
        if top_sum > 1e-12:
            imbalance_top = (best_bid_size - best_ask_size) / top_sum
        elif depth_total_25 > 0.0:
            imbalance_top = (bid_depth_25_raw - ask_depth_25_raw) / depth_total_25

        liquidity_imbalance_25 = 0.0
        if depth_total_25 > 1e-12:
            liquidity_imbalance_25 = (bid_depth_25_raw - ask_depth_25_raw) / depth_total_25

        liquidity_imbalance_50 = 0.0
        if depth_total_50 > 1e-12:
            liquidity_imbalance_50 = (bid_depth_50_raw - ask_depth_50_raw) / depth_total_50

        liquidity_imbalance_100 = 0.0
        if depth_total_100 > 1e-12:
            liquidity_imbalance_100 = (bid_depth_100_raw - ask_depth_100_raw) / depth_total_100

        liquidity_tier_ratio_50_25 = depth_total_50 / depth_total_25 if depth_total_25 > 1e-12 else 0.0
        liquidity_tier_ratio_100_25 = depth_total_100 / depth_total_25 if depth_total_25 > 1e-12 else 0.0
        liquidity_tier_ratio_100_50 = depth_total_100 / depth_total_50 if depth_total_50 > 1e-12 else 0.0

        previous_close = self._prev_close
        if previous_close is None:
            minute_return = 0.0
        else:
            minute_return = math.log(max(close, 1e-12) / max(previous_close, 1e-12))

        ema_fast = self.price_ema_fast.update(close)
        ema_slow = self.price_ema_slow.update(close)
        ema_diff = ema_fast - ema_slow
        ema_ratio = 0.0
        if abs(ema_slow) > 1e-12:
            ema_ratio = (ema_fast / ema_slow) - 1.0

        rsi = self.rsi_tracker.update(close - previous_close if previous_close is not None else 0.0)
        atr = self.atr_tracker.update(high, low, close, previous_close)

        return_fast = self.return_fast.update(minute_return)
        return_medium = self.return_medium.update(minute_return)
        return_slow = self.return_slow.update(minute_return)
        return_long = self.return_long.update(minute_return)
        return_std_short = self.return_std_short.update(minute_return)
        return_std_medium = self.return_std_medium.update(minute_return)
        return_std_long = self.return_std_long.update(minute_return)
        volatility_mean_long = self.volatility_mean_long.update(return_std_long)
        volatility_std_long = self.volatility_std_tracker.update(return_std_long)

        funding_rate_raw = candle.get("funding_rate", 0.0)
        try:
            funding_rate = float(funding_rate_raw) if funding_rate_raw is not None else 0.0
        except (TypeError, ValueError):
            funding_rate = 0.0
        funding_rate_alt = _extract_float(
            [
                "funding_rate_alt",
                "predicted_funding_rate",
                "external_funding_rate",
                "funding_rate_external",
                "mark_funding_rate",
                "funding_next",
            ],
            default=0.0,
        )
        funding_rate_basis = _extract_float(
            [
                "basis_rate",
                "spot_basis",
                "basis",
                "basis_spread",
                "premium_index",
            ],
            default=0.0,
        )
        funding_mean = self.funding_mean_long.update(funding_rate)
        funding_std = self.funding_std_tracker.update(funding_rate)
        funding_ema = self.funding_ema_fast.update(funding_rate)
        funding_available = 1.0 if funding_rate_raw is not None else 0.0
        funding_alt_mean = self.funding_alt_mean_long.update(funding_rate_alt)
        funding_basis_mean = self.funding_basis_mean_long.update(funding_rate_basis)
        funding_diff = funding_rate - funding_mean
        funding_zscore = 0.0
        if funding_std > 1e-12:
            funding_zscore = funding_diff / funding_std
        funding_change = 0.0
        if self._prev_funding_rate is not None:
            funding_change = funding_rate - self._prev_funding_rate
        self._prev_funding_rate = funding_rate
        self._funding_history.append(funding_rate)
        funding_alt_diff = funding_rate - funding_rate_alt
        funding_basis_diff = funding_rate - funding_rate_basis
        funding_alt_change = 0.0
        if self._prev_funding_rate_alt is not None:
            funding_alt_change = funding_rate_alt - self._prev_funding_rate_alt
        self._prev_funding_rate_alt = funding_rate_alt
        funding_basis_change = 0.0
        if self._prev_funding_rate_basis is not None:
            funding_basis_change = funding_rate_basis - self._prev_funding_rate_basis
        self._prev_funding_rate_basis = funding_rate_basis
        funding_alt_diff_mean = funding_rate_alt - funding_alt_mean
        funding_basis_diff_mean = funding_rate_basis - funding_basis_mean

        price_fast = self.price_fast.update(close)
        price_medium = self.price_medium.update(close)
        price_slow = self.price_slow.update(close)
        price_std_fast = self.price_std_fast.update(close)

        volume_fast = self.volume_fast.update(volume)
        volume_medium = self.volume_medium.update(volume)
        volume_slow = self.volume_slow.update(volume)
        volume_std_fast = self.volume_std_fast.update(volume)
        volume_std_slow = self.volume_std_slow.update(volume)

        volatility_slope = 0.0
        if self._prev_return_std_short is not None:
            volatility_slope = return_std_short - self._prev_return_std_short
        self._prev_return_std_short = return_std_short

        volatility_slope_medium = 0.0
        if self._prev_return_std_medium is not None:
            volatility_slope_medium = return_std_medium - self._prev_return_std_medium
        self._prev_return_std_medium = return_std_medium

        realized_vol_trend = return_std_medium - return_std_long

        price_zscore_fast = 0.0
        if price_std_fast > 0:
            price_zscore_fast = (close - price_fast) / price_std_fast

        volume_ratio = 0.0
        if volume_medium > 1e-12:
            volume_ratio = volume / volume_medium

        volume_zscore_fast = 0.0
        if volume_std_fast > 0:
            volume_zscore_fast = (volume - volume_fast) / volume_std_fast

        volume_trend = 0.0
        if volume_slow > 1e-12:
            volume_trend = (volume_fast - volume_slow) / volume_slow

        depth_baseline = max(volume_medium, volume_slow, 1e-12)
        liquidity_density_25 = (bid_depth_25_raw + ask_depth_25_raw) / depth_baseline
        liquidity_density_50 = (bid_depth_50_raw + ask_depth_50_raw) / depth_baseline
        liquidity_density_100 = (bid_depth_100_raw + ask_depth_100_raw) / depth_baseline
        liquidity_pressure = liquidity_imbalance_50 if depth_total_50 > 0.0 else liquidity_imbalance_25
        spread_vol_ratio = 0.0
        if return_std_short > 1e-12:
            spread_vol_ratio = spread_pct / return_std_short

        if self.exclude_microstructure:
            mid_price = close
            best_bid = close
            best_ask = close
            best_bid_size = 0.0
            best_ask_size = 0.0
            spread_abs = 0.0
            spread_bps = 0.0
            spread_vol_ratio = 0.0
            has_spread = 0.0
            imbalance_top = 0.0
            liquidity_imbalance_25 = 0.0
            liquidity_imbalance_50 = 0.0
            liquidity_imbalance_100 = 0.0
            liquidity_density_25 = 0.0
            liquidity_density_50 = 0.0
            liquidity_density_100 = 0.0
            liquidity_pressure = 0.0
            liquidity_tier_ratio_50_25 = 0.0
            liquidity_tier_ratio_100_25 = 0.0
            liquidity_tier_ratio_100_50 = 0.0
            bid_depth_25 = 0.0
            ask_depth_25 = 0.0
            bid_depth_50 = 0.0
            ask_depth_50 = 0.0
            bid_depth_100 = 0.0
            ask_depth_100 = 0.0
            has_depth_25 = 0.0
            has_depth_50 = 0.0
            has_depth_100 = 0.0

        close_history = list(self._close_history)
        volume_history = list(self._volume_history)

        def _log_return_lag(steps: int) -> float:
            if len(close_history) <= steps:
                return 0.0
            base = close_history[-steps - 1]
            return math.log(max(close, 1e-12) / max(base, 1e-12))

        return_lag_5 = _log_return_lag(5)
        return_lag_15 = _log_return_lag(15)
        return_lag_60 = _log_return_lag(60)
        return_lag_240 = _log_return_lag(240)

        def _donchian(window: int) -> tuple[float, float]:
            if not close_history:
                return close, close
            window_data = close_history[-window:] if len(close_history) >= window else close_history
            return max(window_data), min(window_data)

        high_short, low_short = _donchian(60)
        high_long, low_long = _donchian(240)

        def _position(high_val: float, low_val: float) -> float:
            span = max(high_val - low_val, 1e-12)
            return (close - low_val) / span

        donchian_pos_short = _position(high_short, low_short)
        donchian_pos_long = _position(high_long, low_long)

        def _volume_percentile(window: int) -> float:
            if not volume_history:
                return 0.0
            window_data = volume_history[-window:] if len(volume_history) >= window else volume_history
            below = sum(1 for v in window_data if v <= volume)
            return below / len(window_data)

        volume_percentile_short = _volume_percentile(60)
        volume_percentile_long = _volume_percentile(240)

        regime_volatility = return_std_long
        regime_sharpe = 0.0
        if regime_volatility > 1e-8:
            regime_sharpe = (return_long / regime_volatility) * math.sqrt(240)

        trend_bias = math.tanh(regime_sharpe / 2.0)

        volatility_ratio = 0.0
        if regime_volatility > 1e-12:
            volatility_ratio = return_std_medium / max(regime_volatility, 1e-12)

        volatility_gap = return_std_long - volatility_mean_long
        volatility_zscore_long = 0.0
        if volatility_std_long > 1e-12:
            volatility_zscore_long = volatility_gap / volatility_std_long

        atr_pct = 0.0
        if close > 0:
            atr_pct = atr / close

        range_abs = high - low
        range_pct = range_abs / max(close, 1e-8)
        ema_diff_pct = ema_diff / max(abs(price_slow), 1e-8)
        range_vol_ratio = range_abs / max(return_std_short * close, 1e-8)

        regime_label = 0
        regime_state_bull = 0.0
        regime_state_bear = 0.0
        regime_state_sideways = 0.0

        cluster_inputs = {
            "volatility": max(regime_volatility, 0.0),
            "liquidity": max(liquidity_density_50, 0.0),
            "imbalance": liquidity_imbalance_50,
            "trend": trend_bias,
            "funding": funding_diff,
        }

        if self.regime_clusterer is not None:
            try:
                predicted_cluster = self.regime_clusterer.predict_one(cluster_inputs)
            except (AttributeError, OverflowError, ValueError):
                predicted_cluster = None
            assigned_cluster = 0 if predicted_cluster is None else int(predicted_cluster)
            try:
                self.regime_clusterer.learn_one(cluster_inputs)
            except (AttributeError, OverflowError, ValueError):
                self.regime_clusterer = None
            regime_label = assigned_cluster

            stats = self._regime_stats.setdefault(regime_label, {"trend_sum": 0.0, "count": 0.0})
            stats["trend_sum"] += trend_bias
            stats["count"] += 1.0

            averages: Dict[int, float] = {}
            for cluster_id, cluster_stats in self._regime_stats.items():
                avg_trend = 0.0
                if cluster_stats["count"] > 0:
                    avg_trend = cluster_stats["trend_sum"] / cluster_stats["count"]
                averages[cluster_id] = avg_trend

            if averages:
                bull_label = max(averages, key=averages.get)
                bear_label = min(averages, key=averages.get)
                sorted_labels = sorted(averages, key=averages.get)
                sideways_label = sorted_labels[len(sorted_labels) // 2]
                regime_state_bull = 1.0 if regime_label == bull_label else 0.0
                regime_state_bear = 1.0 if regime_label == bear_label else 0.0
                regime_state_sideways = 1.0 if regime_label == sideways_label else 0.0
        else:
            if trend_bias > 0.1:
                regime_label = 2
                regime_state_bull = 1.0
            elif trend_bias < -0.1:
                regime_label = 0
                regime_state_bear = 1.0
            else:
                regime_label = 1
                regime_state_sideways = 1.0

        feature_vector: Dict[str, float] = {
            "return_1m": minute_return,
            "return_fast": return_fast,
            "return_medium": return_medium,
            "return_slow": return_slow,
            "return_long": return_long,
            "return_std_short": return_std_short,
            "return_std_medium": return_std_medium,
            "return_std_long": return_std_long,
            "momentum_fast_slow": return_fast - return_slow,
            "momentum_medium_slow": return_medium - return_slow,
            "momentum_long_slow": return_long - return_slow,
            "momentum_fast_long": return_fast - return_long,
            "price_zscore_fast": price_zscore_fast,
            "volume_ratio": volume_ratio,
            "volume_trend": volume_trend,
            "volume_zscore_fast": volume_zscore_fast,
            "volume_percentile_short": volume_percentile_short,
            "volume_percentile_long": volume_percentile_long,
            "range_pct": range_pct,
            "range_vol_ratio": range_vol_ratio,
            "ema_ratio": ema_ratio,
            "ema_diff_pct": ema_diff_pct,
            "atr_pct": atr_pct,
            "rsi": rsi,
            "regime_sharpe": regime_sharpe,
            "trend_bias": trend_bias,
            "regime_label": float(regime_label),
            "regime_state_bull": regime_state_bull,
            "regime_state_bear": regime_state_bear,
            "regime_state_sideways": regime_state_sideways,
            "volatility_regime": regime_volatility,
            "volatility_ratio": volatility_ratio,
            "volatility_baseline": volatility_mean_long,
            "volatility_gap": volatility_gap,
            "volatility_zscore_long": volatility_zscore_long,
            "volatility_slope": volatility_slope,
            "realized_vol_slope_short": volatility_slope,
            "realized_vol_slope_medium": volatility_slope_medium,
            "realized_vol_trend": realized_vol_trend,
            "return_lag_5": return_lag_5,
            "return_lag_15": return_lag_15,
            "return_lag_60": return_lag_60,
            "return_lag_240": return_lag_240,
            "donchian_pos_short": donchian_pos_short,
            "donchian_pos_long": donchian_pos_long,
            "micro_mid_price": mid_price,
            "micro_best_bid": best_bid,
            "micro_best_ask": best_ask,
            "micro_best_bid_size": best_bid_size,
            "micro_best_ask_size": best_ask_size,
            "micro_spread_abs": spread_abs,
            "micro_spread_bps": spread_bps,
            "micro_spread_vol_ratio": spread_vol_ratio,
            "microstructure_spread_available": has_spread,
            "orderbook_imbalance_top": imbalance_top,
            "liquidity_imbalance_25bps": liquidity_imbalance_25,
            "liquidity_imbalance_50bps": liquidity_imbalance_50,
            "liquidity_imbalance_100bps": liquidity_imbalance_100,
            "liquidity_density_25bps": liquidity_density_25,
            "liquidity_density_50bps": liquidity_density_50,
            "liquidity_density_100bps": liquidity_density_100,
            "liquidity_pressure": liquidity_pressure,
            "liquidity_tier_ratio_50_25": liquidity_tier_ratio_50_25,
            "liquidity_tier_ratio_100_25": liquidity_tier_ratio_100_25,
            "liquidity_tier_ratio_100_50": liquidity_tier_ratio_100_50,
            "liquidity_bid_depth_25bps": bid_depth_25,
            "liquidity_ask_depth_25bps": ask_depth_25,
            "liquidity_bid_depth_50bps": bid_depth_50,
            "liquidity_ask_depth_50bps": ask_depth_50,
            "liquidity_bid_depth_100bps": bid_depth_100,
            "liquidity_ask_depth_100bps": ask_depth_100,
            "liquidity_depth_25_available": has_depth_25,
            "liquidity_depth_50_available": has_depth_50,
            "liquidity_depth_100_available": has_depth_100,
            "orderbook_imbalance_25bps": liquidity_imbalance_25,
            "orderbook_imbalance_50bps": liquidity_imbalance_50,
            "orderbook_imbalance_100bps": liquidity_imbalance_100,
            "funding_rate": funding_rate,
            "funding_rate_mean": funding_mean,
            "funding_rate_diff": funding_diff,
            "funding_rate_change": funding_change,
            "funding_rate_zscore": funding_zscore,
            "funding_rate_ema": funding_ema,
            "funding_available": funding_available,
            "funding_rate_alt": funding_rate_alt,
            "funding_rate_alt_mean": funding_alt_mean,
            "funding_rate_diff_alt": funding_alt_diff,
            "funding_rate_alt_change": funding_alt_change,
            "funding_rate_alt_diff_mean": funding_alt_diff_mean,
            "funding_rate_basis": funding_rate_basis,
            "funding_rate_basis_mean": funding_basis_mean,
            "funding_rate_diff_basis": funding_basis_diff,
            "funding_rate_basis_change": funding_basis_change,
            "funding_rate_basis_diff_mean": funding_basis_diff_mean,
        }

        self._prev_close = close
        return feature_vector