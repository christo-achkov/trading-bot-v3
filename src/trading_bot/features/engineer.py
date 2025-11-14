"""Feature engineering utilities operating on streaming candle data."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict


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
    atr_tracker: AverageTrueRange = field(default_factory=lambda: AverageTrueRange(14))
    rsi_tracker: StreamingRSI = field(default_factory=lambda: StreamingRSI(14))

    _close_history: Deque[float] = field(default_factory=lambda: deque(maxlen=720))
    _volume_history: Deque[float] = field(default_factory=lambda: deque(maxlen=720))
    _prev_close: float | None = None

    def process(self, candle: Dict[str, float]) -> Dict[str, float]:
        """Update internal state and return the latest feature vector."""

        close = float(candle["close"])
        high = float(candle.get("high", close))
        low = float(candle.get("low", close))
        volume = float(candle.get("volume", 0.0))

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

        price_fast = self.price_fast.update(close)
        price_medium = self.price_medium.update(close)
        price_slow = self.price_slow.update(close)
        price_std_fast = self.price_std_fast.update(close)

        volume_fast = self.volume_fast.update(volume)
        volume_medium = self.volume_medium.update(volume)
        volume_slow = self.volume_slow.update(volume)
        volume_std_fast = self.volume_std_fast.update(volume)
        volume_std_slow = self.volume_std_slow.update(volume)

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

        self._close_history.append(close)
        self._volume_history.append(volume)

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

        atr_pct = 0.0
        if close > 0:
            atr_pct = atr / close

        range_abs = high - low
        range_pct = range_abs / max(close, 1e-8)
        ema_diff_pct = ema_diff / max(abs(price_slow), 1e-8)
        range_vol_ratio = range_abs / max(return_std_short * close, 1e-8)

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
            "volatility_regime": regime_volatility,
            "volatility_ratio": volatility_ratio,
            "return_lag_5": return_lag_5,
            "return_lag_15": return_lag_15,
            "return_lag_60": return_lag_60,
            "return_lag_240": return_lag_240,
            "donchian_pos_short": donchian_pos_short,
            "donchian_pos_long": donchian_pos_long,
        }

        self._prev_close = close
        return feature_vector