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
    return_long: RollingMean = field(default_factory=lambda: RollingMean(240))
    return_std: RollingStd = field(default_factory=lambda: RollingStd(120))
    return_std_long: RollingStd = field(default_factory=lambda: RollingStd(240))
    price_fast: RollingMean = field(default_factory=lambda: RollingMean(20))
    price_slow: RollingMean = field(default_factory=lambda: RollingMean(60))
    price_std_fast: RollingStd = field(default_factory=lambda: RollingStd(20))
    volume_mean: RollingMean = field(default_factory=lambda: RollingMean(60))
    volume_std: RollingStd = field(default_factory=lambda: RollingStd(60))
    price_ema_fast: ExponentialMovingAverage = field(default_factory=lambda: ExponentialMovingAverage(12))
    price_ema_slow: ExponentialMovingAverage = field(default_factory=lambda: ExponentialMovingAverage(26))
    atr_tracker: AverageTrueRange = field(default_factory=lambda: AverageTrueRange(14))
    rsi_tracker: StreamingRSI = field(default_factory=lambda: StreamingRSI(14))

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
        if ema_slow != 0:
            ema_ratio = (ema_fast / ema_slow) - 1.0

        rsi = self.rsi_tracker.update(close - previous_close if previous_close is not None else 0.0)
        atr = self.atr_tracker.update(high, low, close, previous_close)

        return_fast = self.return_fast.update(minute_return)
        return_medium = self.return_medium.update(minute_return)
        return_slow = self.return_slow.update(minute_return)
        return_long = self.return_long.update(minute_return)
        return_std = self.return_std.update(minute_return)
        return_std_long = self.return_std_long.update(minute_return)

        price_fast = self.price_fast.update(close)
        price_slow = self.price_slow.update(close)
        price_std_fast = self.price_std_fast.update(close)
        volume_mean = self.volume_mean.update(volume)
        volume_std = self.volume_std.update(volume)

        price_zscore = 0.0
        if price_std_fast > 0:
            price_zscore = (close - price_fast) / price_std_fast

        volume_ratio = 0.0
        if volume_mean > 0:
            volume_ratio = volume / volume_mean

        volume_zscore = 0.0
        if volume_std > 0:
            volume_zscore = (volume - volume_mean) / volume_std

        regime_vol = return_std_long
        regime_sharpe = 0.0
        if regime_vol > 1e-8:
            regime_sharpe = (return_long / regime_vol) * math.sqrt(240)

        trend_bias = math.tanh(regime_sharpe / 2.0)

        atr_pct = 0.0
        if close > 0:
            atr_pct = atr / close

        feature_vector: Dict[str, float] = {
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "return_1m": minute_return,
            "return_fast": return_fast,
            "return_medium": return_medium,
            "return_slow": return_slow,
            "return_long": return_long,
            "return_std": return_std,
            "return_std_long": return_std_long,
            "momentum_diff_fast_slow": return_fast - return_slow,
            "momentum_diff_medium_slow": return_medium - return_slow,
            "price_fast_ma": price_fast,
            "price_slow_ma": price_slow,
            "price_zscore_fast": price_zscore,
            "volume_ratio": volume_ratio,
            "range": high - low,
            "range_pct": (high - low) / max(close, 1e-6),
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "ema_diff": ema_diff,
            "ema_ratio": ema_ratio,
            "atr": atr,
            "atr_pct": atr_pct,
            "rsi": rsi,
            "volume_zscore": volume_zscore,
            "regime_sharpe": regime_sharpe,
            "trend_bias": trend_bias,
            "volatility_regime": regime_vol,
        }

        self._prev_close = close
        return feature_vector