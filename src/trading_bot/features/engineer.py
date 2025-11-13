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


@dataclass
class OnlineFeatureBuilder:
    """Construct derived features incrementally from streaming candles."""

    return_fast: RollingMean = field(default_factory=lambda: RollingMean(5))
    return_medium: RollingMean = field(default_factory=lambda: RollingMean(30))
    return_slow: RollingMean = field(default_factory=lambda: RollingMean(120))
    return_std: RollingStd = field(default_factory=lambda: RollingStd(120))
    price_fast: RollingMean = field(default_factory=lambda: RollingMean(20))
    price_slow: RollingMean = field(default_factory=lambda: RollingMean(60))
    price_std_fast: RollingStd = field(default_factory=lambda: RollingStd(20))
    volume_mean: RollingMean = field(default_factory=lambda: RollingMean(60))

    _prev_close: float | None = None

    def process(self, candle: Dict[str, float]) -> Dict[str, float]:
        """Update internal state and return the latest feature vector."""

        close = float(candle["close"])
        high = float(candle.get("high", close))
        low = float(candle.get("low", close))
        volume = float(candle.get("volume", 0.0))

        if self._prev_close is None:
            minute_return = 0.0
        else:
            minute_return = math.log(max(close, 1e-12) / max(self._prev_close, 1e-12))

        self._prev_close = close

        return_fast = self.return_fast.update(minute_return)
        return_medium = self.return_medium.update(minute_return)
        return_slow = self.return_slow.update(minute_return)
        return_std = self.return_std.update(minute_return)

        price_fast = self.price_fast.update(close)
        price_slow = self.price_slow.update(close)
        price_std_fast = self.price_std_fast.update(close)
        volume_mean = self.volume_mean.update(volume)

        price_zscore = 0.0
        if price_std_fast > 0:
            price_zscore = (close - price_fast) / price_std_fast

        volume_ratio = 0.0
        if volume_mean > 0:
            volume_ratio = volume / volume_mean

        feature_vector: Dict[str, float] = {
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "return_1m": minute_return,
            "return_fast": return_fast,
            "return_medium": return_medium,
            "return_slow": return_slow,
            "return_std": return_std,
            "momentum_diff_fast_slow": return_fast - return_slow,
            "momentum_diff_medium_slow": return_medium - return_slow,
            "price_fast_ma": price_fast,
            "price_slow_ma": price_slow,
            "price_zscore_fast": price_zscore,
            "volume_ratio": volume_ratio,
            "range": high - low,
            "range_pct": (high - low) / max(close, 1e-6),
        }

        return feature_vector