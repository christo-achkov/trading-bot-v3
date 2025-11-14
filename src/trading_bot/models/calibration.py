"""Lightweight online calibration utilities."""
from __future__ import annotations

import math
from bisect import bisect_left
from collections import deque
from typing import Deque, List, Tuple


class OnlineIsotonicCalibrator:
    """Approximate isotonic regression over a sliding window of predictions."""

    def __init__(self, window_size: int = 1024, min_samples: int = 50) -> None:
        self.window_size = max(int(window_size), 1)
        self.min_samples = max(int(min_samples), 1)
        self._buffer: Deque[Tuple[float, float]] = deque(maxlen=self.window_size)
        self._centers: List[float] = []
        self._values: List[float] = []
        self._dirty = True

    def learn_one(self, x, y) -> "OnlineIsotonicCalibrator":
        """Ingest a new (prediction, target) pair."""

        if x is None:
            return self

        value = x.get("prediction") if isinstance(x, dict) else None
        if value is None:
            return self

        try:
            pred = float(value)
            target = float(y)
        except (TypeError, ValueError):
            return self

        if not (math.isfinite(pred) and math.isfinite(target)):
            return self

        self._buffer.append((pred, target))
        self._dirty = True
        return self

    def predict_one(self, x):
        """Calibrate a raw prediction if enough history is available."""

        if x is None:
            return None

        value = x.get("prediction") if isinstance(x, dict) else None
        if value is None:
            return None

        try:
            pred = float(value)
        except (TypeError, ValueError):
            return value

        if not math.isfinite(pred):
            return pred

        if len(self._buffer) < self.min_samples:
            return pred

        if self._dirty:
            self._fit()

        if not self._centers:
            return pred

        idx = bisect_left(self._centers, pred)
        if idx <= 0:
            return self._values[0]
        if idx >= len(self._centers):
            return self._values[-1]

        x0 = self._centers[idx - 1]
        x1 = self._centers[idx]
        y0 = self._values[idx - 1]
        y1 = self._values[idx]

        if x1 == x0:
            return y0

        weight = (pred - x0) / (x1 - x0)
        return y0 + weight * (y1 - y0)

    def _fit(self) -> None:
        """Recompute the isotonic mapping using pool-adjacent-violators."""

        if len(self._buffer) < self.min_samples:
            self._centers = []
            self._values = []
            self._dirty = False
            return

        sorted_pairs = sorted(self._buffer, key=lambda pair: pair[0])
        blocks: List[dict[str, float]] = []

        for pred, target in sorted_pairs:
            block = {
                "min": pred,
                "max": pred,
                "sum_y": target,
                "count": 1.0,
            }
            blocks.append(block)

            while len(blocks) >= 2:
                prev = blocks[-2]
                curr = blocks[-1]
                prev_mean = prev["sum_y"] / prev["count"]
                curr_mean = curr["sum_y"] / curr["count"]
                if prev_mean <= curr_mean:
                    break
                merged = {
                    "min": prev["min"],
                    "max": curr["max"],
                    "sum_y": prev["sum_y"] + curr["sum_y"],
                    "count": prev["count"] + curr["count"],
                }
                blocks[-2:] = [merged]

        self._centers = [(block["min"] + block["max"]) / 2.0 for block in blocks]
        self._values = [block["sum_y"] / block["count"] for block in blocks]
        self._dirty = False


def isotonic_calibrator(window_size: int = 1024, min_samples: int = 50) -> OnlineIsotonicCalibrator:
    """Factory helper returning a configured online isotonic calibrator."""

    return OnlineIsotonicCalibrator(window_size=window_size, min_samples=min_samples)


__all__ = ["OnlineIsotonicCalibrator", "isotonic_calibrator"]
