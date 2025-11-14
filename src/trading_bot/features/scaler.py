"""Online feature scaling utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict


@dataclass
class RunningMoment:
    """Track running mean and second central moment for a single feature."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    def scale(self, value: float, *, epsilon: float) -> float:
        if self.count < 2:
            return 0.0
        std = math.sqrt(max(self.variance(), epsilon))
        if std < epsilon:
            return 0.0
        return (value - self.mean) / std


class OnlineFeatureScaler:
    """Maintain per-feature standardization statistics on the fly."""

    def __init__(self, epsilon: float = 1e-4) -> None:
        self._epsilon = epsilon
        self._moments: Dict[str, RunningMoment] = {}

    def transform(self, features: Dict[str, float]) -> Dict[str, float]:
        """Update internal stats and return standardized values."""

        scaled: Dict[str, float] = {}
        for name, value in features.items():
            moment = self._moments.get(name)
            if moment is None:
                moment = RunningMoment()
                self._moments[name] = moment
            moment.update(value)
            scaled[name] = moment.scale(value, epsilon=self._epsilon)
        return scaled

    def reset(self) -> None:
        """Clear accumulated statistics."""

        self._moments.clear()
