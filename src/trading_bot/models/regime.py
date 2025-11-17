"""Utilities for regime-aware model management."""
from __future__ import annotations

import math
from typing import Callable, Dict, Hashable, Optional, Protocol


class SupportsPredictLearn(Protocol):
    """Protocol for River-like estimators used in the trading engine."""

    def predict_one(self, x):
        ...

    def learn_one(self, x, y):
        ...


class RegimeAwareWrapper:
    """Wrap a base estimator so each regime label maintains an independent model."""

    def __init__(
        self,
        factory: Callable[[], SupportsPredictLearn],
        *,
        regime_feature: str = "regime_label",
        fallback_label: Optional[int] = None,
    ) -> None:
        self._factory = factory
        self._regime_feature = regime_feature
        self._fallback_label = fallback_label
        self._models: Dict[Hashable, SupportsPredictLearn] = {}
        self._fallback_model: SupportsPredictLearn | None = None

    def predict_one(self, x):  # type: ignore[override]
        model = self._resolve_model(x)
        return model.predict_one(x) if model is not None else 0.0

    def learn_one(self, x, y):  # type: ignore[override]
        model = self._resolve_model(x, create=True)
        if model is not None:
            model.learn_one(x, y)
        return self

    # ---------------------------------------------------------------------
    def _resolve_model(self, x, *, create: bool = False) -> SupportsPredictLearn | None:
        label = self._extract_label(x)
        if label is None:
            if self._fallback_model is None and create:
                self._fallback_model = self._factory()
            return self._fallback_model

        if label not in self._models and create:
            self._models[label] = self._factory()
        return self._models.get(label) or self._fallback_model

    def _extract_label(self, x) -> Optional[int]:
        if not isinstance(x, dict):
            return self._fallback_label
        raw_value = x.get(self._regime_feature)
        if raw_value is None:
            return self._fallback_label
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return self._fallback_label
        if not math.isfinite(value):
            return self._fallback_label
        try:
            return int(round(value))
        except (TypeError, ValueError):
            return self._fallback_label


__all__ = ["RegimeAwareWrapper", "SupportsPredictLearn"]
