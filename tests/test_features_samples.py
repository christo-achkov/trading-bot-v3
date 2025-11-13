"""Tests for feature sample generation."""
from __future__ import annotations

from trading_bot.features import OnlineFeatureBuilder, iter_supervised_samples


def test_iter_supervised_samples_produces_labels() -> None:
    candles = [
        {"close": 100.0, "high": 101.0, "low": 99.0, "volume": 10.0},
        {"close": 101.0, "high": 102.0, "low": 100.0, "volume": 12.0},
        {"close": 100.5, "high": 101.5, "low": 99.5, "volume": 11.0},
    ]

    samples = list(iter_supervised_samples(candles, builder=OnlineFeatureBuilder()))

    assert len(samples) == 2
    first_features, first_label = samples[0]
    assert "return_1m" in first_features
    assert first_label == 1
    _, second_label = samples[1]
    assert second_label == -1
