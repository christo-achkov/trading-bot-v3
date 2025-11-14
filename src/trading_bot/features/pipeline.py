"""Feature pipelines built on top of River transformers."""
from __future__ import annotations

from river import compose, feature_extraction, stats


def default_feature_pipeline() -> compose.Pipeline:
    """Compose a default online feature pipeline for BTC minute candles."""

    return compose.Pipeline(
        (
            "returns",
            feature_extraction.Agg(
                feature="close",
                agg=stats.RollingMean(window_size=30),
            ),
        ),
    )
