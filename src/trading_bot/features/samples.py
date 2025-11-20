"""Helpers for turning candle streams into supervised learning samples."""
from __future__ import annotations

import math
from typing import Dict, Iterable, Iterator, Tuple

from trading_bot.features.engineer import OnlineFeatureBuilder


def iter_supervised_samples(
    candles: Iterable[Dict[str, float]],
    *,
    builder: OnlineFeatureBuilder | None = None,
    label_threshold: float = 0.0,
) -> Iterator[Tuple[Dict[str, float], int]]:
    """Yield `(features, label)` tuples from a candle iterator.

    The label encodes the direction of the next return:
      * `1` when the next return exceeds ``label_threshold``
      * `-1` when the next return is below ``-label_threshold``
      * `0` otherwise
    """

    feature_builder = builder or OnlineFeatureBuilder()

    pending_features: Dict[str, float] | None = None
    previous_close: float | None = None

    for candle in candles:
        features = feature_builder.process(candle)
        close = float(candle["close"])

        if pending_features is not None and previous_close is not None:
            next_return = math.log(max(close, 1e-12) / max(previous_close, 1e-12))
            if next_return > label_threshold:
                label = 1
            elif next_return < -label_threshold:
                label = -1
            else:
                label = 0

            yield pending_features, label

        pending_features = features
        previous_close = close

    # the final candle has no lookahead label and is therefore discarded