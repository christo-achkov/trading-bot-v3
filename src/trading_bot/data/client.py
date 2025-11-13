"""Data source clients."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, Protocol


class Candle(Protocol):
    """Protocol describing a single OHLCV candle."""

    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataClient(Protocol):
    """Abstract client interface for fetching market data."""

    def fetch_candles(self, *, symbol: str, interval: str, start: datetime, end: datetime) -> Iterable[Candle]:
        """Yield candles for the given range without loading everything into memory."""
        raise NotImplementedError
