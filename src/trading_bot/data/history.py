"""Helpers for fetching historical candles directly from Binance."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Sequence

from trading_bot.data.binance_client import BinanceRESTClient
from trading_bot.data.binance_downloader import BinanceDownloader, CandleBatch


def fetch_candles(
    client: BinanceRESTClient,
    *,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    chunk_minutes: int,
) -> List[dict]:
    """Retrieve candles between two timestamps (inclusive) in memory."""

    downloader = BinanceDownloader(
        client,
        symbol=symbol,
        interval=interval,
        chunk_minutes=chunk_minutes,
        microstructure_provider=None,
    )

    collected: List[dict] = []
    for batch in downloader.stream(start=start, end=end):
        _append_batch(collected, batch)
    return collected


def _append_batch(target: List[dict], batch: CandleBatch) -> None:
    """Extend the target list with the batch while ensuring ordering."""

    target.extend(batch.candles)


def fetch_candles_iter(
    client: BinanceRESTClient,
    *,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    chunk_minutes: int,
) -> Iterable[dict]:
    """Yield candles lazily using the Binance downloader."""

    downloader = BinanceDownloader(
        client,
        symbol=symbol,
        interval=interval,
        chunk_minutes=chunk_minutes,
        microstructure_provider=None,
    )

    for batch in downloader.stream(start=start, end=end):
        for candle in batch.candles:
            yield candle


__all__: Sequence[str] = ("fetch_candles", "fetch_candles_iter")
