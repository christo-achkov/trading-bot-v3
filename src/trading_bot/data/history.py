"""Helpers for fetching historical candles directly from Binance."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

from trading_bot.data.binance_client import BinanceRESTClient
from trading_bot.data.binance_downloader import BinanceDownloader, CandleBatch
from trading_bot.data.enriched import has_enriched_coverage, iter_enriched_candles
from trading_bot.utils import ensure_utc


def fetch_candles(
    client: BinanceRESTClient,
    *,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    chunk_minutes: int,
    enriched_root: Path | None = None,
    prefer_enriched: bool = False,
) -> List[dict]:
    """Retrieve candles between two timestamps (inclusive) in memory."""

    if prefer_enriched and enriched_root is not None:
        start_utc = ensure_utc(start)
        end_utc = ensure_utc(end)
        if has_enriched_coverage(enriched_root, symbol=symbol, interval=interval, start=start_utc, end=end_utc):
            return list(
                iter_enriched_candles(
                    enriched_root,
                    symbol=symbol,
                    interval=interval,
                    start=start_utc,
                    end=end_utc,
                )
            )

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
    enriched_root: Path | None = None,
    prefer_enriched: bool = False,
) -> Iterable[dict]:
    """Yield candles lazily using the Binance downloader."""

    if prefer_enriched and enriched_root is not None:
        start_utc = ensure_utc(start)
        end_utc = ensure_utc(end)
        if has_enriched_coverage(enriched_root, symbol=symbol, interval=interval, start=start_utc, end=end_utc):
            yield from iter_enriched_candles(
                enriched_root,
                symbol=symbol,
                interval=interval,
                start=start_utc,
                end=end_utc,
            )
            return

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
