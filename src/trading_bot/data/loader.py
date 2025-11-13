"""Utilities for reading stored candle data."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import pandas as pd
import pyarrow.dataset as ds

from trading_bot.utils import ensure_utc

CANDLE_COLUMNS = [
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
]


def iter_candles_from_parquet(
    root_dir: Path,
    *,
    symbol: str,
    interval: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    batch_size: int = 10_000,
) -> Iterator[Dict[str, float]]:
    """Yield candle dictionaries from a partitioned Parquet dataset."""

    dataset, filter_expr = _dataset_and_filter(
        root_dir,
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
    )

    scanner = dataset.scanner(columns=CANDLE_COLUMNS, filter=filter_expr, batch_size=batch_size)

    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        # ensure timezone awareness
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
        frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True)

        for record in frame.to_dict(orient="records"):
            yield record


def count_candles_in_parquet(
    root_dir: Path,
    *,
    symbol: str,
    interval: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> int:
    """Return the number of candles matching the provided filters."""

    dataset, filter_expr = _dataset_and_filter(
        root_dir,
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
    )
    return dataset.count_rows(filter=filter_expr)


def _dataset_and_filter(
    root_dir: Path,
    *,
    symbol: str,
    interval: str,
    start: Optional[datetime],
    end: Optional[datetime],
) -> Tuple[ds.Dataset, ds.Expression]:
    """Build a dataset handle and matching filter expression."""

    dataset = ds.dataset(str(root_dir), format="parquet", partitioning="hive")

    filter_expr: ds.Expression = (ds.field("symbol") == symbol) & (ds.field("interval") == interval)

    if start is not None:
        filter_expr = filter_expr & (ds.field("open_time") >= ensure_utc(start))
    if end is not None:
        filter_expr = filter_expr & (ds.field("open_time") <= ensure_utc(end))

    return dataset, filter_expr