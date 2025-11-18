"""Helpers for reading enriched candles that include order book information."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from trading_bot.utils import parse_iso8601


def has_enriched_coverage(
    root: Path,
    *,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> bool:
    """Return True if enriched data exists for every day in the given range."""

    start_day = _ensure_utc(start).date()
    end_day = _ensure_utc(end).date()
    cursor = start_day

    while cursor <= end_day:
        day_dir = _day_directory(root, symbol, interval, cursor.year, cursor.month, cursor.day)
        if not day_dir.exists() or not any(day_dir.glob("*.parquet")):
            return False
        cursor += timedelta(days=1)

    return True


def iter_enriched_candles(
    root: Path,
    *,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> Iterator[dict]:
    """Yield enriched candle records stored on disk within the requested range."""

    start_utc = _ensure_utc(start)
    end_utc = _ensure_utc(end)

    cursor = start_utc.date()
    end_day = end_utc.date()

    while cursor <= end_day:
        day_dir = _day_directory(root, symbol, interval, cursor.year, cursor.month, cursor.day)
        if day_dir.exists():
            yield from _load_day(day_dir, start_utc, end_utc)
        cursor += timedelta(days=1)


# Internal utilities ----------------------------------------------------


def _day_directory(root: Path, symbol: str, interval: str, year: int, month: int, day: int) -> Path:
    return (
        root
        / f"symbol={symbol.upper()}"
        / f"interval={interval}"
        / f"year={year}"
        / f"month={month:02d}"
        / f"day={day:02d}"
    )


def _load_day(day_dir: Path, start: datetime, end: datetime) -> Iterator[dict]:
    rows: list[dict] = []
    for file_path in sorted(day_dir.glob("*.parquet")):
        table = pq.read_table(file_path)
        rows.extend(table.to_pylist())

    rows.sort(key=lambda row: _extract_time(row) or start)

    seen_times: set[str] = set()
    for row in rows:
        record_time = _extract_time(row)
        if record_time is None:
            continue
        if record_time < start or record_time > end:
            continue
        key = record_time.isoformat()
        if key in seen_times:
            continue
        seen_times.add(key)
        yield row


def _extract_time(record: dict) -> datetime | None:
    timestamp_value = record.get("close_time") or record.get("event_time") or record.get("open_time")
    if timestamp_value is None:
        return None
    if isinstance(timestamp_value, datetime):
        return _ensure_utc(timestamp_value)
    if isinstance(timestamp_value, str):
        try:
            parsed = parse_iso8601(timestamp_value)
        except ValueError:
            return None
        return _ensure_utc(parsed)
    return None


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = ["has_enriched_coverage", "iter_enriched_candles"]
