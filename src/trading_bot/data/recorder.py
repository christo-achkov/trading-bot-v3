"""Utilities for persisting enriched market records to Parquet."""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

import logging

from trading_bot.utils import parse_iso8601

_LOGGER = logging.getLogger(__name__)


class ParquetRecorder:
    """Append-only writer that stores enriched candles grouped by trading day."""

    def __init__(
        self,
        root: Path,
        *,
        symbol: str,
        interval: str,
        batch_size: int = 256,
    ) -> None:
        self._root = Path(root)
        self._symbol = symbol.upper()
        self._interval = interval
        self._batch_size = max(int(batch_size), 1)
        self._buffer: list[dict[str, Any]] = []
        self._current_day: date | None = None
        self._seen_keys: set[str] = set()
        self._root.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        """Queue a record for persistence, flushing when thresholds are met."""

        record_day = self._extract_day(record)
        if self._current_day is None:
            self._current_day = record_day
        elif record_day != self._current_day:
            self._flush()
            self._seen_keys.clear()
            self._current_day = record_day

        key = self._compute_key(record)
        if key is not None and key in self._seen_keys:
            return
        if key is not None:
            self._seen_keys.add(key)

        self._buffer.append(self._normalise_record(record))

        if len(self._buffer) >= self._batch_size:
            self._flush()

    def close(self) -> None:
        """Flush any buffered data to disk."""

        self._flush()
        self._seen_keys.clear()

    def read_recent(self, *, limit: int = 2000) -> list[dict[str, Any]]:
        """Read recent persisted records for this symbol/interval.

        Returns records sorted by timestamp ascending. This is a best-effort reader
        intended for warm-starting live sessions; it does not consult remote REST.
        """
        out: list[dict[str, Any]] = []
        # Walk the directory for the symbol/interval and read parquet files.
        symbol_dir = self._root / f"symbol={self._symbol}" / f"interval={self._interval}"
        if not symbol_dir.exists():
            return []

        # Collect parquet files under the symbol/interval tree
        files = list(symbol_dir.rglob("*.parquet"))
        if not files:
            _LOGGER.debug("ParquetRecorder.read_recent: no parquet files under %s", symbol_dir)
            return []

        # Read newest first until we've gathered enough
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        _LOGGER.debug("ParquetRecorder.read_recent: found %d parquet files under %s", len(files), symbol_dir)
        for path in files:
            try:
                table = pq.read_table(path)
                records = table.to_pylist()
                out.extend(records)
                if len(out) >= limit:
                    break
            except Exception as exc:
                _LOGGER.warning("ParquetRecorder.read_recent: failed to read %s: %s", path, exc)
                # Try a fallback: read all columns except 'symbol' which sometimes has
                # mixed types across files (string vs nested dictionary). Dropping it
                # preserves timestamps/prices for warm-start.
                try:
                    pf = pq.ParquetFile(path)
                    names = list(pf.schema.names)
                    cols = [n for n in names if n != "symbol"]
                    if not cols:
                        continue
                    table = pf.read(columns=cols)
                    records = table.to_pylist()
                    # ensure records don't include 'symbol'
                    out.extend(records)
                    if len(out) >= limit:
                        break
                except Exception as exc2:
                    _LOGGER.warning("ParquetRecorder.read_recent: fallback read failed %s: %s", path, exc2)
                    continue

        if not out:
            return []

        # ensure ascending time order
        def _ts_key(rec: dict[str, Any]):
            return rec.get("close_time") or rec.get("event_time") or rec.get("open_time")

        out.sort(key=_ts_key)
        return out[-limit:]

    # Internal helpers -------------------------------------------------

    def _flush(self) -> None:
        if not self._buffer or self._current_day is None:
            self._buffer.clear()
            return

        day_path = (
            self._root
            / f"symbol={self._symbol}"
            / f"interval={self._interval}"
            / f"year={self._current_day.year}"
            / f"month={self._current_day.month:02d}"
            / f"day={self._current_day.day:02d}"
        )
        day_path.mkdir(parents=True, exist_ok=True)

        file_path = day_path / f"{uuid.uuid4().hex}.parquet"
        table = pa.Table.from_pylist(self._buffer)
        pq.write_table(table, file_path, compression="snappy")

        self._buffer.clear()
        # Retain current_day so subsequent records on the same day reuse the directory.

    def _extract_day(self, record: dict[str, Any]) -> date:
        timestamp_value = record.get("close_time") or record.get("event_time") or record.get("open_time")
        if timestamp_value is None:
            return datetime.now(timezone.utc).date()
        if isinstance(timestamp_value, datetime):
            aware = timestamp_value if timestamp_value.tzinfo else timestamp_value.replace(tzinfo=timezone.utc)
            return aware.astimezone(timezone.utc).date()
        if isinstance(timestamp_value, str):
            try:
                parsed = parse_iso8601(timestamp_value)
            except ValueError:
                return datetime.now(timezone.utc).date()
            return parsed.astimezone(timezone.utc).date()
        raise TypeError(f"Unsupported timestamp type for record grouping: {type(timestamp_value)!r}")

    def _compute_key(self, record: dict[str, Any]) -> str | None:
        timestamp_value = record.get("close_time") or record.get("event_time") or record.get("open_time")
        if timestamp_value is None:
            return None
        if isinstance(timestamp_value, datetime):
            stamp = timestamp_value if timestamp_value.tzinfo else timestamp_value.replace(tzinfo=timezone.utc)
            return stamp.astimezone(timezone.utc).isoformat()
        if isinstance(timestamp_value, str):
            try:
                parsed = parse_iso8601(timestamp_value)
            except ValueError:
                return timestamp_value
            return parsed.astimezone(timezone.utc).isoformat()
        return str(timestamp_value)

    def _normalise_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return {key: self._normalise_value(value) for key, value in record.items()}

    def _normalise_value(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (float, int, str, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [self._normalise_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._normalise_value(val) for key, val in value.items()}
        if isinstance(value, Decimal):
            return float(value)
        return str(value)


__all__ = ["ParquetRecorder"]
