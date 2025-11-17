"""In-memory data structures for live market processing."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Iterable, Iterator, List, Mapping, Sequence

from trading_bot.utils import parse_iso8601


@dataclass(slots=True)
class OrderBookSnapshot:
    """Represent a depth snapshot for a single instrument."""

    first_update_id: int | None
    last_update_id: int | None
    previous_update_id: int | None
    bids: Sequence[tuple[float, float]]
    asks: Sequence[tuple[float, float]]

    def top_of_book(self) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        """Return the best bid/ask levels when available."""

        bid = self.bids[0] if self.bids else None
        ask = self.asks[0] if self.asks else None
        return bid, ask


@dataclass(slots=True)
class Candle:
    """Simplified OHLCV candle representation with optional metadata."""

    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float | None = None
    number_of_trades: int | None = None
    taker_buy_base_volume: float | None = None
    taker_buy_quote_volume: float | None = None
    symbol: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Convert the candle into the dictionary structure expected downstream."""

        record: dict[str, Any] = {
            "open_time": self.open_time,
            "close_time": self.close_time,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
            "quote_asset_volume": float(self.quote_volume or 0.0),
            "number_of_trades": int(self.number_of_trades or 0),
            "taker_buy_base_asset_volume": float(self.taker_buy_base_volume or 0.0),
            "taker_buy_quote_asset_volume": float(self.taker_buy_quote_volume or 0.0),
        }

        if self.symbol is not None:
            record["symbol"] = self.symbol

        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Candle":
        """Instantiate a candle from the downloader/streaming dictionary format."""

        def _extract_float(primary_key: str, *aliases: str, default: float = 0.0) -> float:
            keys = (primary_key, *aliases)
            for key in keys:
                if key not in record:
                    continue
                value = record.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
            return float(default)

        open_time = record.get("open_time")
        close_time = record.get("close_time")
        if isinstance(open_time, str):
            try:
                open_time = parse_iso8601(open_time)
            except ValueError:
                open_time = None
        if isinstance(close_time, str):
            try:
                close_time = parse_iso8601(close_time)
            except ValueError:
                close_time = None

        return cls(
            open_time=open_time,
            close_time=close_time,
            open=_extract_float("open"),
            high=_extract_float("high"),
            low=_extract_float("low"),
            close=_extract_float("close"),
            volume=_extract_float("volume"),
            quote_volume=_extract_float("quote_asset_volume", "quote_volume"),
            number_of_trades=int(record.get("number_of_trades", 0) or 0),
            taker_buy_base_volume=_extract_float("taker_buy_base_asset_volume", "taker_buy_volume"),
            taker_buy_quote_volume=_extract_float("taker_buy_quote_asset_volume", "taker_buy_quote_volume"),
            symbol=record.get("symbol"),
        )


class CandleBuffer:
    """Bounded in-memory store retaining the latest candles in arrival order."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items: Deque[Candle] = deque(maxlen=capacity)

    def append(self, candle: Candle) -> None:
        """Insert a new candle, evicting the oldest when capacity is exceeded."""
        if self._items and self._items[-1].open_time == candle.open_time:
            self._items[-1] = candle
        else:
            self._items.append(candle)

    def extend(self, candles: Iterable[Candle]) -> None:
        """Bulk insert candles preserving their order."""

        for candle in candles:
            self.append(candle)

    def latest(self) -> Candle | None:
        """Return the most recent candle if available."""

        if not self._items:
            return None
        return self._items[-1]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._items)

    def __iter__(self) -> Iterator[Candle]:  # pragma: no cover - trivial
        return iter(self._items)

    def to_list(self) -> List[Candle]:
        """Realise a copy of the buffered candles."""

        return list(self._items)

    def clear(self) -> None:
        """Remove all retained candles."""

        self._items.clear()


class OrderBookBuffer:
    """Track the latest order book snapshot in memory."""

    def __init__(self) -> None:
        self._snapshot: OrderBookSnapshot | None = None

    def update(self, snapshot: OrderBookSnapshot) -> None:
        """Replace the currently stored snapshot."""

        self._snapshot = snapshot

    def latest(self) -> OrderBookSnapshot | None:
        """Return the current snapshot if available."""

        return self._snapshot

    def clear(self) -> None:
        """Forget the stored snapshot."""

        self._snapshot = None
