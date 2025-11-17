"""Assembly utilities for combining Binance stream updates with in-memory buffers."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from trading_bot.data.binance_client import BinanceRESTClient
from trading_bot.data.memory import Candle, CandleBuffer, OrderBookBuffer, OrderBookSnapshot
from trading_bot.data.stream import MarketUpdate

LOGGER = logging.getLogger(__name__)


class OrderBookSyncError(RuntimeError):
    """Raised when incremental depth updates can no longer be trusted."""


@dataclass(slots=True)
class _OrderBookState:
    """Mutable reconstruction of the order book from incremental updates."""

    depth_levels: int
    logger: logging.Logger
    _bids: dict[float, float] = field(init=False, default_factory=dict)
    _asks: dict[float, float] = field(init=False, default_factory=dict)
    _last_update_id: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.depth_levels <= 0:
            raise ValueError("depth_levels must be positive")
        self.reset()

    def reset(self) -> None:
        """Forget all accumulated depth information."""

        self._bids.clear()
        self._asks.clear()
        self._last_update_id = None

    def initialise(self, snapshot: dict) -> OrderBookSnapshot:
        """Seed the order book from a REST snapshot."""

        self.reset()
        bids = snapshot.get("bids", []) or []
        asks = snapshot.get("asks", []) or []

        def _collect(levels) -> dict[float, float]:
            collected: dict[float, float] = {}
            for price_raw, qty_raw in levels:
                try:
                    price = float(price_raw)
                    qty = float(qty_raw)
                except (TypeError, ValueError):
                    continue
                if price <= 0.0 or qty <= 0.0:
                    continue
                collected[price] = qty
            return collected

        self._bids = _collect(bids)
        self._asks = _collect(asks)
        self._last_update_id = int(snapshot.get("lastUpdateId", 0) or 0)

        return self._build_snapshot(
            first_id=self._last_update_id,
            last_id=self._last_update_id,
            previous_id=self._last_update_id - 1 if self._last_update_id else None,
        )

    def apply_delta(self, update: OrderBookSnapshot) -> OrderBookSnapshot | None:
        """Incorporate an incremental depth update."""

        if update.first_update_id is None or update.last_update_id is None:
            return None
        if self._last_update_id is None:
            raise OrderBookSyncError("order book snapshot missing; cannot apply delta")
        if update.last_update_id <= self._last_update_id:
            return None

        expected_next = self._last_update_id + 1
        if update.first_update_id > expected_next:
            raise OrderBookSyncError(
                f"gap detected (expected >= {expected_next}, received {update.first_update_id})"
            )
        if update.previous_update_id is not None and update.previous_update_id != self._last_update_id:
            raise OrderBookSyncError(
                f"out-of-order update (prev {update.previous_update_id} != {self._last_update_id})"
            )

        self._apply_side(self._bids, update.bids)
        self._apply_side(self._asks, update.asks)
        self._last_update_id = update.last_update_id

        return self._build_snapshot(
            first_id=update.first_update_id,
            last_id=self._last_update_id,
            previous_id=update.previous_update_id,
        )

    def _apply_side(self, book: dict[float, float], deltas: Sequence[tuple[float, float]]) -> None:
        for price, qty in deltas:
            if price <= 0.0:
                continue
            if qty <= 0.0:
                book.pop(price, None)
            else:
                book[price] = qty

    def _build_snapshot(
        self,
        *,
        first_id: int | None,
        last_id: int | None,
        previous_id: int | None,
    ) -> OrderBookSnapshot:
        bids_sorted = sorted(((p, q) for p, q in self._bids.items() if q > 0.0), key=lambda x: x[0], reverse=True)[
            : self.depth_levels
        ]
        asks_sorted = sorted(((p, q) for p, q in self._asks.items() if q > 0.0), key=lambda x: x[0])[ : self.depth_levels]

        self._bids = {price: qty for price, qty in bids_sorted}
        self._asks = {price: qty for price, qty in asks_sorted}

        return OrderBookSnapshot(
            first_update_id=first_id,
            last_update_id=last_id,
            previous_update_id=previous_id,
            bids=tuple(bids_sorted),
            asks=tuple(asks_sorted),
        )


class LiveMarketAggregator:
    """Combine WebSocket updates with buffers and emit enriched candle records."""

    def __init__(
        self,
        *,
        symbol: str,
        candle_capacity: int,
        depth_levels: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self._symbol = symbol
        self._candle_buffer = CandleBuffer(candle_capacity)
        self._order_book_buffer = OrderBookBuffer()
        self._logger = logger or LOGGER
        self._state = _OrderBookState(depth_levels=depth_levels, logger=self._logger)
        self._depth_levels = depth_levels
        self._needs_snapshot = True

    @property
    def candle_buffer(self) -> CandleBuffer:
        return self._candle_buffer

    @property
    def order_book_buffer(self) -> OrderBookBuffer:
        return self._order_book_buffer

    def needs_snapshot(self) -> bool:
        return self._needs_snapshot

    def initialise_order_book(self, client: BinanceRESTClient) -> None:
        """Fetch a fresh depth snapshot and seed the internal book."""

        allowed_limits = (5, 10, 20, 50, 100, 500, 1000)
        limit = next((value for value in allowed_limits if value >= self._depth_levels), allowed_limits[-1])
        snapshot = client.get_order_book(symbol=self._symbol, limit=limit)
        ob_snapshot = self._state.initialise(snapshot)
        self._order_book_buffer.update(ob_snapshot)
        self._needs_snapshot = False

    def preload_candles(self, candles: Iterable[dict]) -> None:
        """Populate the candle buffer with historical records."""

        for record in candles:
            try:
                candle = Candle.from_record(record)
            except Exception as error:  # pragma: no cover - defensive guard
                self._logger.debug("Skipping malformed candle during preload: %s", error)
                continue
            self._candle_buffer.append(candle)

    def process_update(self, update: MarketUpdate) -> dict | None:
        """Consume a market update and emit a completed candle record when available."""

        if update.update_type == "depth" and update.depth is not None:
            if self._needs_snapshot:
                return None
            try:
                snapshot = self._state.apply_delta(update.depth)
            except OrderBookSyncError as error:
                self._logger.warning("Order book desynchronised: %s", error)
                self._state.reset()
                self._order_book_buffer.clear()
                self._needs_snapshot = True
                return None
            if snapshot is not None:
                self._order_book_buffer.update(snapshot)
            return None

        if update.update_type == "kline" and update.candle is not None:
            self._candle_buffer.append(update.candle)
            if not update.candle_closed:
                return None
            return self._compose_record(update)

        return None

    def _compose_record(self, update: MarketUpdate) -> dict:
        candle = update.candle
        assert candle is not None  # for type checkers
        record = candle.to_record()

        snapshot = self._order_book_buffer.latest()
        if snapshot is not None and snapshot.bids:
            best_bid_price, best_bid_size = snapshot.bids[0]
        else:
            best_bid_price = float(candle.close)
            best_bid_size = 0.0

        if snapshot is not None and snapshot.asks:
            best_ask_price, best_ask_size = snapshot.asks[0]
        else:
            best_ask_price = float(candle.close)
            best_ask_size = 0.0

        record.update(
            {
                "symbol": candle.symbol or self._symbol,
                "event_time": update.event_time,
                "best_bid": float(best_bid_price),
                "best_ask": float(best_ask_price),
                "best_bid_size": float(best_bid_size),
                "best_ask_size": float(best_ask_size),
                "orderbook_bids": self._serialise_levels(snapshot.bids if snapshot else []),
                "orderbook_asks": self._serialise_levels(snapshot.asks if snapshot else []),
                "orderbook_last_update_id": snapshot.last_update_id if snapshot else None,
            }
        )

        if best_bid_price > 0.0 and best_ask_price > 0.0:
            mid_price = (best_bid_price + best_ask_price) / 2.0
        else:
            mid_price = float(candle.close)
        spread = max(best_ask_price - best_bid_price, 0.0)
        spread_bps = (spread / max(mid_price, 1e-12)) * 10_000.0
        record["microstructure_spread_bps"] = spread_bps

        return record

    def _serialise_levels(self, levels: Sequence[tuple[float, float]]) -> List[List[float]]:
        serialised: List[List[float]] = []
        for price, qty in levels[: self._depth_levels]:
            serialised.append([float(price), float(qty)])
        return serialised


__all__ = ["LiveMarketAggregator", "OrderBookSyncError"]
