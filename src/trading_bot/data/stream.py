"""Asynchronous Binance Futures WebSocket streaming utilities."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

import websockets

from trading_bot.data.memory import Candle, OrderBookSnapshot

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class MarketUpdate:
    """Normalised representation of a streamed market event."""

    event_time: datetime
    symbol: str
    update_type: str
    candle: Candle | None = None
    candle_closed: bool = False
    depth: OrderBookSnapshot | None = None


class BinanceFuturesStream:
    """Stream kline and depth updates for a single symbol via WebSocket."""

    def __init__(
        self,
        *,
        symbol: str,
        interval: str,
        depth_levels: int = 50,
        base_url: str = "wss://fstream.binance.com",
        logger: logging.Logger | None = None,
        reconnect_initial: float = 1.0,
        reconnect_max: float = 30.0,
    ) -> None:
        if depth_levels <= 0:
            raise ValueError("depth_levels must be positive")
        self._symbol = symbol.lower()
        self._interval = interval
        self._depth_levels = depth_levels
        self._base_url = base_url.rstrip("/")
        self._logger = logger or LOGGER
        self._reconnect_initial = reconnect_initial
        self._reconnect_max = reconnect_max
        self._stopped = asyncio.Event()

    async def stop(self) -> None:
        """Signal the streaming loop to terminate."""

        self._stopped.set()

    async def stream(self) -> AsyncIterator[MarketUpdate]:
        """Yield market updates with automatic reconnection."""

        backoff = self._reconnect_initial
        while not self._stopped.is_set():
            url = self._build_url()
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    self._logger.info("Connected to Binance futures stream: %s", url)
                    backoff = self._reconnect_initial
                    async for payload in ws:
                        if self._stopped.is_set():
                            break
                        update = self._parse_message(payload)
                        if update is not None:
                            yield update
            except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
                raise
            except Exception as exc:  # pragma: no cover - network safeguards
                if self._stopped.is_set():
                    break
                self._logger.warning("Binance stream error (%s), reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._reconnect_max)

    def _build_url(self) -> str:
        streams = [
            f"{self._symbol}@kline_{self._interval}",
            f"{self._symbol}@depth{self._depth_levels}@100ms",
        ]
        return f"{self._base_url}/stream?streams={'/'.join(streams)}"

    def _parse_message(self, payload: str) -> Optional[MarketUpdate]:
        message = json.loads(payload)
        data = message.get("data") or {}
        event_type = data.get("e")
        event_time_raw = data.get("E")
        symbol = data.get("s") or self._symbol.upper()
        event_time = self._to_datetime(event_time_raw)

        if event_type == "kline":
            kline = data.get("k", {})
            candle = Candle(
                open_time=self._to_datetime(kline.get("t")),
                close_time=self._to_datetime(kline.get("T")),
                open=float(kline.get("o", 0.0)),
                high=float(kline.get("h", 0.0)),
                low=float(kline.get("l", 0.0)),
                close=float(kline.get("c", 0.0)),
                volume=float(kline.get("v", 0.0)),
                quote_volume=float(kline.get("q", 0.0)),
                number_of_trades=int(kline.get("n", 0)),
                taker_buy_base_volume=float(kline.get("V", 0.0)),
                taker_buy_quote_volume=float(kline.get("Q", 0.0)),
                symbol=symbol,
            )
            candle_closed = bool(kline.get("x", False))
            return MarketUpdate(
                event_time=event_time,
                symbol=symbol,
                update_type="kline",
                candle=candle,
                candle_closed=candle_closed,
            )

        if event_type == "depthUpdate":
            bids = self._normalise_levels(data.get("b", []))
            asks = self._normalise_levels(data.get("a", []))
            snapshot = OrderBookSnapshot(
                first_update_id=int(data.get("U", 0)) if data.get("U") is not None else None,
                last_update_id=int(data.get("u", 0)) if data.get("u") is not None else None,
                previous_update_id=int(data.get("pu", 0)) if data.get("pu") is not None else None,
                bids=bids,
                asks=asks,
            )
            return MarketUpdate(
                event_time=event_time,
                symbol=symbol,
                update_type="depth",
                depth=snapshot,
            )

        return None

    @staticmethod
    def _to_datetime(value: int | float | None) -> datetime:
        if value is None:
            return datetime.now(timezone.utc)
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)

    @staticmethod
    def _normalise_levels(levels: list) -> list[tuple[float, float]]:
        normalised: list[tuple[float, float]] = []
        for price, qty in levels:
            try:
                price_val = float(price)
                qty_val = float(qty)
            except (TypeError, ValueError):
                continue
            if qty_val <= 0.0 or price_val <= 0.0:
                continue
            normalised.append((price_val, qty_val))
        return normalised


__all__ = ["MarketUpdate", "BinanceFuturesStream"]
