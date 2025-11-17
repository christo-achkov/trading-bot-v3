"""Logging-backed exchange client for dry-run execution."""
from __future__ import annotations

import logging
from typing import Any, Dict

from trading_bot.execution.order_router import ExchangeClient

LOGGER = logging.getLogger(__name__)


class LoggingExchangeClient(ExchangeClient):
    """Exchange client that only logs intentions without sending orders."""

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self._logger = logger or LOGGER
        self._counter = 0

    def create_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
        order_type: str = "MARKET",
        reduce_only: bool | None = None,
    ) -> Dict[str, Any]:
        self._counter += 1
        self._logger.info(
            "[DRY-RUN] #%s %s %s qty=%.6f type=%s reduce_only=%s price=%s",
            self._counter,
            symbol,
            side.upper(),
            quantity,
            order_type.upper(),
            reduce_only,
            price,
        )
        return {
            "status": "SIMULATED",
            "order_id": f"sim-{self._counter}",
            "symbol": symbol,
            "side": side.upper(),
            "quantity": float(quantity),
            "order_type": order_type.upper(),
            "reduce_only": reduce_only,
            "price": float(price) if price is not None else None,
        }


__all__ = ["LoggingExchangeClient"]
