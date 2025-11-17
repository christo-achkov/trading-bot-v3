"""Binance USD-M futures execution client."""
from __future__ import annotations

import logging
from typing import Any, Dict

from binance import Client

from trading_bot.execution.order_router import ExchangeClient

LOGGER = logging.getLogger(__name__)


class BinanceFuturesExchange(ExchangeClient):
    """Thin wrapper around python-binance futures order submission."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        futures_url: str | None = None,
        request_timeout: int = 20,
        testnet: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self._logger = logger or LOGGER
        self._client = Client(api_key, api_secret, testnet=testnet, requests_params={"timeout": request_timeout})
        if futures_url:
            self._client.FUTURES_URL = futures_url

    def __enter__(self) -> "BinanceFuturesExchange":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying HTTP session."""

        if hasattr(self._client, "session") and self._client.session:
            self._client.session.close()

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
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": float(quantity),
        }
        if price is not None and params["type"] != "MARKET":
            params["price"] = float(price)
        if reduce_only is not None:
            params["reduceOnly"] = bool(reduce_only)

        self._logger.info(
            "Submitting futures order: %s %s qty=%.6f reduce_only=%s", symbol, side, quantity, reduce_only
        )
        response = self._client.futures_create_order(**params)
        self._logger.info("Order response: %s", response)
        return response


__all__ = ["BinanceFuturesExchange"]
