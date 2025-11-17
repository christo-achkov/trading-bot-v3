"""Thin wrapper around the python-binance REST client."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Literal

from binance import Client


MarketType = Literal["spot", "futures"]


class BinanceRESTClient:
    """Provide a minimal interface for fetching market data with resource cleanup."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: str | None = None,
        futures_url: str | None = None,
        market: MarketType = "spot",
        request_timeout: int = 20,
        requests_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        params = dict(requests_params or {})
        params.setdefault("timeout", request_timeout)

        self._market: MarketType = market
        self._client = Client(api_key, api_secret, requests_params=params)

        if market == "spot":
            if base_url:
                self._client.API_URL = base_url
            self._fetch_klines: Callable[..., List[List[Any]]] = self._client.get_klines
            self._fetch_depth: Callable[..., Dict[str, Any]] = self._client.get_order_book
        else:
            endpoint = futures_url or base_url
            if endpoint:
                self._client.FUTURES_URL = endpoint
            self._fetch_klines = self._client.futures_klines
            # python-binance exposes both futures_depth and futures_order_book across versions; prefer order_book
            depth_fn = getattr(self._client, "futures_order_book", None)
            if depth_fn is None:
                depth_fn = getattr(self._client, "futures_depth")
            if depth_fn is None:
                raise AttributeError("python-binance client missing futures depth method")
            self._fetch_depth = depth_fn

    def __enter__(self) -> "BinanceRESTClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying HTTP session."""

        if hasattr(self._client, "session") and self._client.session:
            self._client.session.close()

    def get_klines(
        self,
        *,
        symbol: str,
        interval: str,
        startTime: int,
        endTime: int,
        limit: int,
    ) -> List[List[Any]]:
        """Delegate to python-binance while keeping typing explicit."""

        return self._fetch_klines(  # type: ignore[no-any-return]
            symbol=symbol,
            interval=interval,
            startTime=startTime,
            endTime=endTime,
            limit=limit,
        )

    def get_order_book(self, *, symbol: str, limit: int) -> Dict[str, Any]:
        """Fetch a depth snapshot for the provided symbol."""

        return self._fetch_depth(symbol=symbol, limit=limit)  # type: ignore[no-any-return]
