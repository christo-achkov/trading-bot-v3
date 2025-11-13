"""Thin wrapper around the python-binance REST client."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from binance import Client


class BinanceRESTClient:
    """Provide a minimal interface for fetching klines with resource cleanup."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: str | None = None,
        request_timeout: int = 20,
        requests_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        params = dict(requests_params or {})
        params.setdefault("timeout", request_timeout)

        self._client = Client(api_key, api_secret, requests_params=params)
        if base_url:
            # python-binance exposes the API URL via an attribute; overriding keeps the same session.
            self._client.API_URL = base_url

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

        return self._client.get_klines(  # type: ignore[no-any-return]
            symbol=symbol,
            interval=interval,
            startTime=startTime,
            endTime=endTime,
            limit=limit,
        )
