"""Microstructure heuristics for enriching candle data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


@dataclass
class MicrostructureSnapshot:
    """Simple container for best bid/ask and heuristic depth levels."""

    best_bid: float
    best_ask: float
    best_bid_size: float
    best_ask_size: float
    bids: List[tuple[float, float]]
    asks: List[tuple[float, float]]
    spread_bps: float

    def to_record(self) -> dict[str, object]:
        """Convert the snapshot into a serialisable dictionary."""

        return {
            "best_bid": float(self.best_bid),
            "best_ask": float(self.best_ask),
            "best_bid_size": float(self.best_bid_size),
            "best_ask_size": float(self.best_ask_size),
            "orderbook_bids": [[float(price), float(size)] for price, size in self.bids],
            "orderbook_asks": [[float(price), float(size)] for price, size in self.asks],
            "microstructure_spread_bps": float(self.spread_bps),
        }


class MicrostructureProvider(Protocol):
    """Contract for objects capable of producing microstructure snapshots."""

    def snapshot_for(self, candle: dict) -> MicrostructureSnapshot | None:
        """Return a snapshot for the provided candle context if available."""


__all__ = ["MicrostructureSnapshot", "MicrostructureProvider"]
