"""Microstructure heuristics for enriching candle data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


@dataclass
class MicrostructureSnapshot:
    """Simple container for best bid/ask and synthetic depth levels."""

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


class HeuristicMicrostructureProvider:
    """Fallback provider that fabricates depth from candle statistics."""

    def __init__(
        self,
        *,
        levels: int = 10,
        price_step_bps: float = 5.0,
        spread_floor_bps: float = 0.5,
        spread_cap_bps: float = 25.0,
        volume_ratio: float = 0.01,
        min_size: float = 0.1,
        decay: float = 0.6,
    ) -> None:
        if levels <= 0:
            raise ValueError("levels must be positive")
        if price_step_bps <= 0.0:
            raise ValueError("price_step_bps must be positive")
        if spread_floor_bps <= 0.0:
            raise ValueError("spread_floor_bps must be positive")
        if spread_cap_bps <= spread_floor_bps:
            raise ValueError("spread_cap_bps must exceed spread_floor_bps")
        if decay <= 0.0 or decay >= 1.0:
            raise ValueError("decay must lie in (0, 1)")

        self._levels = levels
        self._price_step = price_step_bps / 10_000.0
        self._spread_floor = spread_floor_bps
        self._spread_cap = spread_cap_bps
        self._volume_ratio = volume_ratio
        self._min_size = min_size
        self._decay = decay

    def snapshot_for(self, candle: dict) -> MicrostructureSnapshot | None:
        close = self._safe_float(candle.get("close"))
        high = self._safe_float(candle.get("high"), fallback=close)
        low = self._safe_float(candle.get("low"), fallback=close)
        volume = max(self._safe_float(candle.get("volume")), 0.0)

        if close <= 0.0:
            return None

        raw_spread_bps = 0.0
        if high > 0.0 and low > 0.0 and high >= low:
            raw_spread_bps = ((high - low) / max(close, 1e-12)) * 10_000.0
        spread_bps = min(max(raw_spread_bps * 0.25, self._spread_floor), self._spread_cap)
        spread = close * (spread_bps / 10_000.0)
        half_spread = max(spread / 2.0, close * 1e-6)

        best_bid = close - half_spread
        best_ask = close + half_spread

        base_size = max(volume * self._volume_ratio, self._min_size)

        bids: List[tuple[float, float]] = []
        asks: List[tuple[float, float]] = []
        for level in range(self._levels):
            price_offset = self._price_step * level
            size_scale = base_size * (self._decay ** level)
            bid_price = max(best_bid * (1.0 - price_offset), 1e-8)
            ask_price = max(best_ask * (1.0 + price_offset), 1e-8)
            bids.append((bid_price, size_scale))
            asks.append((ask_price, size_scale))

        return MicrostructureSnapshot(
            best_bid=best_bid,
            best_ask=best_ask,
            best_bid_size=bids[0][1],
            best_ask_size=asks[0][1],
            bids=bids,
            asks=asks,
            spread_bps=spread_bps,
        )

    @staticmethod
    def _safe_float(value, fallback: float | None = None) -> float:
        try:
            if value is None:
                if fallback is None:
                    return 0.0
                return float(fallback)
            return float(value)
        except (TypeError, ValueError):
            return 0.0 if fallback is None else float(fallback)
