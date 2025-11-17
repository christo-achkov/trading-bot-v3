"""Order generation helpers for live trading."""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from trading_bot.execution import OrderIntent, OrderRouter
from trading_bot.live.session import LiveSignal


@dataclass(slots=True)
class ThresholdConfig:
    """Edge-driven configuration for position flipping."""

    entry_edge: float
    exit_edge: float
    position_size: float
    cooldown_seconds: float = 0.0


class ThresholdPositionManager:
    """Translate calibrated edges into position adjustments and orders."""

    def __init__(
        self,
        *,
        symbol: str,
        router: OrderRouter | None,
        thresholds: ThresholdConfig,
        dry_run: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        if thresholds.entry_edge < thresholds.exit_edge:
            raise ValueError("entry_edge must be greater than or equal to exit_edge")
        if thresholds.position_size < 0:
            raise ValueError("position_size must be non-negative")
        if thresholds.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")

        self._symbol = symbol
        self._router = router
        self._thresholds = thresholds
        self._dry_run = dry_run or router is None
        self._logger = logger or logging.getLogger(__name__)
        self._state = 0  # -1, 0, +1
        self._lock = asyncio.Lock()
        self._cooldown_seconds = float(thresholds.cooldown_seconds)
        self._last_order_at: datetime | None = None

    async def handle_signal(self, signal: LiveSignal) -> None:
        """React to a new model signal."""

        if not math.isfinite(signal.prediction):
            return

        async with self._lock:
            target_state = self._decide_state(signal.prediction)
            if target_state == self._state:
                return

            changed = await self._rebalance(target_state)
            if changed:
                self._state = target_state

    def _decide_state(self, edge: float) -> int:
        entry = self._thresholds.entry_edge
        exit_edge = self._thresholds.exit_edge
        current = self._state

        if edge >= entry:
            return 1
        if edge <= -entry:
            return -1
        if abs(edge) <= exit_edge:
            return 0
        return current

    async def _rebalance(self, target_state: int) -> bool:
        size = self._thresholds.position_size
        if size <= 0:
            self._logger.debug("Position size is zero; skipping order generation")
            self._state = 0
            return False

        current_state = self._state

        if target_state == current_state:
            return False

        now = datetime.now(timezone.utc)
        if (
            self._cooldown_seconds > 0.0
            and self._last_order_at is not None
            and (now - self._last_order_at).total_seconds() < self._cooldown_seconds
        ):
            remaining = self._cooldown_seconds - (now - self._last_order_at).total_seconds()
            self._logger.info(
                "Skipping rebalance for %s due to cooldown (%.2fs remaining)",
                self._symbol,
                max(remaining, 0.0),
            )
            return False

        placed = False

        if target_state == 0 and current_state != 0:
            side = "SELL" if current_state > 0 else "BUY"
            quantity = abs(current_state) * size
            placed |= await self._submit(side=side, quantity=quantity, reduce_only=True)
            if placed:
                self._last_order_at = datetime.now(timezone.utc)
            return placed

        if current_state == 0:
            side = "BUY" if target_state > 0 else "SELL"
            quantity = abs(target_state) * size
            placed |= await self._submit(side=side, quantity=quantity, reduce_only=False)
            if placed:
                self._last_order_at = datetime.now(timezone.utc)
            return placed

        # Switching direction: flatten first, then open the new side.
        flatten_side = "SELL" if current_state > 0 else "BUY"
        flatten_qty = abs(current_state) * size
        placed |= await self._submit(side=flatten_side, quantity=flatten_qty, reduce_only=True)

        open_side = "BUY" if target_state > 0 else "SELL"
        open_qty = abs(target_state) * size
        placed |= await self._submit(side=open_side, quantity=open_qty, reduce_only=False)

        if placed:
            self._last_order_at = datetime.now(timezone.utc)
        return placed
    async def _submit(self, *, side: str, quantity: float, reduce_only: bool) -> bool:
        if quantity <= 0:
            return False

        if self._dry_run or self._router is None:
            self._logger.info(
                "[DRY-RUN] Would submit %s %s qty=%.6f reduce_only=%s",
                self._symbol,
                side,
                quantity,
                reduce_only,
            )
            return True

        intent = OrderIntent(
            symbol=self._symbol,
            side=side,
            quantity=quantity,
            order_type="MARKET",
            reduce_only=reduce_only,
        )
        await asyncio.to_thread(self._router.submit, intent)
        return True


__all__ = ["ThresholdPositionManager", "ThresholdConfig"]
