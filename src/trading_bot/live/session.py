"""Live streaming session that fuses Binance feeds with the feature pipeline."""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, List

from trading_bot.data import BinanceFuturesStream, BinanceRESTClient, LiveMarketAggregator, fetch_candles_iter
from trading_bot.data.binance_downloader import INTERVAL_TO_TIMDELTA
from trading_bot.features import OnlineFeatureBuilder
from trading_bot.models.regime import SupportsPredictLearn
from trading_bot.utils import parse_iso8601

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LiveSignal:
    """Container for the latest model inference and realised return."""

    symbol: str
    event_time: datetime
    close_time: datetime | None
    prediction: float
    raw_prediction: float
    realised_log_return: float | None
    training_target: float | None
    features: dict[str, float]
    record: dict


class LiveMarketSession:
    """Drive the Binance futures stream and emit calibrated signals."""

    def __init__(
        self,
        *,
        symbol: str,
        interval: str,
        candle_history: int,
        fetch_chunk_minutes: int,
        rest_client: BinanceRESTClient,
        stream: BinanceFuturesStream,
        aggregator: LiveMarketAggregator,
        builder: OnlineFeatureBuilder,
        model: SupportsPredictLearn,
        calibrator: SupportsPredictLearn | None = None,
        trade_cost: float = 0.0,
        cost_adjust_training: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        if interval not in INTERVAL_TO_TIMDELTA:
            raise ValueError(f"Unsupported interval: {interval}")
        self._symbol = symbol
        self._interval = interval
        self._interval_delta = INTERVAL_TO_TIMDELTA[interval]
        self._candle_history = max(int(candle_history), 1)
        self._fetch_chunk_minutes = max(int(fetch_chunk_minutes), 1)
        self._rest_client = rest_client
        self._stream = stream
        self._aggregator = aggregator
        self._builder = builder
        self._model = model
        self._calibrator = calibrator
        self._trade_cost = max(float(trade_cost), 0.0)
        self._cost_adjust = bool(cost_adjust_training)
        self._logger = logger or LOGGER
        self._previous_features: dict[str, float] | None = None
        self._previous_close: float | None = None

    async def run(self, callback: Callable[[LiveSignal], Awaitable[None] | None]) -> None:
        """Start streaming; invoke callback when a new signal is ready."""

        await self._warm_start()
        await self._ensure_snapshot()

        try:
            async for update in self._stream.stream():
                if self._aggregator.needs_snapshot():
                    await self._ensure_snapshot()
                    continue

                record = self._aggregator.process_update(update)
                if record is None:
                    continue

                signal = self._handle_record(record)
                if signal is None:
                    continue

                await self._invoke_callback(callback, signal)
        finally:
            await self._stream.stop()

    async def _warm_start(self) -> None:
        candles = await asyncio.to_thread(self._fetch_history)
        if not candles:
            self._logger.warning("No historical candles returned during warm-up")
            return

        self._aggregator.preload_candles(candles)

        previous_features: dict[str, float] | None = None
        previous_close: float | None = None

        for record in candles:
            features = self._builder.process(record)
            close_price = float(record.get("close", 0.0))

            if previous_features is not None and previous_close is not None:
                prediction_raw = self._model.predict_one(previous_features)
                prediction = float(prediction_raw) if prediction_raw is not None else 0.0
                calibrator_input = self._build_calibrator_input(previous_features, prediction)
                if self._calibrator is not None:
                    calibrated = self._calibrator.predict_one(calibrator_input)
                    if calibrated is not None:
                        prediction = float(calibrated)

                log_return = self._log_return(previous_close, close_price)
                target = log_return - self._trade_cost if self._cost_adjust else log_return

                self._model.learn_one(previous_features, target)
                if self._calibrator is not None:
                    self._calibrator.learn_one(calibrator_input, target)

            previous_features = features
            previous_close = close_price

        self._previous_features = previous_features
        self._previous_close = previous_close

    async def _ensure_snapshot(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self._aggregator.initialise_order_book, self._rest_client)
                return
            except Exception as error:  # pragma: no cover - network safeguard
                self._logger.warning("Failed to refresh order book snapshot: %s", error)
                await asyncio.sleep(2.0)

    def _handle_record(self, record: dict) -> LiveSignal | None:
        features = self._builder.process(record)
        close_price = float(record.get("close", 0.0))
        event_time = self._extract_event_time(record)

        realised_log_return: float | None = None
        training_target: float | None = None

        if self._previous_features is not None and self._previous_close is not None:
            raw_prev = self._model.predict_one(self._previous_features)
            raw_prev_value = float(raw_prev) if raw_prev is not None else 0.0
            calibrator_input_prev = self._build_calibrator_input(self._previous_features, raw_prev_value)
            if self._calibrator is not None:
                calibrated_prev = self._calibrator.predict_one(calibrator_input_prev)
                if calibrated_prev is not None:
                    raw_prev_value = float(calibrated_prev)

            realised_log_return = self._log_return(self._previous_close, close_price)
            training_target = realised_log_return - self._trade_cost if self._cost_adjust else realised_log_return

            self._model.learn_one(self._previous_features, training_target)
            if self._calibrator is not None:
                self._calibrator.learn_one(calibrator_input_prev, training_target)

        raw_prediction = self._model.predict_one(features)
        raw_prediction_value = float(raw_prediction) if raw_prediction is not None else 0.0
        calibrator_input_next = self._build_calibrator_input(features, raw_prediction_value)
        prediction_value = raw_prediction_value
        if self._calibrator is not None:
            calibrated_next = self._calibrator.predict_one(calibrator_input_next)
            if calibrated_next is not None:
                prediction_value = float(calibrated_next)

        self._previous_features = features
        self._previous_close = close_price

        return LiveSignal(
            symbol=record.get("symbol", self._symbol),
            event_time=event_time,
            close_time=record.get("close_time"),
            prediction=prediction_value,
            raw_prediction=raw_prediction_value,
            realised_log_return=realised_log_return,
            training_target=training_target,
            features=features,
            record=record,
        )

    def _fetch_history(self) -> List[dict]:
        end_time = datetime.now(timezone.utc)
        total_span = self._interval_delta * max(self._candle_history + 5, 5)
        start_time = end_time - total_span

        fetched: List[dict] = []
        for candle in fetch_candles_iter(
            self._rest_client,
            symbol=self._symbol,
            interval=self._interval,
            start=start_time,
            end=end_time,
            chunk_minutes=self._fetch_chunk_minutes,
        ):
            fetched.append(candle)

        if not fetched:
            return []

        return fetched[-self._candle_history :]

    def _build_calibrator_input(self, features: dict[str, float], prediction: float) -> dict[str, float]:
        input_payload = {"prediction": prediction}
        regime_label = features.get("regime_label")
        if regime_label is not None:
            try:
                input_payload["regime_label"] = float(regime_label)
            except (TypeError, ValueError):
                pass
        return input_payload

    @staticmethod
    def _log_return(previous_close: float, current_close: float) -> float:
        return math.log(max(current_close, 1e-12) / max(previous_close, 1e-12))

    @staticmethod
    def _extract_event_time(record: dict) -> datetime:
        for key in ("event_time", "close_time", "open_time"):
            value = record.get(key)
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                try:
                    return parse_iso8601(value)
                except ValueError:
                    continue
        return datetime.now(timezone.utc)

    async def _invoke_callback(self, callback: Callable[[LiveSignal], Awaitable[None] | None], signal: LiveSignal) -> None:
        result = callback(signal)
        if asyncio.iscoroutine(result):
            await result


__all__ = ["LiveMarketSession", "LiveSignal"]
