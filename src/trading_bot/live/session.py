"""Live streaming session that fuses Binance feeds with the feature pipeline."""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, List

from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn

from trading_bot.data import (
    BinanceFuturesStream,
    BinanceRESTClient,
    LiveMarketAggregator,
    ParquetRecorder,
    fetch_candles_iter,
)
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
    realised_return: float | None
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
        recorder: ParquetRecorder | None = None,
        normalizer_state_path: str | None = None,
    ) -> None:
        if interval not in INTERVAL_TO_TIMDELTA:
            raise ValueError(f"Unsupported interval: {interval}")
        self._symbol = symbol
        self._interval = interval
        self._interval_delta = INTERVAL_TO_TIMDELTA[interval]
        self._candle_history = max(int(candle_history), 0)
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
        self._recorder = recorder
        self._normalizer_state_path = normalizer_state_path

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

                if self._recorder is not None:
                    self._recorder.append(record)

                signal = self._handle_record(record)
                if signal is None:
                    continue

                await self._invoke_callback(callback, signal)
        finally:
            await self._stream.stop()
            if self._recorder is not None:
                self._recorder.close()

    async def _warm_start(self) -> None:
        # load normalizer state if provided before processing history
        try:
            if self._normalizer_state_path is not None:
                try:
                    self._builder.load_normalizer(self._normalizer_state_path)
                except Exception as error:
                    self._logger.warning("Failed to load normalizer state: %s", error)

        except Exception:
            # defensive: continue warm start even if load fails
            pass

        candles = await asyncio.to_thread(self._fetch_history)
        if not candles:
            self._logger.warning("No historical candles returned during warm-up")
            return

        self._aggregator.preload_candles(candles)

        previous_features: dict[str, float] | None = None
        previous_close: float | None = None

        total_steps = max(len(candles) - 1, 0)
        progress: Progress | None = None
        task_id = None
        if total_steps > 0:
            progress = Progress(
                SpinnerColumn(),
                "Pretraining",
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            )
            progress.start()
            task_id = progress.add_task("Pretraining", total=total_steps)

        # Merge records and impute small gaps so features remain continuous.
        merged: list[dict] = []
        prev_rec: dict | None = None
        total_imputed = 0
        gap_events = 0
        max_gap = 0
        for rec in candles:
            if prev_rec is None:
                merged.append(rec)
                prev_rec = rec
                continue

            # determine timestamp fields (prefer close_time/event_time/open_time)
            def _ts_of(r: dict):
                for k in ("close_time", "event_time", "open_time"):
                    v = r.get(k)
                    if v is not None:
                        return v
                return None

            prev_ts = _ts_of(prev_rec)
            cur_ts = _ts_of(rec)
            # if timestamps are not datetimes, skip imputation
            if isinstance(prev_ts, datetime) and isinstance(cur_ts, datetime):
                expected = prev_ts + self._interval_delta
                # insert imputed candles for short gaps only (<= 3 intervals)
                gap_count = 0
                while expected < cur_ts and gap_count < 3:
                    imputed = dict(prev_rec)
                    imputed["open_time"] = expected
                    imputed["close_time"] = expected
                    imputed["event_time"] = expected
                    # use previous close for price fields
                    prev_close = float(prev_rec.get("close", 0.0))
                    imputed["open"] = prev_close
                    imputed["high"] = prev_close
                    imputed["low"] = prev_close
                    imputed["close"] = prev_close
                    imputed["volume"] = 0.0
                    imputed["_imputed"] = True
                    merged.append(imputed)
                    gap_count += 1
                    total_imputed += 1
                    expected = expected + self._interval_delta
                if gap_count > 0:
                    gap_events += 1
                    max_gap = max(max_gap, gap_count)
                    try:
                        self._logger.info(
                            "Detected gap of %d intervals between %s and %s; imputed %d candles",
                            gap_count,
                            prev_ts.isoformat(),
                            cur_ts.isoformat(),
                            gap_count,
                        )
                    except Exception:
                        self._logger.info("Detected gap and imputed %d candles", gap_count)
                merged.append(rec)
            else:
                merged.append(rec)
            prev_rec = rec

        try:
            for record in merged:
                is_imputed = bool(record.get("_imputed"))
                features = self._builder.process(record, imputed=is_imputed)
                close_price = float(record.get("close", 0.0))

                if previous_features is not None and previous_close is not None and not is_imputed:
                    # prefer builder-level normalization when available (do not mutate features)
                    try:
                        model_input_prev = self._builder.normalize_features(previous_features) if previous_features is not None else {}
                    except Exception:
                        model_input_prev = previous_features

                    prediction_raw = self._model.predict_one(model_input_prev)
                    prediction = float(prediction_raw) if prediction_raw is not None else 0.0
                    calibrator_input = self._build_calibrator_input(previous_features, prediction)
                    if self._calibrator is not None:
                        calibrated = self._calibrator.predict_one(calibrator_input)
                        if calibrated is not None:
                            prediction = float(calibrated)

                    # use simple percent return
                    pct_return = self._pct_return(previous_close, close_price)
                    target = pct_return

                    try:
                        self._model.learn_one(model_input_prev, target)
                    except Exception:
                        self._model.learn_one(previous_features, target)
                    if self._calibrator is not None:
                        self._calibrator.learn_one(calibrator_input, target)

                    if progress is not None and task_id is not None:
                        progress.advance(task_id)

                previous_features = features
                previous_close = close_price
        finally:
            if progress is not None:
                progress.stop()
        if total_imputed:
            self._logger.info(
                "Warm-start imputed %d total candles across %d gap events (largest=%d)",
                total_imputed,
                gap_events,
                max_gap,
            )
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

        realised_return: float | None = None
        training_target: float | None = None

        if self._previous_features is not None and self._previous_close is not None:
            try:
                model_input_prev = self._builder.normalize_features(self._previous_features) if self._previous_features is not None else {}
            except Exception:
                model_input_prev = self._previous_features

            raw_prev = self._model.predict_one(model_input_prev)
            raw_prev_value = float(raw_prev) if raw_prev is not None else 0.0
            calibrator_input_prev = self._build_calibrator_input(self._previous_features, raw_prev_value)
            if self._calibrator is not None:
                calibrated_prev = self._calibrator.predict_one(calibrator_input_prev)
                if calibrated_prev is not None:
                    raw_prev_value = float(calibrated_prev)

            realised_return = self._pct_return(self._previous_close, close_price)

            # use realised return internally for online learning (optionally adjust for cost),
            # but do not expose this as the public training_target on the signal. The CLI
            # computes a display/ledger target from the model prediction and current position.
            learn_target = realised_return - self._trade_cost if self._cost_adjust else realised_return
            try:
                self._model.learn_one(model_input_prev, learn_target)
            except Exception:
                self._model.learn_one(self._previous_features, learn_target)
            if self._calibrator is not None:
                self._calibrator.learn_one(calibrator_input_prev, learn_target)

        try:
            model_input_next = self._builder.normalize_features(features)
        except Exception:
            model_input_next = features

        raw_prediction = self._model.predict_one(model_input_next)
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
            realised_return=realised_return,
            training_target=None,
            features=features,
            record=record,
        )

    def _fetch_history(self) -> List[dict]:
        # For live sessions we must only warm-start from recorded enriched data
        # (Parquet). Do not fetch historical candles via REST during live.
        if self._recorder is None:
            self._logger.warning("No Parquet recorder available; skipping warm-start history")
            return []

        try:
            records = self._recorder.read_recent(limit=max(self._candle_history, 0) + 10)
        except Exception as error:
            self._logger.warning("Failed to read historical records from Parquet: %s", error)
            return []

        if not records:
            self._logger.warning("No recorded historical candles available for warm-start")
            return []

        return records[-self._candle_history :] if self._candle_history > 0 else []

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
    def _pct_return(previous_close: float, current_close: float) -> float:
        return (current_close / max(previous_close, 1e-12)) - 1.0

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
