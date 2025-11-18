"""Historical Binance downloader with retry and rate limiting controls."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Sequence

from loguru import logger
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from trading_bot.utils import ensure_utc
from trading_bot.data.microstructure import MicrostructureProvider

MAX_CANDLES_PER_REQUEST = 1000

INTERVAL_TO_TIMDELTA = {
	"1m": timedelta(minutes=1),
	"3m": timedelta(minutes=3),
	"5m": timedelta(minutes=5),
	"15m": timedelta(minutes=15),
	"30m": timedelta(minutes=30),
	"1h": timedelta(hours=1),
	"2h": timedelta(hours=2),
	"4h": timedelta(hours=4),
	"6h": timedelta(hours=6),
	"8h": timedelta(hours=8),
	"12h": timedelta(hours=12),
	"1d": timedelta(days=1),
}


@dataclass(frozen=True)
class CandleBatch:
	"""Container for a batch of candles returned by the downloader."""

	symbol: str
	interval: str
	candles: Sequence[dict]


class BinanceDownloader:
	"""Fetch Binance market data in controllable batches suitable for online learning."""

	def __init__(
		self,
		client,
		*,
		symbol: str,
		interval: str,
		chunk_minutes: int,
		microstructure_provider: MicrostructureProvider | None = None,
	) -> None:
		if interval not in INTERVAL_TO_TIMDELTA:
			raise ValueError(f"Unsupported interval: {interval}")

		requested_minutes = max(1, chunk_minutes)
		interval_delta = INTERVAL_TO_TIMDELTA[interval]

		approximate_candles = max(1, int((timedelta(minutes=requested_minutes)) / interval_delta))
		if approximate_candles > MAX_CANDLES_PER_REQUEST:
			logger.warning(
				"chunk size reduced from {} candles to Binance limit {}",
				approximate_candles,
				MAX_CANDLES_PER_REQUEST,
			)
			approximate_candles = MAX_CANDLES_PER_REQUEST

		self._client = client
		self._symbol = symbol
		self._interval = interval
		self._interval_delta = interval_delta
		self._step = self._interval_delta * approximate_candles
		self._limit = approximate_candles
		self._microstructure_provider = microstructure_provider

	@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
	def _fetch_page(self, start: datetime, end: datetime) -> List[list]:
		"""Internal helper with retry logic for a single REST request."""

		return self._client.get_klines(
			symbol=self._symbol,
			interval=self._interval,
			startTime=int(start.timestamp() * 1000),
			endTime=int(end.timestamp() * 1000),
			limit=self._limit,
		)

	def _transform(self, raw: Sequence) -> dict:
		"""Normalize raw kline payload into a structured dictionary."""

		record = {
			"open_time": datetime.fromtimestamp(int(raw[0]) / 1000, tz=timezone.utc),
			"open": float(raw[1]),
			"high": float(raw[2]),
			"low": float(raw[3]),
			"close": float(raw[4]),
			"volume": float(raw[5]),
			"close_time": datetime.fromtimestamp(int(raw[6]) / 1000, tz=timezone.utc),
			"quote_asset_volume": float(raw[7]),
			"number_of_trades": int(raw[8]),
			"taker_buy_base_asset_volume": float(raw[9]),
			"taker_buy_quote_asset_volume": float(raw[10]),
		}

		if self._microstructure_provider is not None:
			try:
				snapshot = self._microstructure_provider.snapshot_for(record)
			except Exception as error:  # pragma: no cover - defensive fallback
				logger.warning(
					"Failed to build microstructure snapshot for {ts}: {error}",
					ts=record["close_time"],
					error=error,
				)
			else:
				if snapshot is not None:
					record.update(snapshot.to_record())

		return record

	def stream(self, *, start: datetime, end: datetime) -> Iterable[CandleBatch]:
		"""Stream candle batches between two timestamps inclusive."""

		cursor = ensure_utc(start)
		deadline = ensure_utc(end)

		while cursor < deadline:
			window_end = min(cursor + self._step, deadline)
			try:
				candles = self._fetch_page(cursor, window_end)
			except RetryError as exc:  # pragma: no cover - network fallback
				logger.error("Binance request failed after retries: {error}", error=exc)
				raise

			if not candles:
				cursor = window_end
				continue

			transformed = [self._transform(candle) for candle in candles]
			logger.debug(
				"Fetched {count} candles for {symbol} ({interval}) from {start} to {stop}",
				count=len(transformed),
				symbol=self._symbol,
				interval=self._interval,
				start=transformed[0]["open_time"],
				stop=transformed[-1]["close_time"],
			)
			yield CandleBatch(self._symbol, self._interval, transformed)

			next_cursor = transformed[-1]["open_time"] + self._interval_delta
			if next_cursor <= cursor:  # pragma: no cover - defensive guard
				next_cursor = cursor + self._interval_delta
			cursor = min(next_cursor, deadline)
