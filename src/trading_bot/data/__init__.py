"""Data access layer."""

from trading_bot.data.binance_client import BinanceRESTClient
from trading_bot.data.binance_downloader import BinanceDownloader, CandleBatch
from trading_bot.data.history import fetch_candles, fetch_candles_iter
from trading_bot.data.live import LiveMarketAggregator, OrderBookSyncError
from trading_bot.data.memory import Candle, CandleBuffer, OrderBookBuffer, OrderBookSnapshot
from trading_bot.data.stream import BinanceFuturesStream, MarketUpdate

__all__ = [
	"BinanceRESTClient",
	"BinanceDownloader",
	"CandleBatch",
	"fetch_candles",
	"fetch_candles_iter",
	"Candle",
	"CandleBuffer",
	"OrderBookBuffer",
	"OrderBookSnapshot",
	"MarketUpdate",
	"BinanceFuturesStream",
	"LiveMarketAggregator",
	"OrderBookSyncError",
]
