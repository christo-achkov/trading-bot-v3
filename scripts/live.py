"""Live trading CLI driven by the Binance futures WebSocket feed."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import typer

from trading_bot.config import load_settings
from trading_bot.data import BinanceFuturesStream, BinanceRESTClient, LiveMarketAggregator
from trading_bot.features import OnlineFeatureBuilder
from trading_bot.live.session import LiveMarketSession, LiveSignal
from trading_bot.models import adaptive_regressor, regime_isotonic_calibrator

app = typer.Typer(help="Live trading utilities")


@app.command()
def run(
    symbol: Optional[str] = typer.Option(None, help="Override symbol (default comes from LiveSettings)."),
    interval: Optional[str] = typer.Option(None, help="Override kline interval."),
    candle_history: Optional[int] = typer.Option(None, help="Warm-up candles to fetch before streaming."),
    depth_levels: Optional[int] = typer.Option(None, help="Depth levels captured from the order book."),
    cost_adjust_training: bool = typer.Option(
        True,
        help="Subtract trading costs when updating the model online.",
    ),
) -> None:
    """Start a continuous live session printing calibrated signal estimates."""

    asyncio.run(
        _run_live(
            symbol_override=symbol,
            interval_override=interval,
            history_override=candle_history,
            depth_override=depth_levels,
            cost_adjust_training=cost_adjust_training,
        )
    )


async def _run_live(
    *,
    symbol_override: str | None,
    interval_override: str | None,
    history_override: int | None,
    depth_override: int | None,
    cost_adjust_training: bool,
) -> None:
    settings = load_settings()

    resolved_symbol = (symbol_override or settings.live.symbol).upper()
    resolved_interval = interval_override or settings.data.interval
    resolved_history = history_override or settings.live.candle_history
    resolved_depth = depth_override or settings.live.depth_levels

    logger = logging.getLogger("trading_bot.live")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    stream = BinanceFuturesStream(
        symbol=resolved_symbol,
        interval=resolved_interval,
        depth_levels=resolved_depth,
        logger=logger,
    )
    aggregator = LiveMarketAggregator(
        symbol=resolved_symbol,
        candle_capacity=resolved_history,
        depth_levels=resolved_depth,
        logger=logger,
    )

    model = adaptive_regressor()
    calibrator = regime_isotonic_calibrator(window_size=1024, min_samples=50)
    builder = OnlineFeatureBuilder()

    trade_cost = (settings.backtest.fee_bps + settings.backtest.slippage_bps) / 10_000.0

    with BinanceRESTClient(
        settings.binance.api_key,
        settings.binance.api_secret,
        base_url=settings.binance.base_url,
    ) as rest_client:
        session = LiveMarketSession(
            symbol=resolved_symbol,
            interval=resolved_interval,
            candle_history=resolved_history,
            fetch_chunk_minutes=settings.data.fetch_chunk_minutes,
            rest_client=rest_client,
            stream=stream,
            aggregator=aggregator,
            builder=builder,
            model=model,
            calibrator=calibrator,
            trade_cost=trade_cost,
            cost_adjust_training=cost_adjust_training,
            logger=logger,
        )

        async def handle_signal(signal: LiveSignal) -> None:
            logger.info(
                "edge=%.6f (raw=%.6f) realised=%.6f target=%.6f",
                signal.prediction,
                signal.raw_prediction,
                signal.realised_log_return if signal.realised_log_return is not None else float("nan"),
                signal.training_target if signal.training_target is not None else float("nan"),
            )

        try:
            await session.run(handle_signal)
        except asyncio.CancelledError:  # pragma: no cover - cooperative shutdown
            raise
        except KeyboardInterrupt:  # pragma: no cover - console stop
            logger.info("Live session interrupted by user")
        except Exception as error:  # pragma: no cover - runtime guard
            logger.exception("Live session terminated due to error: %s", error)


if __name__ == "__main__":
    app()
