"""Live trading CLI driven by the Binance futures WebSocket feed."""
from __future__ import annotations

import asyncio
import logging
import math
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from trading_bot.config import load_settings
from trading_bot.data import BinanceFuturesStream, BinanceRESTClient, LiveMarketAggregator, ParquetRecorder
from trading_bot.execution import BinanceFuturesExchange, LoggingExchangeClient, OrderRouter
from trading_bot.features import OnlineFeatureBuilder
from trading_bot.live import (
    LiveMarketSession,
    LiveSignal,
    ThresholdConfig,
    ThresholdPositionManager,
)
from trading_bot.models import adaptive_regressor, regime_isotonic_calibrator

app = typer.Typer(help="Live trading utilities")


@app.command()
def run(
    symbol: Optional[str] = typer.Option(None, help="Override symbol (default comes from LiveSettings)."),
    interval: Optional[str] = typer.Option(None, help="Override kline interval."),
    candle_history: Optional[int] = typer.Option(None, help="Warm-up candles to fetch before streaming."),
    depth_levels: Optional[int] = typer.Option(None, help="Depth levels captured from the order book."),
    position_size: Optional[float] = typer.Option(None, min=0.0, help="Override base-asset position size."),
    entry_edge: Optional[float] = typer.Option(None, min=0.0, help="Override entry edge threshold (log-return)."),
    exit_edge: Optional[float] = typer.Option(None, min=0.0, help="Override exit edge threshold (log-return)."),
    cooldown_seconds: Optional[float] = typer.Option(None, min=0.0, help="Override minimum seconds between orders."),
    cost_adjust_training: bool = typer.Option(
        True,
        help="Subtract trading costs when updating the model online.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute/--dry-run",
        help="Route orders to Binance USD-M futures instead of logging only.",
    ),
    testnet: bool = typer.Option(False, help="Use the Binance USD-M futures testnet."),
    futures_url: Optional[str] = typer.Option(None, help="Optional override for the futures REST endpoint."),
    signal_log: Optional[Path] = typer.Option(
        None,
        help="Optional CSV file that captures each signal, realised return, and rolling equity.",
    ),
    print_equity: bool = typer.Option(
        True,
        "--print-equity/--hide-equity",
        help="Include rolling equity in live log output.",
    ),
) -> None:
    """Start a continuous live session printing calibrated signal estimates."""

    asyncio.run(
        _run_live(
            symbol_override=symbol,
            interval_override=interval,
            history_override=candle_history,
            depth_override=depth_levels,
            position_override=position_size,
            entry_override=entry_edge,
            exit_override=exit_edge,
            cooldown_override=cooldown_seconds,
            cost_adjust_training=cost_adjust_training,
            execute=execute,
            testnet=testnet,
            futures_url=futures_url,
            signal_log=signal_log,
            print_equity=print_equity,
        )
    )


async def _run_live(
    *,
    symbol_override: str | None,
    interval_override: str | None,
    history_override: int | None,
    depth_override: int | None,
    position_override: float | None,
    entry_override: float | None,
    exit_override: float | None,
    cooldown_override: float | None,
    cost_adjust_training: bool,
    execute: bool,
    testnet: bool,
    futures_url: str | None,
    signal_log: Path | None,
    print_equity: bool,
) -> None:
    settings = load_settings()

    resolved_symbol = (symbol_override or settings.live.symbol).upper()
    resolved_interval = interval_override or settings.data.interval
    resolved_history = history_override or settings.live.candle_history
    resolved_depth = depth_override or settings.live.depth_levels
    resolved_position = position_override if position_override is not None else settings.live.position_size
    resolved_entry = entry_override if entry_override is not None else settings.live.entry_edge
    resolved_exit = exit_override if exit_override is not None else settings.live.exit_edge
    resolved_cooldown = cooldown_override if cooldown_override is not None else settings.live.cooldown_seconds

    logger = logging.getLogger("trading_bot.live")

    if resolved_exit > resolved_entry:
        logger.warning(
            "exit edge %.6f is greater than entry edge %.6f; adjusting to match entry",
            resolved_exit,
            resolved_entry,
        )
        resolved_exit = resolved_entry
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
        candle_capacity=max(resolved_history, 100),
        depth_levels=resolved_depth,
        logger=logger,
    )

    recorder: ParquetRecorder | None = None
    if settings.data.enriched_root is not None:
        recorder_root = Path(settings.data.enriched_root)
        recorder = ParquetRecorder(
            recorder_root,
            symbol=resolved_symbol,
            interval=resolved_interval,
        )

    model = adaptive_regressor()
    calibrator = regime_isotonic_calibrator(window_size=1024, min_samples=50)
    builder = OnlineFeatureBuilder()

    trade_cost = (settings.backtest.fee_bps + settings.backtest.slippage_bps) / 10_000.0

    csv_handle = None
    csv_writer = None
    cumulative_log_equity = 0.0
    if signal_log is not None:
        signal_log.parent.mkdir(parents=True, exist_ok=True)
        csv_handle = signal_log.open("a", encoding="utf-8", newline="")
        import csv

        csv_writer = csv.writer(csv_handle)
        if signal_log.stat().st_size == 0:
            csv_writer.writerow(
                [
                    "event_time",
                    "close_time",
                    "prediction",
                    "raw_prediction",
                    "realised_log_return",
                    "training_target",
                    "position_state",
                    "cumulative_equity",
                ]
            )

    with BinanceRESTClient(
        settings.binance.api_key,
        settings.binance.api_secret,
        base_url=settings.binance.base_url,
        futures_url=futures_url or settings.binance.futures_url,
        market="futures",
    ) as rest_client:
        exchange_cm = (
            BinanceFuturesExchange(
                settings.binance.api_key,
                settings.binance.api_secret,
                futures_url=futures_url or settings.binance.futures_url,
                testnet=testnet,
                logger=logger,
            )
            if execute
            else nullcontext(LoggingExchangeClient(logger=logger))
        )

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
            recorder=recorder,
        )

        with exchange_cm as exchange_client:
            router = OrderRouter(exchange_client)
            thresholds = ThresholdConfig(
                entry_edge=resolved_entry,
                exit_edge=resolved_exit,
                position_size=resolved_position,
                cooldown_seconds=resolved_cooldown,
            )
            manager = ThresholdPositionManager(
                symbol=resolved_symbol,
                router=router,
                thresholds=thresholds,
                dry_run=not execute,
                logger=logger,
            )

            async def handle_signal(signal: LiveSignal) -> None:
                nonlocal cumulative_log_equity
                await manager.handle_signal(signal)
                if signal.training_target is not None and math.isfinite(signal.training_target):
                    cumulative_log_equity += signal.training_target

                if csv_writer is not None:
                    cumulative_equity = math.expm1(cumulative_log_equity)
                    csv_writer.writerow(
                        [
                            signal.event_time.isoformat(),
                            signal.close_time.isoformat() if isinstance(signal.close_time, datetime) else signal.close_time,
                            signal.prediction,
                            signal.raw_prediction,
                            signal.realised_log_return,
                            signal.training_target,
                            manager.position_state,
                            cumulative_equity,
                        ]
                    )
                    csv_handle.flush()

                log_realised = signal.realised_log_return if signal.realised_log_return is not None else float("nan")
                log_target = signal.training_target if signal.training_target is not None else float("nan")
                if print_equity:
                    cumulative_equity = math.expm1(cumulative_log_equity)
                    logger.info(
                        "edge=%.6f (raw=%.6f) realised=%.6f target=%.6f equity=%.3f%%",
                        signal.prediction,
                        signal.raw_prediction,
                        log_realised,
                        log_target,
                        cumulative_equity * 100.0,
                    )
                else:
                    logger.info(
                        "edge=%.6f (raw=%.6f) realised=%.6f target=%.6f",
                        signal.prediction,
                        signal.raw_prediction,
                        log_realised,
                        log_target,
                    )

            try:
                await session.run(handle_signal)
            except asyncio.CancelledError:  # pragma: no cover - cooperative shutdown
                raise
            except KeyboardInterrupt:  # pragma: no cover - console stop
                logger.info("Live session interrupted by user")
            except Exception as error:  # pragma: no cover - runtime guard
                logger.exception("Live session terminated due to error: %s", error)
            finally:
                if csv_handle is not None:
                    csv_handle.close()


if __name__ == "__main__":
    app()
