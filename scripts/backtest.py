"""CLI to run offline backtests using stored candle data."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import typer
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from trading_bot.backtest import BacktestEngine
from trading_bot.config import load_settings
from trading_bot.data import count_candles_in_parquet, iter_candles_from_parquet
from trading_bot.features import OnlineFeatureBuilder, iter_supervised_samples
from trading_bot.models import adaptive_classifier, default_metrics
from trading_bot.utils import parse_iso8601

app = typer.Typer(help="Backtesting utilities")


@app.command()
def run(
    symbol: Optional[str] = typer.Option(None, help="Override symbol."),
    interval: Optional[str] = typer.Option(None, help="Override interval."),
    start: Optional[str] = typer.Option(None, help="Filter start timestamp (ISO8601)."),
    end: Optional[str] = typer.Option(None, help="Filter end timestamp (ISO8601)."),
    label_threshold: float = typer.Option(0.0, help="Log-return threshold for labeling."),
) -> None:
    """Run a walk-forward backtest over stored Parquet candles."""

    settings = load_settings()

    resolved_symbol = symbol or settings.data.symbol
    resolved_interval = interval or settings.data.interval
    start_dt = parse_iso8601(start) if start else None
    end_dt = parse_iso8601(end) if end else None

    data_dir = Path(settings.data.raw_data_dir)
    if not data_dir.exists():
        typer.echo(f"Data directory {data_dir} does not exist. Download data first.", err=True)
        raise typer.Exit(code=1)

    candles = iter_candles_from_parquet(
        data_dir,
        symbol=resolved_symbol,
        interval=resolved_interval,
        start=start_dt,
        end=end_dt,
    )

    builder = OnlineFeatureBuilder()
    samples = iter_supervised_samples(candles, builder=builder, label_threshold=label_threshold)

    model = adaptive_classifier()
    metric = default_metrics()

    engine = BacktestEngine(model)

    total_candles = count_candles_in_parquet(
        data_dir,
        symbol=resolved_symbol,
        interval=resolved_interval,
        start=start_dt,
        end=end_dt,
    )
    total_samples = max(total_candles - 1, 0)

    if total_samples > 0:
        with _progress(total_samples) as progress:
            task_id = progress.add_task("Backtest", total=total_samples)
            wrapped_samples = _samples_with_progress(samples, progress, task_id)
            result = engine.run(wrapped_samples, metrics=[metric])
    else:
        result = engine.run(samples, metrics=[metric])

    typer.echo(
        "Backtest complete. PnL: {pnl:.2f}, trades: {trades}, hit_rate: {hit:.2%}, metric={metric}".format(
            pnl=result.pnl,
            trades=result.trades,
            hit=result.hit_rate,
            metric=result.metric_summary,
        )
    )


def _progress(total: int) -> Progress:
    """Construct a configured Rich progress instance."""

    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def _samples_with_progress(
    samples: Iterable[Tuple[dict, int]],
    progress: Progress,
    task_id: int,
) -> Iterator[Tuple[dict, int]]:
    """Yield samples while updating the provided progress task."""

    for features, label in samples:
        yield features, label
        progress.advance(task_id)


if __name__ == "__main__":  # pragma: no cover
    app()
