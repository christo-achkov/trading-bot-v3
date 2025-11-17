"""CLI to backfill heuristic microstructure snapshots into stored candles."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import typer

from trading_bot.config import load_settings
from trading_bot.data import CandleBatch, ParquetBatchWriter
from trading_bot.data.loader import iter_candles_from_parquet
from trading_bot.data.microstructure import HeuristicMicrostructureProvider
from trading_bot.utils import parse_iso8601

app = typer.Typer(help="Enrich existing candle datasets with synthetic microstructure data")


@app.command()
def enrich(
    output_dir: Path = typer.Argument(..., help="Destination directory for the enriched dataset."),
    source_dir: Optional[Path] = typer.Option(None, help="Source parquet root (defaults to config path)."),
    symbol: Optional[str] = typer.Option(None, help="Symbol to process (defaults to config)."),
    interval: Optional[str] = typer.Option(None, help="Interval to process (defaults to config)."),
    start: Optional[str] = typer.Option(None, help="Optional ISO8601 start filter."),
    end: Optional[str] = typer.Option(None, help="Optional ISO8601 end filter."),
    batch_size: int = typer.Option(5_000, help="Number of rows Arrow should deliver per batch."),
) -> None:
    """Create an enriched copy of the dataset containing heuristic depth snapshots."""

    settings = load_settings()

    resolved_source = Path(source_dir or settings.data.raw_data_dir)
    resolved_symbol = symbol or settings.data.symbol
    resolved_interval = interval or settings.data.interval

    if not resolved_symbol or not resolved_interval:
        raise typer.Exit("Symbol and interval must be provided via arguments or config settings.")

    start_dt = parse_iso8601(start) if start else None
    end_dt = parse_iso8601(end) if end else None

    provider = HeuristicMicrostructureProvider()
    writer = ParquetBatchWriter(output_dir)

    buffer: List[dict] = []
    partition_key: Optional[Tuple[str, str, int, int, int]] = None

    def flush() -> None:
        nonlocal buffer, partition_key
        if not buffer or partition_key is None:
            return
        symbol_key, interval_key, *_ = partition_key
        writer.write(CandleBatch(symbol_key, interval_key, list(buffer)))
        buffer = []
        partition_key = None

    for record in iter_candles_from_parquet(
        resolved_source,
        symbol=resolved_symbol,
        interval=resolved_interval,
        start=start_dt,
        end=end_dt,
        batch_size=batch_size,
    ):
        snapshot = provider.snapshot_for(record)
        if snapshot is not None:
            record.update(snapshot.to_record())

        open_time = record["open_time"]
        if hasattr(open_time, "to_pydatetime"):
            open_dt: datetime = open_time.to_pydatetime()  # type: ignore[assignment]
        else:
            open_dt = open_time  # type: ignore[assignment]

        year = int(open_dt.year)
        month = int(open_dt.month)
        day = int(open_dt.day)

        record["symbol"] = resolved_symbol
        record["interval"] = resolved_interval
        record["year"] = year
        record["month"] = month
        record["day"] = day

        key = (resolved_symbol, resolved_interval, year, month, day)
        if partition_key is None:
            partition_key = key
        elif key != partition_key:
            flush()
            partition_key = key

        buffer.append(record)

    flush()


if __name__ == "__main__":  # pragma: no cover
    app()
