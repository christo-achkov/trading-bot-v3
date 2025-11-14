"""CLI entrypoint for downloading historical Binance candles to disk."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from trading_bot.config import load_settings
from trading_bot.data.binance_client import BinanceRESTClient
from trading_bot.data.binance_downloader import BinanceDownloader
from trading_bot.data.storage import ParquetBatchWriter
from trading_bot.utils import ensure_utc, parse_iso8601

app = typer.Typer(help="Historical data operations")


@app.command()
def download(
    start: Optional[str] = typer.Option(None, help="Override start timestamp (ISO8601)."),
    end: Optional[str] = typer.Option(None, help="Override end timestamp (ISO8601)."),
    chunk_minutes: Optional[int] = typer.Option(None, help="Minutes per API request (<= 1000)."),
) -> None:
    """Download Binance candles and persist them as partitioned Parquet files."""

    settings = load_settings()

    start_dt = parse_iso8601(start) if start else parse_iso8601(settings.data.start_date)
    end_dt = parse_iso8601(end) if end else ensure_utc(datetime.now(timezone.utc))

    if start_dt >= end_dt:
        typer.echo("Start timestamp must be before end timestamp.", err=True)
        raise typer.Exit(code=1)

    chunk = chunk_minutes or settings.data.fetch_chunk_minutes
    writer = ParquetBatchWriter(Path(settings.data.raw_data_dir))

    with BinanceRESTClient(
        settings.binance.api_key,
        settings.binance.api_secret,
        base_url=settings.binance.base_url,
    ) as client:
        downloader = BinanceDownloader(
            client,
            symbol=settings.data.symbol,
            interval=settings.data.interval,
            chunk_minutes=chunk,
        )

        total = 0
        last_ts: Optional[datetime] = None
        try:
            for batch in downloader.stream(start=start_dt, end=end_dt):
                writer.write(batch)
                total += len(batch.candles)
                last_ts = batch.candles[-1]["close_time"]
                typer.echo(
                    f"Processed {len(batch.candles):>4} candles (total {total}) up to {last_ts.isoformat()}"
                )
        except KeyboardInterrupt:  # pragma: no cover - user initiated stop
            typer.echo("Download interrupted by user.")

    typer.echo(
        "Completed download. Total candles: {total}{tail}".format(
            total=total,
            tail=f", last close {last_ts.isoformat()}" if last_ts else "",
        )
    )


if __name__ == "__main__":  # pragma: no cover
    app()
