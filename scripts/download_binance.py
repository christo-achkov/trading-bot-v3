"""CLI entrypoint for fetching historical Binance candles via REST."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from trading_bot.config import load_settings
from trading_bot.data import BinanceRESTClient, fetch_candles
from trading_bot.utils import ensure_utc, parse_iso8601

app = typer.Typer(help="Historical data operations")


@app.command()
def download(
    start: Optional[str] = typer.Option(None, help="Override start timestamp (ISO8601)."),
    end: Optional[str] = typer.Option(None, help="Override end timestamp (ISO8601)."),
    chunk_minutes: Optional[int] = typer.Option(None, help="Minutes per API request (<= 1000)."),
    output_path: Optional[str] = typer.Option(
        None,
        help="Optional destination file for JSONL output.",
    ),
) -> None:
    """Fetch Binance candles and optionally persist them as JSON lines."""

    settings = load_settings()

    start_dt = parse_iso8601(start) if start else parse_iso8601(settings.data.start_date)
    end_dt = parse_iso8601(end) if end else ensure_utc(datetime.now(timezone.utc))

    if start_dt >= end_dt:
        typer.echo("Start timestamp must be before end timestamp.", err=True)
        raise typer.Exit(code=1)

    chunk = chunk_minutes or settings.data.fetch_chunk_minutes

    with BinanceRESTClient(
        settings.binance.api_key,
        settings.binance.api_secret,
        base_url=settings.binance.base_url,
    ) as client:
        candles = fetch_candles(
            client,
            symbol=settings.data.symbol,
            interval=settings.data.interval,
            start=start_dt,
            end=end_dt,
            chunk_minutes=chunk,
        )

    total = len(candles)
    last_ts = candles[-1]["close_time"] if candles else None

    typer.echo(
        "Fetched {total} candles for {symbol} ({interval}){tail}.".format(
            total=total,
            symbol=settings.data.symbol,
            interval=settings.data.interval,
            tail=f" ending {last_ts.isoformat()}" if last_ts else "",
        )
    )

    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for candle in candles:
                handle.write(json.dumps(candle, default=str) + "\n")
        typer.echo(f"Saved candles to {destination}")


if __name__ == "__main__":  # pragma: no cover
    app()
