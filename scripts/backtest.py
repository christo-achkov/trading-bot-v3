"""CLI to run offline backtests using stored candle data."""
from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Optional

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

from trading_bot.backtest import BacktestEngine, BacktestResult
from trading_bot.config import load_settings
from trading_bot.data import count_candles_in_parquet, iter_candles_from_parquet
from trading_bot.features import OnlineFeatureBuilder
from trading_bot.models import adaptive_classifier, default_metrics
from trading_bot.utils import parse_iso8601

app = typer.Typer(help="Backtesting utilities")


@app.command()
def run(
    symbol: Optional[str] = typer.Option(None, help="Override symbol."),
    interval: Optional[str] = typer.Option(None, help="Override interval."),
    start: Optional[str] = typer.Option(None, help="Filter start timestamp (ISO8601)."),
    end: Optional[str] = typer.Option(None, help="Filter end timestamp (ISO8601)."),
    label_threshold: float = typer.Option(
        0.001,
        help="Log-return threshold for labeling; higher values reduce trade frequency.",
    ),
    prediction_horizon: int = typer.Option(
        15,
        min=1,
        help="Number of buffered predictions to aggregate before deciding on a trade.",
    ),
    aggregation: str = typer.Option(
        "majority",
        help="How to aggregate buffered predictions: majority, weighted, or unanimous.",
        case_sensitive=False,
    ),
    signal_threshold: float = typer.Option(
        0.7,
        min=0.0,
        max=1.0,
        help="Fraction of agreeing predictions required before entering a trade.",
    ),
    pretrain_days: int = typer.Option(
        0,
        min=0,
        help="Optional number of days before start used for warm-up training.",
    ),
    horizon_grid: Optional[str] = typer.Option(
        None,
        help="Comma-separated list of horizons to sweep; overrides prediction_horizon.",
    ),
    fee_bps: float = typer.Option(
        1.0,
        min=0.0,
        help="Per-trade transaction cost in basis points (applied when a trade is taken).",
    ),
    slippage_bps: float = typer.Option(
        1.0,
        min=0.0,
        help="Additional slippage cost in basis points deducted per trade.",
    ),
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

    aggregation_mode = aggregation.lower()
    if aggregation_mode not in {"majority", "weighted", "unanimous"}:
        typer.echo(
            "Invalid aggregation mode. Choose from 'majority', 'weighted', or 'unanimous'.",
            err=True,
        )
        raise typer.Exit(code=1)

    transaction_cost = fee_bps / 10_000.0
    slippage_cost = slippage_bps / 10_000.0

    if horizon_grid:
        try:
            horizons = [
                int(token.strip())
                for token in horizon_grid.split(",")
                if token.strip()
            ]
        except ValueError:
            typer.echo("Invalid horizon grid; use comma-separated integers.", err=True)
            raise typer.Exit(code=1)
        if not horizons:
            typer.echo("No valid horizons provided.", err=True)
            raise typer.Exit(code=1)
    else:
        horizons = [prediction_horizon]

    for horizon in horizons:
        if horizon <= 0:
            typer.echo("Prediction horizon values must be positive integers.", err=True)
            raise typer.Exit(code=1)

    total_candles = count_candles_in_parquet(
        data_dir,
        symbol=resolved_symbol,
        interval=resolved_interval,
        start=start_dt,
        end=end_dt,
    )
    total_samples = max(total_candles - 1, 0)

    def run_once(horizon: int) -> BacktestResult:
        model = adaptive_classifier()
        metric = default_metrics()
        engine = BacktestEngine(model)
        builder = OnlineFeatureBuilder()

        if pretrain_days > 0 and start_dt is not None:
            pretrain_start = start_dt - timedelta(days=pretrain_days)
            pretrain_candles = iter_candles_from_parquet(
                data_dir,
                symbol=resolved_symbol,
                interval=resolved_interval,
                start=pretrain_start,
                end=start_dt,
            )
            _warm_start_model(
                pretrain_candles,
                builder=builder,
                label_threshold=label_threshold,
                model=model,
            )

        candles_iter = iter_candles_from_parquet(
            data_dir,
            symbol=resolved_symbol,
            interval=resolved_interval,
            start=start_dt,
            end=end_dt,
        )

        if total_samples > 0:
            with _progress(total_samples) as progress:
                task_id = progress.add_task(f"Backtest (h={horizon})", total=total_samples)

                def on_step() -> None:
                    progress.advance(task_id)

                return engine.run(
                    candles_iter,
                    builder=builder,
                    label_threshold=label_threshold,
                    prediction_horizon=horizon,
                    aggregation=aggregation_mode,
                    signal_threshold=signal_threshold,
                    transaction_cost=transaction_cost,
                    slippage_cost=slippage_cost,
                    metrics=[metric],
                    step_callback=on_step,
                )
        return engine.run(
            candles_iter,
            builder=builder,
            label_threshold=label_threshold,
            prediction_horizon=horizon,
            aggregation=aggregation_mode,
            signal_threshold=signal_threshold,
            transaction_cost=transaction_cost,
            slippage_cost=slippage_cost,
            metrics=[metric],
        )

    multiple_horizons = len(horizons) > 1

    for horizon in horizons:
        result = run_once(horizon)
        prefix = f"Horizon {horizon}: " if multiple_horizons else ""
        typer.echo(
            prefix
            + "trades={trades}, hit_rate={hit:.2%}, strat_return={sr:.2%}, buy_hold={bh:.2%}, sharpe={sh:.2f}, costs={costs:.2%}, metric={metric}".format(
                trades=result.trades,
                hit=result.hit_rate,
                sr=math.expm1(result.total_return),
                bh=math.expm1(result.buy_hold_return),
                sh=result.sharpe_ratio,
                costs=-math.expm1(-result.total_costs),
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


def _warm_start_model(
    candles: Iterable[dict],
    *,
    builder: OnlineFeatureBuilder,
    label_threshold: float,
    model,
) -> None:
    """Prime the model with historical candles before evaluation begins."""

    previous_features: dict[str, float] | None = None
    previous_close: float | None = None

    for candle in candles:
        features = builder.process(candle)
        close = float(candle["close"])

        if previous_features is not None and previous_close is not None:
            log_return = math.log(max(close, 1e-12) / max(previous_close, 1e-12))
            if log_return > label_threshold:
                label = 1
            elif log_return < -label_threshold:
                label = -1
            else:
                label = 0
            model.learn_one(previous_features, label)

        previous_features = features
        previous_close = close


if __name__ == "__main__":  # pragma: no cover
    app()
