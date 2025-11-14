"""CLI to run offline backtests using stored candle data."""
from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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
from trading_bot.models import adaptive_regressor, default_metrics, isotonic_calibrator
from trading_bot.models.calibration import OnlineIsotonicCalibrator
from trading_bot.utils import parse_iso8601

app = typer.Typer(help="Backtesting utilities")


@app.command()
def run(
    symbol: Optional[str] = typer.Option(None, help="Override symbol."),
    interval: Optional[str] = typer.Option(None, help="Override interval."),
    start: Optional[str] = typer.Option(None, help="Filter start timestamp (ISO8601)."),
    end: Optional[str] = typer.Option(None, help="Filter end timestamp (ISO8601)."),
    edge_threshold: float = typer.Option(
        0.001,
        min=0.0,
        help="Minimum forecast log-return required (after costs) to open a position.",
    ),
    pretrain_days: int = typer.Option(
        365,
        min=0,
        help="Days of history before the start window used for warm-up training.",
    ),
    fee_bps: float = typer.Option(
        7.5,
        min=0.0,
        help="Per-trade transaction cost in basis points.",
    ),
    slippage_bps: float = typer.Option(
        5.0,
        min=0.0,
        help="Per-trade slippage assumption in basis points.",
    ),
    edge_clip: Optional[float] = typer.Option(
        None,
        min=0.0,
        help="Optional absolute cap applied to predicted edges before decision-making.",
    ),
    diagnostics_path: Optional[Path] = typer.Option(
        None,
        help="Persist predicted vs realised log returns to CSV for calibration analysis.",
    ),
    isotonic_calibration: bool = typer.Option(
        True,
        help="Apply an online isotonic calibrator to model outputs before decisions.",
    ),
    iso_window_size: int = typer.Option(
        1024,
        min=1,
        help="Sliding window size used by the isotonic calibrator.",
    ),
    iso_min_samples: int = typer.Option(
        50,
        min=1,
        help="Minimum observations required before calibration is applied.",
    ),
    optimizer: str = typer.Option(
        "sgd",
        help="Optimizer for the linear regressor (choices: adam, sgd, rmsprop).",
    ),
    learning_rate: float = typer.Option(
        0.001,
        min=1e-6,
        help="Primary learning rate supplied to the optimizer.",
    ),
    intercept_learning_rate: float = typer.Option(
        0.001,
        min=0.0,
        help="Learning rate applied to the intercept term.",
    ),
    l2: float = typer.Option(
        1e-3,
        min=0.0,
        help="L2 regularisation strength for the regressor weights.",
    ),
    clip_gradient: float = typer.Option(
        0.5,
        min=0.0,
        help="Absolute gradient clipping threshold for the regressor.",
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

    transaction_cost = fee_bps / 10_000.0
    slippage_cost = slippage_bps / 10_000.0

    total_candles = count_candles_in_parquet(
        data_dir,
        symbol=resolved_symbol,
        interval=resolved_interval,
        start=start_dt,
        end=end_dt,
    )
    total_samples = max(total_candles - 1, 0)

    pretrain_candles: List[dict] | None = None
    pretrain_cache_count = 0
    if pretrain_days > 0 and start_dt is not None:
        pretrain_start = start_dt - timedelta(days=pretrain_days)
        pretrain_source = iter_candles_from_parquet(
            data_dir,
            symbol=resolved_symbol,
            interval=resolved_interval,
            start=pretrain_start,
            end=start_dt,
        )
        pretrain_candles = list(pretrain_source)
        pretrain_cache_count = len(pretrain_candles)

    diagnostics_buffer: List[Tuple[float, float]] | None = [] if diagnostics_path else None

    def run_once() -> BacktestResult:
        model = adaptive_regressor(
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            intercept_lr=intercept_learning_rate,
            l2=l2,
            clip_gradient=clip_gradient,
        )
        calibrator = (
            isotonic_calibrator(window_size=iso_window_size, min_samples=iso_min_samples)
            if isotonic_calibration
            else None
        )
        metric = default_metrics()
        engine = BacktestEngine(model, calibrator=calibrator)
        builder = OnlineFeatureBuilder()

        if pretrain_candles:
            if pretrain_cache_count > 0:
                with _progress(pretrain_cache_count) as pretrain_progress:
                    task = pretrain_progress.add_task("Pretraining", total=pretrain_cache_count)
                    _warm_start_model(
                        pretrain_candles,
                        builder=builder,
                        model=model,
                        calibrator=calibrator,
                        progress=pretrain_progress,
                        progress_task=task,
                    )
            else:
                _warm_start_model(
                    pretrain_candles,
                    builder=builder,
                    model=model,
                    calibrator=calibrator,
                    progress=None,
                    progress_task=None,
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
                task_id = progress.add_task("Backtest", total=total_samples)

                def on_step() -> None:
                    progress.advance(task_id)

                return engine.run(
                    candles_iter,
                    builder=builder,
                    edge_threshold=edge_threshold,
                    transaction_cost=transaction_cost,
                    slippage_cost=slippage_cost,
                    metrics=[metric],
                    step_callback=on_step,
                    diagnostics=diagnostics_buffer,
                    edge_clip=edge_clip,
                )
        return engine.run(
            candles_iter,
            builder=builder,
            edge_threshold=edge_threshold,
            transaction_cost=transaction_cost,
            slippage_cost=slippage_cost,
            metrics=[metric],
            diagnostics=diagnostics_buffer,
            edge_clip=edge_clip,
        )

    result = run_once()
    typer.echo(
        "trades={trades}, hit_rate={hit:.2%}, strat_return={sr:.2%}, buy_hold={bh:.2%}, "
        "sharpe={sh:.2f}, costs={costs:.2%}, metric={metric}".format(
            trades=result.trades,
            hit=result.hit_rate,
            sr=math.expm1(result.total_return),
            bh=math.expm1(result.buy_hold_return),
            sh=result.sharpe_ratio,
            costs=-math.expm1(-result.total_costs),
            metric=result.metric_summary,
        )
    )

    if diagnostics_path and diagnostics_buffer is not None:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        with diagnostics_path.open("w", encoding="utf-8") as handle:
            handle.write("predicted_edge,realized_log_return\n")
            for predicted, realized in diagnostics_buffer:
                handle.write(f"{predicted},{realized}\n")


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
    model,
    calibrator: OnlineIsotonicCalibrator | None = None,
    progress: Progress | None = None,
    progress_task: int | None = None,
) -> None:
    """Prime the model with historical candles before evaluation begins."""

    previous_features: dict[str, float] | None = None
    previous_close: float | None = None

    for candle in candles:
        features = builder.process(candle)
        close = float(candle["close"])

        if previous_features is not None and previous_close is not None:
            log_return = math.log(max(close, 1e-12) / max(previous_close, 1e-12))
            predicted_edge_raw = model.predict_one(previous_features)
            predicted_edge = 0.0 if predicted_edge_raw is None else float(predicted_edge_raw)
            if calibrator is not None:
                calibrator.learn_one({"prediction": predicted_edge}, log_return)
            model.learn_one(previous_features, log_return)

        previous_features = features
        previous_close = close

        if progress is not None and progress_task is not None:
            progress.advance(progress_task)


if __name__ == "__main__":  # pragma: no cover
    app()
