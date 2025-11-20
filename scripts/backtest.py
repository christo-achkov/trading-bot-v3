"""CLI to run offline backtests using Binance REST data."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
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
from trading_bot.data import BinanceRESTClient, fetch_candles, fetch_candles_iter
from trading_bot.data.binance_downloader import INTERVAL_TO_TIMDELTA
from trading_bot.features import OnlineFeatureBuilder
from trading_bot.models import (
    adaptive_regressor,
    default_metrics,
    isotonic_calibrator,
    regime_isotonic_calibrator,
)
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
        help="Minimum forecast return required (after costs) to open a position.",
    ),
    pretrain_days: int = typer.Option(
        30,
        min=0,
        help="Days of history before the start window used for warm-up training.",
    ),
    fee_bps: float = typer.Option(
        5.0,
        min=0.0,
        help="Per-trade transaction cost in basis points.",
    ),
    slippage_bps: float = typer.Option(
        3.0,
        min=0.0,
        help="Per-trade slippage assumption in basis points.",
        min=0.0,
        help="Optional absolute cap applied to predicted edges before decision-making.",
    ),
    diagnostics_path: Optional[Path] = typer.Option(
        None,
        help="Persist predicted vs realised returns to CSV for calibration analysis.",
    ),
    trades_path: Optional[Path] = typer.Option(
        None,
        help="Persist per-trade execution log including direction, fees, and equity.",
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
        1e-2,
        min=0.0,
        help="L2 regularisation strength for the regressor weights.",
    ),
    l1: float = typer.Option(
        0.0,
        min=0.0,
        help="L1 regularisation strength for the regressor weights (encourages sparsity).",
    ),
    clip_gradient: float = typer.Option(
        0.5,
        min=0.0,
        help="Absolute gradient clipping threshold for the regressor.",
    ),
    long_edge_threshold: Optional[float] = typer.Option(
        None,
        min=0.0,
        help="Optional override for the long-entry edge threshold (defaults to edge_threshold).",
    ),
    short_edge_threshold: Optional[float] = typer.Option(
        None,
        min=0.0,
        help="Optional override for the short-entry edge threshold (defaults to edge_threshold).",
    ),
    adaptive_threshold: bool = typer.Option(
        False,
        help="Scale entry cushions by a volatility feature before executing trades.",
    ),
    volatility_feature: str = typer.Option(
        "volatility_regime",
        help="Feature name used when adaptive thresholding is enabled.",
    ),
    volatility_scale: float = typer.Option(
        5.0,
        min=0.0,
        help="Multiplier applied to the volatility feature when computing adaptive cushions.",
    ),
    volatility_offset: float = typer.Option(
        0.002,
        help="Baseline value subtracted from the volatility feature before scaling.",
    ),
    minimum_cushion: Optional[float] = typer.Option(
        0.0003,
        min=0.0,
        help="Optional minimum cushion (after costs) enforced when adaptive adjustments shrink thresholds.",
    ),
    bull_cushion_offset: float = typer.Option(
        0.0001,
        help="Offset applied to the long cushion when trend bias is non-negative (positive values harden it).",
    ),
    bear_cushion_offset: float = typer.Option(
        0.0001,
        help="Offset applied to the short cushion when trend bias is non-positive (positive values harden it).",
    ),
    use_position_sizing: bool = typer.Option(
        False,
        help="Size positions proportionally to the excess edge above the cushion.",
    ),
    position_scale: float = typer.Option(
        1000.0,
        min=0.0,
        help="Scale translating excess edge into position size when sizing is enabled.",
    ),
    edge_scale: float = typer.Option(
        1.0,
        min=0.0,
        help="Multiplier applied to predicted edges prior to thresholding.",
    ),
    hysteresis: float = typer.Option(
        0.0,
        min=0.0,
        help="Gap that must be crossed before an existing position is closed (provides entry/exit hysteresis).",
    ),
    use_dynamic_cost: bool = typer.Option(
        True,
        help="Blend spread, volatility, and liquidity cues into per-trade cost estimates.",
    ),
    spread_cost_feature: str = typer.Option(
        "micro_spread_bps",
        help="Feature name feeding the spread contribution within dynamic cost estimation.",
    ),
    spread_cost_scale: float = typer.Option(
        0.0,
        min=0.0,
        help="Scale applied to the spread feature when computing dynamic costs (in return units).",
    ),
    volatility_cost_feature: str = typer.Option(
        "return_std_short",
        help="Feature name feeding the volatility contribution within dynamic cost estimation.",
    ),
    volatility_cost_scale: float = typer.Option(
        0.0,
        min=0.0,
        help="Scale applied to the volatility feature when computing dynamic costs.",
    ),
    liquidity_cost_feature: Optional[str] = typer.Option(
        "liquidity_density_50bps",
        help="Feature name reducing dynamic costs when liquidity is abundant (set to null to disable).",
    ),
    liquidity_cost_scale: float = typer.Option(
        0.0,
        min=0.0,
        help="Scale applied to the liquidity feature (positive values reduce estimated costs).",
    ),
    dynamic_cost_floor: float = typer.Option(
        0.0,
        min=0.0,
        help="Lower bound applied to dynamic cost estimates.",
    ),
    dynamic_cost_cap: Optional[float] = typer.Option(
        None,
        min=0.0,
        help="Optional upper bound applied to dynamic cost estimates.",
    ),
    skip_low_edge_trades: bool = typer.Option(
        True,
        help="Skip signals whose predicted edge does not exceed the dynamic cost estimate.",
    ),
    cost_adjust_training: bool = typer.Option(
        True,
        help="Subtract the estimated cost from training targets so the model learns net-of-cost returns (enable to train on net PnL).",
    ),
) -> None:
    """Run a walk-forward backtest using Binance REST data."""

    settings = load_settings()

    resolved_symbol = symbol or settings.data.symbol
    resolved_interval = interval or settings.data.interval

    default_start = parse_iso8601(settings.data.start_date)
    resolved_start = parse_iso8601(start) if start else default_start
    resolved_end = parse_iso8601(end) if end else datetime.now(timezone.utc)

    if resolved_start >= resolved_end:
        typer.echo("Start time must be before end time.", err=True)
        raise typer.Exit(code=1)

    interval_delta = INTERVAL_TO_TIMDELTA.get(resolved_interval)
    if interval_delta is None:
        typer.echo(f"Unsupported interval: {resolved_interval}", err=True)
        raise typer.Exit(code=1)

    transaction_cost = fee_bps / 10_000.0
    slippage_cost = slippage_bps / 10_000.0

    duration = resolved_end - resolved_start
    estimated_candles = max(int(duration.total_seconds() / interval_delta.total_seconds()) + 1, 0)
    total_samples = max(estimated_candles - 1, 0)

    pretrain_candles: List[dict] = []
    pretrain_cache_count = 0
    pretrain_start = resolved_start - timedelta(days=pretrain_days) if pretrain_days > 0 else None

    fetch_chunk = settings.data.fetch_chunk_minutes

    diagnostics_buffer: List[Tuple[float, float]] | None = [] if diagnostics_path else None
    trade_log: List[dict] | None = [] if trades_path else None

    with BinanceRESTClient(
        settings.binance.api_key,
        settings.binance.api_secret,
        base_url=settings.binance.base_url,
    ) as client:
        if pretrain_start is not None and pretrain_start < resolved_start:
            pretrain_candles = fetch_candles(
                client,
                symbol=resolved_symbol,
                interval=resolved_interval,
                start=pretrain_start,
                end=resolved_start,
                chunk_minutes=fetch_chunk,
                enriched_root=settings.data.enriched_root,
                prefer_enriched=True,
            )
            pretrain_cache_count = len(pretrain_candles)

        def run_once() -> BacktestResult:
            model = adaptive_regressor(
                optimizer_name=optimizer,
                learning_rate=learning_rate,
                intercept_lr=intercept_learning_rate,
                l2=l2,
                l1=l1,
                clip_gradient=clip_gradient,
            )
            calibrator = (
                regime_isotonic_calibrator(
                    window_size=iso_window_size,
                    min_samples=iso_min_samples,
                    regime_feature="regime_label",
                )
                if isotonic_calibration
                else None
            )
            metric = default_metrics()
            engine = BacktestEngine(model, calibrator=calibrator)
            builder = OnlineFeatureBuilder(exclude_microstructure=True)

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
                            trade_cost=transaction_cost + slippage_cost,
                            cost_adjust_training=cost_adjust_training,
                        )
                else:
                    _warm_start_model(
                        pretrain_candles,
                        builder=builder,
                        model=model,
                        calibrator=calibrator,
                        progress=None,
                        progress_task=None,
                        trade_cost=transaction_cost + slippage_cost,
                        cost_adjust_training=cost_adjust_training,
                    )

            candles_iter = fetch_candles_iter(
                client,
                symbol=resolved_symbol,
                interval=resolved_interval,
                start=resolved_start,
                end=resolved_end,
                chunk_minutes=fetch_chunk,
                enriched_root=settings.data.enriched_root,
                prefer_enriched=True,
            )

            edge_clip = None
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
                        trade_log=trade_log,
                        long_threshold=long_edge_threshold,
                        short_threshold=short_edge_threshold,
                        adaptive_threshold=adaptive_threshold,
                        volatility_feature=volatility_feature,
                        volatility_scale=volatility_scale,
                        volatility_offset=volatility_offset,
                        minimum_cushion=minimum_cushion,
                        bull_cushion_offset=bull_cushion_offset,
                        bear_cushion_offset=bear_cushion_offset,
                        use_position_sizing=use_position_sizing,
                        position_scale=position_scale,
                        edge_scale=edge_scale,
                        hysteresis=hysteresis,
                        use_dynamic_cost=use_dynamic_cost,
                        spread_cost_feature=spread_cost_feature,
                        spread_cost_scale=spread_cost_scale,
                        volatility_cost_feature=volatility_cost_feature,
                        volatility_cost_scale=volatility_cost_scale,
                        liquidity_cost_feature=liquidity_cost_feature,
                        liquidity_cost_scale=liquidity_cost_scale,
                        dynamic_cost_floor=dynamic_cost_floor,
                        dynamic_cost_cap=dynamic_cost_cap,
                        skip_low_edge_trades=skip_low_edge_trades,
                        cost_adjust_training=cost_adjust_training,
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
                trade_log=trade_log,
                long_threshold=long_edge_threshold,
                short_threshold=short_edge_threshold,
                adaptive_threshold=adaptive_threshold,
                volatility_feature=volatility_feature,
                volatility_scale=volatility_scale,
                volatility_offset=volatility_offset,
                minimum_cushion=minimum_cushion,
                bull_cushion_offset=bull_cushion_offset,
                bear_cushion_offset=bear_cushion_offset,
                use_position_sizing=use_position_sizing,
                position_scale=position_scale,
                edge_scale=edge_scale,
                hysteresis=hysteresis,
                use_dynamic_cost=use_dynamic_cost,
                spread_cost_feature=spread_cost_feature,
                spread_cost_scale=spread_cost_scale,
                volatility_cost_feature=volatility_cost_feature,
                volatility_cost_scale=volatility_cost_scale,
                liquidity_cost_feature=liquidity_cost_feature,
                liquidity_cost_scale=liquidity_cost_scale,
                dynamic_cost_floor=dynamic_cost_floor,
                dynamic_cost_cap=dynamic_cost_cap,
                skip_low_edge_trades=skip_low_edge_trades,
                cost_adjust_training=cost_adjust_training,
            )

        result = run_once()
    typer.echo(
        "trades={trades}, hit_rate={hit:.2%}, strat_return={sr:.2%}, buy_hold={bh:.2%}, "
        "sharpe={sh:.2f}, costs={costs:.2%}, metric={metric}".format(
            trades=result.trades,
            hit=result.hit_rate,
            sr=result.total_return,
            bh=result.buy_hold_return,
            sh=result.sharpe_ratio,
            costs=result.total_costs,
            metric=result.metric_summary,
        )
    )

    if diagnostics_path and diagnostics_buffer is not None:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        with diagnostics_path.open("w", encoding="utf-8") as handle:
            handle.write("predicted_edge,realized_return\n")
            for predicted, realized in diagnostics_buffer:
                handle.write(f"{predicted},{realized}\n")

    if trades_path and trade_log is not None:
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        with trades_path.open("w", encoding="utf-8") as handle:
            field_names = [
                "trade_id",
                "timestamp",
                "side",
                "predicted_edge",
                "pct_return",
                "gross_return",
                "fees",
                "net_return",
                "equity_after",
                "position_size",
                "long_cushion",
                "short_cushion",
                "cost_estimate",
                "micro_spread_bps",
                "microstructure_spread_available",
                "volatility_regime",
                "volatility_slope",
                "regime_label",
                "regime_state_bull",
                "regime_state_bear",
                "regime_state_sideways",
                "funding_rate",
                "funding_rate_diff",
                "funding_available",
                "orderbook_imbalance_top",
                "liquidity_bid_depth_50bps",
                "liquidity_ask_depth_50bps",
                "liquidity_bid_depth_25bps",
                "liquidity_ask_depth_25bps",
                "liquidity_bid_depth_100bps",
                "liquidity_ask_depth_100bps",
                "liquidity_depth_25_available",
                "liquidity_depth_50_available",
                "liquidity_depth_100_available",
            ]
            handle.write(",".join(field_names) + "\n")
            for entry in trade_log:
                row = [str(entry.get(field, "")) for field in field_names]
                handle.write(",".join(row) + "\n")


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
    trade_cost: float = 0.0,
    cost_adjust_training: bool = True,
) -> None:
    """Prime the model with historical candles before evaluation begins."""

    previous_features: dict[str, float] | None = None
    previous_close: float | None = None

    for candle in candles:
        features = builder.process(candle)
        close = float(candle["close"])

        if previous_features is not None and previous_close is not None:
            pct_return = (close / max(previous_close, 1e-12)) - 1.0
            net_return = pct_return - trade_cost if cost_adjust_training else pct_return
            predicted_edge_raw = model.predict_one(previous_features)
            predicted_edge = 0.0 if predicted_edge_raw is None else float(predicted_edge_raw)
            if calibrator is not None:
                calibrator.learn_one({"prediction": predicted_edge}, net_return)
            model.learn_one(previous_features, net_return)

        previous_features = features
        previous_close = close

        if progress is not None and progress_task is not None:
            progress.advance(progress_task)


if __name__ == "__main__":  # pragma: no cover
    app()
