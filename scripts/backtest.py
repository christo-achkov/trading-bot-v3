"""CLI to run offline backtests using stored candle data."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

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
        0.002,
        help="Log-return threshold for labeling; higher values reduce trade frequency.",
    ),
    prediction_horizon: int = typer.Option(
        20,
        min=1,
        help="Number of buffered predictions to aggregate before deciding on a trade.",
    ),
    aggregation: str = typer.Option(
        "majority",
        help="How to aggregate buffered predictions: majority, weighted, or unanimous.",
        case_sensitive=False,
    ),
    signal_threshold: float = typer.Option(
        0.85,
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
    auto_tune: bool = typer.Option(
        False,
        help="Use a learning-based tuner to adapt thresholds from the recent pretraining window.",
    ),
    tune_label_grid: Optional[str] = typer.Option(
        None,
        help="Comma-separated label-threshold candidates for auto tuning.",
    ),
    tune_signal_grid: Optional[str] = typer.Option(
        None,
        help="Comma-separated signal-threshold candidates for auto tuning.",
    ),
    tune_horizon_grid: Optional[str] = typer.Option(
        None,
        help="Comma-separated horizon candidates for auto tuning; defaults to evaluation horizons.",
    ),
    tune_aggregation_grid: Optional[str] = typer.Option(
        None,
        help="Comma-separated aggregation modes for auto tuning (majority, weighted, unanimous).",
    ),
    tune_window_days: int = typer.Option(
        30,
        min=1,
        help="Number of most recent days from the pretraining window used for auto tuning.",
    ),
    tune_iterations: int = typer.Option(
        5,
        min=1,
        help="Number of episodes the bandit tuner should run.",
    ),
    tune_exploration: float = typer.Option(
        0.35,
        min=0.0,
        max=1.0,
        help="Exploration rate for the epsilon-greedy tuner (1.0 explores fully).",
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

    label_override_candidates = _parse_float_grid(tune_label_grid, "tune-label-grid")
    signal_override_candidates = _parse_float_grid(tune_signal_grid, "tune-signal-grid")
    horizon_override_candidates = _parse_int_grid(tune_horizon_grid, "tune-horizon-grid")
    aggregation_override_candidates = _parse_aggregation_grid(tune_aggregation_grid)

    pretrain_candles_cache: List[dict] | None = None
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
        pretrain_candles_cache = list(pretrain_source)
        pretrain_cache_count = len(pretrain_candles_cache)

    if auto_tune and not pretrain_candles_cache:
        typer.echo(
            "Auto tuning requires --pretrain-days and a non-empty warm-up window.",
            err=True,
        )
        raise typer.Exit(code=1)

    if auto_tune and pretrain_candles_cache:
        label_candidates = _resolve_label_candidates(label_threshold, label_override_candidates)
        signal_candidates = _resolve_signal_candidates(signal_threshold, signal_override_candidates)
        horizon_candidates = _resolve_horizon_candidates(
            prediction_horizon,
            horizon_override_candidates,
            horizons,
        )
        aggregation_candidates = _resolve_aggregation_candidates(
            aggregation_mode,
            aggregation_override_candidates,
        )
        tuning_candles = _select_recent_candles(pretrain_candles_cache, tune_window_days)
        tuned_label, tuned_horizon, tuned_signal, tuned_aggregation, tuning_baseline = _auto_tune_parameters(
            tuning_candles,
            aggregation_candidates=aggregation_candidates,
            label_candidates=label_candidates,
            horizon_candidates=horizon_candidates,
            signal_candidates=signal_candidates,
            transaction_cost=transaction_cost,
            slippage_cost=slippage_cost,
            iterations=tune_iterations,
            exploration_rate=tune_exploration,
        )
        label_threshold = tuned_label
        signal_threshold = tuned_signal
        prediction_horizon = tuned_horizon
        aggregation_mode = tuned_aggregation
        horizons = [tuned_horizon]
        typer.echo(
            "Auto-tune chose aggregation={agg}, label_threshold={lt:.6f}, horizon={hz}, signal_threshold={st:.2f}, pretrain_return={ret:.2%}, trades={trades}".format(
                agg=tuned_aggregation,
                lt=tuned_label,
                hz=tuned_horizon,
                st=tuned_signal,
                ret=math.expm1(tuning_baseline.total_return),
                trades=tuning_baseline.trades,
            )
        )

    pretrain_progress_shown = False

    def run_once(horizon: int) -> BacktestResult:
        nonlocal pretrain_progress_shown
        model = adaptive_classifier()
        metric = default_metrics()
        engine = BacktestEngine(model)
        builder = OnlineFeatureBuilder()

        if pretrain_candles_cache:
            if not pretrain_progress_shown and pretrain_cache_count > 0:
                with _progress(pretrain_cache_count) as pretrain_progress:
                    task = pretrain_progress.add_task("Pretraining", total=pretrain_cache_count)
                    _warm_start_model(
                        pretrain_candles_cache,
                        builder=builder,
                        label_threshold=label_threshold,
                        model=model,
                        progress=pretrain_progress,
                        progress_task=task,
                    )
                pretrain_progress_shown = True
            else:
                _warm_start_model(
                    pretrain_candles_cache,
                    builder=builder,
                    label_threshold=label_threshold,
                    model=model,
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
            if log_return > label_threshold:
                label = 1
            elif log_return < -label_threshold:
                label = -1
            else:
                label = 0
            model.learn_one(previous_features, label)

        previous_features = features
        previous_close = close

        if progress is not None and progress_task is not None:
            progress.advance(progress_task)


def _parse_float_grid(raw: Optional[str], param_name: str) -> List[float]:
    if not raw:
        return []

    values: List[float] = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        try:
            values.append(float(stripped))
        except ValueError as exc:
            typer.echo(f"{param_name} must contain numeric values.", err=True)
            raise typer.Exit(code=1) from exc
    return values


def _parse_int_grid(raw: Optional[str], param_name: str) -> List[int]:
    if not raw:
        return []

    values: List[int] = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        try:
            values.append(int(stripped))
        except ValueError as exc:
            typer.echo(f"{param_name} must contain integer values.", err=True)
            raise typer.Exit(code=1) from exc
    return values


def _parse_aggregation_grid(raw: Optional[str]) -> List[str]:
    if not raw:
        return []

    valid_modes = {"majority", "weighted", "unanimous"}
    modes: List[str] = []
    for token in raw.split(","):
        mode = token.strip().lower()
        if not mode:
            continue
        if mode not in valid_modes:
            typer.echo(
                "tune-aggregation-grid values must be majority, weighted, or unanimous.",
                err=True,
            )
            raise typer.Exit(code=1)
        if mode not in modes:
            modes.append(mode)
    return modes


def _resolve_label_candidates(base: float, overrides: Sequence[float]) -> List[float]:
    if overrides:
        candidates = [value for value in overrides if value > 0]
    else:
        candidates = [base * factor for factor in (0.75, 1.0, 1.25, 1.5)]

    candidates.append(base)
    return sorted({max(value, 1e-6) for value in candidates})


def _resolve_signal_candidates(base: float, overrides: Sequence[float]) -> List[float]:
    if overrides:
        candidates = [value for value in overrides if 0.0 <= value <= 1.0]
    else:
        offsets = (-0.05, 0.0, 0.05)
        candidates = [base + offset for offset in offsets]

    candidates.append(base)
    bounded = [min(0.99, max(0.5, value)) for value in candidates]
    return sorted({round(value, 6) for value in bounded})


def _resolve_horizon_candidates(
    base: int,
    overrides: Sequence[int],
    evaluation_horizons: Sequence[int],
) -> List[int]:
    candidates: List[int]
    if overrides:
        candidates = [value for value in overrides if value > 0]
    else:
        candidates = [value for value in evaluation_horizons if value > 0]
        if not candidates:
            candidates.append(base)
        candidates.extend([max(1, base - 5), base, base + 5])

    candidates.append(base)
    return sorted({value for value in candidates if value > 0})


def _resolve_aggregation_candidates(base: str, overrides: Sequence[str]) -> List[str]:
    valid_modes = ("majority", "weighted", "unanimous")
    ordered: List[str] = []

    def add(mode: str) -> None:
        if mode in valid_modes and mode not in ordered:
            ordered.append(mode)

    if overrides:
        for mode in overrides:
            add(mode)
    else:
        for mode in valid_modes:
            add(mode)

    add(base.lower())

    if not ordered:
        typer.echo("No valid aggregation modes available for auto tuning.", err=True)
        raise typer.Exit(code=1)

    return ordered


def _select_recent_candles(candles: Sequence[dict], window_days: int) -> List[dict]:
    if not candles:
        return []

    if window_days <= 0:
        return list(candles)

    cutoff = _infer_time_cutoff(candles, window_days)
    if cutoff is None:
        approx_count = min(len(candles), window_days * 1_440)
        return list(candles[-approx_count:])

    filtered: List[dict] = []
    for candle in candles:
        ts = _extract_timestamp(candle)
        if ts is not None and ts >= cutoff:
            filtered.append(candle)
    if filtered:
        return filtered

    approx_count = min(len(candles), window_days * 1_440)
    return list(candles[-approx_count:])


@dataclass(frozen=True)
class _ParameterCombo:
    aggregation: str
    label_threshold: float
    prediction_horizon: int
    signal_threshold: float


class _EpsilonGreedyBandit:
    def __init__(self, arms: int, epsilon: float) -> None:
        self._epsilon = max(0.0, min(1.0, epsilon))
        self._values = [0.0 for _ in range(arms)]
        self._counts = [0 for _ in range(arms)]
        self._rng = random.Random()

    def select(self) -> int:
        if self._rng.random() < self._epsilon or not any(self._counts):
            return self._rng.randrange(len(self._values))
        return max(range(len(self._values)), key=self._values.__getitem__)

    def update(self, arm: int, reward: float) -> None:
        self._counts[arm] += 1
        value = self._values[arm]
        self._values[arm] = value + (reward - value) / self._counts[arm]


def _auto_tune_parameters(
    candles: Sequence[dict],
    *,
    aggregation_candidates: Sequence[str],
    label_candidates: Sequence[float],
    horizon_candidates: Sequence[int],
    signal_candidates: Sequence[float],
    transaction_cost: float,
    slippage_cost: float,
    iterations: int,
    exploration_rate: float,
) -> Tuple[float, int, float, str, BacktestResult]:
    if not candles:
        typer.echo("Auto tuning window is empty.", err=True)
        raise typer.Exit(code=1)

    combos: List[_ParameterCombo] = []
    for agg in aggregation_candidates:
        for label in label_candidates:
            if label <= 0:
                continue
            for horizon in horizon_candidates:
                if horizon <= 0:
                    continue
                for signal in signal_candidates:
                    if not 0.0 <= signal <= 1.0:
                        continue
                    combos.append(
                        _ParameterCombo(
                            aggregation=agg,
                            label_threshold=label,
                            prediction_horizon=horizon,
                            signal_threshold=signal,
                        )
                    )

    # evaluate each arm once to seed estimates (reduces time wasted on duplicates)
    initial_evaluations = min(len(combos), iterations)
    first_pass = combos[:initial_evaluations]

    bandit = _EpsilonGreedyBandit(len(combos), exploration_rate)
    best_combo: _ParameterCombo | None = None
    best_result: BacktestResult | None = None
    best_reward = float("-inf")

    for index, combo in enumerate(first_pass):
        result = _evaluate_combo(
            candles,
            combo,
            transaction_cost=transaction_cost,
            slippage_cost=slippage_cost,
        )
        reward = _reward_from_result(result)
        bandit.update(index, reward)
        if reward > best_reward:
            best_reward = reward
            best_combo = combo
            best_result = result

    remaining_iterations = max(0, iterations - initial_evaluations)

    if not combos:
        typer.echo("Auto tuning did not generate any valid parameter combinations.", err=True)
        raise typer.Exit(code=1)

    for _ in range(remaining_iterations):
        arm_index = bandit.select()
        combo = combos[arm_index]

        typer.echo(
            "Auto-tune evaluating aggregation={agg}, label_threshold={lt:.6f}, horizon={hz}, signal_threshold={st:.2f}".format(
                agg=combo.aggregation,
                lt=combo.label_threshold,
                hz=combo.prediction_horizon,
                st=combo.signal_threshold,
            )
        )

        result = _evaluate_combo(
            candles,
            combo,
            transaction_cost=transaction_cost,
            slippage_cost=slippage_cost,
        )

        reward = _reward_from_result(result)
        bandit.update(arm_index, reward)

        if reward > best_reward:
            best_reward = reward
            best_combo = combo
            best_result = result

    if best_combo is None or best_result is None:
        typer.echo("Auto tuning failed to identify an improved parameter set.", err=True)
        raise typer.Exit(code=1)

    return (
        best_combo.label_threshold,
        best_combo.prediction_horizon,
        best_combo.signal_threshold,
        best_combo.aggregation,
        best_result,
    )


def _evaluate_combo(
    candles: Sequence[dict],
    combo: _ParameterCombo,
    *,
    transaction_cost: float,
    slippage_cost: float,
) -> BacktestResult:
    model = adaptive_classifier()
    builder = OnlineFeatureBuilder()
    engine = BacktestEngine(model)
    return engine.run(
        (dict(candle) for candle in candles),
        builder=builder,
        label_threshold=combo.label_threshold,
        prediction_horizon=combo.prediction_horizon,
        aggregation=combo.aggregation,
        signal_threshold=combo.signal_threshold,
        transaction_cost=transaction_cost,
        slippage_cost=slippage_cost,
    )


def _reward_from_result(result: BacktestResult) -> float:
    net_return = result.total_return
    excess_return = net_return - result.buy_hold_return

    reward = net_return + 0.1 * excess_return

    if result.trades == 0:
        reward -= 0.02
    elif result.trades < 5:
        reward -= 0.005 * (5 - result.trades)

    return reward


def _infer_time_cutoff(candles: Sequence[dict], window_days: int) -> Optional[datetime]:
    timestamps = [ts for ts in (_extract_timestamp(c) for c in candles) if ts is not None]
    if not timestamps:
        return None
    latest = max(timestamps)
    return latest - timedelta(days=window_days)


def _extract_timestamp(candle: dict) -> Optional[datetime]:
    for key in ("timestamp", "open_time", "close_time"):
        if key not in candle:
            continue
        raw = candle[key]
        if isinstance(raw, datetime):
            return raw
        if isinstance(raw, (int, float)):
            if raw > 1e12:
                return datetime.utcfromtimestamp(raw / 1_000.0)
            return datetime.utcfromtimestamp(raw)
        if isinstance(raw, str):
            parsed = _parse_timestamp(raw)
            if parsed is not None:
                return parsed
    return None


def _parse_timestamp(raw: str) -> Optional[datetime]:
    value = raw.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


if __name__ == "__main__":  # pragma: no cover
    app()
