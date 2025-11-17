"""Unified autotuning CLI powered by the knob registry."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

try:
    import optuna  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    optuna = None

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from trading_bot.backtest import BacktestEngine
from trading_bot.config import load_settings
from trading_bot.data import BinanceRESTClient, fetch_candles
from trading_bot.features import OnlineFeatureBuilder
from trading_bot.models import adaptive_regressor, default_metrics, isotonic_calibrator
from trading_bot.models.calibration import OnlineIsotonicCalibrator
from trading_bot.tuning import KnobDefinition, KnobRegistry
from trading_bot.utils import parse_iso8601


@dataclass
class Guardrails:
    min_trades: int
    min_sharpe: float | None = None


@dataclass
class SegmentSummary:
    index: int
    trades: int
    hit_rate: float
    strat_return: float
    buy_hold: float
    sharpe: float
    costs: float
    pnl: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "trades": self.trades,
            "hit_rate": self.hit_rate,
            "strat_return_log": self.strat_return,
            "strat_return": math.expm1(self.strat_return),
            "buy_hold_log": self.buy_hold,
            "buy_hold": math.expm1(self.buy_hold),
            "sharpe": self.sharpe,
            "costs_log": self.costs,
            "costs": math.expm1(self.costs) - 1.0,
            "pnl_log": self.pnl,
            "pnl": math.expm1(self.pnl) - 1.0,
        }


@dataclass
class SweepResult:
    params: Dict[str, Any]
    overrides: Dict[str, Any]
    origin: str
    trades: int
    hit_rate: float
    strat_return: float
    buy_hold: float
    sharpe: float
    costs: float
    pnl: float
    walk_forward: int
    segments: List[SegmentSummary] = field(default_factory=list)

    def to_row(self) -> str:
        knob_bits = ", ".join(f"{key}={self.overrides[key]}" for key in sorted(self.overrides)) or "defaults"
        return (
            f"origin={self.origin}, knobs={{{knob_bits}}}, trades={self.trades}, "
            f"hit={self.hit_rate:.2%}, strat={math.expm1(self.strat_return):.2%}, "
            f"sharpe={self.sharpe:.2f}, costs={math.expm1(self.costs) - 1:.2%}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.origin,
            "walk_forward": self.walk_forward,
            "params": self.params,
            "overrides": self.overrides,
            "metrics": {
                "trades": self.trades,
                "hit_rate": self.hit_rate,
                "strat_return_log": self.strat_return,
                "strat_return": math.expm1(self.strat_return),
                "buy_hold_log": self.buy_hold,
                "buy_hold": math.expm1(self.buy_hold),
                "sharpe": self.sharpe,
                "costs_log": self.costs,
                "costs": math.expm1(self.costs) - 1.0,
                "pnl_log": self.pnl,
                "pnl": math.expm1(self.pnl) - 1.0,
            },
            "segments": [segment.to_dict() for segment in self.segments],
        }


@dataclass
class PreparedData:
    pretrain_candles: List[dict]
    backtest_candles: List[dict]
    transaction_cost: float
    slippage_cost: float


class _TextProgress:
    """Minimal console progress reporter when tqdm is unavailable."""

    def __init__(self, description: str, total: int) -> None:
        self._description = description
        self._total = max(total, 0)
        self._count = 0
        if self._total > 0:
            print(f"{self._description}: 0/{self._total} started", flush=True)

    def update(self, step: int = 1) -> None:
        if self._total <= 0:
            return
        self._count = min(self._count + max(step, 0), self._total)
        milestone = max(1, self._total // 10)
        if self._count in {self._total} or self._count % milestone == 0:
            print(
                f"{self._description}: {self._count}/{self._total} complete",
                flush=True,
            )

    def close(self) -> None:
        if self._total > 0 and self._count < self._total:
            print(
                f"{self._description}: {self._count}/{self._total} complete",
                flush=True,
            )


@contextmanager
def _progress_tracker(total: int, *, description: str) -> Iterator[Any] | None:
    if total <= 0:
        yield None
        return

    if tqdm is not None:
        progress = tqdm(total=total, desc=description, leave=False)
        try:
            yield progress
        finally:
            progress.close()
    else:
        tracker = _TextProgress(description, total)
        try:
            yield tracker
        finally:
            tracker.close()


def _warm_start_model(
    candles: Iterable[dict],
    *,
    builder: OnlineFeatureBuilder,
    model,
    calibrator: OnlineIsotonicCalibrator | None,
    trade_cost: float,
    cost_adjust_training: bool,
) -> None:
    previous_features: dict[str, float] | None = None
    previous_close: float | None = None

    total = len(candles) if isinstance(candles, Sequence) else -1

    with _progress_tracker(total, description="Warm starting model") as progress:
        for candle in candles:
            features = builder.process(candle)
            close = float(candle["close"])

            if previous_features is not None and previous_close is not None:
                log_return = math.log(max(close, 1e-12) / max(previous_close, 1e-12))
                target = log_return - trade_cost if cost_adjust_training else log_return
                raw_pred = model.predict_one(previous_features)
                predicted_edge = float(raw_pred) if raw_pred is not None else 0.0
                if calibrator is not None:
                    calibrator.learn_one({"prediction": predicted_edge}, target)
                model.learn_one(previous_features, target)

            previous_features = features
            previous_close = close

            if progress is not None:
                progress.update(1)


def _split_segments(candles: Sequence[dict], splits: int) -> List[List[dict]]:
    if splits <= 1 or not candles:
        return [list(candles)] if candles else []
    total = len(candles)
    segment_size = max(1, total // splits)
    segments: List[List[dict]] = []
    idx = 0
    for _ in range(splits - 1):
        next_idx = min(idx + segment_size, total)
        chunk = candles[idx:next_idx]
        if chunk:
            segments.append(list(chunk))
        idx = next_idx
    if idx < total:
        segments.append(list(candles[idx:]))
    return segments


def _simulate_once(
    params: Dict[str, Any],
    *,
    pretrain_candles: Sequence[dict],
    backtest_candles: Sequence[dict],
    transaction_cost: float,
    slippage_cost: float,
    trade_cost: float,
) -> SegmentSummary:
    builder = OnlineFeatureBuilder()
    model = adaptive_regressor(
        optimizer_name="sgd",
        learning_rate=float(params.get("learning_rate", 0.001)),
        intercept_lr=float(params.get("intercept_lr", 0.001)),
        l2=float(params.get("l2", 0.001)),
        clip_gradient=float(params.get("clip_gradient", 0.5)),
    )

    calibrator: OnlineIsotonicCalibrator | None = None
    if bool(params.get("use_calibrator", True)):
        calibrator = isotonic_calibrator(
            window_size=int(params.get("iso_window_size", 1024)),
            min_samples=int(params.get("iso_min_samples", 160)),
        )

    engine = BacktestEngine(model, calibrator=calibrator)

    cost_adjust_training = bool(params.get("cost_adjust_training", False))
    if pretrain_candles:
        _warm_start_model(
            pretrain_candles,
            builder=builder,
            model=model,
            calibrator=calibrator,
            trade_cost=trade_cost,
            cost_adjust_training=cost_adjust_training,
        )

    spread_cost_scale = float(params.get("spread_cost_scale", 0.005))
    volatility_cost_scale = float(params.get("volatility_cost_scale", 0.1))
    liquidity_cost_scale = float(params.get("liquidity_cost_scale", 0.0))
    use_dynamic_cost = bool(params.get("use_dynamic_cost", True)) or any(
        scale > 0 for scale in (spread_cost_scale, volatility_cost_scale, liquidity_cost_scale)
    )

    edge_clip_value = params.get("edge_clip", 0.0)
    edge_clip = float(edge_clip_value) if float(edge_clip_value or 0.0) > 0.0 else None
    long_threshold = params.get("long_threshold")
    short_threshold = params.get("short_threshold")

    result = engine.run(
        backtest_candles,
        builder=builder,
        edge_threshold=float(params.get("edge_threshold", 0.0005)),
        transaction_cost=transaction_cost,
        slippage_cost=slippage_cost,
        metrics=[default_metrics()],
        diagnostics=None,
        edge_clip=edge_clip,
        trade_log=None,
        long_threshold=float(long_threshold) if long_threshold is not None else None,
        short_threshold=float(short_threshold) if short_threshold is not None else None,
        adaptive_threshold=bool(params.get("adaptive_threshold", True)),
        volatility_feature=str(params.get("volatility_feature", "volatility_regime")),
        volatility_scale=float(params.get("volatility_scale", 0.06)),
        volatility_offset=float(params.get("volatility_offset", 0.006)),
        minimum_cushion=float(params.get("minimum_cushion", 0.0022)),
        bull_cushion_offset=float(params.get("bull_cushion_offset", 0.0)),
        bear_cushion_offset=float(params.get("bear_cushion_offset", 0.0)),
        use_position_sizing=bool(params.get("use_position_sizing", True)),
        position_scale=float(params.get("position_scale", 150.0)),
        edge_scale=float(params.get("edge_scale", 1.0)),
        hysteresis=float(params.get("hysteresis", 0.0003)),
        use_dynamic_cost=use_dynamic_cost,
        spread_cost_feature=str(params.get("spread_cost_feature", "micro_spread_bps")),
        spread_cost_scale=spread_cost_scale,
        volatility_cost_feature=str(params.get("volatility_cost_feature", "return_std_short")),
        volatility_cost_scale=volatility_cost_scale,
        liquidity_cost_feature=params.get("liquidity_cost_feature", "liquidity_density_50bps"),
        liquidity_cost_scale=liquidity_cost_scale,
        dynamic_cost_floor=float(params.get("dynamic_cost_floor", trade_cost)),
        dynamic_cost_cap=float(params.get("dynamic_cost_cap")) if params.get("dynamic_cost_cap") else None,
        skip_low_edge_trades=bool(params.get("skip_low_edge_trades", True)),
        turnover_penalty=float(params.get("turnover_penalty", 0.0)),
        cost_adjust_training=cost_adjust_training,
    )

    return SegmentSummary(
        index=0,
        trades=result.trades,
        hit_rate=result.hit_rate,
        strat_return=result.total_return,
        buy_hold=result.buy_hold_return,
        sharpe=result.sharpe_ratio,
        costs=result.total_costs,
        pnl=result.pnl,
    )


def _run_walk_forward(
    params: Dict[str, Any],
    *,
    data: PreparedData,
    trade_cost: float,
    walk_forward: int,
) -> Tuple[Dict[str, float], List[SegmentSummary]]:
    if walk_forward <= 1:
        with _progress_tracker(1, description="Simulating walk-forward segments") as progress:
            segment = _simulate_once(
                params,
                pretrain_candles=data.pretrain_candles,
                backtest_candles=data.backtest_candles,
                transaction_cost=data.transaction_cost,
                slippage_cost=data.slippage_cost,
                trade_cost=trade_cost,
            )
            segment.index = 0
            if progress is not None:
                progress.update(1)
        aggregated = {
            "trades": segment.trades,
            "hit_rate": segment.hit_rate,
            "strat_return": segment.strat_return,
            "buy_hold": segment.buy_hold,
            "sharpe": segment.sharpe,
            "costs": segment.costs,
            "pnl": segment.pnl,
        }
        return aggregated, [segment]

    rolling_pretrain = list(data.pretrain_candles)
    segments: List[SegmentSummary] = []

    chunks = _split_segments(data.backtest_candles, walk_forward)

    with _progress_tracker(len(chunks), description="Simulating walk-forward segments") as progress:
        for idx, chunk in enumerate(chunks):
            segment = _simulate_once(
                params,
                pretrain_candles=rolling_pretrain,
                backtest_candles=chunk,
                transaction_cost=data.transaction_cost,
                slippage_cost=data.slippage_cost,
                trade_cost=trade_cost,
            )
            segment.index = idx
            segments.append(segment)
            rolling_pretrain.extend(chunk)
            if progress is not None:
                progress.update(1)

    aggregated = _aggregate_segments(segments)
    return aggregated, segments


def _aggregate_segments(segments: Sequence[SegmentSummary]) -> Dict[str, float]:
    if not segments:
        return {
            "trades": 0,
            "hit_rate": 0.0,
            "strat_return": 0.0,
            "buy_hold": 0.0,
            "sharpe": 0.0,
            "costs": 0.0,
            "pnl": 0.0,
        }

    total_trades = sum(segment.trades for segment in segments)
    total_wins = sum(segment.hit_rate * segment.trades for segment in segments)
    total_return = sum(segment.strat_return for segment in segments)
    total_buy_hold = sum(segment.buy_hold for segment in segments)
    total_costs = sum(segment.costs for segment in segments)
    total_pnl = sum(segment.pnl for segment in segments)
    avg_sharpe = sum(segment.sharpe for segment in segments) / len(segments)

    hit_rate = total_wins / total_trades if total_trades else 0.0

    return {
        "trades": total_trades,
        "hit_rate": hit_rate,
        "strat_return": total_return,
        "buy_hold": total_buy_hold,
        "sharpe": avg_sharpe,
        "costs": total_costs,
        "pnl": total_pnl,
    }


def prepare_data(*, start: str | None, end: str | None, warmup_days: int) -> PreparedData:
    settings = load_settings()
    symbol = settings.data.symbol
    interval = settings.data.interval

    default_start = parse_iso8601(settings.data.start_date)
    start_dt = parse_iso8601(start) if start else default_start
    end_dt = parse_iso8601(end) if end else datetime.now(timezone.utc)
    pretrain_start = start_dt - timedelta(days=warmup_days) if warmup_days > 0 else None

    transaction_cost = settings.backtest.fee_bps / 10_000.0
    slippage_cost = settings.backtest.slippage_bps / 10_000.0

    chunk_minutes = settings.data.fetch_chunk_minutes

    pretrain: List[dict] = []

    with BinanceRESTClient(
        settings.binance.api_key,
        settings.binance.api_secret,
        base_url=settings.binance.base_url,
    ) as client:
        if pretrain_start is not None and pretrain_start < start_dt:
            pretrain = fetch_candles(
                client,
                symbol=symbol,
                interval=interval,
                start=pretrain_start,
                end=start_dt,
                chunk_minutes=chunk_minutes,
            )

        backtest = fetch_candles(
            client,
            symbol=symbol,
            interval=interval,
            start=start_dt,
            end=end_dt,
            chunk_minutes=chunk_minutes,
        )

    return PreparedData(
        pretrain_candles=pretrain,
        backtest_candles=backtest,
        transaction_cost=transaction_cost,
        slippage_cost=slippage_cost,
    )


def _normalise_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    return value


def _make_key(params: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((name, _normalise_value(value)) for name, value in params.items()))


def generate_grid_configs(knobs: Sequence[KnobDefinition], limit: int | None) -> List[Dict[str, Any]]:
    if not knobs:
        return []
    value_lists: List[Tuple[str, Sequence[Any]]] = []
    for knob in knobs:
        values = knob.grid_values()
        if values:
            value_lists.append((knob.name, values))
    if not value_lists:
        return []

    configs: List[Dict[str, Any]] = []
    for combo in product(*[values for _, values in value_lists]):
        overrides = {name: value for (name, _), value in zip(value_lists, combo)}
        configs.append(overrides)
        if limit and limit > 0 and len(configs) >= limit:
            break
    return configs


def generate_random_configs(
    knobs: Sequence[KnobDefinition],
    *,
    samples: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if not knobs or samples <= 0:
        return []
    configs: List[Dict[str, Any]] = []
    for _ in range(samples):
        overrides = {knob.name: knob.random_value(rng) for knob in knobs if knob.can_sample()}
        configs.append(overrides)
    return configs


def evaluate_configuration(
    *,
    params: Dict[str, Any],
    overrides: Dict[str, Any],
    origin: str,
    data: PreparedData,
    trade_cost: float,
    walk_forward: int,
    guardrails: Guardrails,
) -> SweepResult | None:
    aggregated, segments = _run_walk_forward(params, data=data, trade_cost=trade_cost, walk_forward=walk_forward)

    trades = int(aggregated["trades"])
    if trades < guardrails.min_trades:
        return None
    sharpe = aggregated["sharpe"]
    if guardrails.min_sharpe is not None and sharpe < guardrails.min_sharpe:
        return None

    return SweepResult(
        params=dict(params),
        overrides=dict(overrides),
        origin=origin,
        trades=trades,
        hit_rate=aggregated["hit_rate"],
        strat_return=aggregated["strat_return"],
        buy_hold=aggregated["buy_hold"],
        sharpe=sharpe,
        costs=aggregated["costs"],
        pnl=aggregated["pnl"],
        walk_forward=walk_forward,
        segments=segments,
    )


def run_grid_search(
    *,
    knobs: Sequence[KnobDefinition],
    base_params: Dict[str, Any],
    data: PreparedData,
    guardrails: Guardrails,
    trade_cost: float,
    walk_forward: int,
    limit: int | None,
    seen: set[Tuple[Tuple[str, Any], ...]],
) -> List[SweepResult]:
    results: List[SweepResult] = []
    configs = generate_grid_configs(knobs, limit)

    with _progress_tracker(len(configs), description="Grid search") as progress:
        for overrides in configs:
            try:
                params = dict(base_params)
                params.update(overrides)
                key = _make_key(params)
                if key in seen:
                    continue
                result = evaluate_configuration(
                    params=params,
                    overrides=overrides,
                    origin="grid",
                    data=data,
                    trade_cost=trade_cost,
                    walk_forward=walk_forward,
                    guardrails=guardrails,
                )
                if result is None:
                    continue
                seen.add(key)
                results.append(result)
            finally:
                if progress is not None:
                    progress.update(1)
    return results


def run_random_search(
    *,
    knobs: Sequence[KnobDefinition],
    base_params: Dict[str, Any],
    data: PreparedData,
    guardrails: Guardrails,
    trade_cost: float,
    walk_forward: int,
    samples: int,
    seed: int,
    seen: set[Tuple[Tuple[str, Any], ...]],
) -> List[SweepResult]:
    rng = random.Random(seed)
    results: List[SweepResult] = []
    candidates = generate_random_configs(knobs, samples=samples, rng=rng)

    with _progress_tracker(len(candidates), description="Random search") as progress:
        for overrides in candidates:
            try:
                params = dict(base_params)
                params.update(overrides)
                key = _make_key(params)
                if key in seen:
                    continue
                result = evaluate_configuration(
                    params=params,
                    overrides=overrides,
                    origin="random",
                    data=data,
                    trade_cost=trade_cost,
                    walk_forward=walk_forward,
                    guardrails=guardrails,
                )
                if result is None:
                    continue
                seen.add(key)
                results.append(result)
            finally:
                if progress is not None:
                    progress.update(1)
    return results


def run_optuna_search(
    *,
    knobs: Sequence[KnobDefinition],
    base_params: Dict[str, Any],
    data: PreparedData,
    guardrails: Guardrails,
    trade_cost: float,
    walk_forward: int,
    trials: int,
    seed: int,
    seen: set[Tuple[Tuple[str, Any], ...]],
) -> List[SweepResult]:
    if optuna is None or not knobs or trials <= 0:
        return []

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="autotune")
    collected: List[SweepResult] = []

    def objective(trial: Any) -> float:
        overrides: Dict[str, Any] = {}
        for knob in knobs:
            overrides[knob.name] = knob.optuna_suggest(trial)
        params = dict(base_params)
        params.update(overrides)
        key = _make_key(params)
        if key in seen:
            trial.set_user_attr("duplicate", True)
            return float("-inf")
        result = evaluate_configuration(
            params=params,
            overrides=overrides,
            origin="optuna",
            data=data,
            trade_cost=trade_cost,
            walk_forward=walk_forward,
            guardrails=guardrails,
        )
        if result is None:
            trial.set_user_attr("violated_guard", True)
            return float("-inf")
        seen.add(key)
        collected.append(result)
        trial.set_user_attr("sweep_result", result)
        return result.strat_return

    study.optimize(objective, n_trials=trials, show_progress_bar=tqdm is not None)

    # Sort unique results by their achieved return.
    unique: Dict[Tuple[Tuple[str, Any], ...], SweepResult] = {}
    for result in collected:
        unique[_make_key(result.params)] = result

    ordered = sorted(unique.values(), key=lambda item: item.strat_return, reverse=True)
    return ordered


def write_outputs(results: Sequence[SweepResult], *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_payload = [result.to_dict() for result in results]
    json_path = output_dir / "sweep_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, indent=2)

    param_names = sorted({name for result in results for name in result.params.keys()})
    csv_path = output_dir / "sweep_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "origin",
                "walk_forward",
                "trades",
                "hit_rate",
                "strat_return_log",
                "strat_return",
                "buy_hold_log",
                "buy_hold",
                "sharpe",
                "costs_log",
                "costs",
                *param_names,
            ]
        )
        for result in results:
            row = [
                result.origin,
                result.walk_forward,
                result.trades,
                result.hit_rate,
                result.strat_return,
                math.expm1(result.strat_return),
                result.buy_hold,
                math.expm1(result.buy_hold),
                result.sharpe,
                result.costs,
                math.expm1(result.costs) - 1.0,
            ]
            row.extend(result.params.get(name) for name in param_names)
            writer.writerow(row)


def _parse_overrides(raw: Sequence[str], registry: KnobRegistry) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Overrides must use name=value syntax (received '{item}')")
        name, value = item.split("=", 1)
        knob = registry.get(name)
        overrides[name] = knob.cast(value)
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autotune trading knobs with grid, random, and Optuna search")
    parser.add_argument("--registry", default="config/knobs.toml", help="Path to the knob registry TOML")
    parser.add_argument("--groups", nargs="*", help="Knob groups to include (omit or use 'all' for every group)")
    parser.add_argument("--include", nargs="*", default=[], help="Explicit knob names to force include")
    parser.add_argument("--exclude", nargs="*", default=[], help="Knob names to drop from tuning")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Pin knob value via name=value")
    parser.add_argument("--search", nargs="*", choices=["grid", "random", "optuna"], default=["grid", "optuna"], help="Search strategies to run")
    parser.add_argument("--grid-limit", type=int, default=64, help="Maximum grid combinations to evaluate")
    parser.add_argument("--random-samples", type=int, default=0, help="Number of random samples to draw")
    parser.add_argument("--optuna-trials", type=int, default=int(os.getenv("OPTUNA_TRIALS", "40")), help="Optuna trial budget")
    parser.add_argument("--seed", type=int, default=7123, help="Deterministic seed for random draws")
    parser.add_argument("--start", type=str, help="Optional ISO8601 start for the backtest window")
    parser.add_argument("--end", type=str, help="Optional ISO8601 end for the backtest window")
    parser.add_argument("--warmup-days", type=int, default=1520, help="Warmup days loaded before the start date")
    parser.add_argument("--walk-forward", type=int, default=1, help="Number of walk-forward splits for evaluation")
    parser.add_argument("--min-trades", type=int, default=5, help="Minimum trades required for a configuration")
    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=-5.0,
        help="Sharpe floor for accepting configurations (default: -5.0)",
    )
    parser.add_argument("--sort-by", choices=["strat_return", "sharpe", "hit_rate"], default="strat_return", help="Sort key for final report")
    parser.add_argument("--output-dir", default="diagnostics", help="Destination directory for results")
    parser.add_argument("--skip-default", dest="evaluate_default", action="store_false", help="Skip evaluating the base configuration")
    parser.set_defaults(evaluate_default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    registry = KnobRegistry.load(args.registry)
    group_selection = None
    if args.groups:
        lowered = {group.lower() for group in args.groups}
        if "all" not in lowered:
            group_selection = args.groups

    selected_knobs = registry.collect(groups=group_selection, include=args.include, exclude=args.exclude)
    if not selected_knobs:
        raise RuntimeError("No knobs selected for tuning")

    base_defaults = {knob.name: knob.cast(knob.default) for knob in selected_knobs}
    overrides = _parse_overrides(args.overrides, registry) if args.overrides else {}
    base_params = dict(base_defaults)
    base_params.update(overrides)

    adjustable_knobs = [knob for knob in selected_knobs if knob.name not in overrides]

    data = prepare_data(start=args.start, end=args.end, warmup_days=args.warmup_days)
    trade_cost = data.transaction_cost + data.slippage_cost
    guardrails = Guardrails(min_trades=args.min_trades, min_sharpe=args.min_sharpe)

    seen: set[Tuple[Tuple[str, Any], ...]] = set()
    results: List[SweepResult] = []

    if args.evaluate_default:
        print("Evaluating baseline configuration...", flush=True)
        key = _make_key(base_params)
        result = evaluate_configuration(
            params=base_params,
            overrides=overrides,
            origin="baseline",
            data=data,
            trade_cost=trade_cost,
            walk_forward=args.walk_forward,
            guardrails=guardrails,
        )
        if result is not None:
            seen.add(key)
            results.append(result)

    if "grid" in args.search and adjustable_knobs:
        print("Running grid search...", flush=True)
        results.extend(
            run_grid_search(
                knobs=adjustable_knobs,
                base_params=base_params,
                data=data,
                guardrails=guardrails,
                trade_cost=trade_cost,
                walk_forward=args.walk_forward,
                limit=args.grid_limit,
                seen=seen,
            )
        )

    if "random" in args.search and args.random_samples > 0 and adjustable_knobs:
        print("Running random search...", flush=True)
        results.extend(
            run_random_search(
                knobs=adjustable_knobs,
                base_params=base_params,
                data=data,
                guardrails=guardrails,
                trade_cost=trade_cost,
                walk_forward=args.walk_forward,
                samples=args.random_samples,
                seed=args.seed,
                seen=seen,
            )
        )

    if "optuna" in args.search and optuna is None:
        print("Optuna not installed; skipping Optuna search.")

    if "optuna" in args.search and args.optuna_trials > 0 and optuna is not None and adjustable_knobs:
        print("Running Optuna search...", flush=True)
        results.extend(
            run_optuna_search(
                knobs=adjustable_knobs,
                base_params=base_params,
                data=data,
                guardrails=guardrails,
                trade_cost=trade_cost,
                walk_forward=args.walk_forward,
                trials=args.optuna_trials,
                seed=args.seed,
                seen=seen,
            )
        )

    if not results:
        print("No configurations satisfied guardrails; nothing to report.")
        return

    sort_key = {
        "strat_return": lambda item: item.strat_return,
        "sharpe": lambda item: item.sharpe,
        "hit_rate": lambda item: item.hit_rate,
    }[args.sort_by]

    results.sort(key=sort_key, reverse=True)

    for result in results:
        print(result.to_row())

    output_dir = Path(args.output_dir)
    write_outputs(results, output_dir=output_dir)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
