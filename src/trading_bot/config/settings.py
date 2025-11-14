"""Configuration management for the trading bot."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

try:  # pragma: no cover - import shim for Python < 3.11
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class BinanceSettings(BaseModel):
    """Binance REST/WebSocket connection parameters."""

    api_key: str = Field(..., min_length=10)
    api_secret: str = Field(..., min_length=10)
    base_url: str = Field("https://api.binance.com")


class DataSettings(BaseModel):
    """Data acquisition configuration."""

    symbol: str = Field("BTCUSDT")
    interval: str = Field("1m")
    start_date: str = Field(..., description="ISO8601 inclusive start timestamp")
    fetch_chunk_minutes: int = Field(1000, ge=1)
    raw_data_dir: Path = Field(Path("data/raw/binance"))


class ModelSettings(BaseModel):
    """Model registry and persistence configuration."""

    state_dir: Path = Field(Path("data/models"))
    model_name: str = Field("river_adaptive_ensemble")


class BacktestSettings(BaseModel):
    """Backtesting defaults for capital and frictions."""

    initial_equity: float = Field(100_000.0, ge=0.0)
    fee_bps: float = Field(7.5, ge=0.0)
    slippage_bps: float = Field(5.0, ge=0.0)


class AppSettings(BaseSettings):
    """Application-wide configuration composed from individual domains."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_nested_delimiter="__",
        env_file=".env",
    )

    binance: BinanceSettings
    data: DataSettings
    model: ModelSettings = ModelSettings()
    backtest: BacktestSettings = BacktestSettings()


def load_settings(path: Optional[Path] = None) -> AppSettings:
    """Load settings from a TOML file, falling back to environment variables."""

    if path is None:
        path = Path("config/settings.toml")

    if path.exists():
        raw_data = tomllib.loads(path.read_text())
        return AppSettings.model_validate(raw_data)

    try:
        return AppSettings()
    except ValidationError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            f"Unable to load configuration. Provide {path} or the relevant environment variables."
        ) from exc
