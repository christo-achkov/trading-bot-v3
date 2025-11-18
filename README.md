# trading-bot-v3

Adaptive crypto trading bot built with online machine learning and production-grade architecture.

## Features
- Streaming data ingestion from Binance
- Feature generation with incremental statistics
- Drift-aware machine learning models using the [River](https://github.com/online-ml/river) library
- Walk-forward backtesting engine with execution simulation
- Optional Parquet capture of live-enriched candles (price + order book)
- Modular configuration and logging for production deployment

## Getting Started

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]

# fetch historical candles into data/raw/binance
python -m scripts.download_binance download

# run a walk-forward backtest on stored data
python -m scripts.backtest run
```

### Environment variables
Copy `config/settings.example.toml` to `config/settings.toml` and adjust to your credentials.

## Project Layout
```
├── config/             # Configuration templates and secrets (gitignored)
├── data/               # Data lake partitioned by source and stage
├── notebooks/          # Research notebooks
├── scripts/            # Operational entrypoints (downloaders, runners)
├── src/trading_bot/    # Production code (ingestion, features, models, execution)
└── tests/              # Automated test suite
```

## Quality Gates
- `ruff` for linting
- `black` for formatting (line length 100)
- `mypy` for static typing
- `pytest` with coverage for regression protection

## Roadmap
1. Implement historical data downloader and persistence
2. Build online feature pipeline and model ensemble
3. Develop backtesting engine with execution and risk modules
4. Integrate live trading adapter with monitoring and alerting
