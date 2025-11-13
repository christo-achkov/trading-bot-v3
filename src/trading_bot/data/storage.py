"""Persistence helpers for market data batches."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trading_bot.data.binance_downloader import CandleBatch


@dataclass
class ParquetBatchWriter:
    """Persist candle batches to a partitioned Parquet dataset."""

    root_dir: Path
    partition_cols: tuple[str, ...] = ("symbol", "interval", "year", "month", "day")

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def write(self, batch: CandleBatch) -> List[Path]:
        """Write a batch to the partitioned dataset and return affected directories."""

        if not batch.candles:
            return []

        frame = pd.DataFrame(batch.candles)
        frame["symbol"] = batch.symbol
        frame["interval"] = batch.interval
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
        frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True)
        frame["year"] = frame["open_time"].dt.year.astype("int16")
        frame["month"] = frame["open_time"].dt.month.astype("int8")
        frame["day"] = frame["open_time"].dt.day.astype("int8")

        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(self.root_dir),
            partition_cols=list(self.partition_cols),
            existing_data_behavior="overwrite_or_ignore",
        )

        partitions = frame[[col for col in self.partition_cols if col in frame.columns]].drop_duplicates()
        written_paths: List[Path] = []
        for _, row in partitions.iterrows():
            path = self.root_dir
            for col in self.partition_cols:
                if col not in row:
                    continue
                value = row[col]
                path = path / f"{col}={str(value)}"
            written_paths.append(path)

        return written_paths
