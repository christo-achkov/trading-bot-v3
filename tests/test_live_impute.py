from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta

from trading_bot.features.engineer import OnlineFeatureBuilder
from trading_bot.live.session import LiveMarketSession


class DummyModel:
    def __init__(self):
        self.learn_calls = []

    def predict_one(self, x):
        return 0.0

    def learn_one(self, x, y):
        self.learn_calls.append((x, y))


class DummyAggregator:
    def __init__(self):
        self.preloaded = None

    def preload_candles(self, candles):
        self.preloaded = list(candles)


class FakeRecorder:
    def __init__(self, records):
        self._records = records

    def read_recent(self, *, limit: int = 2000):
        return list(self._records)


def make_candle(ts: datetime, close: float = 100.0):
    return {
        "open_time": ts,
        "close_time": ts,
        "event_time": ts,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1.0,
    }


def test_builder_process_imputed_skips_history():
    builder = OnlineFeatureBuilder()
    ts = datetime.now(timezone.utc)
    c = make_candle(ts, close=100.0)
    # normal process updates histories
    builder.process(c)
    len_before = len(builder._close_history)
    # imputed process should not append to _close_history
    c2 = make_candle(ts + timedelta(minutes=1), close=101.0)
    builder.process(c2, imputed=True)
    assert len(builder._close_history) == len_before


def test_warm_start_imputes_and_skips_learning():
    # create two candles separated by 2 intervals -> one imputed row expected
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    c1 = make_candle(now, close=100.0)
    c2 = make_candle(now + timedelta(minutes=3), close=102.0)

    recorder = FakeRecorder([c1, c2])
    builder = OnlineFeatureBuilder()
    model = DummyModel()
    agg = DummyAggregator()

    session = LiveMarketSession(
        symbol="BTCUSDC",
        interval="1m",
        candle_history=10,
        fetch_chunk_minutes=100,
        rest_client=None,
        stream=None,
        aggregator=agg,
        builder=builder,
        model=model,
        calibrator=None,
        trade_cost=0.0,
        cost_adjust_training=False,
        logger=None,
        recorder=recorder,
    )

    asyncio.run(session._warm_start())

    # Only one learn call should have occurred (transition from real->real), imputed skipped
    assert len(model.learn_calls) == 1
