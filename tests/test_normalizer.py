from trading_bot.features.engineer import OnlineFeatureBuilder, UniversalOnlineNormalizer


def test_eps_fallback_and_bounded():
    builder = OnlineFeatureBuilder()
    # empty histories -> MAD==0 -> scale should be floored to eps
    norm = UniversalOnlineNormalizer(builder, eps=1e-3)
    # transform a small and a large value; both must be finite and in (-1,1)
    small = norm.transform("volume", 0.0)
    large = norm.transform("return_1m", 123456.0)
    assert isinstance(small, float)
    assert isinstance(large, float)
    import math
    assert math.isfinite(small)
    assert math.isfinite(large)
    assert -1.0 <= small <= 1.0
    assert -1.0 <= large <= 1.0


def test_serialization_roundtrip(tmp_path):
    builder = OnlineFeatureBuilder()
    # seed builder histories
    for v in [1.0, 2.0, 3.0, 4.0]:
        builder._close_history.append(v)
    for v in [10.0, 20.0, 30.0]:
        builder._volume_history.append(v)
    norm = UniversalOnlineNormalizer(builder, eps=1e-4, window_map={"return_1m": 3})
    p = tmp_path / "norm.json"
    norm.save(str(p))

    # create new builder and load
    builder2 = OnlineFeatureBuilder()
    loaded = UniversalOnlineNormalizer.load(builder2, str(p))
    # ensure eps and window_map restored
    assert abs(loaded._eps - 1e-4) < 1e-12
    assert loaded._window_map.get("return_1m") == 3
    # histories restored
    assert list(builder2._close_history) == [1.0, 2.0, 3.0, 4.0]
    assert list(builder2._volume_history) == [10.0, 20.0, 30.0]
