"""Tests for datetime utilities."""
from datetime import datetime, timezone

import pytest

from trading_bot.utils import ensure_utc, parse_iso8601


def test_parse_iso8601_supports_z_suffix() -> None:
    parsed = parse_iso8601("2020-01-01T00:00:00Z")
    assert parsed == datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)


def test_parse_iso8601_raises_for_invalid_input() -> None:
    with pytest.raises(ValueError):
        parse_iso8601("not-a-date")


def test_ensure_utc_adds_timezone() -> None:
    naive = datetime(2020, 1, 1, 12, 0)
    result = ensure_utc(naive)
    assert result.tzinfo is timezone.utc
