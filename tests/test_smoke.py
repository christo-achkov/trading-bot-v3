"""Smoke tests ensuring project wiring works."""
from trading_bot import __version__


def test_version() -> None:
    assert __version__ == "0.1.0"
