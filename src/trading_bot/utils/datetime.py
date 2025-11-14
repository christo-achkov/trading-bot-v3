"""Datetime helpers shared across the project."""
from __future__ import annotations

from datetime import datetime, timezone


def ensure_utc(ts: datetime) -> datetime:
    """Force a naive datetime into UTC for API compatibility."""

    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def parse_iso8601(value: str) -> datetime:
    """Parse an ISO8601 string and return a timezone-aware datetime in UTC."""

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid ISO8601 datetime: {value}") from exc

    return ensure_utc(parsed)
