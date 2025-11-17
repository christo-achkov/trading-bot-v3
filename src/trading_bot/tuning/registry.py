"""Knob registry utilities used by the tuning CLI."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import tomllib


@dataclass(frozen=True)
class KnobDefinition:
    """Metadata describing how a single knob should be tuned."""

    name: str
    group: str
    type: str
    stage: str
    default: Any
    objective: str | None
    description: str | None
    grid: Sequence[Any]
    minimum: float | None
    maximum: float | None
    step: float | None
    choices: Sequence[Any]
    distribution: str | None

    def cast(self, value: Any) -> Any:
        """Convert ``value`` to the declared knob type."""

        if self.type == "float":
            return float(value)
        if self.type == "int":
            return int(value)
        if self.type == "bool":
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes", "on"}:
                    return True
                if lowered in {"0", "false", "no", "off"}:
                    return False
                raise ValueError(f"Cannot coerce value '{value}' into boolean for knob '{self.name}'")
            return bool(value)
        if self.type == "categorical":
            return value
        raise ValueError(f"Unsupported knob type '{self.type}' for knob '{self.name}'")

    def grid_values(self) -> Sequence[Any]:
        """Return explicit grid values if defined, otherwise a sensible default."""

        if self.grid:
            return [self.cast(v) for v in self.grid]
        if self.type == "bool":
            return [False, True]
        if self.type == "categorical" and self.choices:
            return list(self.choices)
        return []

    def can_sample(self) -> bool:
        if self.type in {"float", "int"}:
            return self.minimum is not None and self.maximum is not None and self.minimum < self.maximum
        if self.type == "categorical":
            return bool(self.choices)
        if self.type == "bool":
            return True
        return False

    def random_value(self, rng: random.Random) -> Any:
        """Draw a random value respecting the knob's metadata."""

        if self.type == "bool":
            return rng.choice([False, True])

        if self.type == "categorical":
            if not self.choices:
                return self.default
            return rng.choice(list(self.choices))

        if self.type == "float":
            if not self.can_sample():
                return float(self.default)
            if self.distribution == "log-uniform":
                low = math.log(self.minimum)
                high = math.log(self.maximum)
                return math.exp(rng.uniform(low, high))
            value = rng.uniform(self.minimum, self.maximum)
            if self.step:
                steps = round((value - self.minimum) / self.step)
                value = self.minimum + steps * self.step
            return float(value)

        if self.type == "int":
            if not self.can_sample():
                return int(self.default)
            if self.step and self.step > 0:
                count = int((self.maximum - self.minimum) / self.step) + 1
                idx = rng.randrange(count)
                return int(self.minimum + idx * self.step)
            return rng.randint(int(self.minimum), int(self.maximum))

        raise ValueError(f"Unsupported knob type '{self.type}' for knob '{self.name}'")

    def optuna_suggest(self, trial: Any) -> Any:
        """Ask Optuna for a sample matching the knob definition."""

        if self.type == "bool":
            return trial.suggest_categorical(self.name, [False, True])

        if self.type == "categorical":
            if not self.choices:
                return self.default
            return trial.suggest_categorical(self.name, list(self.choices))

        if self.type == "float":
            if not self.can_sample():
                return float(self.default)
            kwargs: MutableMapping[str, Any] = {"name": self.name, "low": float(self.minimum), "high": float(self.maximum)}
            if self.step:
                kwargs["step"] = float(self.step)
            if self.distribution == "log-uniform":
                kwargs["log"] = True
            return trial.suggest_float(**kwargs)

        if self.type == "int":
            if not self.can_sample():
                return int(self.default)
            kwargs = {"name": self.name, "low": int(self.minimum), "high": int(self.maximum)}
            if self.step and self.step > 0:
                kwargs["step"] = int(self.step)
            return trial.suggest_int(**kwargs)

        raise ValueError(f"Unsupported knob type '{self.type}' for knob '{self.name}'")


class KnobRegistry:
    """Loads and serves knob definitions from ``config/knobs.toml``."""

    def __init__(self, *, knobs: Sequence[KnobDefinition], group_descriptions: Mapping[str, str]) -> None:
        self._knobs = list(knobs)
        self._group_descriptions = dict(group_descriptions)
        self._by_name = {knob.name: knob for knob in self._knobs}

    @classmethod
    def load(cls, path: str | Path) -> "KnobRegistry":
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Knob registry file not found: {resolved}")
        with resolved.open("rb") as handle:
            payload = tomllib.load(handle)

        raw_groups = payload.get("groups", {})
        group_descriptions: Dict[str, str] = {
            key: str(value.get("description", "")) for key, value in raw_groups.items()
        }

        raw_knobs: Iterable[Mapping[str, Any]] = payload.get("knobs", [])
        knob_definitions: List[KnobDefinition] = []
        for entry in raw_knobs:
            try:
                knob = KnobDefinition(
                    name=str(entry["name"]),
                    group=str(entry["group"]),
                    type=str(entry.get("type", "float")).lower(),
                    stage=str(entry.get("stage", "trading")),
                    default=entry.get("default"),
                    objective=str(entry.get("objective")) if entry.get("objective") else None,
                    description=str(entry.get("description")) if entry.get("description") else None,
                    grid=list(entry.get("grid", [])),
                    minimum=float(entry["min"]) if entry.get("min") is not None else None,
                    maximum=float(entry["max"]) if entry.get("max") is not None else None,
                    step=float(entry["step"]) if entry.get("step") is not None else None,
                    choices=list(entry.get("choices", [])),
                    distribution=str(entry.get("distribution")) if entry.get("distribution") else None,
                )
            except KeyError as exc:  # pragma: no cover - defensive
                raise KeyError(f"Missing required field {exc} in knob entry: {entry}") from exc

            if knob.group not in group_descriptions:
                group_descriptions.setdefault(knob.group, "")

            knob_definitions.append(knob)

        return cls(knobs=knob_definitions, group_descriptions=group_descriptions)

    def groups(self) -> Mapping[str, str]:
        return dict(sorted(self._group_descriptions.items()))

    def get(self, name: str) -> KnobDefinition:
        try:
            return self._by_name[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown knob '{name}'") from exc

    def defaults(self, *, groups: Sequence[str] | None = None) -> Dict[str, Any]:
        selected = self.collect(groups=groups)
        return {knob.name: knob.cast(knob.default) for knob in selected}

    def collect(
        self,
        *,
        groups: Sequence[str] | None = None,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> List[KnobDefinition]:
        include_names = {name for name in (include or [])}
        exclude_names = {name for name in (exclude or [])}
        group_filter = {group for group in (groups or [])}

        collected: List[KnobDefinition] = []
        seen: set[str] = set()

        def _add(knob: KnobDefinition) -> None:
            if knob.name in seen:
                return
            if knob.name in exclude_names:
                return
            collected.append(knob)
            seen.add(knob.name)

        if include_names:
            for name in include_names:
                if name in self._by_name:
                    _add(self._by_name[name])

        for knob in self._knobs:
            if group_filter and knob.group not in group_filter:
                continue
            _add(knob)

        if not group_filter and not include_names:
            for knob in self._knobs:
                _add(knob)

        return collected