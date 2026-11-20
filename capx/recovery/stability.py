"""Temporal aggregation for noisy text world-state signals."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any


@dataclass(frozen=True)
class StateWindowConfig:
    window_size: int = 3
    min_stable_fraction: float = 2 / 3
    numeric_abs_tolerance: float = 0.01
    numeric_rel_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not 0.0 < self.min_stable_fraction <= 1.0:
            raise ValueError("min_stable_fraction must be in (0, 1]")
        if self.numeric_abs_tolerance < 0 or self.numeric_rel_tolerance < 0:
            raise ValueError("numeric tolerances must be non-negative")


@dataclass(frozen=True)
class StateWindowSummary:
    state: dict[str, Any]
    confidence: float
    stable: bool
    unstable_signals: tuple[str, ...]
    sample_count: int


class StateWindowAggregator:
    """Reduce the latest K text-state frames using median/majority consensus."""

    def __init__(self, config: StateWindowConfig | None = None) -> None:
        self.config = config or StateWindowConfig()

    def summarize(
        self,
        frames: Sequence[Mapping[str, Any]],
        *,
        fallback: Mapping[str, Any] | None = None,
    ) -> StateWindowSummary:
        selected = [dict(frame) for frame in frames[-self.config.window_size:]]
        if not selected and fallback is not None:
            selected = [dict(fallback)]
        if not selected:
            return StateWindowSummary({}, 0.0, False, (), 0)

        flattened = [self._flatten(frame) for frame in selected]
        keys = sorted(set().union(*(frame.keys() for frame in flattened)))
        aggregate: dict[str, Any] = {}
        confidences: list[float] = []
        unstable: list[str] = []
        for key in keys:
            values = [frame[key] for frame in flattened if key in frame]
            value, confidence = self._aggregate_leaf(values, len(selected))
            self._assign(aggregate, key, value)
            confidences.append(confidence)
            if confidence < self.config.min_stable_fraction:
                unstable.append(key)

        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return StateWindowSummary(
            state=aggregate,
            confidence=round(confidence, 4),
            stable=not unstable,
            unstable_signals=tuple(unstable),
            sample_count=len(selected),
        )

    def _aggregate_leaf(self, values: list[Any], sample_count: int) -> tuple[Any, float]:
        presence = len(values) / sample_count
        if values and self._numeric_sequences(values):
            components = [
                self._aggregate_leaf([value[index] for value in values], len(values))
                for index in range(len(values[0]))
            ]
            return (
                [value for value, _confidence in components],
                presence * sum(confidence for _value, confidence in components) / len(components),
            )
        if values and all(self._is_number(value) for value in values):
            numeric = [float(value) for value in values]
            centre = float(median(numeric))
            tolerance = max(
                self.config.numeric_abs_tolerance,
                abs(centre) * self.config.numeric_rel_tolerance,
            )
            agreeing = sum(abs(value - centre) <= tolerance for value in numeric)
            confidence = presence * agreeing / len(numeric)
            if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
                return int(round(centre)), confidence
            return centre, confidence

        counts = Counter(self._canonical(value) for value in values)
        winner, count = counts.most_common(1)[0]
        chosen = next(value for value in reversed(values) if self._canonical(value) == winner)
        return chosen, presence * count / len(values)

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @classmethod
    def _numeric_sequences(cls, values: list[Any]) -> bool:
        if not all(isinstance(value, (list, tuple)) for value in values):
            return False
        lengths = {len(value) for value in values}
        return len(lengths) == 1 and bool(lengths) and next(iter(lengths)) > 0 and all(
            cls._is_number(item) for value in values for item in value
        )

    @classmethod
    def _canonical(cls, value: Any) -> str:
        if isinstance(value, Mapping):
            return repr(sorted((str(key), cls._canonical(item)) for key, item in value.items()))
        if isinstance(value, (list, tuple)):
            return repr([cls._canonical(item) for item in value])
        return repr(value)

    @classmethod
    def _flatten(cls, value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, Mapping):
                result.update(cls._flatten(item, path))
            else:
                result[path] = item
        return result

    @staticmethod
    def _assign(target: dict[str, Any], path: str, value: Any) -> None:
        current = target
        parts = path.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
