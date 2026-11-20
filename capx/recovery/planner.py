"""Three-step sliding-window planning and ordered execution state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from capx.recovery.schema import PlannedAction


@dataclass
class ThreeStepWindowPlanner:
    """Expose at most three future actions while executing in order."""

    window_size: int = 3

    def __post_init__(self) -> None:
        if self.window_size != 3:
            raise ValueError("The initial harness requires a three-step window")
        self._actions: list[PlannedAction] = []
        self._cursor = 0

    def load(self, actions: Iterable[PlannedAction]) -> None:
        self._actions = list(actions)
        self._cursor = 0

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def done(self) -> bool:
        return self._cursor >= len(self._actions)

    @property
    def remaining(self) -> tuple[PlannedAction, ...]:
        return tuple(self._actions[self._cursor :])

    def current_window(self) -> tuple[PlannedAction, ...]:
        return tuple(self._actions[self._cursor : self._cursor + self.window_size])

    def advance(self) -> PlannedAction:
        if self.done:
            raise IndexError("Cannot advance a completed plan")
        action = self._actions[self._cursor]
        self._cursor += 1
        return action

    def rewind_one(self) -> None:
        if self._cursor > 0:
            self._cursor -= 1

    def replace_remaining(self, actions: Iterable[PlannedAction]) -> None:
        self._actions = self._actions[: self._cursor] + list(actions)
