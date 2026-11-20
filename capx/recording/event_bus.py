"""Small synchronous event bus for recorder-independent instrumentation."""

from __future__ import annotations

from collections.abc import Callable

from capx.recording.schema import WorldStateEvent


class WorldStateEventBus:
    def __init__(self) -> None:
        self._subscribers: list[Callable[[WorldStateEvent], None]] = []

    def subscribe(self, callback: Callable[[WorldStateEvent], None]) -> None:
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def emit(self, event_type: str, payload: dict | None = None) -> None:
        event = WorldStateEvent(event_type=event_type, payload=payload or {})
        for callback in tuple(self._subscribers):
            callback(event)
