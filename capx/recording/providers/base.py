from __future__ import annotations

from typing import Any, Protocol

from capx.recording.schema import WorldStateSnapshot


class WorldStateProvider(Protocol):
    def capture(self, *, episode_id: str, sequence_id: int,
                observation: dict[str, Any] | None = None,
                action: Any = None) -> WorldStateSnapshot: ...
