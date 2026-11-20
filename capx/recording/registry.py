"""Stable semantic entity registration shared by simulator providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EntitySpec:
    entity_id: str
    semantic_name: str
    entity_type: str
    source_name: str
    source_kind: str = "body"
    static: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)


class EntityRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, EntitySpec] = {}

    def register(self, spec: EntitySpec) -> None:
        existing = self._specs.get(spec.entity_id)
        if existing is not None and existing != spec:
            raise ValueError(f"Entity ID already registered: {spec.entity_id}")
        self._specs[spec.entity_id] = spec

    def register_body(self, *, entity_id: str, semantic_name: str, body_name: str,
                      entity_type: str = "rigid_object", static: bool = False,
                      attributes: dict[str, Any] | None = None) -> None:
        self.register(EntitySpec(entity_id=entity_id, semantic_name=semantic_name,
                                 entity_type=entity_type, source_name=body_name,
                                 source_kind="body", static=static,
                                 attributes=attributes or {}))

    def all(self) -> list[EntitySpec]:
        return list(self._specs.values())

    def source_names(self) -> set[str]:
        return {spec.source_name for spec in self._specs.values()}
