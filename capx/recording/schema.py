"""Backend-neutral schemas for recorded world state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Pose:
    position: np.ndarray
    quaternion_wxyz: np.ndarray


@dataclass
class EntityState:
    entity_id: str
    entity_type: str
    semantic_name: str
    pose: Pose | None = None
    linear_velocity: np.ndarray | None = None
    angular_velocity: np.ndarray | None = None
    joint_positions: np.ndarray | None = None
    joint_velocities: np.ndarray | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraState:
    camera_id: str
    pose: Pose | None
    intrinsics: np.ndarray | None
    width: int | None
    height: int | None
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None


@dataclass
class WorldStateSnapshot:
    schema_version: int
    episode_id: str
    sequence_id: int
    simulator_step: int
    simulator_time: float
    coordinate_frame: str
    entities: list[EntityState]
    cameras: list[CameraState]
    action: Any = None
    reward: float | None = None
    terminated: bool | None = None
    truncated: bool | None = None
    task_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldStateEvent:
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
