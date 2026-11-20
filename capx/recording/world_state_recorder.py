"""Backend-neutral side-channel world-state trajectory recorder."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from capx.recording.schema import WorldStateSnapshot
from capx.recording.sinks import JsonlNpzSink


@dataclass(frozen=True)
class WorldStateRecorderConfig:
    enabled: bool = False
    record_every_n_sim_steps: int = 5
    record_rgb: bool = True
    record_depth: bool = True
    compress_arrays: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "WorldStateRecorderConfig":
        value = value or {}
        known = {field.name for field in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"Unknown world_state_recorder options: {sorted(unknown)}")
        cfg = cls(**value)
        if cfg.record_every_n_sim_steps < 1:
            raise ValueError("record_every_n_sim_steps must be >= 1")
        return cfg


class WorldStateRecorder:
    """Serialize provider snapshots and semantic events through a pluggable sink."""

    schema_version = 2

    def __init__(self, output_dir: str | Path, config: WorldStateRecorderConfig,
                 *, episode_id: str = "episode") -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.episode_id = episode_id
        self.sink = JsonlNpzSink(self.output_dir, compress_arrays=config.compress_arrays)
        self._record_index = 0
        self._closed = False
        (self.output_dir / "manifest.json").write_text(json.dumps({
            "schema_version": self.schema_version,
            "created_at_unix": time.time(),
            "episode_id": episode_id,
            "config": self._jsonable(config.__dict__),
            "privileged_side_channel": True,
            "agent_visible": False,
            "storage": "jsonl_npz",
        }, indent=2), encoding="utf-8")

    @property
    def next_sequence_id(self) -> int:
        return self._record_index

    def should_record_sim_step(self, sim_step: int) -> bool:
        return sim_step % self.config.record_every_n_sim_steps == 0

    def record_snapshot(self, event: str, snapshot: WorldStateSnapshot,
                        *, metadata: dict[str, Any] | None = None) -> None:
        if self._closed:
            return
        arrays: dict[str, np.ndarray] = {}
        if snapshot.action is not None:
            arrays["action"] = np.asarray(snapshot.action)
        for camera in snapshot.cameras:
            safe_name = camera.camera_id.replace("/", "_")
            if self.config.record_rgb and camera.rgb is not None:
                arrays[f"{safe_name}__rgb"] = np.asarray(camera.rgb)
            if self.config.record_depth and camera.depth is not None:
                arrays[f"{safe_name}__depth"] = np.asarray(camera.depth)

        world_state = asdict(snapshot)
        for camera in world_state["cameras"]:
            camera.pop("rgb", None)
            camera.pop("depth", None)
        entry = {
            "schema_version": self.schema_version,
            "record_index": self._record_index,
            "wall_time_unix": time.time(),
            "event": event,
            "sim_step": snapshot.simulator_step,
            "sim_time": snapshot.simulator_time,
            "world_state": self._jsonable(world_state),
            "metadata": self._jsonable(metadata or {}),
        }
        self.sink.write(entry, arrays)
        self._record_index += 1

    def record(self, event: str, *, sim_step: int | None = None,
               sim_time: float | None = None, state: dict[str, Any] | None = None,
               observation: dict[str, Any] | None = None, action: Any = None,
               metadata: dict[str, Any] | None = None) -> None:
        """Compatibility writer for callers that have not migrated to a provider."""
        if self._closed:
            return
        arrays: dict[str, np.ndarray] = {}
        if action is not None:
            arrays["action"] = np.asarray(action)
        if observation is not None:
            for name, camera in observation.items():
                if not isinstance(camera, dict) or not isinstance(camera.get("images"), dict):
                    continue
                safe_name = str(name).replace("/", "_")
                if self.config.record_rgb and "rgb" in camera["images"]:
                    arrays[f"{safe_name}__rgb"] = np.asarray(camera["images"]["rgb"])
                if self.config.record_depth and "depth" in camera["images"]:
                    arrays[f"{safe_name}__depth"] = np.asarray(camera["images"]["depth"])
        entry = {"schema_version": self.schema_version, "record_index": self._record_index,
                 "wall_time_unix": time.time(), "event": event,
                 "sim_step": sim_step, "sim_time": sim_time,
                 "state": self._jsonable(state or {}), "metadata": self._jsonable(metadata or {})}
        self.sink.write(entry, arrays)
        self._record_index += 1

    def close(self) -> None:
        if not self._closed:
            self.sink.close()
            self._closed = True

    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, Path): return str(value)
        if isinstance(value, dict): return {str(k): cls._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [cls._jsonable(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None: return value
        return repr(value)

    def __enter__(self) -> "WorldStateRecorder": return self
    def __exit__(self, *_args: Any) -> None: self.close()
