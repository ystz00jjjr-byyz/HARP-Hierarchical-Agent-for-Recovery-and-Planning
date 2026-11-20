"""Opt-in text world-state feedback for CaP-X code-block trials."""

from __future__ import annotations

import json
import hashlib
from collections import deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from capx.recovery.loop_guard import (
    RecoveryDirective,
    RecoveryLoopGuard,
    RecoveryLoopGuardConfig,
)
from capx.recovery.stability import (
    StateWindowAggregator,
    StateWindowConfig,
    StateWindowSummary,
)


@dataclass(frozen=True)
class TextWorldStateFeedbackConfig:
    enabled: bool = False
    window_size: int = 3
    min_stable_fraction: float = 2 / 3
    numeric_abs_tolerance: float = 0.01
    numeric_rel_tolerance: float = 0.05
    include_entities: bool = True
    fail_open: bool = True
    loop_guard_enabled: bool = False
    max_same_failure_retries: int = 2
    min_progress_delta: float = 0.01
    action_aligned_window: bool = False
    wait_for_stability: bool = True
    stability_wait_steps: int = 20
    sample_sim_steps: int = 3

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "TextWorldStateFeedbackConfig":
        value = value or {}
        known = {field.name for field in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"Unknown text_world_state_feedback options: {sorted(unknown)}")
        config = cls(**value)
        StateWindowConfig(
            window_size=config.window_size,
            min_stable_fraction=config.min_stable_fraction,
            numeric_abs_tolerance=config.numeric_abs_tolerance,
            numeric_rel_tolerance=config.numeric_rel_tolerance,
        )
        RecoveryLoopGuardConfig(
            enabled=config.loop_guard_enabled,
            max_same_failure_retries=config.max_same_failure_retries,
            min_progress_delta=config.min_progress_delta,
        )
        if config.stability_wait_steps < 0:
            raise ValueError("stability_wait_steps must be >= 0")
        if config.sample_sim_steps < 1:
            raise ValueError("sample_sim_steps must be >= 1")
        return config


class TrialWorldStateFeedback:
    """Capture live state and expose only a compact text summary."""

    def __init__(self, env: Any, config: TextWorldStateFeedbackConfig) -> None:
        self.env = env
        self.config = config
        self.frames: deque[dict[str, Any]] = deque(maxlen=config.window_size)
        self.aggregator = StateWindowAggregator(StateWindowConfig(
            window_size=config.window_size,
            min_stable_fraction=config.min_stable_fraction,
            numeric_abs_tolerance=config.numeric_abs_tolerance,
            numeric_rel_tolerance=config.numeric_rel_tolerance,
        ))
        self.history: list[dict[str, Any]] = []
        self.last_error: str | None = None
        self.loop_guard = RecoveryLoopGuard(RecoveryLoopGuardConfig(
            enabled=config.loop_guard_enabled,
            max_same_failure_retries=config.max_same_failure_retries,
            min_progress_delta=config.min_progress_delta,
        ))
        self.last_directive = RecoveryDirective("observe", 0, False, "initial state")

    def capture(self) -> StateWindowSummary | None:
        try:
            state = self._read_compact_state()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            if self.config.fail_open:
                return None
            raise
        self.frames.append(state)
        summary = self.aggregator.summarize(tuple(self.frames))
        self.last_directive = self.loop_guard.observe(summary.state)
        self.history.append({
            "raw_state": state,
            "summary": asdict(summary),
            "recovery_directive": self.last_directive.to_dict(),
        })
        self.last_error = None
        return summary

    def capture_action_window(self) -> StateWindowSummary | None:
        """Clear stale frames and sample K post-action states from one phase."""
        if not self.config.action_aligned_window:
            return self.capture()
        try:
            self.frames.clear()
            state = self._read_compact_state()
            if self.config.wait_for_stability:
                for _ in range(self.config.stability_wait_steps):
                    if bool(state.get("object_stable", False)):
                        break
                    self._advance_simulation(1)
                    state = self._read_compact_state()

            sampled: list[dict[str, Any]] = []
            for index in range(self.config.window_size):
                if index:
                    self._advance_simulation(self.config.sample_sim_steps)
                sampled.append(self._read_compact_state())
            self.frames.extend(sampled)
            summary = self.aggregator.summarize(tuple(self.frames))
            if not summary.stable or not bool(summary.state.get("object_stable", False)):
                self.last_directive = self.loop_guard.wait(
                    "post-action state is still moving or lacks K-frame consensus"
                )
            else:
                self.last_directive = self.loop_guard.observe(summary.state)
            self.history.append({
                "capture_mode": "action_aligned",
                "raw_state": sampled[-1],
                "raw_window": sampled,
                "summary": asdict(summary),
                "recovery_stage": derive_recovery_stage(summary.state),
                "recovery_directive": self.last_directive.to_dict(),
            })
            self.last_error = None
            return summary
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            if self.config.fail_open:
                return None
            raise

    def capture_static_window(self) -> StateWindowSummary | None:
        """Build a fresh K-frame window without stepping a terminated episode."""
        try:
            self.frames.clear()
            sampled = [
                self._read_compact_state()
                for _ in range(self.config.window_size)
            ]
            self.frames.extend(sampled)
            summary = self.aggregator.summarize(tuple(self.frames))
            self.last_directive = self.loop_guard.observe(summary.state)
            self.history.append({
                "capture_mode": "static_final",
                "raw_state": sampled[-1],
                "raw_window": sampled,
                "summary": asdict(summary),
                "recovery_stage": derive_recovery_stage(summary.state),
                "recovery_directive": self.last_directive.to_dict(),
            })
            self.last_error = None
            return summary
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            if self.config.fail_open:
                return None
            raise

    def _read_compact_state(self) -> dict[str, Any]:
        state = capture_text_world_state(
            self.env, include_entities=self.config.include_entities
        )
        # These fields are useful for trajectory replay but are volatile,
        # extremely verbose, and not semantic recovery evidence.
        state.pop("simulator_state", None)
        state.pop("simulator_step", None)
        state.pop("simulator_time", None)
        return state

    def _advance_simulation(self, steps: int) -> None:
        low_level = getattr(self.env, "low_level_env", self.env)
        step_once = getattr(low_level, "_step_once", None)
        if not callable(step_once):
            return
        for _ in range(steps):
            step_once()

    def prompt_text(self, summary: StateWindowSummary | None) -> str | None:
        if summary is None:
            return None
        payload = {
            "sample_count": summary.sample_count,
            "stable": summary.stable,
            "confidence": summary.confidence,
            "unstable_signals": list(summary.unstable_signals),
            "state": summary.state,
            "recovery_directive": self.last_directive.to_dict(),
            "recovery_stage": derive_recovery_stage(summary.state),
        }
        source_label = (
            "sensor-derived recovery signal"
            if summary.state.get("uses_simulator_truth") is False
            else "privileged recovery signal"
        )
        prompt = (
            f"Text world-state feedback ({source_label}; verify before acting):\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )
        if self.last_directive.action == "switch_strategy":
            prompt += (
                "\nHARD RECOVERY CONSTRAINT: the same failure has exhausted its two retries. "
                "REGENERATE only with a structurally different robot-action sequence; changing "
                "only offsets, heights, comments, or variable names is not a strategy switch."
            )
        return prompt

    @property
    def should_abort(self) -> bool:
        return self.last_directive.action == "abort"

    @property
    def should_wait(self) -> bool:
        return self.last_directive.action == "wait"

    def accepts_regeneration(self, candidate_code: str, previous_code: str) -> bool:
        return self.loop_guard.accepts_candidate(candidate_code, previous_code)


def capture_text_world_state(env: Any, *, include_entities: bool = True) -> dict[str, Any]:
    """Capture one provider snapshot and convert it to JSON-friendly semantics."""
    low_level = getattr(env, "low_level_env", env)
    custom = getattr(low_level, "get_text_world_state", None)
    if callable(custom):
        return dict(custom())

    provider = getattr(low_level, "world_state_provider", None)
    if provider is None:
        raise RuntimeError("environment does not expose a world-state provider")
    snapshot = provider.capture(
        episode_id="live_recovery",
        sequence_id=0,
        observation=None,
    )
    state: dict[str, Any] = dict(snapshot.task_state)
    state["simulator_step"] = snapshot.simulator_step
    state["simulator_time"] = snapshot.simulator_time
    if include_entities:
        entities: dict[str, Any] = {}
        for entity in snapshot.entities:
            item: dict[str, Any] = {
                "entity_type": entity.entity_type,
                "attributes": _jsonable(entity.attributes),
            }
            if entity.pose is not None:
                item["position"] = _jsonable(entity.pose.position)
                item["quaternion_wxyz"] = _jsonable(entity.pose.quaternion_wxyz)
            entities[entity.semantic_name or entity.entity_id] = item
        state["entities"] = entities
    return _jsonable(state)


def append_text_world_state_feedback(prompt: str, feedback: str | None) -> str:
    return prompt if not feedback else f"{prompt}\n\n{feedback}"


def derive_recovery_stage(state: dict[str, Any]) -> str:
    if state.get("task_completed"):
        return "TASK_COMPLETED"
    if state.get("wrong_object_grasped"):
        return "WRONG_OBJECT_GRASPED"
    if state.get("object_at_target") and not state.get("gripper_open"):
        return "PLACED_NOT_RELEASED"
    if state.get("object_at_target") and not state.get("object_stable"):
        return "RELEASED_UNSTABLE"
    if state.get("object_at_target"):
        return "VERIFY_STACK"
    if state.get("target_grasped") and state.get("object_lifted"):
        return "LIFTED_READY_TO_PLACE"
    if state.get("target_grasped"):
        return "GRASPED_NOT_LIFTED"
    if state.get("restack_precondition_cleared"):
        return "ACQUIRE_TARGET_AFTER_CLEAR"
    return "ACQUIRE_GRASP"


def save_initial_state_fingerprint(env: Any, output_dir: str | Path, trial: int) -> dict[str, Any] | None:
    """Persist a compact reset-state identity without exposing it to the model."""
    try:
        state = capture_text_world_state(env, include_entities=False)
    except Exception:
        return None
    poses = state.get("cube_poses_robot_base")
    if not isinstance(poses, dict):
        return None
    canonical = {
        str(name): [round(float(value), 6) for value in pose]
        for name, pose in sorted(poses.items())
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    payload = {
        "trial": trial,
        "fingerprint": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "cube_poses_robot_base": canonical,
    }
    path = Path(output_dir) / f"initial_state_fingerprint_trial_{trial:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
