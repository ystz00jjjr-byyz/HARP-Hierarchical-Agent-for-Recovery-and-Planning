"""Deterministic guard against repeated, non-progressing recovery loops."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


GuardAction = Literal["observe", "wait", "retry", "switch_strategy", "abort", "finish"]


@dataclass(frozen=True)
class RecoveryLoopGuardConfig:
    enabled: bool = False
    max_same_failure_retries: int = 2
    min_progress_delta: float = 0.01

    def __post_init__(self) -> None:
        if self.max_same_failure_retries < 0:
            raise ValueError("max_same_failure_retries must be >= 0")
        if self.min_progress_delta < 0:
            raise ValueError("min_progress_delta must be >= 0")


@dataclass(frozen=True)
class RecoveryDirective:
    action: GuardAction
    repeated_failures: int
    progress_detected: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RecoveryLoopGuard:
    """Track semantic progress and prevent execution of a repeated strategy."""

    def __init__(self, config: RecoveryLoopGuardConfig) -> None:
        self.config = config
        self._initialized = False
        self._last_signature: tuple[bool, ...] | None = None
        self._best_rank = -1
        self._best_distance = float("inf")
        self._repeated_failures = 0
        self.directive = RecoveryDirective("observe", 0, False, "initial state")

    def observe(self, state: Mapping[str, Any]) -> RecoveryDirective:
        if not self.config.enabled:
            self.directive = RecoveryDirective("observe", 0, False, "guard disabled")
            return self.directive

        rank = self._progress_rank(state)
        distance = self._distance(state)
        signature = self._failure_signature(state)
        # Placement alone is not completion for tasks that also require release.
        completed = bool(state.get("task_completed"))

        if not self._initialized:
            self._initialized = True
            self._last_signature = signature
            self._best_rank = rank
            self._best_distance = distance
            self.directive = RecoveryDirective("observe", 0, False, "initial state")
            return self.directive

        if completed:
            self.directive = RecoveryDirective("finish", 0, True, "task completion observed")
            return self.directive

        rank_progress = rank > self._best_rank
        distance_progress = distance + self.config.min_progress_delta < self._best_distance
        progress = rank_progress or distance_progress
        if progress:
            self._best_rank = max(self._best_rank, rank)
            self._best_distance = min(self._best_distance, distance)
            self._repeated_failures = 0
            action: GuardAction = "observe"
            reason = "semantic task progress observed"
        else:
            if signature == self._last_signature:
                self._repeated_failures += 1
            else:
                self._repeated_failures = 1
            if self._repeated_failures <= self.config.max_same_failure_retries:
                action = "retry"
                reason = "same failure may be retried within the configured budget"
            elif self._repeated_failures == self.config.max_same_failure_retries + 1:
                action = "switch_strategy"
                reason = "retry budget exhausted; use a structurally different action sequence"
            else:
                action = "abort"
                reason = "strategy switch made no semantic progress"

        self._last_signature = signature
        self.directive = RecoveryDirective(
            action, self._repeated_failures, progress, reason
        )
        return self.directive

    def accepts_candidate(self, candidate_code: str, previous_code: str) -> bool:
        if not self.config.enabled or self.directive.action != "switch_strategy":
            return True
        if not candidate_code.strip():
            return False
        return strategy_fingerprint(candidate_code) != strategy_fingerprint(previous_code)

    def wait(self, reason: str) -> RecoveryDirective:
        """Request verification without consuming the failure retry budget."""
        self.directive = RecoveryDirective(
            "wait", self._repeated_failures, False, reason
        )
        return self.directive

    @staticmethod
    def _progress_rank(state: Mapping[str, Any]) -> int:
        if state.get("task_completed") or state.get("object_at_target"):
            return 4
        if state.get("target_grasped") and state.get("object_lifted"):
            return 3
        if state.get("object_lifted"):
            return 2
        if state.get("target_grasped"):
            return 1
        if state.get("restack_precondition_cleared"):
            return 1
        return 0

    @staticmethod
    def _distance(state: Mapping[str, Any]) -> float:
        value = state.get("distance_to_target", float("inf"))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")

    @staticmethod
    def _failure_signature(state: Mapping[str, Any]) -> tuple[bool, ...]:
        keys = (
            "wrong_object_grasped",
            "target_grasped",
            "object_lifted",
            "object_at_target",
            "object_stable",
            "gripper_open",
            "gripper_closed",
            "restack_precondition_cleared",
        )
        return tuple(bool(state.get(key, False)) for key in keys)


def strategy_fingerprint(code: str) -> tuple[str, ...]:
    """Return the ordered robot-API call structure, ignoring numeric tweaks."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ("<invalid-python>",)

    calls: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in {
            "home_pose", "open_gripper", "close_gripper", "goto_pose",
            "get_object_pose", "sample_grasp_pose",
        }:
            calls.append((getattr(node, "lineno", 0), getattr(node, "col_offset", 0), name))
    return tuple(name for _line, _column, name in sorted(calls))


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""
