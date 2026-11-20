"""Transparent per-step scoring for text world-state signals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from capx.recovery.schema import ExecutionOutcome, PlannedAction, StepScore, ValidationResult
from capx.recovery.stability import StateWindowAggregator, StateWindowSummary


def _value(state: Mapping[str, Any], key: str) -> Any:
    current: Any = state
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _first(state: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = _value(state, key)
        if value is not None:
            return value
    return None


def _unit(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _error_score(value: Any, maximum: float) -> float | None:
    if value is None:
        return None
    try:
        return min(1.0, max(0.0, 1.0 - float(value) / maximum))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _weighted_available(parts: list[tuple[float, float | None]], fallback: float) -> float:
    available = [(weight, value) for weight, value in parts if value is not None]
    if not available:
        return fallback
    weight_sum = sum(weight for weight, _ in available)
    return sum(weight * float(value) for weight, value in available) / weight_sum


@dataclass(frozen=True)
class ScoreWeights:
    goal: float = 0.50
    progress: float = 0.20
    execution: float = 0.15
    confidence: float = 0.15


class PrimitiveStepScorer:
    """Compute a 0-100 score from pre/post text world state.

    A safety violation always scores zero. A failed post-condition caps the
    total below the pass boundary even if softer components look favourable.
    """

    SAFETY_SIGNALS = (
        "collision",
        "safety_violation",
        "out_of_bounds",
        "wrong_object_grasped",
    )

    def __init__(self, weights: ScoreWeights | None = None,
                 state_aggregator: StateWindowAggregator | None = None) -> None:
        self.weights = weights or ScoreWeights()
        self.state_aggregator = state_aggregator or StateWindowAggregator()

    def summarize_outcome(self, outcome: ExecutionOutcome) -> StateWindowSummary:
        return self.state_aggregator.summarize(outcome.state_window, fallback=outcome.state)

    def score(
        self,
        action: PlannedAction,
        *,
        pre_state: Mapping[str, Any],
        outcome: ExecutionOutcome,
        validation: ValidationResult,
        state_summary: StateWindowSummary | None = None,
    ) -> StepScore:
        summary = state_summary or self.summarize_outcome(outcome)
        state = summary.state
        safety_evidence = {
            key: bool(_value(state, key))
            for key in self.SAFETY_SIGNALS
            if _value(state, key) is not None
        }
        safety_violation = any(safety_evidence.values())
        execution_score = self._execution_score(outcome)
        goal_score = self._goal_score(action.primitive, state, validation)
        progress_score = self._progress_score(action.primitive, pre_state, state)
        confidence_score = self._confidence_score(state) * summary.confidence

        raw = (
            self.weights.goal * goal_score
            + self.weights.progress * progress_score
            + self.weights.execution * execution_score
            + self.weights.confidence * confidence_score
        )
        total = round(100.0 * min(1.0, max(0.0, raw)), 2)
        if not validation.ok:
            total = min(total, 69.0)
        if safety_violation:
            total = 0.0

        decision = self._decision(
            total=total,
            validation=validation,
            progress_score=progress_score,
            execution_score=execution_score,
            safety_violation=safety_violation,
            state=state,
        )
        failed_rules = () if validation.ok else (validation.message,)
        return StepScore(
            action_id=action.action_id,
            primitive=action.primitive,
            total=total,
            hard_pass=validation.ok and not safety_violation,
            execution_score=round(execution_score, 4),
            goal_score=round(goal_score, 4),
            progress_score=round(progress_score, 4),
            confidence_score=round(confidence_score, 4),
            safety_penalty=1.0 if safety_violation else 0.0,
            decision=decision,
            failed_rules=failed_rules,
            evidence={
                "pre_state": dict(pre_state),
                "post_state": dict(state),
                "safety": safety_evidence,
                "state_window_size": summary.sample_count,
                "state_window_stable": summary.stable,
                "state_window_confidence": summary.confidence,
                "unstable_signals": list(summary.unstable_signals),
            },
        )

    @staticmethod
    def _execution_score(outcome: ExecutionOutcome) -> float:
        explicit = outcome.metadata.get("execution_ok")
        if explicit is not None:
            return 1.0 if explicit else 0.0
        if outcome.stderr or outcome.truncated:
            return 0.0
        return 1.0

    def _goal_score(
        self, primitive: str, state: Mapping[str, Any], validation: ValidationResult
    ) -> float:
        primitive = primitive.strip().lower()
        fallback = 1.0 if validation.ok else 0.0
        if primitive in {"move_to", "goto_pose", "align"}:
            return _weighted_available([
                (0.55, _unit(_first(state, ("eef_aligned", "target_aligned")))),
                (0.30, _error_score(_first(state, ("alignment_error", "eef_alignment_error")), 0.10)),
                (0.15, _error_score(_value(state, "orientation_error"), 0.50)),
            ], fallback)
        if primitive == "rotate":
            return _weighted_available([
                (0.65, _unit(_value(state, "orientation_aligned"))),
                (0.35, _error_score(_value(state, "orientation_error"), 0.50)),
            ], fallback)
        if primitive == "grasp":
            return _weighted_available([
                (0.50, _unit(_first(state, ("target_grasped", "object_grasped")))),
                (0.25, _unit(_value(state, "relative_pose_stable"))),
                (0.15, _unit(_value(state, "object_lifted"))),
                (0.10, _unit(_value(state, "gripper_contact"))),
            ], fallback)
        if primitive == "place":
            return _weighted_available([
                (0.50, _unit(_first(state, ("object_at_target", "placed")))),
                (0.25, _unit(_value(state, "object_stable"))),
                (0.15, _unit(_value(state, "orientation_aligned"))),
                (0.10, _error_score(_value(state, "distance_to_target"), 0.20)),
            ], fallback)
        if primitive in {"release", "open_gripper"}:
            return _weighted_available([
                (0.45, _unit(_value(state, "gripper_open"))),
                (0.35, _unit(_first(state, ("object_at_target", "placed")))),
                (0.20, _unit(_value(state, "object_stable"))),
            ], fallback)
        if primitive == "close_gripper":
            return _weighted_available([
                (0.70, _unit(_value(state, "gripper_closed"))),
                (0.30, _unit(_value(state, "gripper_contact"))),
            ], fallback)
        return fallback

    @staticmethod
    def _progress_score(
        primitive: str, pre_state: Mapping[str, Any], post_state: Mapping[str, Any]
    ) -> float:
        primitive = primitive.strip().lower()
        keys = {
            "move_to": ("alignment_error", "eef_alignment_error"),
            "goto_pose": ("alignment_error", "eef_alignment_error"),
            "align": ("alignment_error", "eef_alignment_error"),
            "rotate": ("orientation_error",),
            "place": ("distance_to_target",),
        }.get(primitive)
        if keys:
            before = _first(pre_state, keys)
            after = _first(post_state, keys)
            if before is not None and after is not None:
                try:
                    before_f, after_f = float(before), float(after)
                    if before_f <= 1e-9:
                        return 1.0 if after_f <= before_f else 0.0
                    return min(1.0, max(0.0, (before_f - after_f) / before_f))
                except (TypeError, ValueError):
                    pass
        transition_keys = {
            "grasp": ("target_grasped", "object_grasped"),
            "place": ("object_at_target", "placed"),
            "release": ("gripper_open",),
            "open_gripper": ("gripper_open",),
            "close_gripper": ("gripper_closed",),
        }.get(primitive, ())
        before = _first(pre_state, transition_keys)
        after = _first(post_state, transition_keys)
        if after is not None:
            if bool(after) and not bool(before):
                return 1.0
            if bool(after):
                return 0.75
            return 0.0
        return 0.5

    @staticmethod
    def _confidence_score(state: Mapping[str, Any]) -> float:
        confidence = _value(state, "confidence")
        if confidence is None:
            return 1.0
        if isinstance(confidence, Mapping):
            values = [_unit(value) for value in confidence.values()]
            valid = [value for value in values if value is not None]
            return sum(valid) / len(valid) if valid else 1.0
        return _unit(confidence, 1.0) or 0.0

    @staticmethod
    def _decision(*, total: float, validation: ValidationResult, progress_score: float,
                  execution_score: float, safety_violation: bool,
                  state: Mapping[str, Any]) -> str:
        if safety_violation:
            return "abort"
        if execution_score == 0.0:
            return "replan"
        if validation.ok:
            if total >= 85.0:
                return "pass"
            if total >= 70.0:
                return "verify"
            return "retry"
        if progress_score >= 0.5 or bool(_value(state, "recoverable_environment")):
            return "retry"
        return "replan"
