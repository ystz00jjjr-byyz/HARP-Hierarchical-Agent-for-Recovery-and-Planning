"""Minimal three-step recovery harness for existing CaP-X multi-turn execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from capx.recovery.planner import ThreeStepWindowPlanner
from capx.recovery.schema import (
    ExecutionOutcome,
    HarnessEvent,
    HarnessResult,
    PlannedAction,
    StepScore,
    ValidationResult,
)
from capx.recovery.scoring import PrimitiveStepScorer
from capx.recovery.validators import RuleBasedValidator, default_validator

RecoveryDecision = Literal["retry", "replan", "abort"]
Executor = Callable[[PlannedAction], ExecutionOutcome | Mapping[str, Any]]
RecoveryPolicy = Callable[[PlannedAction, ValidationResult, Mapping[str, Any]], RecoveryDecision]


def _normalize_outcome(value: ExecutionOutcome | Mapping[str, Any]) -> ExecutionOutcome:
    if isinstance(value, ExecutionOutcome):
        return value
    return ExecutionOutcome(
        state=dict(value.get("state", value)),
        stdout=str(value.get("stdout", "")),
        stderr=str(value.get("stderr", "")),
        reward=value.get("reward"),
        terminated=bool(value.get("terminated", False)),
        truncated=bool(value.get("truncated", False)),
        metadata=dict(value.get("metadata", {})),
        state_window=tuple(dict(item) for item in value.get("state_window", ())),
    )


class ThreeStepRecoveryHarness:
    """Execute a plan in windows of three, validating each action in order."""

    def __init__(self, executor: Executor, *, validator: RuleBasedValidator | None = None,
                 scorer: PrimitiveStepScorer | None = None,
                 recovery_policy: RecoveryPolicy | None = None, max_retries: int = 1,
                 event_sink: Callable[[HarnessEvent], None] | None = None) -> None:
        self.executor = executor
        self.validator = validator or default_validator
        self.scorer = scorer or PrimitiveStepScorer()
        self.recovery_policy = recovery_policy
        self.max_retries = max_retries
        self.event_sink = event_sink

    def run(
        self,
        actions: list[PlannedAction],
        *,
        initial_state: Mapping[str, Any] | None = None,
    ) -> HarnessResult:
        planner = ThreeStepWindowPlanner()
        planner.load(actions)
        events: list[HarnessEvent] = []
        executed: list[str] = []
        retries: dict[str, int] = {}
        scores: list[StepScore] = []
        previous_state: Mapping[str, Any] = dict(initial_state or {})

        def emit(event: HarnessEvent) -> None:
            events.append(event)
            if self.event_sink:
                self.event_sink(event)

        while not planner.done:
            window = planner.current_window()
            emit(HarnessEvent("planning.window_ready", window_start=planner.cursor,
                              window_size=len(window), payload={"action_ids": [a.action_id for a in window]}))
            action = planner.advance()
            emit(HarnessEvent("execution.start", action_id=action.action_id,
                              window_start=planner.cursor - 1, window_size=len(window)))
            outcome = _normalize_outcome(self.executor(action))
            state_summary = self.scorer.summarize_outcome(outcome)
            effective_state = state_summary.state
            validation = self.validator.validate(action, effective_state)
            score = self.scorer.score(
                action,
                pre_state=previous_state,
                outcome=outcome,
                validation=validation,
                state_summary=state_summary,
            )
            scores.append(score)
            emit(HarnessEvent("validation.result", action_id=action.action_id,
                              window_start=planner.cursor - 1, window_size=len(window),
                              payload={"ok": validation.ok, "rule": validation.rule_name,
                                       "message": validation.message, "state": effective_state,
                                       "state_window_stable": state_summary.stable,
                                       "state_window_confidence": state_summary.confidence,
                                       "unstable_signals": list(state_summary.unstable_signals),
                                       "missing_signals": list(validation.missing_signals)}))
            emit(HarnessEvent(
                "scoring.result",
                action_id=action.action_id,
                window_start=planner.cursor - 1,
                window_size=len(window),
                payload={
                    "total": score.total,
                    "hard_pass": score.hard_pass,
                    "decision": score.decision,
                    "execution_score": score.execution_score,
                    "goal_score": score.goal_score,
                    "progress_score": score.progress_score,
                    "confidence_score": score.confidence_score,
                    "safety_penalty": score.safety_penalty,
                },
            ))
            previous_state = effective_state
            if validation.ok and score.decision in {"pass", "verify"}:
                executed.append(action.action_id)
                continue
            decision = (
                self.recovery_policy(action, validation, effective_state)
                if self.recovery_policy is not None
                else score.decision
            )
            if decision in {"pass", "verify"}:
                decision = "retry"
            emit(HarnessEvent("recovery.decision", action_id=action.action_id,
                              payload={"decision": decision, "reason": validation.message}))
            if decision == "retry" and retries.get(action.action_id, 0) < self.max_retries:
                retries[action.action_id] = retries.get(action.action_id, 0) + 1
                planner.rewind_one()
                emit(HarnessEvent("recovery.retry", action_id=action.action_id,
                                  payload={"attempt": retries[action.action_id]}))
                continue
            if decision == "replan":
                return HarnessResult(
                    status="blocked",
                    executed_actions=tuple(executed),
                    failed_action=action.action_id,
                    validation=validation,
                    events=tuple(events),
                    scores=tuple(scores),
                )
            return HarnessResult(
                status="aborted",
                executed_actions=tuple(executed),
                failed_action=action.action_id,
                validation=validation,
                events=tuple(events),
                scores=tuple(scores),
            )
        return HarnessResult(
            status="success",
            executed_actions=tuple(executed),
            events=tuple(events),
            scores=tuple(scores),
        )


def build_recovery_prompt(*, task: str, current_action: PlannedAction,
                          upcoming_actions: tuple[PlannedAction, ...], state: Mapping[str, Any],
                          validation: ValidationResult, score: StepScore | None = None) -> str:
    """Build a compact text-only prompt for CaP-X's existing multi-turn model."""
    lines = [
        "You are deciding the next safe recovery action for a CaP-X robot.",
        f"Task: {task}",
        f"Failed action: {current_action.action_id} ({current_action.primitive})",
        f"Validation: {validation.rule_name} -> {validation.message}",
        f"Missing signals: {', '.join(validation.missing_signals) or 'none'}",
        "Current text world state:",
        _format_state(state),
        "Upcoming nominal actions:",
    ]
    if score is not None:
        lines[5:5] = [
            f"Step score: {score.total:.2f}/100",
            f"Score decision: {score.decision}",
            "Score components: "
            f"goal={score.goal_score:.2f}, progress={score.progress_score:.2f}, "
            f"execution={score.execution_score:.2f}, confidence={score.confidence_score:.2f}, "
            f"safety_penalty={score.safety_penalty:.2f}",
        ]
    lines.extend(f"- {a.action_id}: {a.primitive}({a.args})" for a in upcoming_actions)
    lines.append("Choose one: retry, replan, abort. Return a concise reason and next action intent.")
    return "\n".join(lines)


def _format_state(state: Mapping[str, Any], prefix: str = "") -> str:
    lines: list[str] = []
    for key in sorted(state):
        value = state[key]
        if isinstance(value, Mapping):
            lines.append(f"{prefix}{key}:")
            lines.extend(_format_state(value, prefix + "  ").splitlines())
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines) or f"{prefix}<empty>"
