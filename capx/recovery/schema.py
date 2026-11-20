"""Data contracts for the minimal three-step recovery harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class PlannedAction:
    """One executable primitive and its expected post-condition."""

    action_id: str
    primitive: str
    args: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionOutcome:
    """Normalized result returned by an executor after one primitive."""

    state: dict[str, Any]
    stdout: str = ""
    stderr: str = ""
    reward: float | None = None
    terminated: bool = False
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    state_window: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    primitive: str
    action_id: str
    rule_name: str
    message: str
    missing_signals: tuple[str, ...] = ()
    observed: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepScore:
    """Per-primitive quality score; hard rules remain authoritative."""

    action_id: str
    primitive: str
    total: float
    hard_pass: bool
    execution_score: float
    goal_score: float
    progress_score: float
    confidence_score: float
    safety_penalty: float
    decision: Literal["pass", "verify", "retry", "replan", "abort"]
    failed_rules: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HarnessEvent:
    event_type: str
    action_id: str | None = None
    window_start: int | None = None
    window_size: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HarnessResult:
    status: Literal["success", "recovered", "blocked", "aborted"]
    executed_actions: tuple[str, ...]
    failed_action: str | None = None
    validation: ValidationResult | None = None
    events: tuple[HarnessEvent, ...] = ()
    scores: tuple[StepScore, ...] = ()
