"""Deterministic execution and decision gates for multi-turn robot trials."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from capx.recovery.candidate_reranking import estimate_execution_cost
from capx.recovery.stability import StateWindowSummary


@dataclass(frozen=True)
class GuardResult:
    """Auditable result from a code, stage, or decision gate."""

    ok: bool
    gate: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionGuardConfig:
    """Opt-in hard gates applied immediately before and after execution."""

    enabled: bool = False
    reject_empty_code: bool = True
    max_robot_api_calls_per_block: int = 8
    max_expensive_loop_iterations: int = 8
    finish_requires_task_completed: bool = True
    max_decision_reprompts: int = 2
    robot_action_calls: tuple[str, ...] = (
        "goto_pose",
        "home_pose",
        "open_gripper",
        "close_gripper",
    )
    stage_expectations: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "ExecutionGuardConfig":
        raw = dict(value or {})
        known = {item.name for item in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown execution_guard options: {sorted(unknown)}")
        if "robot_action_calls" in raw:
            raw["robot_action_calls"] = tuple(raw["robot_action_calls"])
        if "stage_expectations" in raw:
            raw["stage_expectations"] = tuple(raw["stage_expectations"])
        config = cls(**raw)
        if config.max_robot_api_calls_per_block < 1:
            raise ValueError("max_robot_api_calls_per_block must be >= 1")
        if config.max_expensive_loop_iterations < 0:
            raise ValueError("max_expensive_loop_iterations must be >= 0")
        if config.max_decision_reprompts < 0:
            raise ValueError("max_decision_reprompts must be >= 0")
        for index, expectation in enumerate(config.stage_expectations):
            if not isinstance(expectation, Mapping):
                raise ValueError(f"stage_expectations[{index}] must be a mapping")
            required = expectation.get("required", {})
            if not isinstance(required, Mapping):
                raise ValueError(
                    f"stage_expectations[{index}].required must be a mapping"
                )
        return config


def task_completion_confirmed(
    *,
    task_completed: bool | None,
    state_summary: StateWindowSummary | None = None,
    required_samples: int = 3,
) -> bool:
    """Accept only an authoritative completion flag or a stable K-state consensus."""
    if task_completed is True:
        return True
    return bool(
        state_summary is not None
        and state_summary.sample_count >= required_samples
        and state_summary.stable
        and state_summary.state.get("task_completed") is True
    )


def validate_code_block(code: str, config: ExecutionGuardConfig) -> GuardResult:
    """Reject malformed or unexpectedly expensive code before robot APIs run."""
    if not config.enabled:
        return GuardResult(True, "code_preflight", "execution guard disabled")
    if config.reject_empty_code and not code.strip():
        return GuardResult(False, "code_preflight", "generated code block is empty")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        location = f"line {exc.lineno or '?'}:{exc.offset or '?'}"
        return GuardResult(
            False,
            "code_preflight",
            f"invalid Python syntax at {location}: {exc.msg}",
            {"syntax_error": exc.msg, "line": exc.lineno, "offset": exc.offset},
        )

    action_names = set(config.robot_action_calls)
    action_calls = [
        _call_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node.func) in action_names
    ]
    if len(action_calls) > config.max_robot_api_calls_per_block:
        return GuardResult(
            False,
            "code_preflight",
            "robot action block is too long for atomic verification: "
            f"{len(action_calls)} calls > {config.max_robot_api_calls_per_block}",
            {"robot_action_calls": action_calls},
        )

    cost = estimate_execution_cost(code)
    if cost.get("unbounded_expensive_loop"):
        return GuardResult(
            False,
            "code_preflight",
            "expensive robot/perception API appears in an unbounded loop",
            {"execution_cost": cost},
        )
    loop_iterations = int(cost.get("max_expensive_loop_iterations", 0) or 0)
    if loop_iterations > config.max_expensive_loop_iterations:
        return GuardResult(
            False,
            "code_preflight",
            "expensive API loop exceeds the configured iteration limit: "
            f"{loop_iterations} > {config.max_expensive_loop_iterations}",
            {"execution_cost": cost},
        )
    return GuardResult(
        True,
        "code_preflight",
        "code passed static execution checks",
        {"robot_action_calls": action_calls, "execution_cost": cost},
    )


def validate_stage(
    stage_index: int,
    world_state: Mapping[str, Any] | None,
    config: ExecutionGuardConfig,
) -> GuardResult:
    """Validate the configured post-condition for one nominal plan stage."""
    if not config.enabled or stage_index >= len(config.stage_expectations):
        return GuardResult(True, "stage_postcondition", "no stage gate configured")
    expectation = config.stage_expectations[stage_index]
    name = str(expectation.get("name", f"stage_{stage_index + 1}"))
    required = expectation.get("required", {})
    state = dict(world_state or {})
    failures: list[str] = []
    observed: dict[str, Any] = {}
    for path, expected in required.items():
        actual, present = _lookup(state, str(path))
        observed[str(path)] = actual if present else "<missing>"
        if not present:
            failures.append(f"{path} is missing")
        elif not _matches(actual, expected):
            failures.append(f"{path}={actual!r} does not satisfy {expected!r}")
    if failures:
        return GuardResult(
            False,
            "stage_postcondition",
            f"{name} failed: " + "; ".join(failures),
            {"stage_index": stage_index, "stage_name": name, "observed": observed},
        )
    return GuardResult(
        True,
        "stage_postcondition",
        f"{name} passed",
        {"stage_index": stage_index, "stage_name": name, "observed": observed},
    )


def validate_decision(
    decision: str,
    *,
    task_completed: bool | None,
    stage_result: GuardResult,
    config: ExecutionGuardConfig,
) -> GuardResult:
    """Prevent model self-reports from overriding environment post-conditions."""
    if not config.enabled:
        return GuardResult(True, "decision", "execution guard disabled")
    normalized = decision.lower()
    if normalized == "finish" and config.finish_requires_task_completed:
        if task_completed is not True:
            return GuardResult(
                False,
                "finish_completion",
                "FINISH rejected because environment task_completed is not true",
                {"task_completed": task_completed},
            )
    if normalized in {"continue", "finish"} and not stage_result.ok:
        return GuardResult(
            False,
            "stage_decision",
            f"{normalized.upper()} rejected because {stage_result.reason}",
            {"stage_result": stage_result.details},
        )
    return GuardResult(True, "decision", f"{normalized} accepted")


def format_guard_feedback(result: GuardResult) -> str:
    """Create a compact, explicit correction for the next model decision."""
    return (
        "HARD EXECUTION GATE: the previous model decision was rejected. "
        f"Reason: {result.reason}. Do not repeat that decision. Return CONTINUE only "
        "if the current stage passed and the next preplanned block is safe; otherwise "
        "return REGENERATE with one short, syntactically complete recovery block that "
        "satisfies the current world-state post-condition."
    )


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _lookup(state: Mapping[str, Any], path: str) -> tuple[Any, bool]:
    current: Any = state
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None, False
        current = current[part]
    return current, True


def _matches(actual: Any, expected: Any) -> bool:
    if not isinstance(expected, Mapping):
        return actual == expected
    try:
        if "eq" in expected and actual != expected["eq"]:
            return False
        if "lte" in expected and float(actual) > float(expected["lte"]):
            return False
        if "gte" in expected and float(actual) < float(expected["gte"]):
            return False
    except (TypeError, ValueError):
        return False
    allowed = {"eq", "lte", "gte"}
    return bool(expected) and not (set(expected) - allowed)


def assign_stage_indices(
    count: int,
    *,
    start: int = 0,
    generation: int = 0,
    regenerated: bool = False,
    regenerated_at_idx: int | None = None,
) -> list[dict[str, Any]]:
    """Build consistent metadata for initial or regenerated code blocks."""
    items = []
    for offset in range(count):
        item: dict[str, Any] = {
            "generation": generation,
            "regenerated": regenerated,
            "stage_index": start + offset,
        }
        if regenerated_at_idx is not None:
            item["regenerated_at_idx"] = regenerated_at_idx
        items.append(item)
    return items
