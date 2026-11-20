"""Deterministic offline task-graph execution and recovery.

The offline controller is deliberately independent from the LLM loop.  Both
nominal actions and recovery actions are declared in configuration, while the
online controller only checks preconditions, executes one primitive, validates
the K-state consensus, and selects a predeclared recovery route.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from capx.recovery.stability import StateWindowAggregator, StateWindowConfig


def _lookup(state: Mapping[str, Any], path: str) -> tuple[Any, bool]:
    current: Any = state
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None, False
        current = current[part]
    return current, True


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        if "eq" in expected and actual != expected["eq"]:
            return False
        try:
            if "gte" in expected and float(actual) < float(expected["gte"]):
                return False
            if "lte" in expected and float(actual) > float(expected["lte"]):
                return False
        except (TypeError, ValueError):
            return False
        return bool(expected) and not (set(expected) - {"eq", "gte", "lte"})
    return actual == expected


def _check_conditions(state: Mapping[str, Any], conditions: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    failures: list[str] = []
    for path, expected in conditions.items():
        actual, present = _lookup(state, str(path))
        if not present:
            failures.append(f"{path}:missing")
        elif not _matches(actual, expected):
            failures.append(f"{path}:{actual!r}!={expected!r}")
    return not failures, tuple(failures)


def _postcondition_failure_route(
    action: "OfflineAction", state: Mapping[str, Any]
) -> str | None:
    if state.get("recovery_budget_low") is True:
        return "RECOVERY_BUDGET_LOW"
    if state.get("object_stable") is False:
        unstable = action.failure_routes.get("OBJECT_UNSTABLE")
        if unstable:
            return unstable
    if (
        action.success_conditions.get("target_grasped") is True
        and state.get("target_grasped") is False
        and state.get("object_near_workspace_boundary") is True
    ):
        return "WORKSPACE_EDGE_GRASP_MISS"
    aspect_ratio = float(state.get("max_horizontal_aspect_ratio", 0.0) or 0.0)
    handled = str(action.metadata.get("handles_exception", ""))
    if (
        aspect_ratio >= 2.0
        and action.success_conditions.get("target_grasped") is True
        and state.get("target_grasped") is False
        and handled != "SLENDER_GRASP_POSE_UNSAFE"
    ):
        return "SLENDER_GRASP_POSE_UNSAFE"
    raw_distance = state.get("distance_to_target", float("inf"))
    try:
        distance_to_target = float(raw_distance)
    except (TypeError, ValueError):
        distance_to_target = float("inf")
    if (
        aspect_ratio >= 2.0
        and action.success_conditions.get("task_completed") is True
        and distance_to_target <= 0.12
        and handled != "PLACE_ALIGNMENT_FAILED"
    ):
        return "PLACE_ALIGNMENT_FAILED"
    return action.failure_routes.get("POSTCONDITION_FAILED")


class StateParameterError(ValueError):
    """An offline action could not bind a required value from current state."""


def bind_state_parameters(
    action: "OfflineAction", state: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Bind action parameters from fresh world state into static action code."""
    bound: dict[str, Any] = {}
    for name, raw_path in action.args.items():
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise StateParameterError(
                f"offline action {action.action_id!r} parameter {name!r} requires a state path"
            )
        value, present = _lookup(state, raw_path)
        if not present:
            raise StateParameterError(
                f"offline action {action.action_id!r} parameter {name!r} missing state path {raw_path!r}"
            )
        bound[str(name)] = value
    if not bound:
        return action.code, bound
    try:
        payload = json.dumps(bound, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise StateParameterError(
            f"offline action {action.action_id!r} parameters are not JSON-serializable"
        ) from exc
    prefix = f"import json\n_offline_params = json.loads({payload!r})"
    return f"{prefix}\n\n{action.code}", bound


@dataclass(frozen=True)
class OfflineAction:
    """One static executable action in the task graph or recovery library."""

    action_id: str
    primitive: str
    code: str
    args: dict[str, Any] = field(default_factory=dict)
    preconditions: dict[str, Any] = field(default_factory=dict)
    success_conditions: dict[str, Any] = field(default_factory=dict)
    failure_routes: dict[str, str] = field(default_factory=dict)
    next_action: str | None = None
    max_attempts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OfflineAction":
        limits = dict(value.get("limits", {}))
        action_id = str(value.get("id", value.get("action_id", ""))).strip()
        if not action_id:
            raise ValueError("offline action requires id")
        code = str(value.get("code", "")).strip()
        if not code:
            raise ValueError(f"offline action {action_id!r} requires static code")
        max_attempts = int(limits.get("max_attempts", value.get("max_attempts", 1)))
        if max_attempts < 1:
            raise ValueError(f"offline action {action_id!r} max_attempts must be >= 1")
        return cls(
            action_id=action_id,
            primitive=str(value.get("primitive", "unknown")),
            code=code,
            args=dict(value.get("parameters", value.get("args", {}))),
            preconditions=dict(value.get("preconditions", {})),
            success_conditions=dict(value.get("success_conditions", value.get("expected", {}))),
            failure_routes={str(k): str(v) for k, v in dict(value.get("failure_routes", {})).items()},
            next_action=value.get("next", value.get("next_action")),
            max_attempts=max_attempts,
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class OfflineTaskGraph:
    actions: dict[str, OfflineAction]
    initial: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OfflineTaskGraph":
        raw_actions = value.get("actions", ())
        if isinstance(raw_actions, Mapping):
            actions = {
                str(action_id): OfflineAction.from_mapping({"id": action_id, **dict(raw)})
                for action_id, raw in raw_actions.items()
            }
        else:
            parsed = [OfflineAction.from_mapping(item) for item in raw_actions]
            actions = {item.action_id: item for item in parsed}
        if not actions:
            raise ValueError("offline task graph must contain actions")
        initial = tuple(str(item) for item in value.get("initial", ()))
        if not initial:
            initial = (next(iter(actions)),)
        missing = set(initial) - set(actions)
        if missing:
            raise ValueError(f"offline task graph initial actions missing: {sorted(missing)}")
        return cls(actions=actions, initial=initial)


def attach_action_library(
    graph: OfflineTaskGraph, value: Mapping[str, Any] | None,
) -> OfflineTaskGraph:
    """Attach reusable recovery actions without moving task-specific routes."""
    if not value:
        return graph
    library_graph = OfflineTaskGraph.from_mapping(value)
    duplicates = set(graph.actions) & set(library_graph.actions)
    if duplicates:
        raise ValueError(f"offline action library duplicates actions: {sorted(duplicates)}")
    return OfflineTaskGraph(
        actions={**graph.actions, **library_graph.actions},
        initial=graph.initial,
    )


def prepend_failure_trigger(
    graph: OfflineTaskGraph,
    exception_id: str,
    state_overrides: Mapping[str, Any] | None = None,
    prepare_actions: tuple[str, ...] = (),
) -> OfflineTaskGraph:
    """Prepend one deterministic exception trigger for directed validation."""
    exception_id = str(exception_id).strip()
    if not exception_id:
        return graph
    action_id = "injected_failure_trigger"
    if action_id in graph.actions:
        raise ValueError(f"offline task graph already contains {action_id!r}")
    if len(prepare_actions) > 2 or any(item not in graph.actions for item in prepare_actions):
        raise ValueError("offline failure injection accepts at most two known preparation actions")
    trigger = OfflineAction(
        action_id=action_id,
        primitive="replay_failure_trigger",
        code="pass",
        success_conditions={"injected_failure_consumed": True},
        metadata={
            "exception_id": exception_id,
            "state_overrides": dict(state_overrides or {}),
        },
    )
    return OfflineTaskGraph(
        actions={action_id: trigger, **graph.actions},
        initial=(*prepare_actions, action_id),
    )


@dataclass(frozen=True)
class RecoveryLibrary:
    routes: dict[str, tuple[str, ...]]
    target_routes: dict[str, dict[str, tuple[str, ...]]] = field(
        default_factory=dict
    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RecoveryLibrary":
        routes: dict[str, tuple[str, ...]] = {}
        target_routes: dict[str, dict[str, tuple[str, ...]]] = {}
        raw_value = dict(value or {})
        raw_value = dict(raw_value.get("recovery_library", raw_value))
        for exception_id, raw_actions in raw_value.items():
            if isinstance(raw_actions, Mapping):
                variants = raw_actions.get("by_target_geometry", {})
                if isinstance(variants, Mapping):
                    target_routes[str(exception_id)] = {
                        str(target): tuple(str(item) for item in actions)
                        for target, actions in variants.items()
                    }
                raw_actions = raw_actions.get("actions", ())
            routes[str(exception_id)] = tuple(str(item) for item in raw_actions)
        return cls(routes, target_routes)

    def lookup(
        self,
        exception_id: str,
        state: Mapping[str, Any] | None = None,
    ) -> tuple[str, ...] | None:
        target = str((state or {}).get("target_geometry_type", ""))
        route_id = exception_id
        if (
            exception_id == "OUTSIDE_CONTAINER"
            and target == "container"
            and "OUTSIDE_TARGET_REGION" in self.target_routes
        ):
            route_id = "OUTSIDE_TARGET_REGION"
        route = self.target_routes.get(route_id, {}).get(target)
        return route or self.routes.get(route_id) or self.routes.get(exception_id) or self.routes.get("*")


def select_hybrid_recovery(
    graph: OfflineTaskGraph,
    library: RecoveryLibrary,
    stage_actions: tuple[str, ...],
    stage_index: int,
    *,
    sandbox_rc: int = 0,
    state: Mapping[str, Any] | None = None,
) -> tuple[str | None, tuple[str, ...]]:
    """Select one bounded offline route before an online model retry."""
    if stage_index < 0 or stage_index >= len(stage_actions):
        return None, ()
    nominal = graph.actions.get(stage_actions[stage_index])
    if nominal is None:
        return None, ()
    current_state = state or {}
    explicit_exception = current_state.get("active_exception_id")
    exception_id = (
        str(explicit_exception)
        if explicit_exception
        else nominal.failure_routes.get("SANDBOX_ERROR")
        if sandbox_rc
        else _postcondition_failure_route(nominal, current_state)
    )
    route = library.lookup(exception_id, current_state) if exception_id else None
    if not route or any(action_id not in graph.actions for action_id in route):
        return exception_id, ()

    window: list[str] = []
    seen: set[str] = set()
    pending = list(route)
    while pending and len(window) < 3:
        action_id = pending.pop(0)
        if action_id in seen or action_id not in graph.actions:
            continue
        seen.add(action_id)
        action = graph.actions[action_id]
        skip_conditions = action.metadata.get("skip_if_conditions", {})
        skip, _ = _check_conditions(current_state, skip_conditions)
        if not skip_conditions or not skip:
            window.append(action_id)
        if action.next_action and action.next_action not in seen:
            pending.insert(0, action.next_action)
    return exception_id, tuple(window)


@dataclass(frozen=True)
class OfflineExecution:
    state: dict[str, Any]
    state_window: tuple[dict[str, Any], ...] = ()
    stdout: str = ""
    stderr: str = ""
    reward: float | None = None
    sandbox_rc: int = 0
    task_completed: bool | None = None
    terminated: bool = False
    truncated: bool = False
    exception_id: str | None = None


@dataclass(frozen=True)
class OfflineEvent:
    event_type: str
    action_id: str | None = None
    exception_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OfflineRunResult:
    status: Literal["success", "aborted", "blocked"]
    executed_actions: tuple[str, ...]
    failed_action: str | None
    exception_id: str | None
    final_state: dict[str, Any]
    events: tuple[OfflineEvent, ...]
    reward: float | None = None
    task_completed: bool | None = None


class OfflineWindowController:
    """Run a static graph with K_action=3 and deterministic recovery routes."""

    def __init__(
        self,
        graph: OfflineTaskGraph,
        recovery_library: RecoveryLibrary,
        *,
        k_action: int = 3,
        k_state: int = 3,
        min_stable_fraction: float = 2 / 3,
        event_sink: Callable[[OfflineEvent], None] | None = None,
    ) -> None:
        if k_action != 3 or k_state != 3:
            raise ValueError("offline architecture requires K_action=3 and K_state=3")
        self.graph = graph
        self.recovery_library = recovery_library
        self.event_sink = event_sink
        self.aggregator = StateWindowAggregator(StateWindowConfig(
            window_size=k_state, min_stable_fraction=min_stable_fraction,
        ))

    def run(
        self,
        execute: Callable[[OfflineAction], OfflineExecution | Mapping[str, Any]],
        *,
        initial_state: Mapping[str, Any] | None = None,
    ) -> OfflineRunResult:
        actions = self.graph.actions
        window = list(self.graph.initial)
        executed: list[str] = []
        attempts: dict[str, int] = {}
        events: list[OfflineEvent] = []
        state = dict(initial_state or {})
        last: OfflineExecution | None = None

        def emit(event: OfflineEvent) -> None:
            events.append(event)
            if self.event_sink:
                self.event_sink(event)

        while window:
            if any(action_id not in actions for action_id in window):
                return self._abort(executed, state, events, None, "UNKNOWN_ACTION", last)
            current = actions[window[0]]
            emit(OfflineEvent("window.ready", current.action_id, payload={"actions": list(window)}))
            pre_ok, pre_failures = _check_conditions(state, current.preconditions)
            if not pre_ok:
                exception_id = current.failure_routes.get("PRECONDITION_FAILED")
                emit(OfflineEvent("precondition.failed", current.action_id, exception_id,
                                  {"failures": list(pre_failures)}))
                recovery = self._recovery_window(exception_id)
                if recovery is None:
                    return self._abort(executed, state, events, current.action_id,
                                       exception_id or "UNKNOWN_FAILURE", last)
                window = list(recovery)
                continue

            attempts[current.action_id] = attempts.get(current.action_id, 0) + 1
            emit(OfflineEvent("action.start", current.action_id,
                              payload={"attempt": attempts[current.action_id]}))
            outcome = execute(current)
            if not isinstance(outcome, OfflineExecution):
                outcome = OfflineExecution(**dict(outcome))
            last = outcome
            samples = outcome.state_window or (outcome.state,)
            summary = self.aggregator.summarize(samples)
            state = dict(summary.state)
            if outcome.task_completed is not None:
                state["task_completed"] = outcome.task_completed
            exception_id = outcome.exception_id
            if current.primitive == "replay_failure_trigger":
                exception_id = str(current.metadata.get("exception_id", "")).strip() or exception_id
                state.update(dict(current.metadata.get("state_overrides", {})))
            if (
                exception_id == "COMPLETED_BEFORE_TERMINATION"
                and outcome.task_completed is True
            ):
                state["task_completed"] = True
                executed.append(current.action_id)
                emit(OfflineEvent(
                    "environment.completed", current.action_id,
                    "COMPLETED_BEFORE_TERMINATION",
                ))
                return OfflineRunResult(
                    "success", tuple(executed), None,
                    "COMPLETED_BEFORE_TERMINATION", state, tuple(events),
                    outcome.reward, True,
                )
            post_ok, post_failures = _check_conditions(state, current.success_conditions)
            semantic_stability = bool(
                post_ok
                and current.metadata.get("accept_semantic_stability", False)
                and state.get("object_stable") is True
            )
            semantic_failure = bool(
                not post_ok
                and current.metadata.get("accept_semantic_failure", False)
                and state.get("object_stable") is True
                and state.get("measurement_fresh") is True
            )
            semantic_success = bool(
                post_ok
                and current.metadata.get("accept_semantic_success", False)
                and state.get("measurement_fresh") is True
            )
            if outcome.sandbox_rc != 0:
                exception_id = current.failure_routes.get(
                    "SANDBOX_ERROR", exception_id or "SANDBOX_ERROR"
                )
                post_ok = False
            if (
                not summary.stable
                and not semantic_stability
                and not semantic_failure
                and not semantic_success
            ):
                exception_id = exception_id or "STATE_UNCERTAIN"
                post_ok = False
            emit(OfflineEvent("action.validated", current.action_id, exception_id,
                              {"ok": post_ok, "stable": summary.stable,
                               "confidence": summary.confidence,
                               "failures": list(post_failures), "state": state}))
            if (
                (outcome.terminated or outcome.truncated)
                and outcome.task_completed is not True
            ):
                emit(OfflineEvent(
                    "environment.terminated", current.action_id,
                    "ENVIRONMENT_TERMINATED",
                ))
                return self._abort(
                    executed, state, events, current.action_id,
                    "ENVIRONMENT_TERMINATED", last,
                )
            if post_ok:
                executed.append(current.action_id)
                tail = [item for item in window[1:] if item in actions]
                if tail:
                    kept: list[str] = []
                    for item in tail:
                        ok, _ = _check_conditions(state, actions[item].preconditions)
                        if not ok:
                            break
                        kept.append(item)
                    tail = kept
                next_id = current.next_action
                if next_id and next_id not in tail:
                    tail.append(next_id)
                window = tail[:3]
                if len(window) < 3 and next_id:
                    self._refill(window, actions, 3)
                emit(OfflineEvent("window.advanced", current.action_id,
                                  payload={"actions": list(window)}))
                if outcome.task_completed is True:
                    break
                continue

            route_key = exception_id or _postcondition_failure_route(current, state)
            if route_key == "STATE_UNCERTAIN":
                if attempts[current.action_id] < current.max_attempts:
                    emit(OfflineEvent("state.wait", current.action_id, route_key))
                    window = [current.action_id, *window[1:]][:3]
                    continue
            recovery = self._recovery_window(route_key)
            if recovery is not None:
                emit(OfflineEvent("recovery.selected", current.action_id, route_key,
                                  {"actions": list(recovery)}))
                window = list(recovery)
                continue
            return self._abort(executed, state, events, current.action_id,
                               route_key or "UNKNOWN_FAILURE", last)

        status: Literal["success", "aborted", "blocked"] = (
            "success" if (last is not None and last.task_completed is True) else "blocked"
        )
        return OfflineRunResult(status, tuple(executed), None, None, state, tuple(events),
                                 last.reward if last else None, last.task_completed if last else None)

    def _recovery_window(self, exception_id: str | None) -> tuple[str, ...] | None:
        if not exception_id:
            return None
        route = self.recovery_library.lookup(exception_id)
        if route is None or any(item not in self.graph.actions for item in route):
            return None
        return route[:3]

    @staticmethod
    def _refill(window: list[str], actions: Mapping[str, OfflineAction], size: int) -> None:
        seen = set(window)
        cursor = actions[window[-1]].next_action if window else None
        while cursor and len(window) < size and cursor not in seen:
            if cursor not in actions:
                break
            window.append(cursor)
            seen.add(cursor)
            cursor = actions[cursor].next_action

    @staticmethod
    def _abort(executed: list[str], state: dict[str, Any], events: list[OfflineEvent],
               action_id: str | None, exception_id: str, last: OfflineExecution | None) -> OfflineRunResult:
        events.append(OfflineEvent("safe_stop", action_id, exception_id))
        return OfflineRunResult("aborted", tuple(executed), action_id, exception_id, state,
                                tuple(events), last.reward if last else None,
                                last.task_completed if last else None)
