"""Inference-time candidate selection for the phase-1 PRM experiment.

The ``prm_proxy`` mode is deliberately a deterministic rule scorer, not a
trained process-reward model. It provides the same candidate/reranking data
contract that a learned checkpoint can replace later.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, Mapping, Sequence

from capx.recovery.loop_guard import strategy_fingerprint


SelectionMode = Literal["random", "prm_proxy", "prm_proxy_v2"]

_CALL_COSTS = {
    "sample_grasp_pose": 12.0,
    "goto_pose": 6.0,
    "get_object_pose": 3.0,
    "home_pose": 5.0,
    "open_gripper": 1.0,
    "close_gripper": 1.0,
}
_EXPENSIVE_LOOP_APIS = {"sample_grasp_pose", "goto_pose", "get_object_pose"}
_STAGE_EXPECTED_CALLS: dict[str, set[str]] = {
    "ACQUIRE_GRASP": {"sample_grasp_pose", "goto_pose", "close_gripper"},
    "ACQUIRE_TARGET_AFTER_CLEAR": {"sample_grasp_pose", "goto_pose", "close_gripper"},
    "GRASPED_NOT_LIFTED": {"goto_pose"},
    "LIFTED_READY_TO_PLACE": {"get_object_pose", "goto_pose", "open_gripper"},
    "PLACED_NOT_RELEASED": {"open_gripper", "goto_pose"},
    "RELEASED_UNSTABLE": {"goto_pose"},
    "VERIFY_STACK": {"home_pose"},
    "WRONG_OBJECT_GRASPED": {"open_gripper", "goto_pose"},
}
_STAGE_TARGET_OBJECT = {
    "ACQUIRE_GRASP": "green",
    "ACQUIRE_TARGET_AFTER_CLEAR": "red",
    "GRASPED_NOT_LIFTED": "red",
    "LIFTED_READY_TO_PLACE": "red",
    "WRONG_OBJECT_GRASPED": "red",
}
_REQUIRE_SAMPLE_STAGE = {"ACQUIRE_GRASP", "ACQUIRE_TARGET_AFTER_CLEAR"}


@dataclass(frozen=True)
class CandidateRerankingConfig:
    enabled: bool = False
    candidate_count: int = 4
    mode: SelectionMode = "prm_proxy"
    seed: int = 20260904
    max_resample_rounds: int = 0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "CandidateRerankingConfig":
        value = dict(value or {})
        known = {field.name for field in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"Unknown candidate_reranking options: {sorted(unknown)}")
        config = cls(**value)
        if config.candidate_count < 2:
            raise ValueError("candidate_count must be >= 2")
        if config.max_resample_rounds < 0:
            raise ValueError("max_resample_rounds must be >= 0")
        if config.mode not in {"random", "prm_proxy", "prm_proxy_v2"}:
            raise ValueError(
                "candidate_reranking.mode must be random, prm_proxy, or prm_proxy_v2"
            )
        return config


def aggregate_token_usage(responses: Sequence[Mapping[str, Any]]) -> dict[str, int] | None:
    usages = [item.get("token_usage") for item in responses]
    usages = [item for item in usages if isinstance(item, Mapping)]
    if not usages:
        return None
    return {
        key: sum(int(item.get(key, 0) or 0) for item in usages)
        for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens")
    }


def select_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    config: CandidateRerankingConfig,
    trial: int,
    decision_index: int,
    recovery_stage: str = "ACQUIRE_GRASP",
    recovery_directive: str = "observe",
    task_completed: bool = False,
    previous_code: str = "",
    world_state: Mapping[str, Any] | None = None,
    state_confidence: float = 1.0,
    state_stable: bool = True,
) -> tuple[int, dict[str, Any]]:
    """Select one parsed candidate and return a fully auditable selection record."""
    if not candidates:
        raise ValueError("at least one candidate is required")

    posterior = infer_stage_posterior(
        world_state,
        reported_stage=recovery_stage,
        confidence=state_confidence,
        stable=state_stable,
    )
    scored = []
    for item in candidates:
        if config.mode == "prm_proxy_v2":
            score = _score_candidate_v2(
                item,
                recovery_directive=recovery_directive,
                task_completed=task_completed,
                previous_code=previous_code,
                stage_posterior=posterior,
            )
        else:
            score = _score_candidate(
                item,
                recovery_stage=recovery_stage,
                recovery_directive=recovery_directive,
                task_completed=task_completed,
                previous_code=previous_code,
            )
        scored.append(score)
    selection_abstained = False
    if config.mode == "random":
        material = f"{config.seed}:{trial}:{decision_index}:{len(candidates)}"
        selected_index = int(hashlib.sha256(material.encode()).hexdigest(), 16) % len(candidates)
    elif config.mode == "prm_proxy_v2":
        eligible = [
            index for index, score in enumerate(scored) if not score["hard_reject"]
        ]
        selection_abstained = not eligible
        pool = eligible or list(range(len(scored)))
        selected_index = max(pool, key=lambda index: scored[index]["total"])
    else:
        selected_index = max(range(len(scored)), key=lambda index: scored[index]["total"])

    audit_candidates = []
    for index, (candidate, score) in enumerate(zip(candidates, scored, strict=True)):
        audit_candidates.append({
            "index": index,
            "selected": index == selected_index,
            "decision": candidate.get("decision", "initial"),
            "code": candidate.get("code", ""),
            "content": candidate.get("content", ""),
            "reasoning": candidate.get("reasoning") or "",
            "token_usage": candidate.get("token_usage"),
            "score": score,
        })
    return selected_index, {
        "selector": config.mode,
        "proxy_kind": {
            "prm_proxy": "deterministic_rule_guided",
            "prm_proxy_v2": "deterministic_rule_guided_v2",
        }.get(config.mode),
        "candidate_count": len(candidates),
        "selection_seed": config.seed,
        "trial": trial,
        "decision_index": decision_index,
        "context": {
            "recovery_stage": recovery_stage,
            "recovery_directive": recovery_directive,
            "task_completed": task_completed,
            "stage_posterior": posterior,
            "state_confidence": state_confidence,
            "state_stable": state_stable,
        },
        "selected_index": selected_index,
        "selected_score": scored[selected_index]["total"],
        "selection_abstained": selection_abstained,
        "candidates": audit_candidates,
    }


def parse_world_state_feedback(feedback: str | None) -> dict[str, Any]:
    """Recover the compact JSON payload embedded in the text feedback prompt."""
    if not feedback:
        return {}
    start = feedback.find("{")
    if start < 0:
        return {}
    try:
        payload, _ = json.JSONDecoder().raw_decode(feedback[start:])
    except (json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def infer_stage_posterior(
    world_state: Mapping[str, Any] | None,
    *,
    reported_stage: str = "ACQUIRE_GRASP",
    confidence: float = 1.0,
    stable: bool = True,
) -> dict[str, Any]:
    """Fuse the reported stage with direct K=3 world-state evidence."""
    state = dict(world_state or {})
    if not state:
        distribution = {reported_stage: 1.0}
        return {
            "reported_stage": reported_stage,
            "evidence_stage": reported_stage,
            "map_stage": reported_stage,
            "distribution": distribution,
        }

    evidence_stage = _derive_stage_from_state(state)
    bounded_confidence = max(0.0, min(1.0, float(confidence)))
    if not stable:
        bounded_confidence = min(bounded_confidence, 0.55)
    if evidence_stage == reported_stage:
        distribution = {evidence_stage: 1.0}
    else:
        evidence_weight = bounded_confidence
        distribution = {
            evidence_stage: round(evidence_weight, 4),
            reported_stage: round(1.0 - evidence_weight, 4),
        }
        distribution = {key: value for key, value in distribution.items() if value > 0}
    map_stage = max(distribution, key=distribution.get)
    return {
        "reported_stage": reported_stage,
        "evidence_stage": evidence_stage,
        "map_stage": map_stage,
        "distribution": distribution,
    }


def _derive_stage_from_state(state: Mapping[str, Any]) -> str:
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


def _score_candidate(
    candidate: Mapping[str, Any],
    *,
    recovery_stage: str,
    recovery_directive: str,
    task_completed: bool,
    previous_code: str,
) -> dict[str, Any]:
    decision = str(candidate.get("decision", "initial")).lower()
    code = str(candidate.get("code", "") or "")
    valid_python = _valid_python(code) if decision in {"initial", "regenerate"} else True
    components = {
        "static_validity": 25.0 if valid_python else -100.0,
        "decision_alignment": _decision_alignment(decision, recovery_directive, task_completed),
        "stage_alignment": _stage_alignment(code, decision, recovery_stage),
        "strategy_novelty": _strategy_novelty(code, decision, recovery_directive, previous_code),
        "efficiency": _efficiency_score(code),
    }
    hard_reject = not valid_python or (decision == "finish" and not task_completed)
    total = sum(components.values())
    if hard_reject:
        total = min(total, -100.0)
    return {
        "total": round(total, 3),
        "hard_reject": hard_reject,
        "valid_python": valid_python,
        "components": components,
    }


def _score_candidate_v2(
    candidate: Mapping[str, Any],
    *,
    recovery_directive: str,
    task_completed: bool,
    previous_code: str,
    stage_posterior: Mapping[str, Any],
) -> dict[str, Any]:
    decision = str(candidate.get("decision", "initial")).lower()
    code = str(candidate.get("code", "") or "")
    valid_python = _valid_python(code) if decision in {"initial", "regenerate"} else True
    map_stage = str(stage_posterior.get("map_stage", "ACQUIRE_GRASP"))
    distribution = stage_posterior.get("distribution", {map_stage: 1.0})
    analysis_code = _current_stage_code(code, decision)
    cost = estimate_execution_cost(code) if valid_python else _empty_cost_analysis()
    object_check = _target_object_alignment(analysis_code, decision, map_stage)
    stage_score = _posterior_stage_alignment(
        analysis_code, decision, distribution if isinstance(distribution, Mapping) else {}
    )
    components = {
        "static_validity": 25.0 if valid_python else -100.0,
        "decision_alignment": _decision_alignment(
            decision, recovery_directive, task_completed
        ),
        "stage_posterior_alignment": stage_score,
        "target_object_alignment": object_check["score"],
        "strategy_novelty": _strategy_novelty(
            code, decision, recovery_directive, previous_code
        ),
        "predicted_execution_cost": _execution_cost_score(cost),
    }
    rejection_reasons = []
    if not valid_python:
        rejection_reasons.append("invalid_python")
    if decision == "finish" and not task_completed:
        rejection_reasons.append("premature_finish")
    if object_check["hard_reject"]:
        rejection_reasons.append(str(object_check["reason"]))
    if cost["unbounded_expensive_loop"]:
        rejection_reasons.append("unbounded_expensive_api_loop")
    elif cost["max_expensive_loop_iterations"] > 8:
        rejection_reasons.append("expensive_api_loop_exceeds_8_iterations")
    hard_reject = bool(rejection_reasons)
    total = sum(components.values())
    if hard_reject:
        total = min(total, -100.0)
    return {
        "total": round(total, 3),
        "hard_reject": hard_reject,
        "rejection_reasons": rejection_reasons,
        "valid_python": valid_python,
        "map_stage": map_stage,
        "components": components,
        "target_object_check": object_check,
        "execution_cost": cost,
    }


def _current_stage_code(code: str, decision: str) -> str:
    if decision != "initial":
        return code
    marker = "breakpoint_code_block()"
    return code.split(marker, 1)[0]


def _posterior_stage_alignment(
    code: str,
    decision: str,
    distribution: Mapping[str, Any],
) -> float:
    if decision in {"finish", "continue"}:
        return 10.0
    calls = strategy_fingerprint(code)
    call_set = set(calls)
    if calls == ("<invalid-python>",) or not calls:
        return -30.0
    weighted = 0.0
    for stage, probability in distribution.items():
        required = _STAGE_EXPECTED_CALLS.get(str(stage), {"goto_pose"})
        coverage = len(required & call_set) / len(required)
        weighted += float(probability) * 35.0 * coverage
    return round(weighted, 3)


def _target_object_alignment(code: str, decision: str, stage: str) -> dict[str, Any]:
    expected = _STAGE_TARGET_OBJECT.get(stage)
    if decision in {"finish", "continue"} or expected is None:
        return {
            "score": 0.0,
            "expected": expected,
            "observed": [],
            "hard_reject": False,
            "reason": None,
        }
    observed = _literal_sampled_objects(code)
    wrong = sorted({item for item in observed if item != expected})
    missing = stage in _REQUIRE_SAMPLE_STAGE and expected not in observed
    if wrong:
        score, reason = -60.0, f"wrong_target_object:{','.join(wrong)}!=expected_{expected}"
    elif missing:
        score, reason = -25.0, f"missing_required_target_object:{expected}"
    elif expected in observed:
        score, reason = 25.0, None
    else:
        score, reason = 0.0, None
    return {
        "score": score,
        "expected": expected,
        "observed": observed,
        "hard_reject": bool(wrong or missing),
        "reason": reason,
    }


def _literal_sampled_objects(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    objects = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) != "sample_grasp_pose":
            continue
        value_node = node.args[0] if node.args else None
        if value_node is None:
            for keyword in node.keywords:
                if keyword.arg in {"object_name", "object", "name"}:
                    value_node = keyword.value
                    break
        if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
            value = value_node.value.lower()
            if "green" in value:
                objects.append("green")
            elif "red" in value:
                objects.append("red")
            else:
                objects.append(value)
        else:
            objects.append("<dynamic>")
    return objects


def _call_name(node: ast.Call) -> str:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return "<dynamic>"


def _empty_cost_analysis() -> dict[str, Any]:
    return {
        "estimated_execution_cost": 0.0,
        "weighted_call_counts": {},
        "expensive_loop_calls": [],
        "max_expensive_loop_iterations": 0,
        "unbounded_expensive_loop": False,
    }


def estimate_execution_cost(code: str) -> dict[str, Any]:
    """Estimate runtime/API cost, expanding statically bounded loop bodies."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _empty_cost_analysis()
    counts: dict[str, float] = {}
    expensive_loop_calls: list[dict[str, Any]] = []
    max_expensive_iterations = 0
    unbounded_expensive_loop = False
    function_defs = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    active_functions: set[str] = set()

    def loop_scale(iterations: int | None, parent_iterations: int) -> tuple[int, int]:
        effective = 100 if iterations is None else min(iterations, 1000)
        marker = (
            -1
            if iterations is None or parent_iterations < 0
            else effective * max(1, parent_iterations)
        )
        return effective, marker

    def expand_function(name: str, multiplier: int, loop_iterations: int) -> None:
        definition = function_defs.get(name)
        if definition is None or name in active_functions:
            return
        active_functions.add(name)
        try:
            for statement in definition.body:
                visit(statement, multiplier, loop_iterations)
        finally:
            active_functions.remove(name)

    def visit(node: ast.AST, multiplier: int = 1, loop_iterations: int = 0) -> None:
        nonlocal max_expensive_iterations, unbounded_expensive_loop
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Definitions are free until invoked; local function calls are expanded below.
            return
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in {"map", "filter"} and len(node.args) >= 2:
                for iterable in node.args[1:]:
                    visit(iterable, multiplier, loop_iterations)
                effective, marker = loop_scale(
                    _range_iterations(node.args[1]), loop_iterations
                )
                callable_node = node.args[0]
                if isinstance(callable_node, ast.Lambda):
                    visit(callable_node.body, multiplier * effective, marker)
                elif isinstance(callable_node, ast.Name):
                    expand_function(callable_node.id, multiplier * effective, marker)
                return
            if name in function_defs:
                for argument in node.args:
                    visit(argument, multiplier, loop_iterations)
                for keyword in node.keywords:
                    visit(keyword.value, multiplier, loop_iterations)
                expand_function(name, multiplier, loop_iterations)
                return
            if name in _CALL_COSTS:
                counts[name] = counts.get(name, 0.0) + multiplier
                if loop_iterations and name in _EXPENSIVE_LOOP_APIS:
                    expensive_loop_calls.append({
                        "api": name,
                        "iterations": loop_iterations,
                        "line": getattr(node, "lineno", None),
                    })
                    if loop_iterations < 0:
                        unbounded_expensive_loop = True
                    else:
                        max_expensive_iterations = max(
                            max_expensive_iterations, loop_iterations
                        )
            for child in ast.iter_child_nodes(node):
                visit(child, multiplier, loop_iterations)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            comprehension_iterations = 1
            unbounded_comprehension = loop_iterations < 0
            for generator in node.generators:
                visit(generator.iter, multiplier, loop_iterations)
                iterations = _range_iterations(generator.iter)
                if iterations is None:
                    unbounded_comprehension = True
                    effective_iterations = 100
                else:
                    effective_iterations = min(iterations, 1000)
                comprehension_iterations *= effective_iterations
                comprehension_iterations = min(comprehension_iterations, 1000)
                for condition in generator.ifs:
                    visit(
                        condition,
                        multiplier * comprehension_iterations,
                        -1 if unbounded_comprehension else comprehension_iterations,
                    )
            combined_iterations = (
                -1
                if unbounded_comprehension
                else comprehension_iterations * max(1, loop_iterations)
            )
            body_multiplier = multiplier * comprehension_iterations
            if isinstance(node, ast.DictComp):
                visit(node.key, body_multiplier, combined_iterations)
                visit(node.value, body_multiplier, combined_iterations)
            else:
                visit(node.elt, body_multiplier, combined_iterations)
            return
        if isinstance(node, ast.For):
            iterations = _range_iterations(node.iter)
            effective_iterations, loop_marker = loop_scale(iterations, loop_iterations)
            visit(node.iter, multiplier, loop_iterations)
            for statement in node.body:
                visit(statement, multiplier * effective_iterations, loop_marker)
            for statement in node.orelse:
                visit(statement, multiplier, loop_iterations)
            return
        if isinstance(node, ast.While):
            for statement in node.body:
                visit(statement, multiplier * 100, -1)
            for statement in node.orelse:
                visit(statement, multiplier, loop_iterations)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, multiplier, loop_iterations)

    visit(tree)
    estimated = sum(count * _CALL_COSTS[name] for name, count in counts.items())
    return {
        "estimated_execution_cost": round(estimated, 3),
        "weighted_call_counts": {
            key: int(value) if value.is_integer() else value
            for key, value in sorted(counts.items())
        },
        "expensive_loop_calls": expensive_loop_calls,
        "max_expensive_loop_iterations": max_expensive_iterations,
        "unbounded_expensive_loop": unbounded_expensive_loop,
    }


def _range_iterations(node: ast.AST) -> int | None:
    if not isinstance(node, ast.Call) or _call_name(node) != "range":
        return None
    try:
        values = [int(ast.literal_eval(argument)) for argument in node.args]
        return len(range(*values))
    except (TypeError, ValueError):
        return None


def _execution_cost_score(cost: Mapping[str, Any]) -> float:
    value = float(cost.get("estimated_execution_cost", 0.0))
    if value <= 40:
        return 15.0
    if value <= 80:
        return 5.0
    if value <= 160:
        return -15.0
    return -40.0


def _valid_python(code: str) -> bool:
    if not code.strip():
        return False
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _decision_alignment(decision: str, directive: str, task_completed: bool) -> float:
    if task_completed:
        return 35.0 if decision == "finish" else -20.0
    if decision == "finish":
        return -80.0
    if directive in {"retry", "switch_strategy"}:
        return 30.0 if decision == "regenerate" else -15.0
    if directive == "wait":
        return 20.0 if decision == "continue" else 0.0
    return 15.0 if decision in {"initial", "continue", "regenerate"} else 0.0


def _stage_alignment(code: str, decision: str, stage: str) -> float:
    if decision in {"finish", "continue"}:
        return 10.0
    calls = strategy_fingerprint(code)
    call_set = set(calls)
    if calls == ("<invalid-python>",) or not calls:
        return -30.0
    expected: dict[str, set[str]] = {
        "ACQUIRE_GRASP": {"sample_grasp_pose", "goto_pose", "close_gripper"},
        "ACQUIRE_TARGET_AFTER_CLEAR": {"sample_grasp_pose", "goto_pose", "close_gripper"},
        "GRASPED_NOT_LIFTED": {"goto_pose"},
        "LIFTED_READY_TO_PLACE": {"get_object_pose", "goto_pose", "open_gripper"},
        "PLACED_NOT_RELEASED": {"open_gripper", "goto_pose"},
        "RELEASED_UNSTABLE": {"goto_pose"},
        "VERIFY_STACK": {"home_pose"},
        "WRONG_OBJECT_GRASPED": {"open_gripper", "goto_pose"},
    }
    required = expected.get(stage, {"goto_pose"})
    coverage = len(required & call_set) / len(required)
    score = 35.0 * coverage
    if decision == "initial" and code.count("breakpoint_code_block") == 2:
        score += 10.0
    return round(score, 3)


def _strategy_novelty(code: str, decision: str, directive: str, previous_code: str) -> float:
    if decision != "regenerate" or not previous_code:
        return 5.0
    changed = strategy_fingerprint(code) != strategy_fingerprint(previous_code)
    if directive == "switch_strategy":
        return 20.0 if changed else -50.0
    return 8.0 if changed else 0.0


def _efficiency_score(code: str) -> float:
    calls = strategy_fingerprint(code)
    if calls == ("<invalid-python>",):
        return -10.0
    excess_grasp_sampling = max(0, calls.count("sample_grasp_pose") - 3)
    excess_calls = max(0, len(calls) - 24)
    return max(-10.0, 10.0 - 2.0 * excess_grasp_sampling - 0.5 * excess_calls)


def config_to_dict(config: CandidateRerankingConfig) -> dict[str, Any]:
    return asdict(config)
