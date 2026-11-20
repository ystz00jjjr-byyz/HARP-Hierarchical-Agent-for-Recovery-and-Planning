"""Single-trial execution for CaP-X environments.

This module handles single trial execution including code generation,
multi-turn decisions, and visual feedback. It contains the core trial
loop extracted from launch.py, covering:

- Initial code generation and oracle code handling
- Code block execution with multi-turn regeneration
- Visual feedback capture and image/video differencing
- Trial artifact saving (code, logs, per-turn videos, combined video)
"""

from __future__ import annotations

import ast
import base64
import copy
import gc
import io
import json
import os
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from capx.envs.configs.instantiate import instantiate
from capx.envs.tasks.base import CodeExecutionEnvBase

from capx.llm.client import (
    VLM_MODELS,
    ModelQueryArgs,
    query_model as _query_model,
    query_model_ensemble as _query_model_ensemble,
    query_single_model_ensemble as _query_single_model_ensemble,
)
from capx.recovery.trial_feedback import (
    TextWorldStateFeedbackConfig,
    TrialWorldStateFeedback,
    append_text_world_state_feedback,
    capture_text_world_state,
    save_initial_state_fingerprint,
)
from capx.recovery.candidate_reranking import (
    CandidateRerankingConfig,
    aggregate_token_usage,
    parse_world_state_feedback,
    select_candidate,
)
from capx.recovery.execution_guard import (
    ExecutionGuardConfig,
    GuardResult,
    assign_stage_indices,
    format_guard_feedback,
    task_completion_confirmed,
    validate_code_block,
    validate_decision,
    validate_stage,
)
from capx.recovery.offline import (
    _check_conditions,
    OfflineAction,
    OfflineExecution,
    OfflineRunResult,
    OfflineTaskGraph,
    OfflineWindowController,
    RecoveryLibrary,
    StateParameterError,
    attach_action_library,
    bind_state_parameters,
    prepend_failure_trigger,
    select_hybrid_recovery,
)
from capx.utils.launch_utils import (
    TrialSummary,
    _build_multi_turn_decision_prompt,
    _build_multi_turn_decision_prompt_legacy,
    _extract_code,
    _get_visual_feedback,
    _parse_multi_turn_decision,
    _save_trial_artifacts,
)
from capx.utils.video_utils import _encode_video_base64, _write_video

# Use TYPE_CHECKING to avoid circular imports for type hints only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs


MULTITURN_LIMIT = 10


def _model_requests_used(events: list[dict[str, Any]]) -> int:
    """Return the number of external model requests recorded so far."""
    return sum(int(event.get("model_request_count", 0) or 0) for event in events)


def _attach_hybrid_stage_monitors(
    metadata_items: list[dict[str, Any]], stage_actions: tuple[str, ...]
) -> None:
    """Keep every model block covered by the nearest task-graph stage."""
    if not stage_actions:
        return
    last_stage = len(stage_actions) - 1
    for metadata in metadata_items:
        stage_index = int(metadata["stage_index"])
        metadata["hybrid_monitor_action_id"] = stage_actions[
            min(max(stage_index, 0), last_stage)
        ]


def _selection_log_fields(selection_data: dict | None) -> dict[str, Any]:
    """Flatten candidate-selection accounting into one response-log event."""
    fields: dict[str, Any] = {}
    elapsed = (selection_data or {}).get("model_response_seconds")
    if elapsed is not None:
        fields["model_response_seconds"] = float(elapsed)
    audit = (selection_data or {}).get("candidate_reranking")
    if not isinstance(audit, dict):
        return fields
    candidates = audit.get("candidates", [])
    fields.update({
        "candidate_reranking": audit,
        "model_request_count": int(audit.get("candidate_count", len(candidates))),
        "token_accounted_requests": sum(
            isinstance(candidate.get("token_usage"), dict)
            for candidate in candidates
            if isinstance(candidate, dict)
        ),
    })
    return fields


def _model_query_log_fields(selection_data: dict | None) -> dict[str, Any]:
    """Account for every model query, including a single non-ensemble call."""
    fields = _selection_log_fields(selection_data)
    fields.setdefault("model_request_count", 1)
    return fields


def _parse_l2_advisory(content: str, phase: str) -> dict[str, Any]:
    """Parse one code-free L2 decision against a phase-specific whitelist."""
    if not isinstance(content, str):
        raise ValueError("L2 advisory content must be text")
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("L2 advisory must contain one JSON object")
    try:
        selection = json.loads(content[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"L2 advisory is not valid JSON: {exc}") from exc
    schemas = {
        "initial": (
            {"decision", "template", "confidence"},
            {"EXECUTE", "SAFE_STOP"},
            {"bin_place_nominal_v1"},
        ),
        "stage": (
            {"decision", "next_template", "confidence"},
            {"CONTINUE", "SAFE_STOP"},
            {"bin_place_nominal_place_v1"},
        ),
        "final": (
            {"decision", "confidence"},
            {"ACCEPT_COMPLETION", "REQUEST_RECHECK"},
            None,
        ),
    }
    if phase not in schemas:
        raise ValueError(f"unsupported L2 advisory phase: {phase}")
    required, decisions, templates = schemas[phase]
    if not isinstance(selection, dict) or set(selection) != required:
        raise ValueError(
            f"L2 {phase} fields must be exactly {', '.join(sorted(required))}"
        )
    if selection["decision"] not in decisions:
        raise ValueError(f"unsupported L2 {phase} decision")
    if templates is not None:
        template_key = "template" if phase == "initial" else "next_template"
        if selection[template_key] not in templates:
            raise ValueError(f"unsupported L2 {phase} template")
    confidence = selection["confidence"]
    if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        raise ValueError("L2 confidence must be between 0 and 1")
    return selection


def _query_l2_advisory(
    args: LaunchArgs,
    *,
    phase: str,
    task_description: str,
    world_state_feedback: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Ask the model for a bounded task-graph decision, never executable code."""
    schemas = {
        "initial": (
            '{"decision":"EXECUTE","template":"bin_place_nominal_v1",'
            '"confidence":0.0}'
        ),
        "stage": (
            '{"decision":"CONTINUE","next_template":'
            '"bin_place_nominal_place_v1","confidence":0.0}'
        ),
        "final": '{"decision":"ACCEPT_COMPLETION","confidence":0.0}',
    }
    prompt = [
        {
            "role": "system",
            "content": (
                "You are the L2 supervisor for a robot task graph. Never write "
                "or suggest executable code. Select only the trusted template "
                "named in the requested JSON schema. Return exactly one JSON "
                "object and no markdown. Use SAFE_STOP when the current state is "
                "unsafe or geometrically infeasible. Environment completion is "
                "authoritative."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Phase: {phase}\nTask: {task_description}\n"
                f"K=3 world state:\n{world_state_feedback}\n"
                f"Required schema: {schemas[phase]}"
            ),
        },
    ]
    responses = []
    parse_errors = []
    for attempt in range(2):
        response = _query_model(args, prompt)
        responses.append(response)
        try:
            selection = _parse_l2_advisory(response["content"], phase)
            break
        except (KeyError, ValueError) as exc:
            parse_errors.append(str(exc))
            if attempt:
                selection = {
                    "initial": {
                        "decision": "SAFE_STOP",
                        "template": "bin_place_nominal_v1",
                        "confidence": 0.0,
                    },
                    "stage": {
                        "decision": "SAFE_STOP",
                        "next_template": "bin_place_nominal_place_v1",
                        "confidence": 0.0,
                    },
                    "final": {
                        "decision": "ACCEPT_COMPLETION",
                        "confidence": 0.0,
                    },
                }[phase]
                break
            prompt = [
                *prompt,
                {"role": "assistant", "content": response["content"]},
                {
                    "role": "user",
                    "content": (
                        "Invalid decision schema. Return only the exact JSON "
                        f"schema already requested for phase {phase}."
                    ),
                },
            ]
    combined = dict(response)
    combined["model_request_count"] = len(responses)
    combined["model_response_seconds"] = sum(
        float(item.get("model_response_seconds", 0.0)) for item in responses
    )
    combined["token_usage"] = aggregate_token_usage(responses)
    combined["l2_schema_fallback"] = len(parse_errors) == 2
    combined["l2_parse_errors"] = parse_errors
    combined["l2_attempts"] = [
        {
            "content": item.get("content"),
            "reasoning": item.get("reasoning"),
            "token_usage": item.get("token_usage"),
            "model_response_seconds": item.get("model_response_seconds", 0.0),
        }
        for item in responses
    ]
    return selection, combined, prompt


def _remaining_sim_steps(env: CodeExecutionEnvBase) -> int | None:
    """Return the simulator horizon left, when the backend exposes it."""
    low_level = getattr(env, "low_level_env", env)
    maximum = getattr(low_level, "max_steps", None)
    current = getattr(low_level, "_sim_step_count", None)
    if maximum is None or current is None:
        return None
    try:
        return max(0, int(maximum) - int(current))
    except (TypeError, ValueError):
        return None


def _recovery_role_for_failed_action(
    action_id: str | None, world_state: dict[str, Any]
) -> str:
    """Keep model fallback bound to the role of the failed trusted action."""
    roles = ("primary", "secondary", "tertiary", "quaternary")
    if action_id:
        for role in roles:
            if str(action_id).endswith(f"_{role}"):
                return role
    active_role = str(world_state.get("active_object_role", "primary"))
    return active_role if active_role in roles else "primary"


def _recovery_binding(
    active_role: str | None, world_state: dict[str, Any]
) -> tuple[str, str]:
    """Bind a structured recovery to the failed role and its target region."""
    role = active_role or _recovery_role_for_failed_action(None, world_state)
    target = "target"
    target_regions = world_state.get("target_regions")
    if isinstance(target_regions, dict):
        target_state = target_regions.get(role)
        if isinstance(target_state, dict):
            target = str(target_state.get("target_name", f"target_{role}"))
    return role, target


def _recovery_monitor_actions(
    graph: Any, role: str, fallback: tuple[str, ...]
) -> tuple[str, ...]:
    """Use role-specific grasp/place postconditions for staged recovery."""
    action_ids = (f"candidate_grasp_{role}", f"candidate_replace_{role}")
    if graph is not None and all(action_id in graph.actions for action_id in action_ids):
        return action_ids
    return fallback


def _parse_structured_recovery_selection(
    selection: str | dict[str, Any] | None,
) -> dict[str, Any]:
    """Accept exactly one raw JSON or Python dict, never executable code."""
    if isinstance(selection, dict):
        return dict(selection)
    if not isinstance(selection, str) or not selection.strip():
        raise ValueError("template selection must be one structured object")
    text = selection.strip()
    if "```" in text:
        raise ValueError("template selection must not use markdown or fenced Python")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"template selection is not valid JSON or a Python literal dict: {exc}"
            ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("template selection must be a dict")
    return dict(parsed)


def _compile_bin_place_recovery_template(
    selection_text: str | dict[str, Any] | None,
    object_role: str = "primary",
    target_region: str = "target",
) -> tuple[list[str], dict[str, Any]]:
    """Compile one bounded model selection into trusted regrasp/replace code."""
    allowed_roles = {"primary", "secondary", "tertiary", "quaternary"}
    allowed_targets = {"target", *(f"target_{role}" for role in allowed_roles)}
    if object_role not in allowed_roles or target_region not in allowed_targets:
        raise ValueError("unsupported recovery role or target region")
    role_literal = json.dumps(object_role)
    target_literal = json.dumps(target_region)
    selection = _parse_structured_recovery_selection(selection_text)
    required = {
        "template",
        "grasp_mode",
        "approach_clearance",
        "lift_clearance",
        "entry_clearance",
        "pregrasp_home",
    }
    if set(selection) != required:
        raise ValueError(
            "template selection fields must be exactly " + ", ".join(sorted(required))
        )
    if selection["template"] != "bin_place_regrasp_replace_v1":
        raise ValueError("unsupported recovery template")
    grasp_mode = str(selection["grasp_mode"])
    if grasp_mode not in {
        "sample_grasp_pose",
        "yaw_topdown",
        "boundary_yaw_topdown",
        "orthogonal_cuboid",
    }:
        raise ValueError(
            "grasp_mode must be sample_grasp_pose, yaw_topdown, "
            "boundary_yaw_topdown, or orthogonal_cuboid"
        )
    pregrasp_home = selection["pregrasp_home"]
    if not isinstance(pregrasp_home, bool):
        raise ValueError("pregrasp_home must be True or False")

    def discrete(name: str, allowed: tuple[float, ...]) -> float:
        try:
            value = float(selection[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not any(abs(value - candidate) < 1e-9 for candidate in allowed):
            raise ValueError(f"{name} must be one of {allowed}")
        return value

    approach = discrete("approach_clearance", (0.08, 0.10, 0.12))
    lift = discrete("lift_clearance", (0.15, 0.18))
    entry = discrete("entry_clearance", (0.08, 0.10, 0.12))
    if grasp_mode == "orthogonal_cuboid":
        grasp_code = (
            f"grasp_pos, grasp_quat = sample_cuboid_grasp_pose("
            f"{role_literal}, candidate_index=1)"
        )
    elif grasp_mode == "yaw_topdown":
        boundary_yaw_offset = ""
        boundary_inset = ""
        grasp_code = f"""w, x, y, z = current_quat / np.linalg.norm(current_quat)
yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
{boundary_yaw_offset}grasp_depth = np.clip(0.20 * float(primary_extent[2]), 0.004, 0.008)
grasp_pos = current_pos - np.array([0.0, 0.0, grasp_depth])
{boundary_inset}grasp_quat = np.array([0.0, -np.sin(0.5 * yaw), np.cos(0.5 * yaw), 0.0])"""
    else:
        boundary_yaw_offset = (
            "yaw += 0.5 * np.pi\n"
        )
        boundary_inset = (
            "grasp_pos[1] -= np.copysign(0.01, current_pos[1])\n"
        )
        boundary_code = f"""w, x, y, z = current_quat / np.linalg.norm(current_quat)
yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
{boundary_yaw_offset}grasp_depth = np.clip(0.20 * float(primary_extent[2]), 0.004, 0.008)
grasp_pos = current_pos - np.array([0.0, 0.0, grasp_depth])
{boundary_inset}grasp_quat = np.array([0.0, -np.sin(0.5 * yaw), np.cos(0.5 * yaw), 0.0])"""
        grasp_code = f"""if abs(float(current_pos[1])) >= 0.27:
{textwrap.indent(boundary_code, '    ')}
else:
    grasp_pos, grasp_quat = sample_grasp_pose({role_literal})"""
    home_code = "home_pose()\n" if pregrasp_home else ""
    regrasp = f"""import numpy as np
current_pos, current_quat, primary_extent = get_object_pose({role_literal}, return_bbox_extent=True)
{grasp_code}
approach = max({approach!r}, 2.0 * float(primary_extent[2]))
lift = max({lift!r}, 3.0 * float(primary_extent[2]))
{home_code}open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=approach)
close_gripper()
lift_pos = grasp_pos + np.array([0.0, 0.0, lift])
goto_pose(lift_pos, grasp_quat)"""
    replace = f"""target_object_center, target_extent, rim_z = get_target_region({target_literal})
target_gripper = target_object_center + (grasp_pos - current_pos)
entry = target_gripper.copy()
entry[2] = max(rim_z + {entry!r}, target_gripper[2] + {entry!r})
transit = 0.5 * (lift_pos + entry)
transit[2] = max(float(lift_pos[2]), float(entry[2]))
goto_pose(transit, grasp_quat)
goto_pose(entry, grasp_quat)
goto_pose(target_gripper, grasp_quat)
open_gripper()"""
    normalized = {
        "template": "bin_place_regrasp_replace_v1",
        "grasp_mode": grasp_mode,
        "approach_clearance": approach,
        "lift_clearance": lift,
        "entry_clearance": entry,
        "pregrasp_home": pregrasp_home,
    }
    return [regrasp, replace], normalized


def _compile_bin_place_recovery_candidates(
    selection_text: str | dict[str, Any] | None,
    candidate_limit: int,
    object_role: str = "primary",
    target_region: str = "target",
) -> tuple[list[str], dict[str, Any]]:
    """Compile one model selection plus one deterministic alternate candidate."""
    blocks, selection = _compile_bin_place_recovery_template(
        selection_text, object_role=object_role, target_region=target_region
    )
    if candidate_limit == 1:
        return blocks, selection
    if candidate_limit != 2:
        raise ValueError("model recovery candidate limit must be 1 or 2")

    alternate = {
        **selection,
        "grasp_mode": (
            "orthogonal_cuboid"
            if selection["grasp_mode"] == "sample_grasp_pose"
            else "sample_grasp_pose"
        ),
        "approach_clearance": 0.12,
        "lift_clearance": 0.18,
        "entry_clearance": 0.12,
        "pregrasp_home": True,
    }
    alternate_blocks, alternate_selection = _compile_bin_place_recovery_template(
        alternate, object_role=object_role, target_region=target_region
    )
    selection = {
        **selection,
        "candidate_count": 2,
        "derived_second_candidate": alternate_selection,
    }
    return [*blocks, *alternate_blocks], selection


def _compile_recovery_graph_v1(
    selection_text: str | dict[str, Any] | None,
    object_role: str = "primary",
    target_region: str = "target",
) -> tuple[list[str], dict[str, Any]]:
    """Compile a bounded skill graph into the existing trusted two-stage path."""
    selection = _parse_structured_recovery_selection(selection_text)
    if set(selection) != {"version", "steps"}:
        raise ValueError("recovery graph fields must be exactly steps and version")
    if selection["version"] != "recovery_graph_v1":
        raise ValueError("unsupported recovery graph version")
    steps = selection["steps"]
    if not isinstance(steps, list):
        raise ValueError("recovery graph steps must be a list")

    expected = [
        "reobserve_object",
        "sample_grasp",
        "regrasp",
        "reobserve_target",
        "replace",
        "verify",
    ]
    skills: list[str] = []
    normalized_steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, dict):
            raise ValueError(f"recovery graph step {index} must be an object")
        step = dict(raw_step)
        skill = str(step.get("skill", ""))
        skills.append(skill)
        normalized_steps.append(step)
    pregrasp_home = bool(skills and skills[0] == "safe_home")
    if pregrasp_home:
        if normalized_steps[0] != {"skill": "safe_home"}:
            raise ValueError("safe_home accepts no parameters")
        skills = skills[1:]
        normalized_steps = normalized_steps[1:]
    if skills != expected:
        raise ValueError(
            "recovery graph skills must be "
            + " -> ".join(expected)
            + " with optional leading safe_home"
        )

    object_step, grasp_step, regrasp_step, target_step, replace_step, verify_step = (
        normalized_steps
    )
    if object_step != {"skill": "reobserve_object"}:
        raise ValueError("reobserve_object accepts no parameters")
    if target_step != {"skill": "reobserve_target"}:
        raise ValueError("reobserve_target accepts no parameters")
    if verify_step != {"skill": "verify"}:
        raise ValueError("verify accepts no parameters")
    if set(grasp_step) != {"skill", "mode"}:
        raise ValueError("sample_grasp fields must be exactly mode and skill")
    if set(regrasp_step) != {
        "skill", "approach_clearance", "lift_clearance"
    }:
        raise ValueError(
            "regrasp fields must be exactly approach_clearance, lift_clearance, and skill"
        )
    if set(replace_step) != {"skill", "entry_clearance"}:
        raise ValueError("replace fields must be exactly entry_clearance and skill")

    template_selection = {
        "template": "bin_place_regrasp_replace_v1",
        "grasp_mode": grasp_step["mode"],
        "approach_clearance": regrasp_step["approach_clearance"],
        "lift_clearance": regrasp_step["lift_clearance"],
        "entry_clearance": replace_step["entry_clearance"],
        "pregrasp_home": pregrasp_home,
    }
    blocks, compiled = _compile_bin_place_recovery_template(
        template_selection,
        object_role=object_role,
        target_region=target_region,
    )
    return blocks, {
        "version": "recovery_graph_v1",
        "steps": ([{"skill": "safe_home"}] if pregrasp_home else [])
        + normalized_steps,
        "compiled_template": compiled,
    }


def _compile_model_recovery_selection(
    mode: str,
    selection_text: str | dict[str, Any] | None,
    candidate_limit: int,
    *,
    object_role: str = "primary",
    target_region: str = "target",
) -> tuple[list[str], dict[str, Any]]:
    """Dispatch one structured model plan to a trusted local compiler."""
    if mode == "bin_place_regrasp_replace_v1":
        return _compile_bin_place_recovery_candidates(
            selection_text,
            candidate_limit,
            object_role=object_role,
            target_region=target_region,
        )
    if mode != "recovery_graph_v1":
        raise ValueError("unsupported structured recovery mode")
    blocks, graph = _compile_recovery_graph_v1(
        selection_text,
        object_role=object_role,
        target_region=target_region,
    )
    if candidate_limit == 1:
        return blocks, graph
    if candidate_limit != 2:
        raise ValueError("model recovery candidate limit must be 1 or 2")
    alternate = {
        **graph["compiled_template"],
        "grasp_mode": (
            "orthogonal_cuboid"
            if graph["compiled_template"]["grasp_mode"] == "sample_grasp_pose"
            else "sample_grasp_pose"
        ),
        "approach_clearance": 0.12,
        "lift_clearance": 0.18,
        "entry_clearance": 0.12,
        "pregrasp_home": True,
    }
    alternate_blocks, alternate_selection = _compile_bin_place_recovery_template(
        alternate,
        object_role=object_role,
        target_region=target_region,
    )
    graph["candidate_count"] = 2
    graph["derived_second_candidate"] = alternate_selection
    return [*blocks, *alternate_blocks], graph


def _structured_recovery_schema(mode: str) -> str:
    if mode == "recovery_graph_v1":
        return json.dumps({
            "version": "recovery_graph_v1",
            "steps": [
                {"skill": "safe_home"},
                {"skill": "reobserve_object"},
                {"skill": "sample_grasp", "mode": "sample_grasp_pose"},
                {
                    "skill": "regrasp",
                    "approach_clearance": 0.10,
                    "lift_clearance": 0.15,
                },
                {"skill": "reobserve_target"},
                {"skill": "replace", "entry_clearance": 0.10},
                {"skill": "verify"},
            ],
        }, separators=(",", ":"))
    return (
        '{"template":"bin_place_regrasp_replace_v1",'
        '"grasp_mode":"sample_grasp_pose",'
        '"approach_clearance":0.10,"lift_clearance":0.15,'
        '"entry_clearance":0.10,"pregrasp_home":true}'
    )


def _current_task_completed(env: CodeExecutionEnvBase) -> bool | None:
    """Read the authoritative environment completion predicate without acting."""
    checker = getattr(env.low_level_env, "task_completed", None)
    return bool(checker()) if callable(checker) else None


def _policy_task_completed(
    state: dict[str, Any], evaluation_task_completed: bool | None
) -> bool | None:
    """Keep evaluator truth out of explicitly non-privileged policy state."""
    if state.get("uses_simulator_truth") is False:
        value = state.get("task_completed")
        return bool(value) if value is not None else None
    return evaluation_task_completed


def _completed_before_termination(
    env: CodeExecutionEnvBase, exc: Exception, completion_latched: bool
) -> bool:
    """Reconcile completion after an action reports a terminated episode."""
    return bool(
        "terminated episode" in str(exc).lower()
        and (completion_latched or _current_task_completed(env) is True)
    )


def _is_completed_terminal_result(
    task_completed: bool | None, terminated: bool, stderr: str
) -> bool:
    return bool(
        task_completed is True
        and (terminated or "terminated episode" in stderr.lower())
    )


def _load_offline_value(value: Any) -> dict[str, Any]:
    """Load an inline or YAML-backed offline graph/library declaration."""
    if isinstance(value, str):
        import yaml

        path = Path(value).expanduser()
        with path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if not isinstance(loaded, dict):
            raise ValueError(f"offline recovery file must contain a mapping: {path}")
        return loaded
    if not isinstance(value, dict):
        raise ValueError("offline recovery declarations must be mappings or YAML paths")
    return dict(value)


def _run_offline_task_graph_trial(
    env: CodeExecutionEnvBase,
    trial: int,
    config: dict[str, Any],
    offline_cfg: dict[str, Any],
    partial_artifacts: dict[str, Any] | None = None,
) -> TrialSummary:
    """Execute a fully static task graph without a recovery-model request."""
    trial_start = time.time()
    output_dir = config.get("output_dir")
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    recorder_cfg_dict = config.get("world_state_recorder", {"enabled": False})
    if recorder_cfg_dict.get("enabled", False):
        from capx.recording import WorldStateRecorder, WorldStateRecorderConfig

        if not output_dir:
            raise ValueError("world_state_recorder requires output_dir")
        recorder_cfg = WorldStateRecorderConfig.from_dict(recorder_cfg_dict)
        env.set_world_state_recorder(WorldStateRecorder(
            Path(output_dir) / "world_state" / f"trial_{trial:03d}",
            recorder_cfg,
            episode_id=f"trial_{trial:03d}",
        ))
    else:
        env.set_world_state_recorder(None)

    obs, _ = env.reset(options={"trial": trial}, seed=trial)
    if output_dir:
        save_initial_state_fingerprint(env, output_dir, trial)
    if config.get("record_video") and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(True, clear=True)

    state_cfg = dict(offline_cfg.get("state_sampling", {}))
    state_cfg.setdefault("enabled", True)
    state_cfg.setdefault("window_size", 3)
    state_cfg.setdefault("action_aligned_window", True)
    state_cfg.setdefault("fail_open", False)
    state_feedback = TrialWorldStateFeedback(
        env, TextWorldStateFeedbackConfig.from_dict(state_cfg)
    )
    initial_summary = state_feedback.capture()
    initial_state = initial_summary.state if initial_summary is not None else {}

    graph_value = _load_offline_value(offline_cfg.get("task_graph", offline_cfg))
    graph = OfflineTaskGraph.from_mapping(graph_value)
    action_library = offline_cfg.get("action_library")
    if action_library:
        graph = attach_action_library(graph, _load_offline_value(action_library))
    graph = prepend_failure_trigger(
        graph,
        str(offline_cfg.get("inject_exception", "")),
        offline_cfg.get("inject_state"),
        tuple(str(item) for item in offline_cfg.get("inject_prepare_actions", ())),
    )
    library_value = offline_cfg.get("recovery_library", {})
    library = RecoveryLibrary.from_mapping(_load_offline_value(library_value))
    event_log: list[dict[str, Any]] = []

    def record_event(event: Any) -> None:
        event_log.append({
            "event_type": event.event_type,
            "action_id": event.action_id,
            "exception_id": event.exception_id,
            "payload": event.payload,
        })
        env.record_world_state_event(event.event_type, metadata=event.payload)

    controller = OfflineWindowController(
        graph,
        library,
        k_action=int(offline_cfg.get("k_action", 3)),
        k_state=int(offline_cfg.get("k_state", 3)),
        min_stable_fraction=float(offline_cfg.get("min_stable_fraction", 2 / 3)),
        event_sink=record_event,
    )
    last_info: dict[str, Any] = {
        "sandbox_rc": -1, "stdout": "", "stderr": "", "task_completed": False,
    }
    last_reward = 0.0
    last_terminated = False
    last_truncated = False
    completion_latched = False

    def execute(action: OfflineAction) -> OfflineExecution:
        nonlocal obs, last_info, last_reward, last_terminated, last_truncated, completion_latched
        env.record_world_state_event("offline_action_start", metadata={"action_id": action.action_id})
        try:
            binding_summary = state_feedback.capture_action_window()
            current_state = (
                dict(binding_summary.state)
                if binding_summary is not None
                else capture_text_world_state(env, include_entities=False)
            )
            nonprivileged_state = current_state.get("uses_simulator_truth") is False
            bound_code, bound_parameters = bind_state_parameters(action, current_state)
            event_log.append({
                "event_type": "parameters.bound",
                "action_id": action.action_id,
                "exception_id": None,
                "payload": {"parameters": bound_parameters},
            })
            env.record_world_state_event(
                "offline_parameters_bound",
                metadata={"action_id": action.action_id, "parameters": bound_parameters},
            )
            obs_next, reward, terminated, truncated, info = env.step(bound_code)
        except StateParameterError as exc:
            nonprivileged_state = initial_state.get("uses_simulator_truth") is False
            event_log.append({
                "event_type": "parameters.failed",
                "action_id": action.action_id,
                "exception_id": "STATE_PARAMETER_MISSING",
                "payload": {"error": str(exc)},
            })
            info = {
                "sandbox_rc": 1,
                "stdout": "",
                "stderr": str(exc),
                "task_completed": False,
                "offline_exception_id": "STATE_PARAMETER_MISSING",
            }
            obs_next, reward, terminated, truncated = obs, 0.0, False, False
        except Exception as exc:
            nonprivileged_state = initial_state.get("uses_simulator_truth") is False
            episode_terminated = "terminated episode" in str(exc).lower()
            completed_before_termination = _completed_before_termination(
                env, exc, completion_latched
            )
            info = {
                "sandbox_rc": 1,
                "stdout": "",
                "stderr": f"offline action raised {type(exc).__name__}: {exc}",
                "task_completed": completed_before_termination,
                "offline_exception_id": (
                    "COMPLETED_BEFORE_TERMINATION"
                    if completed_before_termination
                    else None
                ),
            }
            obs_next, reward, terminated, truncated = (
                obs, float(completed_before_termination), episode_terminated, False
            )
        obs = obs_next
        last_info = dict(info)
        last_reward = float(reward)
        last_terminated = bool(terminated)
        last_truncated = bool(truncated)
        evaluation_task_completed = info.get("task_completed")
        if evaluation_task_completed is None:
            evaluation_task_completed = _current_task_completed(env)
        completion_latched = bool(
            completion_latched or evaluation_task_completed is True
        )
        if _is_completed_terminal_result(
            evaluation_task_completed,
            bool(terminated),
            str(info.get("stderr", "")),
        ):
            info["offline_exception_id"] = "COMPLETED_BEFORE_TERMINATION"
        terminal_state = bool(
            truncated
            or terminated
            or (
                not nonprivileged_state
                and evaluation_task_completed is True
            )
        )
        if terminal_state:
            state_feedback.frames.clear()
            summary = state_feedback.capture()
        else:
            summary = state_feedback.capture_action_window()
        state = dict(summary.state) if summary is not None else {}
        if not nonprivileged_state and evaluation_task_completed is not None:
            state["task_completed"] = bool(evaluation_task_completed)
        policy_task_completed = _policy_task_completed(
            state, evaluation_task_completed
        )
        if info.get("offline_exception_id") == "COMPLETED_BEFORE_TERMINATION":
            policy_task_completed = True
        raw_window: tuple[dict[str, Any], ...] = ()
        if not terminal_state and state_feedback.history:
            raw = state_feedback.history[-1].get("raw_window", ())
            if isinstance(raw, list):
                raw_window = tuple(dict(item) for item in raw if isinstance(item, dict))
        env.record_world_state_event(
            "offline_action_end",
            metadata={"action_id": action.action_id, "sandbox_rc": info.get("sandbox_rc"),
                      "evaluation_task_completed": evaluation_task_completed,
                      "policy_task_completed": policy_task_completed},
        )
        return OfflineExecution(
            state=state,
            state_window=raw_window or (state,),
            stdout=str(info.get("stdout", "")),
            stderr=str(info.get("stderr", "")),
            reward=last_reward,
            sandbox_rc=int(info.get("sandbox_rc", 1)),
            task_completed=policy_task_completed,
            terminated=(last_terminated if not nonprivileged_state else bool(policy_task_completed)),
            truncated=last_truncated,
            exception_id=info.get("offline_exception_id"),
        )

    result: OfflineRunResult = controller.run(execute, initial_state=initial_state)
    final_code = "\n\n".join(
        f"# Offline action {action.action_id}\n{action.code}"
        for action in graph.actions.values()
    )
    policy_task_completed = result.task_completed is True
    evaluation_task_completed = last_info.get("task_completed")
    if evaluation_task_completed is None:
        evaluation_task_completed = _current_task_completed(env)
    task_completed = evaluation_task_completed is True
    success = result.status == "success" and task_completed
    stderr = str(last_info.get("stderr", ""))
    if result.status != "success":
        stderr = "\n".join(part for part in (stderr, f"OFFLINE_SAFE_STOP: {result.exception_id}") if part)
    log_lines = [
        "Offline task-graph execution",
        f"Status: {result.status}",
        f"Executed actions: {', '.join(result.executed_actions)}",
        f"Failed action: {result.failed_action or '-'}",
        f"Exception: {result.exception_id or '-'}",
        f"Sandbox rc: {last_info.get('sandbox_rc', -1)}",
        f"Reward: {last_reward}",
        f"Task completed: {task_completed}",
        f"Policy task completed: {policy_task_completed}",
        f"K_action: {offline_cfg.get('k_action', 3)}, K_state: {offline_cfg.get('k_state', 3)}",
    ]
    code_path = _save_trial_artifacts(
        config, trial, int(last_info.get("sandbox_rc", 1)), last_reward,
        task_completed, final_code, final_code, [{"decision": "offline_task_graph", "events": event_log}],
        log_lines, [],
    )
    _save_trial_video(
        env,
        config,
        trial,
        {**last_info, "task_completed": task_completed},
        last_reward,
        len(result.executed_actions),
    )
    if output_dir:
        (Path(output_dir) / f"offline_recovery_trial_{trial:03d}.json").write_text(
            json.dumps({"config": offline_cfg, "events": event_log,
                        "result": {"status": result.status,
                                   "executed_actions": result.executed_actions,
                                   "failed_action": result.failed_action,
                                   "exception_id": result.exception_id}},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    env.record_world_state_event("trial_end", metadata={
        "trial": trial, "success": success, "reward": last_reward,
        "evaluation_task_completed": task_completed,
        "policy_task_completed": policy_task_completed, "offline": True,
    })
    env.set_world_state_recorder(None)
    if partial_artifacts is not None:
        partial_artifacts.update({"final_code": final_code, "num_code_blocks": len(result.executed_actions)})
    gc.collect()
    return TrialSummary(
        trial=trial, success=success, reward=last_reward,
        terminated=last_terminated, truncated=last_truncated or not success,
        sandbox_rc=int(last_info.get("sandbox_rc", 1)), log="\n".join(log_lines + [stderr]),
        task_completed=task_completed, code_path=code_path,
        num_regenerations=0, num_finishes=0, num_code_blocks=len(result.executed_actions),
    )

# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

def _annotate_code_blocks(
    code_blocks: list[str],
    code_block_metadata: list[dict[str, Any]],
) -> str:
    """Join code blocks into a single string with ``# Code block N`` headers."""
    annotated = []
    for i, (block, metadata) in enumerate(zip(code_blocks, code_block_metadata, strict=False)):
        annotated.append(f"# Code block {i}\n{block}")
    return "\n\n".join(annotated)


def _build_log_lines(
    final_code: str,
    info_step: dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
    num_regenerations: int,
    num_finishes: int,
    num_code_blocks: int,
    *,
    prefix: str = "",
    stderr_override: str | None = None,
) -> list[str]:
    """Build the standard log-line list used for both normal and timeout summaries."""
    stderr = stderr_override if stderr_override is not None else info_step.get("stderr", "")
    lines = ["-" * 100]
    if prefix:
        lines.append(prefix)
    lines.extend([
        "Generated program:",
        final_code if final_code else "(no program available)",
        "\n\nEnvironment response:",
        f"  Sandbox failed: {info_step.get('sandbox_rc', 1)}",
        f"  Stdout: {info_step.get('stdout', '')}",
        f"  Stderr: {stderr}",
        f"  Reward: {reward}",
        f"  Task Completed: {info_step.get('task_completed', False)}",
        f"  Terminated: {terminated}, Truncated: {truncated}",
        f"  Num Regenerations: {num_regenerations}",
        f"  Num Finishes: {num_finishes}",
        f"  Num Code Blocks: {num_code_blocks}",
        "-" * 100,
    ])
    return lines


# ---------------------------------------------------------------------------
# Trial video directory helper
# ---------------------------------------------------------------------------

def _trial_video_dir(
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
) -> str:
    """Return the trial output directory path used for video saving."""
    return os.path.join(
        config["output_dir"],
        f"trial_{trial:02d}_sandboxrc_{info_step['sandbox_rc']}_reward_{reward:.3f}"
        f"_taskcompleted_{int(info_step.get('task_completed', False))}",
    )


def _save_trial_video(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    num_code_blocks: int,
    *,
    suffix_extra: str = "",
) -> None:
    """Save recorded video frames from the environment, if available."""
    if not config["record_video"] or not hasattr(env, "get_video_frames"):
        return
    frames = env.get_video_frames(clear=True)
    if not frames or not config["output_dir"]:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)
    suffix = f"{reward:.3f}"
    if suffix_extra:
        suffix += f"_{suffix_extra}"

    if isinstance(frames, list):
        _write_video(frames, base_dir, suffix=suffix)
    elif isinstance(frames, dict):
        for key, frame in frames.items():
            _write_video(frame, base_dir, suffix=f"{suffix}_{key}")


def _save_turn_and_combined_videos(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    turn_frame_ranges: list[tuple[int, int]],
) -> None:
    """Save per-turn videos and a combined video of all turns.

    Gets all frames from the environment (clearing the buffer), then writes:
      - ``video_turn_00.mp4``, ``video_turn_01.mp4``, ... for each turn
      - ``video_combined.mp4`` for the full trial
      - If wrist camera is enabled: ``video_turn_00_wrist.mp4``, etc.
    """
    if not config["record_video"] or not config["output_dir"]:
        return
    if not hasattr(env, "get_video_frames"):
        return

    all_frames = env.get_video_frames(clear=True)
    if not all_frames:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)

    # all_frames may be a list (Robosuite) or a dict of lists (R1Pro multi-camera).
    # Normalise to a list for slicing; dict case is handled by _write_multi_video.
    if isinstance(all_frames, dict):
        # Multi-camera: write each camera stream as a combined video
        for key, frames in all_frames.items():
            if frames:
                _write_video(frames, base_dir, suffix=f"combined_{key}")
        return

    # Per-turn videos
    for i, (start, end) in enumerate(turn_frame_ranges):
        turn_frames = all_frames[start:end]
        if turn_frames:
            _write_video(turn_frames, base_dir, suffix=f"turn_{i:02d}")

    # Combined video
    _write_video(all_frames, base_dir, suffix="combined")

    # Wrist camera videos
    if config.get("use_wrist_camera") and hasattr(env, "get_wrist_video_frames"):
        wrist_frames = env.get_wrist_video_frames(clear=True)
        if wrist_frames:
            for i, (start, end) in enumerate(turn_frame_ranges):
                wrist_turn = wrist_frames[start:end]
                if wrist_turn:
                    _write_video(wrist_turn, base_dir, suffix=f"turn_{i:02d}_wrist")
            _write_video(wrist_frames, base_dir, suffix="combined_wrist")


# ---------------------------------------------------------------------------
# Visual feedback and image differencing
# ---------------------------------------------------------------------------

def _capture_initial_visual_feedback(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    config: dict[str, Any],
    args: LaunchArgs,
    visual_differencing_args: ModelQueryArgs,
) -> tuple[list, list[str], str]:
    """Capture the initial environment image and optionally describe it.

    Returns:
        (visual_feedback_imgs, visual_feedback_base64_history, task_description)
    """
    visual_feedback_imgs: list = []
    visual_feedback_base64_history: list[str] = []
    task_description = ""

    use_wrist = config.get("use_wrist_camera", False)

    needs_visual = (
        (config["use_visual_feedback"] and args.model in VLM_MODELS)
        or (config["use_img_differencing"] and visual_differencing_args.model in VLM_MODELS)
        or config.get("use_video_differencing", False)
    )
    if not (needs_visual and hasattr(env, "render")):
        return visual_feedback_imgs, visual_feedback_base64_history, task_description

    initial_base64, initial_img = _get_visual_feedback(env)
    visual_feedback_imgs.append(initial_img)
    visual_feedback_base64_history.append(initial_base64)
    task_description = copy.deepcopy(obs["full_prompt"][-1]["content"][0]["text"])

    # Also capture wrist camera image for multiview initial description
    initial_wrist_base64 = None
    if use_wrist and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            initial_wrist_base64 = (
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )
            visual_feedback_imgs.append(pil_wrist)

    # Append image to the prompt for VLM visual feedback
    if config["use_visual_feedback"]:
        obs["full_prompt"][-1]["content"][0]["text"] += (
            "\n\nIncluded below is an image of the initial state of the environment."
        )
        obs["full_prompt"][-1]["content"].append(
            {"type": "image_url", "image_url": {"url": initial_base64}}
        )
        if initial_wrist_base64 is not None:
            obs["full_prompt"][-1]["content"].append(
                {
                    "type": "text",
                    "text": "Included below is an image from the robot's wrist camera.",
                }
            )
            obs["full_prompt"][-1]["content"].append(
                {"type": "image_url", "image_url": {"url": initial_wrist_base64}}
            )

    # Image differencing: ask a VLM to describe the initial scene
    if config["use_img_differencing"] or config.get("use_video_differencing", False):
        description = _describe_initial_scene(
            visual_differencing_args, task_description, initial_base64,
            wrist_image_base64=initial_wrist_base64,
        )
        feedback = f"The initial state of the environment is described as follows:\n{description}"
        obs["full_prompt"][-1]["content"][0]["text"] += f"\n\n{feedback}"
        if args.debug:
            print(description)

    return visual_feedback_imgs, visual_feedback_base64_history, task_description


def _describe_initial_scene(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    image_base64: str,
    wrist_image_base64: str | None = None,
) -> str:
    """Query a VLM to describe the initial environment state."""
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the initial state of the environment with the goal of the "
                "task in mind. You should try to provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera view:"},
        {"type": "image_url", "image_url": {"url": image_base64}},
    ]
    if wrist_image_base64 is not None:
        user_content.extend([
            {"type": "text", "text": "Wrist camera view:"},
            {"type": "image_url", "image_url": {"url": wrist_image_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the initial state of the "
                "environment with the goal of the task in mind. You should try to provide "
                "objective information and no assumptions. Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


def _get_visual_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    visual_feedback_base64_history: list[str],
    wrist_base64_history: list[str] | None = None,
) -> str | None:
    """Query a VLM to describe what changed between the two most recent frames.

    Args:
        wrist_base64_history: Optional history of wrist camera images.  When provided
            and has >=2 entries, the before/after wrist images are included in the prompt.
    """
    if len(visual_feedback_base64_history) < 2:
        return None

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the difference between the current state of the "
                "environment and the previous state of the environment with the "
                "goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code.."
            ),
        },
        {"type": "text", "text": "Previous state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-2]}},
        {"type": "text", "text": "Current state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-1]}},
    ]

    if wrist_base64_history and len(wrist_base64_history) >= 2:
        user_content.extend([
            {"type": "text", "text": "Previous state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-2]}},
            {"type": "text", "text": "Current state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-1]}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the difference between the "
                "current state of the environment and the previous state of the environment "
                "with the goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


# ---------------------------------------------------------------------------
# Video differencing
# ---------------------------------------------------------------------------

def _get_video_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    turn_frames: list[np.ndarray],
    wrist_turn_frames: list[np.ndarray] | None = None,
) -> str | None:
    """Query a VLM with a video of the turn execution to describe what happened.

    Args:
        visual_differencing_args: Model query args for the VDM model.
        task_description: The task goal.
        turn_frames: RGB frames from the main camera for this turn.
        wrist_turn_frames: RGB frames from the wrist camera for this turn (optional).

    Returns:
        Text description of the execution, or None if no frames.
    """
    if not turn_frames:
        return None

    video_base64 = _encode_video_base64(turn_frames)

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "The following video shows the robot executing code in the "
                "environment from the main camera view. Describe what happened "
                "during execution, including what actions the robot took, how "
                "the objects in the scene changed, and whether the task appears "
                "to have been completed. Provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera video:"},
        {"type": "image_url", "image_url": {"url": video_base64}},
    ]

    if wrist_turn_frames:
        wrist_video_base64 = _encode_video_base64(wrist_turn_frames)
        user_content.extend([
            {
                "type": "text",
                "text": (
                    "The following video shows the same execution from the "
                    "robot's wrist-mounted camera (eye-in-hand view), providing "
                    "a close-up perspective of the gripper and objects being "
                    "manipulated."
                ),
            },
            {"type": "text", "text": "Wrist camera video:"},
            {"type": "image_url", "image_url": {"url": wrist_video_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that analyzes robot execution "
                "videos. You describe what happened during the robot's code "
                "execution, what actions were taken, how the environment "
                "changed, and whether the task appears to have been completed. "
                "Provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


# ---------------------------------------------------------------------------
# Initial code generation
# ---------------------------------------------------------------------------

def _query_initial_code(
    args: LaunchArgs,
    config: dict[str, Any],
    obs: dict[str, Any],
    *,
    trial: int = 0,
    text_world_state_feedback: str | None = None,
) -> tuple[str, str | None, dict | None, dict[str, int] | None]:
    """Query the model for the initial code generation.

    Returns:
        (raw_code, reasoning, ensemble_data, token_usage)
    """
    # Save the initial prompt
    with open(os.path.join(config["output_dir"], "initial_prompt.txt"), "w") as f:
        f.write(str(obs["full_prompt"]))

    ensemble_data = None
    reranking = CandidateRerankingConfig.from_dict(config.get("candidate_reranking"))
    if reranking.enabled:
        if config["use_parallel_ensemble"]:
            raise ValueError("candidate_reranking and use_parallel_ensemble are mutually exclusive")
        state_payload = parse_world_state_feedback(text_world_state_feedback)
        responses = []
        for resample_round in range(reranking.max_resample_rounds + 1):
            responses.extend(
                _query_model(args, obs["full_prompt"])
                for _ in range(reranking.candidate_count)
            )
            candidates = []
            for response in responses:
                blocks = _extract_code(response.get("content"))
                candidates.append({
                    "decision": "initial",
                    "code": "\nbreakpoint_code_block()\n".join(blocks),
                    "content": response.get("content") or "",
                    "reasoning": response.get("reasoning"),
                    "token_usage": response.get("token_usage"),
                })
            selected_index, audit = select_candidate(
                candidates,
                config=reranking,
                trial=trial,
                decision_index=0,
                recovery_stage=str(state_payload.get("recovery_stage", "ACQUIRE_GRASP")),
                recovery_directive=str(
                    (state_payload.get("recovery_directive") or {}).get("action", "observe")
                ),
                task_completed=bool((state_payload.get("state") or {}).get("task_completed")),
                world_state=state_payload.get("state") or {},
                state_confidence=float(state_payload.get("confidence", 1.0)),
                state_stable=bool(state_payload.get("stable", True)),
            )
            if not audit["selection_abstained"]:
                break
        audit["resample_rounds_used"] = resample_round
        out = dict(responses[selected_index])
        if audit["selection_abstained"]:
            out["content"] = (
                "```python\nraise RuntimeError('PRM-v2 abstained: no safe candidate')\n```"
            )
            out["reasoning"] = "No hard-constraint-compliant initial candidate."
            audit["safe_abstention_action"] = "raise_before_robot_api"
        ensemble_data = {
            "candidate_reranking": audit,
            "model_response_seconds": sum(
                float(response.get("model_response_seconds", 0.0))
                for response in responses
            ),
        }
        token_usage = aggregate_token_usage(responses)
    elif config["use_parallel_ensemble"]:
        if config.get("use_multimodel", False):
            print("RUNNING MULTIMODEL ENSEMBLE QUERY")
            out = _query_model_ensemble(args, obs["full_prompt"], is_multiturn=False)
        else:
            print("RUNNING SINGLE MODEL ENSEMBLE QUERY")
            out = _query_single_model_ensemble(args, obs["full_prompt"], args.model, is_multiturn=False)
        ensemble_data = {
            "ensemble_candidates_txt": out["ensemble_candidates_txt"],
            "ensemble_synthesis_txt": out["ensemble_synthesis_txt"],
        }
    else:
        out = _query_model(args, obs["full_prompt"])
        ensemble_data = {
            "model_response_seconds": out.get("model_response_seconds", 0.0)
        }

    return (
        out["content"], out["reasoning"], ensemble_data,
        token_usage if reranking.enabled else out.get("token_usage"),
    )


# ---------------------------------------------------------------------------
# Multi-turn decision handling
# ---------------------------------------------------------------------------

def _handle_multi_turn_step(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    args: LaunchArgs,
    config: dict[str, Any],
    visual_differencing_args: ModelQueryArgs,
    multi_turn_prompt: str,
    code_blocks: list[str],
    code_block_idx: int,
    info_step: dict[str, Any],
    task_description: str,
    visual_feedback_imgs: list,
    visual_feedback_base64_history: list[str],
    stderr_history: list[str],
    text_world_state_feedback: str | None = None,
    turn_frames: list[np.ndarray] | None = None,
    wrist_turn_frames: list[np.ndarray] | None = None,
    wrist_base64_history: list[str] | None = None,
    trial: int = 0,
    structured_recovery_schema: str | None = None,
    structured_recovery_context: str | None = None,
) -> tuple[str, str | None, str | None, dict | None, list | None, dict[str, int] | None]:
    """Execute one multi-turn decision step.

    Captures visual feedback, builds the decision prompt, queries the model,
    and returns the parsed decision.

    Args:
        turn_frames: Frames from the main camera for this turn (for video differencing).
        wrist_turn_frames: Frames from the wrist camera for this turn (for video differencing).
        wrist_base64_history: History of wrist camera base64 images for image-based
            differencing with multiview.

    Returns:
        (decision, new_code, reasoning, multiturn_ensemble_entry)
        where decision is "regenerate", "finish", or "continue".
    """
    use_wrist = config.get("use_wrist_camera", False)

    executed_code = "\n".join(code_blocks[:code_block_idx])
    complete_multi_turn_prompt = multi_turn_prompt.format(
        executed_code=executed_code,
        console_stdout=info_step["stdout"],
        console_stderr=info_step["stderr"],
    )
    complete_multi_turn_prompt = append_text_world_state_feedback(
        complete_multi_turn_prompt, text_world_state_feedback
    )

    if info_step["stderr"] != "":
        stderr_history.append(info_step["stderr"])

    # Capture visual feedback if applicable
    visual_feedback_base64 = None
    needs_visual = (
        (config["use_visual_feedback"] and args.model in VLM_MODELS)
        or (config["use_img_differencing"] and visual_differencing_args.model in VLM_MODELS)
    )
    if needs_visual and hasattr(env, "render"):
        vf_base64, vf_img = _get_visual_feedback(env)
        visual_feedback_imgs.append(vf_img)
        visual_feedback_base64_history.append(vf_base64)

        # Also capture wrist camera snapshot for image-based multiview
        if use_wrist and hasattr(env, "render_wrist") and wrist_base64_history is not None:
            wrist_result = _get_visual_feedback(env, use_wrist_camera=True)
            if wrist_result[0] is not None and isinstance(wrist_result[0], list) and len(wrist_result[0]) > 1:
                wrist_base64_history.append(wrist_result[0][1])  # index 1 = wrist image

    # Determine differencing feedback
    differencing_feedback = None
    is_video_feedback = False

    if config.get("use_video_differencing") and turn_frames:
        # Video-based differencing: pass video of this turn to VDM
        differencing_feedback = _get_video_differencing_feedback(
            visual_differencing_args, task_description, turn_frames, wrist_turn_frames,
        )
        is_video_feedback = True
    elif config["use_img_differencing"] and len(visual_feedback_base64_history) >= 2:
        # Image-based differencing: pass before/after images to VDM
        differencing_feedback = _get_visual_differencing_feedback(
            visual_differencing_args, task_description, visual_feedback_base64_history,
            wrist_base64_history=wrist_base64_history,
        )

    # Only pass visual feedback to prompt if visual_feedback is enabled
    if not config["use_visual_feedback"]:
        visual_feedback_base64 = None
    elif needs_visual and hasattr(env, "render"):
        visual_feedback_base64 = visual_feedback_base64_history[-1] if visual_feedback_base64_history else None

    if structured_recovery_schema is not None:
        user_content: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                f"Task: {task_description}\n"
                f"Failure: {structured_recovery_context or 'unknown'}\n"
                f"Execution stdout: {info_step['stdout']}\n"
                f"Execution stderr: {info_step['stderr']}\n"
                f"K=3 world state:\n{text_world_state_feedback or '<unavailable>'}\n"
                f"Required object schema:\n{structured_recovery_schema}"
            ),
        }]
        if visual_feedback_base64 is not None:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": visual_feedback_base64},
            })
        decision_prompt = [
            {
                "role": "system",
                "content": (
                    "Return one trusted structured robot plan matching the supplied "
                    "schema. Return exactly one raw JSON or Python dictionary "
                    "and nothing else. Do not return REGENERATE, markdown, fenced "
                    "Python, executable code, API calls, comments, or prose."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        content = _query_model(args, decision_prompt)
        return (
            "regenerate",
            content.get("content"),
            content.get("reasoning"),
            {"model_response_seconds": content.get("model_response_seconds", 0.0)},
            decision_prompt,
            content.get("token_usage"),
        )

    # Build decision prompt
    if args.use_legacy_multi_turn_decision_prompt:
        print("Using legacy multi-turn decision prompt")
        decision_prompt = _build_multi_turn_decision_prompt_legacy(
            obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
            is_video_feedback=is_video_feedback,
        )
    else:
        decision_prompt = _build_multi_turn_decision_prompt(
            obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
            is_video_feedback=is_video_feedback,
        )

    # Query model
    multiturn_ensemble_entry = None
    reranking = CandidateRerankingConfig.from_dict(config.get("candidate_reranking"))
    if reranking.enabled:
        if config["use_parallel_ensemble"]:
            raise ValueError("candidate_reranking and use_parallel_ensemble are mutually exclusive")
        state_payload = parse_world_state_feedback(text_world_state_feedback)
        responses = []
        for resample_round in range(reranking.max_resample_rounds + 1):
            responses.extend(
                _query_model(args, decision_prompt)
                for _ in range(reranking.candidate_count)
            )
            candidates = []
            for response in responses:
                candidate_decision, candidate_code = _parse_multi_turn_decision(
                    response.get("content") or ""
                )
                blocks = _extract_code(candidate_code) if candidate_code else []
                candidates.append({
                    "decision": candidate_decision,
                    "code": "\n\n".join(blocks),
                    "content": response.get("content") or "",
                    "reasoning": response.get("reasoning"),
                    "token_usage": response.get("token_usage"),
                })
            selected_index, audit = select_candidate(
                candidates,
                config=reranking,
                trial=trial,
                decision_index=code_block_idx,
                recovery_stage=str(state_payload.get("recovery_stage", "ACQUIRE_GRASP")),
                recovery_directive=str(
                    (state_payload.get("recovery_directive") or {}).get("action", "observe")
                ),
                task_completed=bool((state_payload.get("state") or {}).get("task_completed")),
                previous_code=code_blocks[code_block_idx - 1] if code_block_idx else "",
                world_state=state_payload.get("state") or {},
                state_confidence=float(state_payload.get("confidence", 1.0)),
                state_stable=bool(state_payload.get("stable", True)),
            )
            if not audit["selection_abstained"]:
                break
        audit["resample_rounds_used"] = resample_round
        content = dict(responses[selected_index])
        if audit["selection_abstained"]:
            content["content"] = "FINISH"
            content["reasoning"] = "No hard-constraint-compliant recovery candidate."
            audit["safe_abstention_action"] = "terminate_trial_without_execution"
        multiturn_ensemble_entry = {
            "candidate_reranking": audit,
            "model_response_seconds": sum(
                float(response.get("model_response_seconds", 0.0))
                for response in responses
            ),
        }
        token_usage = aggregate_token_usage(responses)
    elif config["use_parallel_ensemble"]:
        if config.get("use_multimodel", False):
            print("RUNNING MULTITURN MULTIMODEL ENSEMBLE QUERY")
            content = _query_model_ensemble(args, decision_prompt, is_multiturn=True)
        else:
            print("RUNNING MULTITURN SINGLE MODEL ENSEMBLE QUERY")
            content = _query_single_model_ensemble(args, decision_prompt, args.model, is_multiturn=True)
        multiturn_ensemble_entry = {
            "ensemble_candidates_txt": content.get("ensemble_candidates_txt", ""),
            "ensemble_synthesis_txt": content.get("ensemble_synthesis_txt", ""),
        }
    else:
        content = _query_model(args, decision_prompt)
        multiturn_ensemble_entry = {
            "model_response_seconds": content.get("model_response_seconds", 0.0)
        }

    reasoning = content["reasoning"]
    decision, new_code = _parse_multi_turn_decision(content["content"])

    return (
        decision, new_code, reasoning, multiturn_ensemble_entry, decision_prompt,
        token_usage if reranking.enabled else content.get("token_usage"),
    )


# ---------------------------------------------------------------------------
# Core single-trial execution
# ---------------------------------------------------------------------------

def _run_single_trial(
    env: CodeExecutionEnvBase,
    trial: int,
    args: LaunchArgs,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
    partial_artifacts: dict[str, Any] | None = None,
) -> TrialSummary:
    """Execute a single trial end-to-end.

    Steps:
        1. Reset the environment.
        2. Capture initial visual feedback (if configured).
        3. Query the model for initial code generation.
        4. Execute code blocks one-by-one, with optional multi-turn regeneration.
        5. Save artifacts (code, logs, per-turn videos, combined video) and return a TrialSummary.
    """
    trial_start_time = time.time()

    offline_cfg = config.get("offline_recovery", {})
    if isinstance(offline_cfg, dict) and offline_cfg.get("enabled", False):
        return _run_offline_task_graph_trial(
            env, trial, config, offline_cfg, partial_artifacts=partial_artifacts
        )

    use_video_diff = config.get("use_video_differencing", False)
    use_wrist = config.get("use_wrist_camera", False)

    recorder_cfg_dict = config.get("world_state_recorder", {"enabled": False})
    if recorder_cfg_dict.get("enabled", False):
        from capx.recording import WorldStateRecorder, WorldStateRecorderConfig

        if not config.get("output_dir"):
            raise ValueError("world_state_recorder requires output_dir")
        recorder_cfg = WorldStateRecorderConfig.from_dict(recorder_cfg_dict)
        recorder_dir = Path(config["output_dir"]) / "world_state" / f"trial_{trial:03d}"
        env.set_world_state_recorder(WorldStateRecorder(
            recorder_dir, recorder_cfg, episode_id=f"trial_{trial:03d}"
        ))
    else:
        env.set_world_state_recorder(None)

    text_feedback_cfg = TextWorldStateFeedbackConfig.from_dict(
        config.get("text_world_state_feedback", {"enabled": False})
    )
    execution_guard_cfg = ExecutionGuardConfig.from_dict(
        config.get("execution_guard", {"enabled": False})
    )
    hybrid_cfg = config.get("hybrid_recovery", {})
    hybrid_graph: OfflineTaskGraph | None = None
    hybrid_library: RecoveryLibrary | None = None
    hybrid_stage_actions: tuple[str, ...] = ()
    hybrid_policy_first = False
    hybrid_model_led_policy_recovery = False
    hybrid_policy_preconditions: dict[str, Any] = {}
    hybrid_model_fallback_limit = 0
    hybrid_model_fallback_total_limit = 0
    hybrid_model_recovery_stage_indices: tuple[int, ...] = ()
    hybrid_model_recovery_template: str | None = None
    hybrid_model_initial_plan_template: str | None = None
    hybrid_model_recovery_candidate_limit = 1
    hybrid_min_remaining_steps_before_regrasp = 0
    hybrid_l2_enabled = False
    hybrid_l2_max_requests = 0
    hybrid_l2_review_after_actions: tuple[str, ...] = ()
    hybrid_l2_final_review = False
    if isinstance(hybrid_cfg, dict) and hybrid_cfg.get("enabled", False):
        if not execution_guard_cfg.enabled or not text_feedback_cfg.enabled:
            raise ValueError(
                "hybrid_recovery requires execution_guard and text_world_state_feedback"
            )
        hybrid_graph = OfflineTaskGraph.from_mapping(
            _load_offline_value(hybrid_cfg.get("task_graph", {}))
        )
        action_library = hybrid_cfg.get("action_library")
        if action_library:
            hybrid_graph = attach_action_library(
                hybrid_graph, _load_offline_value(action_library)
            )
        hybrid_library = RecoveryLibrary.from_mapping(
            _load_offline_value(hybrid_cfg.get("recovery_library", {}))
        )
        hybrid_stage_actions = tuple(
            str(item) for item in hybrid_cfg.get("stage_actions", ())
        )
        if not hybrid_stage_actions:
            raise ValueError("hybrid_recovery requires stage_actions")
        hybrid_policy_first = bool(hybrid_cfg.get("policy_first", False))
        hybrid_model_led_policy_recovery = bool(
            hybrid_cfg.get("model_led_policy_recovery", False)
        )
        if hybrid_model_led_policy_recovery and hybrid_policy_first:
            raise ValueError(
                "model_led_policy_recovery requires policy_first=false"
            )
        hybrid_policy_preconditions = dict(
            hybrid_cfg.get("policy_preconditions", {})
        )
        hybrid_model_fallback_limit = int(
            hybrid_cfg.get("max_model_fallbacks_per_exception", 0)
        )
        if hybrid_model_fallback_limit not in {0, 1, 2}:
            raise ValueError(
                "max_model_fallbacks_per_exception must be 0, 1, or 2"
            )
        hybrid_model_fallback_total_limit = int(
            hybrid_cfg.get("max_model_fallbacks_total", 0)
        )
        if hybrid_model_fallback_total_limit < 0:
            raise ValueError("max_model_fallbacks_total must be non-negative")
        if hybrid_model_fallback_limit or hybrid_model_fallback_total_limit:
            safe_stop_actions = hybrid_library.lookup("RECOVERY_BUDGET_LOW") or ()
            if not safe_stop_actions or any(
                action_id not in hybrid_graph.actions for action_id in safe_stop_actions
            ):
                raise ValueError(
                    "model fallback budget requires a valid RECOVERY_BUDGET_LOW route"
                )
        hybrid_model_recovery_stage_indices = tuple(
            int(index)
            for index in hybrid_cfg.get("model_recovery_stage_indices", ())
        )
        if any(index < 0 for index in hybrid_model_recovery_stage_indices):
            raise ValueError("model_recovery_stage_indices must be non-negative")
        raw_model_recovery_template = hybrid_cfg.get("model_recovery_template")
        if raw_model_recovery_template is not None:
            hybrid_model_recovery_template = str(raw_model_recovery_template)
            if hybrid_model_recovery_template not in {
                "bin_place_regrasp_replace_v1", "recovery_graph_v1"
            }:
                raise ValueError("unsupported model_recovery_template")
        raw_model_initial_plan_template = hybrid_cfg.get(
            "model_initial_plan_template"
        )
        if raw_model_initial_plan_template is not None:
            hybrid_model_initial_plan_template = str(
                raw_model_initial_plan_template
            )
            if hybrid_model_initial_plan_template != "recovery_graph_v1":
                raise ValueError("unsupported model_initial_plan_template")
            if hybrid_policy_first:
                raise ValueError(
                    "model_initial_plan_template requires policy_first=false"
                )
        hybrid_model_recovery_candidate_limit = int(
            hybrid_cfg.get("model_recovery_candidate_limit", 1)
        )
        if hybrid_model_recovery_candidate_limit not in {1, 2}:
            raise ValueError("model_recovery_candidate_limit must be 1 or 2")
        if (
            hybrid_model_recovery_candidate_limit > 1
            and hybrid_model_recovery_template is None
        ):
            raise ValueError(
                "multiple model recovery candidates require model_recovery_template"
            )
        hybrid_min_remaining_steps_before_regrasp = int(
            hybrid_cfg.get("min_remaining_sim_steps_before_regrasp", 0)
        )
        if hybrid_min_remaining_steps_before_regrasp < 0:
            raise ValueError(
                "min_remaining_sim_steps_before_regrasp must be non-negative"
            )
        l2_cfg = hybrid_cfg.get("l2_model_participation", {})
        if isinstance(l2_cfg, dict) and l2_cfg.get("enabled", False):
            if not hybrid_policy_first:
                raise ValueError("L2 model participation requires policy_first")
            hybrid_l2_enabled = True
            hybrid_l2_max_requests = int(l2_cfg.get("max_requests", 3))
            if hybrid_l2_max_requests not in {2, 3, 4}:
                raise ValueError("L2 max_requests must be 2, 3, or 4")
            hybrid_l2_review_after_actions = tuple(
                str(item) for item in l2_cfg.get("review_after_actions", ())
            )
            hybrid_l2_final_review = bool(l2_cfg.get("final_review", True))
    text_feedback: TrialWorldStateFeedback | None = None

    # --- 1. Reset environment ---
    obs, _ = env.reset(options={"trial": trial}, seed=trial)
    if config.get("output_dir"):
        save_initial_state_fingerprint(env, config["output_dir"], trial)
    initial_text_feedback_prompt = None
    if text_feedback_cfg.enabled:
        text_feedback = TrialWorldStateFeedback(env, text_feedback_cfg)
        initial_text_feedback_prompt = text_feedback.prompt_text(text_feedback.capture())
    # Reset the SIGALRM timer AFTER env.reset() so the timeout only covers
    # actual task execution, not scene loading / cuRobo JIT compilation.
    import signal
    remaining = signal.alarm(0)  # cancel current alarm
    if remaining > 0:
        signal.alarm(1000)  # restart fresh 1000s from now
    obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])
    _patch_libero_goal(env, obs)

    if config["record_video"] and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)
    elif use_video_diff and hasattr(env, "enable_video_capture"):
        # Video differencing needs frame recording even without record_video
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)

    # --- Shared trial state ---
    code_blocks: list[str] = []
    code_block_metadata: list[dict[str, Any]] = []
    all_responses: list[dict[str, Any]] = []
    stderr_history: list[str] = []
    num_regenerations = 0
    num_finishes = 0
    info_step: dict[str, Any] = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    sandbox_rc_override = None
    ensemble_data = None
    multiturn_ensemble_data: list[dict[str, Any]] = []
    hybrid_model_fallback_counts: dict[str, int] = {}
    hybrid_model_fallback_total = 0
    active_hybrid_exception: str | None = None
    active_hybrid_role: str | None = None
    hybrid_l2_requests = 0
    max_model_requests = int(config.get("max_model_requests_per_trial", 0) or 0)
    if max_model_requests < 0:
        raise ValueError("max_model_requests_per_trial must be non-negative")

    # Per-turn frame tracking (for video differencing and per-turn video saving)
    turn_frame_ranges: list[tuple[int, int]] = []

    # Wrist camera base64 history for image-based multiview differencing
    wrist_base64_history: list[str] | None = [] if use_wrist else None

    visual_differencing_args = ModelQueryArgs(
        model=args.visual_differencing_model,
        server_url=args.visual_differencing_model_server_url,
        api_key=args.visual_differencing_model_api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        debug=args.debug,
    )

    if config["use_img_differencing"] or use_video_diff:
        assert visual_differencing_args.model in VLM_MODELS, (
            "Image/video differencing model must be in the list of VLM models"
        )

    # --- 2. Capture initial visual feedback ---
    visual_feedback_imgs, visual_feedback_base64_history, task_description = (
        _capture_initial_visual_feedback(env, obs, config, args, visual_differencing_args)
    )

    selected_hybrid_stage_actions = hybrid_stage_actions
    if hybrid_l2_enabled:
        selection, response, l2_prompt = _query_l2_advisory(
            args,
            phase="initial",
            task_description=task_description,
            world_state_feedback=initial_text_feedback_prompt or "unavailable",
        )
        l2_request_count = int(response.get("model_request_count", 1))
        hybrid_l2_requests += l2_request_count
        all_responses.append({
            "decision": "l2_initial_review",
            "selection": selection,
            "model_request_count": l2_request_count,
            "model_response_seconds": response.get("model_response_seconds", 0.0),
            "token_usage": response.get("token_usage"),
            "l2_attempts": response.get("l2_attempts"),
            "l2_schema_fallback": response.get("l2_schema_fallback", False),
            "l2_parse_errors": response.get("l2_parse_errors", []),
            "l2_prompt": l2_prompt if config.get("save_multiturn_prompts", False) else None,
        })
        if selection["decision"] == "SAFE_STOP":
            if hybrid_library is None:
                raise ValueError("L2 safe stop requires a recovery library")
            selected_hybrid_stage_actions = tuple(
                hybrid_library.lookup("RECOVERY_BUDGET_LOW") or ()
            )
            if not selected_hybrid_stage_actions:
                raise ValueError("L2 safe stop route is unavailable")

    # Seed wrist base64 history with initial wrist image
    if use_wrist and wrist_base64_history is not None and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            wrist_base64_history.append(
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )

    # --- 3. Initial code generation ---
    initial_plan_selection: dict[str, Any] | None = None
    initial_plan_raw_selection: str | None = None
    if hybrid_policy_first:
        if hybrid_graph is None:
            raise ValueError("hybrid policy_first requires a task graph")
        raw_code = "\nbreakpoint_code_block()\n".join(
            hybrid_graph.actions[action_id].code
            for action_id in selected_hybrid_stage_actions
        )
        reasoning = None
        ensemble_data = None
        initial_token_usage = None
        with open(os.path.join(config["output_dir"], "offline_policy_code.py"), "w") as f:
            f.write(raw_code)
    elif config["use_oracle_code"]:
        raw_code = env.oracle_code
        with open(os.path.join(config["output_dir"], "oracle_code.py"), "w") as f:
            f.write(raw_code)
        reasoning = None
        ensemble_data = None
        initial_token_usage = None
    else:
        raw_code, reasoning, ensemble_data, initial_token_usage = _query_initial_code(
            args, config, obs, trial=trial,
            text_world_state_feedback=initial_text_feedback_prompt,
        )
        if hybrid_model_initial_plan_template:
            initial_plan_raw_selection = raw_code
            initial_blocks, initial_plan_selection = (
                _compile_model_recovery_selection(
                    hybrid_model_initial_plan_template,
                    raw_code,
                    1,
                )
            )
            raw_code = "\nbreakpoint_code_block()\n".join(initial_blocks)
    env.record_world_state_event(
        "llm_response",
        metadata={
            "trial": trial,
            "used_oracle_code": bool(config["use_oracle_code"]),
            "plan_source": (
                "offline_policy"
                if hybrid_policy_first
                else "oracle" if config["use_oracle_code"] else "model"
            ),
        },
    )

    # Initialize partial artifacts for timeout recovery
    if partial_artifacts is not None:
        partial_artifacts.update({
            "raw_code": raw_code,
            "code_blocks": code_blocks,
            "code_block_metadata": code_block_metadata,
            "all_responses": all_responses,
            "visual_feedback_imgs": visual_feedback_imgs,
            "info_step": info_step,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "num_regenerations": num_regenerations,
            "num_finishes": num_finishes,
            "num_code_blocks": 0,
            "ensemble_data": ensemble_data,
            "multiturn_ensemble_data": multiturn_ensemble_data,
        })

    # Parse initial code into blocks
    initial_blocks = _extract_code(raw_code)
    code_blocks.extend(initial_blocks)
    if hybrid_policy_first:
        if len(initial_blocks) != len(selected_hybrid_stage_actions):
            raise ValueError("hybrid policy_first requires one code block per stage action")
        code_block_metadata.extend({
            "generation": 0,
            "regenerated": False,
            "stage_index": index,
            "hybrid_policy": True,
            "hybrid_nominal": selected_hybrid_stage_actions == hybrid_stage_actions,
            "hybrid_terminal_safe_stop": (
                selected_hybrid_stage_actions != hybrid_stage_actions
                and index == len(selected_hybrid_stage_actions) - 1
            ),
            "offline_action_id": action_id,
        } for index, action_id in enumerate(selected_hybrid_stage_actions))
    else:
        initial_metadata = assign_stage_indices(len(initial_blocks))
        if hybrid_model_led_policy_recovery:
            _attach_hybrid_stage_monitors(
                initial_metadata, hybrid_stage_actions
            )
        if initial_plan_selection is not None:
            for metadata in initial_metadata:
                metadata["hybrid_structured_plan"] = True
        code_block_metadata.extend(initial_metadata)
    plan_source = (
        "offline_policy"
        if hybrid_policy_first
        else "oracle" if config["use_oracle_code"] else "model"
    )
    all_responses.append({
        "block_idx": [0],
        "code_blocks": initial_blocks,
        "decision": "initial",
        "plan_source": plan_source,
        "model_request_count": 0 if plan_source != "model" else 1,
        "initial_prompt": copy.deepcopy(obs["full_prompt"]),
        "reasoning": reasoning if reasoning is not None else "",
        "token_usage": initial_token_usage,
        "model_initial_raw_selection": initial_plan_raw_selection,
        "model_initial_plan": initial_plan_selection,
        **_selection_log_fields(ensemble_data),
    })

    with open(os.path.join(config["output_dir"], "all_responses.json"), "w") as f:
        json.dump(all_responses, f)

    if args.debug:
        with open(os.path.join(config["output_dir"], "code_init.txt"), "w") as f:
            f.write("\n".join(initial_blocks))

    # --- 4. Execute code blocks (with optional multi-turn) ---
    info_step = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    code_block_idx = 0

    # Track whether we're recording frames (for video diff or record_video)
    recording_frames = (
        (config["record_video"] or use_video_diff)
        and hasattr(env, "get_video_frame_count")
    )

    while code_block_idx < len(code_blocks) and (
        code_block_idx <= MULTITURN_LIMIT
        or code_block_metadata[code_block_idx].get("hybrid_policy", False)
        or code_block_metadata[code_block_idx].get(
            "hybrid_model_recovery", False
        )
    ):
        block_position = code_block_idx
        code = code_blocks[code_block_idx]
        block_metadata = code_block_metadata[code_block_idx]
        stage_index = int(block_metadata.get("stage_index", code_block_idx))
        code_block_idx += 1
        env.record_world_state_event(
            "code_execution_start", metadata={"code_block_index": code_block_idx - 1}
        )

        # Record frame index before step
        frame_start = env.get_video_frame_count() if recording_frames else 0

        policy_action_id = block_metadata.get("offline_action_id")
        if policy_action_id:
            active_hybrid_role = _recovery_role_for_failed_action(
                str(policy_action_id), {}
            )
        monitor_action_id = block_metadata.get("hybrid_monitor_action_id")
        policy_parameters: dict[str, Any] = {}
        policy_action = (
            hybrid_graph.actions.get(str(policy_action_id))
            if policy_action_id and hybrid_graph is not None
            else None
        )
        monitor_action = (
            hybrid_graph.actions.get(str(monitor_action_id))
            if monitor_action_id and hybrid_graph is not None
            else None
        )
        policy_failure_next_action: str | None = None
        is_regrasp = bool(
            (
                block_metadata.get("hybrid_model_recovery", False)
                and block_metadata.get("model_recovery_block_index") == 0
            )
            or (
                policy_action is not None
                and any(
                    marker in policy_action.primitive.lower()
                    for marker in ("grasp", "acquire")
                )
            )
        )
        remaining_sim_steps = _remaining_sim_steps(env)
        if (
            is_regrasp
            and hybrid_min_remaining_steps_before_regrasp
            and remaining_sim_steps is not None
            and remaining_sim_steps < hybrid_min_remaining_steps_before_regrasp
        ):
            message = (
                "Recovery safe-stopped before regrasp: "
                f"{remaining_sim_steps} simulator steps remain, fewer than "
                f"the required {hybrid_min_remaining_steps_before_regrasp}."
            )
            info_step["stderr"] = "\n".join(
                part for part in (info_step.get("stderr", ""), message) if part
            )
            all_responses.append({
                "decision": "regrasp_horizon_safe_stop",
                "offline_action_id": policy_action_id,
                "remaining_sim_steps": remaining_sim_steps,
                "required_sim_steps": hybrid_min_remaining_steps_before_regrasp,
                "model_request_count": 0,
            })
            env.record_world_state_event(
                "regrasp_horizon_safe_stop",
                metadata={
                    "offline_action_id": policy_action_id,
                    "remaining_sim_steps": remaining_sim_steps,
                    "required_sim_steps": hybrid_min_remaining_steps_before_regrasp,
                },
            )
            truncated = True
            break
        try:
            if policy_action_id and hybrid_graph is not None:
                policy_action = hybrid_graph.actions[str(policy_action_id)]
                policy_state = capture_text_world_state(env, include_entities=False)
                global_preconditions = (
                    {}
                    if (
                        policy_action.metadata.get("terminal") == "safe_stop"
                        or policy_action.metadata.get("observation_refresh", False)
                    )
                    else hybrid_policy_preconditions
                )
                policy_ok, policy_failures = _check_conditions(
                    policy_state,
                    {**global_preconditions, **policy_action.preconditions},
                )
                if not policy_ok:
                    raise StateParameterError(
                        "offline policy preconditions failed: "
                        + ", ".join(policy_failures)
                    )
                code, policy_parameters = bind_state_parameters(
                    policy_action, policy_state,
                )
                code_blocks[block_position] = code
        except (KeyError, StateParameterError) as exc:
            preflight = GuardResult(False, "state_parameter_binding", str(exc))
        else:
            preflight = validate_code_block(code, execution_guard_cfg)
        if policy_action is not None:
            next_action_key = (
                "precondition_failure_next_action"
                if not preflight.ok
                else "failure_next_action"
            )
            configured_next_action = policy_action.metadata.get(next_action_key)
            if configured_next_action:
                policy_failure_next_action = str(configured_next_action)
        if preflight.ok:
            obs_next, reward, terminated, truncated, info_step = env.step(code)
        else:
            reward = float(env.compute_reward())
            task_completed = _current_task_completed(env)
            terminated = bool(task_completed is True or reward == 1.0)
            truncated = False
            obs_next = obs
            info_step = {
                "sandbox_rc": 1,
                "stdout": "",
                "stderr": f"Static execution gate rejected code: {preflight.reason}",
                "task_completed": task_completed,
            }
            all_responses.append({
                "decision": "code_preflight_rejected",
                "block_idx": [code_block_idx - 1],
                "code_blocks": [code],
                "execution_guard": {
                    "gate": preflight.gate,
                    "reason": preflight.reason,
                    "details": preflight.details,
                },
                "offline_action_id": policy_action_id,
                "policy_parameters": policy_parameters,
            })
            env.record_world_state_event(
                "code_execution_rejected",
                metadata={
                    "code_block_index": code_block_idx - 1,
                    "gate": preflight.gate,
                    "reason": preflight.reason,
                },
            )
        text_feedback_prompt = None
        text_summary = None
        latest_world_state: dict[str, Any] = {
            "task_completed": info_step.get("task_completed")
        }
        if text_feedback is not None:
            # A long robot program can consume the simulator horizon.  The
            # action-aligned sampler advances the simulator between frames, so
            # do not ask it to step an episode that has already terminated or
            # truncated.  A static capture still preserves the final textual
            # state for metrics and diagnostics.
            if terminated or truncated:
                text_summary = text_feedback.capture_static_window()
            else:
                text_summary = text_feedback.capture_action_window()
                if text_feedback.should_wait:
                    text_summary = text_feedback.capture_action_window()
            text_feedback_prompt = text_feedback.prompt_text(text_summary)
            if text_summary is not None:
                latest_world_state = dict(text_summary.state)
            # Simulator stepping during K=3 sampling can cross the completion
            # threshold after env.step() produced its info snapshot.
            if _current_task_completed(env) is True:
                info_step["task_completed"] = True
                if latest_world_state.get("uses_simulator_truth") is not False:
                    latest_world_state["task_completed"] = True
        stage_result = validate_stage(
            stage_index, latest_world_state, execution_guard_cfg
        )
        postcondition_action = (
            policy_action
            if block_metadata.get("hybrid_policy", False)
            else monitor_action
        )
        if postcondition_action is not None and postcondition_action.success_conditions:
            policy_ok, policy_failures = _check_conditions(
                latest_world_state, postcondition_action.success_conditions
            )
            stage_result = GuardResult(
                policy_ok,
                (
                    "policy_postcondition"
                    if block_metadata.get("hybrid_policy", False)
                    else "model_stage_postcondition"
                ),
                (
                    f"{postcondition_action.action_id} passed"
                    if policy_ok
                    else f"{postcondition_action.action_id} failed: "
                    + "; ".join(policy_failures)
                ),
                {
                    "offline_action_id": postcondition_action.action_id,
                    "required": postcondition_action.success_conditions,
                },
            )
        if execution_guard_cfg.enabled and execution_guard_cfg.stage_expectations:
            stage_feedback = (
                "HARD STAGE POST-CONDITION: "
                f"{'PASS' if stage_result.ok else 'FAIL'} — {stage_result.reason}"
            )
            text_feedback_prompt = append_text_world_state_feedback(
                text_feedback_prompt or "", stage_feedback
            )
        env.record_world_state_event(
            "code_execution_end",
            metadata={
                "code_block_index": code_block_idx - 1,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "sandbox_rc": info_step.get("sandbox_rc"),
                "task_completed": info_step.get("task_completed"),
                "preflight_ok": preflight.ok,
                "stage_gate_ok": stage_result.ok,
                "stage_index": stage_index,
            },
        )

        # Record frame index after step
        frame_end = env.get_video_frame_count() if recording_frames else 0
        turn_frame_ranges.append((frame_start, frame_end))

        if partial_artifacts is not None:
            partial_artifacts.update({
                "info_step": info_step,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            })

        obs = obs_next

        completion_from_environment = info_step.get("task_completed") is True
        completion_confirmed = task_completion_confirmed(
            task_completed=info_step.get("task_completed"),
            state_summary=text_summary,
        )
        if completion_confirmed:
            info_step["task_completed"] = True
            env.record_world_state_event(
                "deterministic_completion",
                metadata={
                    "code_block_index": code_block_idx - 1,
                    "source": (
                        "environment"
                        if completion_from_environment
                        else "stable_state_window"
                    ),
                    "sample_count": getattr(text_summary, "sample_count", 0),
                },
            )
            if (
                hybrid_l2_enabled
                and hybrid_l2_final_review
                and hybrid_l2_requests < hybrid_l2_max_requests
            ):
                selection, response, l2_prompt = _query_l2_advisory(
                    args,
                    phase="final",
                    task_description=task_description,
                    world_state_feedback=text_feedback_prompt or "unavailable",
                )
                l2_request_count = int(response.get("model_request_count", 1))
                hybrid_l2_requests += l2_request_count
                all_responses.append({
                    "decision": "l2_final_review",
                    "selection": selection,
                    "model_request_count": l2_request_count,
                    "model_response_seconds": response.get(
                        "model_response_seconds", 0.0
                    ),
                    "token_usage": response.get("token_usage"),
                    "l2_attempts": response.get("l2_attempts"),
                    "l2_schema_fallback": response.get(
                        "l2_schema_fallback", False
                    ),
                    "l2_parse_errors": response.get("l2_parse_errors", []),
                    "l2_prompt": (
                        l2_prompt
                        if config.get("save_multiturn_prompts", False)
                        else None
                    ),
                })
                env.record_world_state_event(
                    "l2_final_review", metadata={"selection": selection}
                )
                if (
                    selection["decision"] == "REQUEST_RECHECK"
                    and text_feedback is not None
                    and not (terminated or truncated)
                ):
                    text_summary = text_feedback.capture_action_window()
                    text_feedback_prompt = text_feedback.prompt_text(text_summary)
                    completion_confirmed = task_completion_confirmed(
                        task_completed=_current_task_completed(env),
                        state_summary=text_summary,
                    )
                    info_step["task_completed"] = completion_confirmed
            if completion_confirmed:
                break

        # Never dispatch another physical block after the simulator has ended.
        if terminated or truncated:
            break

        if block_metadata.get("hybrid_terminal_safe_stop", False):
            safe_stop_completed = int(info_step.get("sandbox_rc", 1)) == 0
            all_responses.append({
                "decision": "model_fallback_budget_safe_stop",
                "exception_id": block_metadata.get("hybrid_exception_id"),
                "offline_action_id": policy_action_id,
                "safe_stop_completed": safe_stop_completed,
                "model_request_count": 0,
            })
            env.record_world_state_event(
                "model_fallback_budget_safe_stop",
                metadata={
                    "exception_id": block_metadata.get("hybrid_exception_id"),
                    "offline_action_id": policy_action_id,
                    "safe_stop_completed": safe_stop_completed,
                },
            )
            truncated = True
            break

        if (
            hybrid_l2_enabled
            and stage_result.ok
            and str(policy_action_id) in hybrid_l2_review_after_actions
            and hybrid_l2_requests < hybrid_l2_max_requests
        ):
            selection, response, l2_prompt = _query_l2_advisory(
                args,
                phase="stage",
                task_description=task_description,
                world_state_feedback=text_feedback_prompt or "unavailable",
            )
            l2_request_count = int(response.get("model_request_count", 1))
            hybrid_l2_requests += l2_request_count
            all_responses.append({
                "decision": "l2_stage_review",
                "offline_action_id": policy_action_id,
                "selection": selection,
                "model_request_count": l2_request_count,
                "model_response_seconds": response.get(
                    "model_response_seconds", 0.0
                ),
                "token_usage": response.get("token_usage"),
                "l2_attempts": response.get("l2_attempts"),
                "l2_schema_fallback": response.get("l2_schema_fallback", False),
                "l2_parse_errors": response.get("l2_parse_errors", []),
                "l2_prompt": (
                    l2_prompt
                    if config.get("save_multiturn_prompts", False)
                    else None
                ),
            })
            env.record_world_state_event(
                "l2_stage_review",
                metadata={
                    "offline_action_id": policy_action_id,
                    "selection": selection,
                },
            )
            if selection["decision"] == "SAFE_STOP":
                safe_actions = hybrid_library.lookup("RECOVERY_BUDGET_LOW") or ()
                del code_blocks[code_block_idx:]
                del code_block_metadata[code_block_idx:]
                for offset, action_id in enumerate(safe_actions):
                    code_blocks.append(hybrid_graph.actions[action_id].code)
                    code_block_metadata.append({
                        "generation": num_regenerations,
                        "regenerated": True,
                        "stage_index": stage_index + offset,
                        "hybrid_policy": True,
                        "offline_action_id": action_id,
                        "hybrid_terminal_safe_stop": offset == len(safe_actions) - 1,
                        "hybrid_exception_id": "L2_SAFE_STOP",
                    })
                continue

        hybrid_dispatched = False
        hybrid_auto_continue = False
        hybrid_fallback_pending = False
        if (
            hybrid_graph is not None
            and hybrid_library is not None
            and not stage_result.ok
            and not block_metadata.get("hybrid_model_recovery", False)
            and policy_failure_next_action is None
            and (
                not block_metadata.get("hybrid_policy", False)
                or block_metadata.get("hybrid_nominal", False)
            )
        ):
            exception_id, policy_actions = select_hybrid_recovery(
                hybrid_graph,
                hybrid_library,
                hybrid_stage_actions,
                stage_index,
                sandbox_rc=int(info_step.get("sandbox_rc", 1)),
                state=latest_world_state,
            )
            active_hybrid_exception = exception_id or active_hybrid_exception
            policy_gate_ok, policy_gate_failures = _check_conditions(
                latest_world_state, hybrid_policy_preconditions
            )
            first_policy_action = (
                hybrid_graph.actions.get(str(policy_actions[0]))
                if policy_actions
                else None
            )
            refreshes_observation = bool(
                first_policy_action
                and first_policy_action.metadata.get("observation_refresh", False)
            )
            if not policy_gate_ok and not refreshes_observation:
                policy_actions = ()
            if policy_actions:
                del code_blocks[code_block_idx:]
                del code_block_metadata[code_block_idx:]
                recovery_stage_indices = (
                    hybrid_model_recovery_stage_indices
                    if len(hybrid_model_recovery_stage_indices) == len(policy_actions)
                    else tuple(stage_index + offset for offset in range(len(policy_actions)))
                )
                for offset, action_id in enumerate(policy_actions):
                    code_blocks.append(hybrid_graph.actions[action_id].code)
                    code_block_metadata.append({
                        "generation": num_regenerations,
                        "regenerated": True,
                        "stage_index": recovery_stage_indices[offset],
                        "hybrid_policy": True,
                        "offline_action_id": action_id,
                    })
                all_responses.append({
                    "decision": "policy_recovery",
                    "exception_id": exception_id,
                    "policy_actions": list(policy_actions),
                    "block_idx": list(range(code_block_idx, len(code_blocks))),
                    "model_request_count": 0,
                })
                env.record_world_state_event(
                    "hybrid_policy_selected",
                    metadata={
                        "exception_id": exception_id,
                        "policy_actions": list(policy_actions),
                        "stage_index": stage_index,
                    },
                )
                hybrid_dispatched = True
            else:
                all_responses.append({
                    "decision": "policy_miss",
                    "exception_id": exception_id,
                    "stage_index": stage_index,
                    "fallback_to_model": bool(multi_turn_prompt),
                    "policy_precondition_failures": list(policy_gate_failures),
                })
                hybrid_fallback_pending = True
        elif (
            block_metadata.get("hybrid_policy", False)
            and not stage_result.ok
            and policy_failure_next_action == "recover_budget_safe_stop"
            and (hybrid_model_fallback_limit or hybrid_model_fallback_total_limit)
            and bool(multi_turn_prompt)
        ):
            if code_block_idx < len(code_blocks):
                del code_blocks[code_block_idx:]
                del code_block_metadata[code_block_idx:]
            all_responses.append({
                "decision": "policy_failed",
                "offline_action_id": policy_action_id,
                "stage_index": stage_index,
                "reason": stage_result.reason,
                "fallback_to_model": True,
                "safe_stop_action_id": policy_failure_next_action,
            })
            env.record_world_state_event(
                "hybrid_policy_failed",
                metadata={
                    "offline_action_id": policy_action_id,
                    "stage_index": stage_index,
                    "reason": stage_result.reason,
                    "fallback_to_model": True,
                },
            )
            hybrid_fallback_pending = True
        elif (
            block_metadata.get("hybrid_policy", False)
            and not stage_result.ok
            and policy_failure_next_action is not None
            and policy_failure_next_action in hybrid_graph.actions
        ):
            retry_action_id = policy_failure_next_action
            del code_blocks[code_block_idx:]
            del code_block_metadata[code_block_idx:]
            code_blocks.append(hybrid_graph.actions[retry_action_id].code)
            code_block_metadata.append({
                "generation": num_regenerations,
                "regenerated": True,
                "stage_index": stage_index,
                "hybrid_policy": True,
                "offline_action_id": retry_action_id,
                "hybrid_terminal_safe_stop": bool(
                    hybrid_graph.actions[retry_action_id].metadata.get("terminal")
                    == "safe_stop"
                ),
            })
            hybrid_auto_continue = True
            all_responses.append({
                "decision": (
                    "policy_precondition_safe_stop"
                    if not preflight.ok
                    else "policy_bounded_retry"
                ),
                "offline_action_id": policy_action_id,
                "next_offline_action_id": retry_action_id,
                "model_request_count": 0,
            })
            env.record_world_state_event(
                "hybrid_policy_bounded_retry",
                metadata={
                    "offline_action_id": policy_action_id,
                    "next_offline_action_id": retry_action_id,
                    "stage_index": stage_index,
                },
            )
        elif (
            block_metadata.get("hybrid_policy", False)
            and not stage_result.ok
            and code_block_idx < len(code_blocks)
            and code_block_metadata[code_block_idx].get("hybrid_policy", False)
            and not bool(
                policy_action
                and policy_action.metadata.get("interrupt_on_failure", False)
            )
        ):
            hybrid_auto_continue = True
            all_responses.append({
                "decision": "policy_retry",
                "offline_action_id": policy_action_id,
                "next_offline_action_id": code_block_metadata[code_block_idx].get(
                    "offline_action_id"
                ),
                "model_request_count": 0,
            })
        elif block_metadata.get("hybrid_policy", False) and not stage_result.ok:
            if code_block_idx < len(code_blocks):
                del code_blocks[code_block_idx:]
                del code_block_metadata[code_block_idx:]
            all_responses.append({
                "decision": "policy_failed",
                "offline_action_id": policy_action_id,
                "stage_index": stage_index,
                "reason": stage_result.reason,
                "fallback_to_model": bool(multi_turn_prompt),
            })
            env.record_world_state_event(
                "hybrid_policy_failed",
                metadata={
                    "offline_action_id": policy_action_id,
                    "stage_index": stage_index,
                    "reason": stage_result.reason,
                },
            )
            hybrid_fallback_pending = True
        elif (
            block_metadata.get("hybrid_policy", False)
            and stage_result.ok
            and policy_action is not None
            and policy_action.next_action
            and code_block_idx >= len(code_blocks)
        ):
            next_action_id = str(policy_action.next_action)
            next_action = hybrid_graph.actions.get(next_action_id)
            if next_action is not None:
                code_blocks.append(next_action.code)
                code_block_metadata.append({
                    "generation": num_regenerations,
                    "regenerated": True,
                    "stage_index": stage_index,
                    "hybrid_policy": True,
                    "offline_action_id": next_action_id,
                })
                hybrid_auto_continue = True
                all_responses.append({
                    "decision": "policy_continue",
                    "offline_action_id": policy_action_id,
                    "next_offline_action_id": next_action_id,
                    "model_request_count": 0,
                })
                env.record_world_state_event(
                    "hybrid_policy_continued",
                    metadata={
                        "offline_action_id": policy_action_id,
                        "next_offline_action_id": next_action_id,
                    },
                )
        elif (
            block_metadata.get("hybrid_structured_plan", False)
            and stage_result.ok
            and code_block_idx < len(code_blocks)
            and code_block_metadata[code_block_idx].get(
                "hybrid_structured_plan", False
            )
        ):
            hybrid_auto_continue = True
            all_responses.append({
                "decision": "structured_plan_continue",
                "completed_stage_index": stage_index,
                "next_stage_index": code_block_metadata[code_block_idx].get(
                    "stage_index"
                ),
                "model_request_count": 0,
            })
            env.record_world_state_event(
                "structured_plan_continue",
                metadata={
                    "completed_stage_index": stage_index,
                    "next_stage_index": code_block_metadata[code_block_idx].get(
                        "stage_index"
                    ),
                },
            )
        elif (
            block_metadata.get("hybrid_model_recovery", False)
            and not stage_result.ok
            and int(block_metadata.get("model_recovery_candidate_index", 0))
            + 1 < hybrid_model_recovery_candidate_limit
        ):
            candidate_index = int(
                block_metadata.get("model_recovery_candidate_index", 0)
            )
            while (
                code_block_idx < len(code_blocks)
                and int(code_block_metadata[code_block_idx].get(
                    "model_recovery_candidate_index", candidate_index
                )) == candidate_index
            ):
                del code_blocks[code_block_idx]
                del code_block_metadata[code_block_idx]
            hybrid_auto_continue = code_block_idx < len(code_blocks)
            all_responses.append({
                "decision": "model_recovery_candidate_retry",
                "completed_candidate_index": candidate_index,
                "next_candidate_index": candidate_index + 1,
                "reason": stage_result.reason,
                "model_request_count": 0,
            })
            env.record_world_state_event(
                "model_recovery_candidate_retry",
                metadata={
                    "completed_candidate_index": candidate_index,
                    "next_candidate_index": candidate_index + 1,
                    "reason": stage_result.reason,
                },
            )
        elif (
            block_metadata.get("hybrid_model_recovery", False)
            and stage_result.ok
            and code_block_idx < len(code_blocks)
            and int(code_block_metadata[code_block_idx].get(
                "model_recovery_candidate_index", 0
            )) > int(block_metadata.get("model_recovery_candidate_index", 0))
        ):
            candidate_index = int(
                block_metadata.get("model_recovery_candidate_index", 0)
            )
            hybrid_auto_continue = True
            all_responses.append({
                "decision": "model_recovery_candidate_retry",
                "completed_candidate_index": candidate_index,
                "next_candidate_index": candidate_index + 1,
                "reason": "K=3 postcondition did not confirm task completion",
                "model_request_count": 0,
            })
            env.record_world_state_event(
                "model_recovery_candidate_retry",
                metadata={
                    "completed_candidate_index": candidate_index,
                    "next_candidate_index": candidate_index + 1,
                    "reason": "K=3 postcondition did not confirm task completion",
                },
            )
        elif (
            block_metadata.get("hybrid_model_recovery", False)
            and (
                not stage_result.ok
                or (
                    code_block_idx >= len(code_blocks)
                    and not _current_task_completed(env)
                )
            )
        ):
            if code_block_idx < len(code_blocks):
                del code_blocks[code_block_idx:]
                del code_block_metadata[code_block_idx:]
            all_responses.append({
                "decision": "model_recovery_failed",
                "completed_recovery_block": block_metadata.get(
                    "model_recovery_block_index"
                ),
                "stage_index": stage_index,
                "reason": (
                    stage_result.reason
                    if not stage_result.ok
                    else "model recovery ended before task completion"
                ),
                "fallback_to_model": bool(multi_turn_prompt),
            })
            env.record_world_state_event(
                "model_recovery_failed",
                metadata={
                    "model_recovery_block_index": block_metadata.get(
                        "model_recovery_block_index"
                    ),
                    "stage_index": stage_index,
                    "reason": (
                        stage_result.reason
                        if not stage_result.ok
                        else "model recovery ended before task completion"
                    ),
                },
            )
            hybrid_fallback_pending = True
        elif (
            block_metadata.get("hybrid_policy", False)
            and stage_result.ok
            and code_block_idx < len(code_blocks)
            and code_block_metadata[code_block_idx].get("hybrid_policy", False)
        ):
            hybrid_auto_continue = True
            all_responses.append({
                "decision": "policy_continue",
                "offline_action_id": policy_action_id,
                "model_request_count": 0,
            })
        elif (
            block_metadata.get("hybrid_model_recovery", False)
            and stage_result.ok
            and code_block_idx < len(code_blocks)
            and code_block_metadata[code_block_idx].get(
                "hybrid_model_recovery", False
            )
            and code_block_metadata[code_block_idx].get("generation")
            == block_metadata.get("generation")
        ):
            hybrid_auto_continue = True
            all_responses.append({
                "decision": "model_recovery_continue",
                "completed_recovery_block": block_metadata.get(
                    "model_recovery_block_index"
                ),
                "next_recovery_block": code_block_metadata[code_block_idx].get(
                    "model_recovery_block_index"
                ),
                "model_request_count": 0,
            })

        if (
            hybrid_fallback_pending
            and (hybrid_model_fallback_limit or hybrid_model_fallback_total_limit)
            and multi_turn_prompt
        ):
            exception_key = active_hybrid_exception or "UNCLASSIFIED_FAILURE"
            fallback_count = hybrid_model_fallback_counts.get(exception_key, 0)
            budget_exhausted = bool(
                (
                    hybrid_model_fallback_limit
                    and fallback_count >= hybrid_model_fallback_limit
                )
                or (
                    hybrid_model_fallback_total_limit
                    and hybrid_model_fallback_total
                    >= hybrid_model_fallback_total_limit
                )
            )
            if budget_exhausted:
                safe_actions = hybrid_library.lookup("RECOVERY_BUDGET_LOW") or ()
                del code_blocks[code_block_idx:]
                del code_block_metadata[code_block_idx:]
                for offset, action_id in enumerate(safe_actions):
                    code_blocks.append(hybrid_graph.actions[action_id].code)
                    code_block_metadata.append({
                        "generation": num_regenerations,
                        "regenerated": True,
                        "stage_index": stage_index + offset,
                        "hybrid_policy": True,
                        "offline_action_id": action_id,
                        "hybrid_terminal_safe_stop": offset == len(safe_actions) - 1,
                        "hybrid_exception_id": exception_key,
                    })
                if all_responses:
                    all_responses[-1]["fallback_to_model"] = False
                all_responses.append({
                    "decision": "model_fallback_budget_exhausted",
                    "exception_id": exception_key,
                    "fallback_attempts": fallback_count,
                    "fallback_total": hybrid_model_fallback_total,
                    "policy_actions": list(safe_actions),
                    "model_request_count": 0,
                })
                env.record_world_state_event(
                    "model_fallback_budget_exhausted",
                    metadata={
                        "exception_id": exception_key,
                        "fallback_attempts": fallback_count,
                        "fallback_total": hybrid_model_fallback_total,
                        "policy_actions": list(safe_actions),
                    },
                )
                hybrid_dispatched = True
                hybrid_fallback_pending = False
            else:
                hybrid_model_fallback_counts[exception_key] = fallback_count + 1
                hybrid_model_fallback_total += 1
                if all_responses:
                    all_responses[-1]["fallback_attempt"] = fallback_count + 1
                    all_responses[-1]["fallback_limit"] = hybrid_model_fallback_limit
                    all_responses[-1]["fallback_total"] = hybrid_model_fallback_total
                    all_responses[-1]["fallback_total_limit"] = (
                        hybrid_model_fallback_total_limit
                    )

        # Multi-turn decision
        if multi_turn_prompt and not hybrid_dispatched and not hybrid_auto_continue:
            if "terminated episode" in info_step["stderr"]:
                truncated = True
                break
            if (
                text_feedback is not None
                and text_feedback.should_abort
                and not hybrid_fallback_pending
            ):
                message = "Recovery loop guard aborted after a failed strategy switch."
                info_step["stderr"] = "\n".join(
                    part for part in (info_step.get("stderr", ""), message) if part
                )
                all_responses.append({
                    "decision": "guard_abort",
                    "recovery_directive": text_feedback.last_directive.to_dict(),
                    "reasoning": message,
                })
                truncated = True
                break

            # Get turn frames for video differencing
            turn_frames = None
            wrist_turn_frames = None
            if use_video_diff and recording_frames:
                turn_frames = env.get_video_frames_range(frame_start, frame_end)
                if use_wrist and hasattr(env, "get_wrist_video_frames_range"):
                    wrist_turn_frames = env.get_wrist_video_frames_range(
                        frame_start, frame_end,
                    )

            decision_guard_aborted = False
            decision_reprompts = 0
            while True:
                model_requests_used = _model_requests_used(all_responses)
                if max_model_requests and model_requests_used >= max_model_requests:
                    message = (
                        "Model request budget exhausted: "
                        f"{model_requests_used}/{max_model_requests} requests used."
                    )
                    all_responses.append({
                        "decision": "model_request_budget_exhausted",
                        "model_requests_used": model_requests_used,
                        "model_request_limit": max_model_requests,
                        "model_request_count": 0,
                        "reasoning": message,
                    })
                    info_step["stderr"] = "\n".join(
                        part for part in (info_step.get("stderr", ""), message)
                        if part
                    )
                    truncated = True
                    decision_guard_aborted = True
                    break
                structured_model_recovery = bool(
                    hybrid_fallback_pending
                    and hybrid_model_recovery_stage_indices
                    and hybrid_model_recovery_template
                )
                (
                    decision, new_code, mt_reasoning, mt_ensemble, decision_prompt,
                    mt_token_usage,
                ) = _handle_multi_turn_step(
                    env, obs, args, config, visual_differencing_args,
                    multi_turn_prompt, code_blocks, code_block_idx, info_step,
                    task_description, visual_feedback_imgs, visual_feedback_base64_history,
                    stderr_history,
                    text_world_state_feedback=text_feedback_prompt,
                    turn_frames=turn_frames,
                    wrist_turn_frames=wrist_turn_frames,
                    wrist_base64_history=wrist_base64_history,
                    trial=trial,
                    structured_recovery_schema=(
                        _structured_recovery_schema(
                            hybrid_model_recovery_template
                        )
                        if structured_model_recovery else None
                    ),
                    structured_recovery_context=(
                        f"{active_hybrid_exception or 'UNCLASSIFIED_FAILURE'}: "
                        f"{stage_result.reason}"
                        if structured_model_recovery else None
                    ),
                )

                if mt_ensemble is not None:
                    mt_ensemble["regeneration"] = num_regenerations + 1
                    multiturn_ensemble_data.append(mt_ensemble)

                selection_abstained = bool(
                    ((mt_ensemble or {}).get("candidate_reranking") or {}).get(
                        "selection_abstained", False
                    )
                )
                if selection_abstained:
                    message = "Candidate selector abstained because no safe decision was available."
                    all_responses.append({
                        "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                        "decision": "guard_abort",
                        "reasoning": message,
                        "token_usage": mt_token_usage,
                        **_model_query_log_fields(mt_ensemble),
                    })
                    info_step["stderr"] = "\n".join(
                        part for part in (info_step.get("stderr", ""), message) if part
                    )
                    truncated = True
                    decision_guard_aborted = True
                    break

                decision_result = validate_decision(
                    decision,
                    task_completed=latest_world_state.get(
                        "task_completed", info_step.get("task_completed")
                    ),
                    stage_result=stage_result,
                    config=execution_guard_cfg,
                )
                staged_model_recovery = bool(
                    hybrid_fallback_pending
                    and hybrid_model_recovery_stage_indices
                    and decision == "regenerate"
                )
                if decision_result.ok and staged_model_recovery:
                    try:
                        if hybrid_model_recovery_template:
                            recovery_role, recovery_target = _recovery_binding(
                                active_hybrid_role, latest_world_state
                            )
                            staged_blocks, _ = (
                                _compile_model_recovery_selection(
                                    hybrid_model_recovery_template,
                                    new_code,
                                    hybrid_model_recovery_candidate_limit,
                                    object_role=recovery_role,
                                    target_region=recovery_target,
                                )
                            )
                        else:
                            staged_blocks = _extract_code(new_code)
                    except ValueError as exc:
                        staged_blocks = []
                        decision_result = GuardResult(
                            False,
                            "model_recovery_template",
                            str(exc),
                        )
                    expected_recovery_blocks = (
                        len(hybrid_model_recovery_stage_indices)
                        * hybrid_model_recovery_candidate_limit
                    )
                    if decision_result.ok and len(staged_blocks) != expected_recovery_blocks:
                        decision_result = GuardResult(
                            False,
                            "model_recovery_shape",
                            "expected "
                            f"{expected_recovery_blocks} code blocks, "
                            f"received {len(staged_blocks)}",
                        )
                    elif decision_result.ok:
                        for staged_index, staged_block in enumerate(staged_blocks):
                            block_guard = validate_code_block(
                                staged_block, execution_guard_cfg
                            )
                            if not block_guard.ok:
                                decision_result = GuardResult(
                                    False,
                                    "model_recovery_preflight",
                                    f"recovery block {staged_index} failed static "
                                    f"validation: {block_guard.reason}",
                                    {
                                        "block_index": staged_index,
                                        "block_gate": block_guard.gate,
                                        "block_details": block_guard.details,
                                    },
                                )
                                break
                if decision_result.ok:
                    break

                rejected_decision = (
                    f"{decision_result.gate}_rejected"
                    if decision_result.gate.startswith("model_recovery_")
                    else f"{decision}_rejected"
                )
                all_responses.append({
                    "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                    "decision": rejected_decision,
                    "reasoning": mt_reasoning if mt_reasoning is not None else (new_code or ""),
                    "token_usage": mt_token_usage,
                    "execution_guard": {
                        "gate": decision_result.gate,
                        "reason": decision_result.reason,
                        "details": decision_result.details,
                    },
                    **_model_query_log_fields(mt_ensemble),
                })
                env.record_world_state_event(
                    "model_decision_rejected",
                    metadata={
                        "decision": decision,
                        "gate": decision_result.gate,
                        "reason": decision_result.reason,
                        "stage_index": stage_index,
                    },
                )
                if decision_reprompts >= execution_guard_cfg.max_decision_reprompts:
                    message = (
                        "Execution guard aborted after exhausting decision reprompts: "
                        f"{decision_result.reason}"
                    )
                    all_responses.append({
                        "decision": "guard_abort",
                        "reasoning": message,
                        "execution_guard": {
                            "gate": decision_result.gate,
                            "reason": decision_result.reason,
                        },
                    })
                    info_step["stderr"] = "\n".join(
                        part for part in (info_step.get("stderr", ""), message) if part
                    )
                    truncated = True
                    decision_guard_aborted = True
                    break

                if hybrid_fallback_pending and (
                    hybrid_model_fallback_limit
                    or hybrid_model_fallback_total_limit
                ):
                    exception_key = active_hybrid_exception or "UNCLASSIFIED_FAILURE"
                    fallback_count = hybrid_model_fallback_counts.get(exception_key, 0)
                    budget_exhausted = bool(
                        (
                            hybrid_model_fallback_limit
                            and fallback_count >= hybrid_model_fallback_limit
                        )
                        or (
                            hybrid_model_fallback_total_limit
                            and hybrid_model_fallback_total
                            >= hybrid_model_fallback_total_limit
                        )
                    )
                    if budget_exhausted:
                        message = (
                            "Model fallback budget exhausted before decision reprompt: "
                            f"{decision_result.reason}"
                        )
                        all_responses.append({
                            "decision": "model_fallback_budget_exhausted",
                            "exception_id": exception_key,
                            "fallback_attempts": fallback_count,
                            "fallback_total": hybrid_model_fallback_total,
                            "reasoning": message,
                            "model_request_count": 0,
                        })
                        env.record_world_state_event(
                            "model_fallback_budget_exhausted",
                            metadata={
                                "exception_id": exception_key,
                                "fallback_attempts": fallback_count,
                                "fallback_total": hybrid_model_fallback_total,
                                "reason": decision_result.reason,
                            },
                        )
                        info_step["stderr"] = "\n".join(
                            part for part in (info_step.get("stderr", ""), message)
                            if part
                        )
                        truncated = True
                        decision_guard_aborted = True
                        break
                    hybrid_model_fallback_counts[exception_key] = fallback_count + 1
                    hybrid_model_fallback_total += 1

                if (
                    hybrid_model_recovery_template
                    and decision_result.gate.startswith("model_recovery_")
                ):
                    correction = (
                        "HARD CONSTRAINT: return exactly one raw structured object "
                        f"matching {_structured_recovery_schema(hybrid_model_recovery_template)}. "
                        "Do not return REGENERATE, markdown, fenced Python, prose, "
                        f"or robot code. Rejected selection: {decision_result.reason}"
                    )
                else:
                    correction = format_guard_feedback(decision_result)
                info_step["stderr"] = "\n".join(
                    part for part in (info_step.get("stderr", ""), correction) if part
                )
                text_feedback_prompt = append_text_world_state_feedback(
                    text_feedback_prompt or "", correction
                )
                decision_reprompts += 1

            if decision_guard_aborted:
                break

            if decision == "regenerate":
                print("Model chose to regenerate code")
                staged_model_recovery = bool(
                    hybrid_fallback_pending
                    and hybrid_model_recovery_stage_indices
                )
                template_selection = None
                if staged_model_recovery and hybrid_model_recovery_template:
                    recovery_role, recovery_target = _recovery_binding(
                        active_hybrid_role, latest_world_state
                    )
                    new_blocks, template_selection = (
                        _compile_model_recovery_selection(
                            hybrid_model_recovery_template,
                            new_code,
                            hybrid_model_recovery_candidate_limit,
                            object_role=recovery_role,
                            target_region=recovery_target,
                        )
                    )
                else:
                    new_blocks = _extract_code(new_code)
                if (
                    staged_model_recovery
                    and len(new_blocks)
                    != len(hybrid_model_recovery_stage_indices)
                    * hybrid_model_recovery_candidate_limit
                ):
                    message = (
                        "Execution guard rejected model recovery: expected "
                        f"{len(hybrid_model_recovery_stage_indices) * hybrid_model_recovery_candidate_limit} code blocks, "
                        f"received {len(new_blocks)}."
                    )
                    info_step["stderr"] = "\n".join(
                        part for part in (info_step.get("stderr", ""), message)
                        if part
                    )
                    all_responses.append({
                        "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                        "block_idx": [code_block_idx],
                        "code_blocks": new_blocks,
                        "decision": "model_recovery_shape_rejected",
                        "reasoning": message,
                        "token_usage": mt_token_usage,
                        **_model_query_log_fields(mt_ensemble),
                    })
                    truncated = True
                    break
                candidate = "\n\n".join(new_blocks)
                if (
                    text_feedback is not None
                    and not text_feedback.accepts_regeneration(candidate, code)
                ):
                    message = (
                        "Recovery loop guard rejected a structurally repeated strategy "
                        "after the retry budget was exhausted."
                    )
                    info_step["stderr"] = "\n".join(
                        part for part in (info_step.get("stderr", ""), message) if part
                    )
                    all_responses.append({
                        "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                        "block_idx": [code_block_idx],
                        "code_blocks": new_blocks,
                        "decision": "guard_abort",
                        "recovery_directive": text_feedback.last_directive.to_dict(),
                        "reasoning": message,
                        "token_usage": mt_token_usage,
                        **_model_query_log_fields(mt_ensemble),
                    })
                    truncated = True
                    break
                all_responses.append({
                    "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                    "block_idx": [code_block_idx],
                    "code_blocks": new_blocks,
                    "decision": "regenerate",
                    "reasoning": mt_reasoning if mt_reasoning is not None else "",
                    "token_usage": mt_token_usage,
                    "model_recovery_raw_selection": (
                        new_code
                        if staged_model_recovery
                        and hybrid_model_recovery_template
                        else None
                    ),
                    "model_recovery_template": template_selection,
                    **_model_query_log_fields(mt_ensemble),
                })
                del code_blocks[code_block_idx:]
                del code_block_metadata[code_block_idx:]
                code_blocks.extend(new_blocks)
                next_stage_index = stage_index + 1 if stage_result.ok else stage_index
                new_metadata = assign_stage_indices(
                    len(new_blocks),
                    start=next_stage_index,
                    generation=num_regenerations + 1,
                    regenerated=True,
                    regenerated_at_idx=code_block_idx,
                )
                if hybrid_model_led_policy_recovery and not staged_model_recovery:
                    _attach_hybrid_stage_monitors(
                        new_metadata, hybrid_stage_actions
                    )
                if staged_model_recovery:
                    expanded_stage_indices = (
                        hybrid_model_recovery_stage_indices
                        * hybrid_model_recovery_candidate_limit
                    )
                    recovery_monitors = _recovery_monitor_actions(
                        hybrid_graph,
                        active_hybrid_role or "primary",
                        hybrid_stage_actions,
                    )
                    for block_index, (metadata, recovery_stage_index) in enumerate(
                        zip(
                            new_metadata,
                            expanded_stage_indices,
                            strict=True,
                        )
                    ):
                        metadata.update({
                            "stage_index": recovery_stage_index,
                            "hybrid_model_recovery": True,
                            "model_recovery_block_index": (
                                block_index
                                % len(hybrid_model_recovery_stage_indices)
                            ),
                            "model_recovery_candidate_index": (
                                block_index
                                // len(hybrid_model_recovery_stage_indices)
                            ),
                        })
                        if (
                            not execution_guard_cfg.stage_expectations
                            and recovery_monitors
                        ):
                            metadata["hybrid_monitor_action_id"] = (
                                recovery_monitors[
                                    block_index % len(recovery_monitors)
                                ]
                            )
                code_block_metadata.extend(new_metadata)
                num_regenerations += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_regenerations"] = num_regenerations

            elif decision == "finish":
                all_responses.append({
                    "decision": "finish",
                    "reasoning": mt_reasoning if mt_reasoning is not None else (new_code or ""),
                    "token_usage": mt_token_usage,
                    **_model_query_log_fields(mt_ensemble),
                })
                print("Model chose to finish")
                num_finishes += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_finishes"] = num_finishes
                break

            elif decision == "continue":
                all_responses.append({
                    "decision": "continue",
                    "reasoning": mt_reasoning if mt_reasoning is not None else "",
                    "token_usage": mt_token_usage,
                    **_model_query_log_fields(mt_ensemble),
                })

        print(f"Code block {code_block_idx} done")
        print(f"Number of code blocks: {len(code_blocks)}")

        # Save intermediate artifacts (code, logs) per code block
        final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
        _save_trial_artifacts(
            config, trial, info_step["sandbox_rc"], reward,
            info_step.get("task_completed", False), final_code, raw_code,
            all_responses, ["-" * 100, "Generated program:", final_code],
            visual_feedback_imgs,
        )

        # Only save intermediate video if NOT doing per-turn saving
        # (per-turn saving is deferred to after the loop to avoid clearing the buffer)
        if not recording_frames:
            _save_trial_video(
                env, config, trial, info_step, reward, len(code_blocks),
                suffix_extra=str(len(code_blocks)),
            )

    print("Code blocks done")

    # --- 5. Build final summary ---
    final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
    num_code_blocks = len(code_blocks)

    if partial_artifacts is not None:
        partial_artifacts["final_code"] = final_code
        partial_artifacts["num_code_blocks"] = num_code_blocks

    # Override sandbox_rc for terminated-episode stderr
    if "executing action in terminated episode" in info_step["stderr"]:
        sandbox_rc_override = 0
    if sandbox_rc_override is not None:
        info_step["sandbox_rc"] = sandbox_rc_override

    stderr = "\n\n".join(stderr_history) if stderr_history else info_step["stderr"]
    log_lines = _build_log_lines(
        final_code, info_step, reward, terminated, truncated,
        num_regenerations, num_finishes, num_code_blocks,
        stderr_override=stderr,
    )

    code_path = _save_trial_artifacts(
        config, trial, info_step["sandbox_rc"], reward,
        info_step.get("task_completed", False), final_code, raw_code,
        all_responses, log_lines, visual_feedback_imgs,
        ensemble_data=ensemble_data,
        multiturn_ensemble_data=multiturn_ensemble_data,
    )

    # Save per-turn and combined videos
    if recording_frames and turn_frame_ranges:
        _save_turn_and_combined_videos(
            env, config, trial, info_step, reward, turn_frame_ranges,
        )
    else:
        _save_trial_video(env, config, trial, info_step, reward, num_code_blocks)

    success = info_step["sandbox_rc"] == 0
    if text_feedback is not None and config.get("output_dir"):
        feedback_path = (
            Path(config["output_dir"])
            / f"text_world_state_feedback_trial_{trial:03d}.json"
        )
        feedback_path.write_text(
            json.dumps({
                "config": text_feedback_cfg.__dict__,
                "history": text_feedback.history,
                "last_error": text_feedback.last_error,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    env.record_world_state_event(
        "trial_end",
        metadata={"trial": trial, "success": success, "reward": reward,
                  "task_completed": info_step.get("task_completed")},
    )
    env.set_world_state_recorder(None)

    # --- Evolving skill library integration (opt-in) ---
    if config.get("evolve_skill_library", False) and info_step.get("task_completed", False):
        try:
            from capx.skills import SkillLibrary

            skill_lib_path = config.get("skill_library_path", None)
            skill_lib = SkillLibrary(path=skill_lib_path)
            task_name = config.get("task_name", f"trial_{trial}")
            new_skills = skill_lib.extract_from_code(final_code, task_name=task_name)
            skill_lib.save()
            if new_skills:
                print(f"[SkillLibrary] Extracted {len(new_skills)} new skill(s): {new_skills}")
        except Exception as exc:
            print(f"[SkillLibrary] Skill extraction failed: {exc}")

    print(f"Trial {trial} took {time.time() - trial_start_time:.2f} seconds")

    gc.collect()

    return TrialSummary(
        trial=trial,
        success=success,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        sandbox_rc=info_step["sandbox_rc"],
        log="\n".join(log_lines),
        task_completed=info_step.get("task_completed", None),
        code_path=code_path,
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=num_code_blocks,
    )


def _patch_libero_goal(env: CodeExecutionEnvBase, obs: dict[str, Any]) -> None:
    """Inject the LIBERO task language into the prompt template if applicable."""
    if not hasattr(env.low_level_env, "handle"):
        return
    handle = env.low_level_env.handle
    if (
        hasattr(handle, "task_language")
        and "libero_environment_goal" in obs["full_prompt"][-1]["content"][0]["text"]
    ):
        goal = getattr(handle, "task_language")
        obs["full_prompt"][-1]["content"][0]["text"] = (
            obs["full_prompt"][-1]["content"][0]["text"].format(
                libero_environment_goal=goal
            )
        )
