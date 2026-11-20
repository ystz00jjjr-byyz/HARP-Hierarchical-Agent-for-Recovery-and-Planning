"""Compile recorded CAPX trials into an offline graph and recovery library.

This is intentionally conservative: successful traces build the nominal graph;
recovery routes are emitted only when a trial explicitly declares an
``exception_id`` in ``failure.json``. Unlabelled failures remain review data.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_TASK_DONE = re.compile(r"taskcompleted_(\d+)")
_BLOCK_HEADER = re.compile(r"^# Code block \d+.*$", re.MULTILINE)

_CANDIDATE_ACTIONS = {
    "CLEAR_RECOVERY_FAILED": "recover_clear_recovery_radial",
    "SLENDER_GRASP_POSE_UNSAFE": "recover_slender_grasp_pose",
    "UNSTABLE_RELEASE": "recover_slender_unstable_release",
    "PLACE_ALIGNMENT_FAILED": "recover_slender_place_alignment",
    "RECOVERY_BUDGET_LOW": "recover_budget_safe_stop",
}


@dataclass(frozen=True)
class Trace:
    path: Path
    success: bool
    blocks: tuple[str, ...]
    failure: dict[str, Any]


def _success_from_path(path: Path) -> bool:
    match = _TASK_DONE.search(path.name)
    return bool(match and match.group(1) == "1")


def _load_blocks(path: Path) -> tuple[str, ...]:
    response_path = path / "all_responses.json"
    if response_path.exists():
        try:
            responses = json.loads(response_path.read_text(encoding="utf-8"))
            for item in responses:
                if item.get("decision") == "initial" and item.get("code_blocks"):
                    return tuple(str(block).strip() for block in item["code_blocks"] if str(block).strip())
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    code_path = path / "code.py"
    if not code_path.exists():
        return ()
    text = code_path.read_text(encoding="utf-8")
    parts = _BLOCK_HEADER.split(text)
    return tuple(part.strip() for part in parts if part.strip())


def _load_failure(path: Path) -> dict[str, Any]:
    failure_path = path / "failure.json"
    if not failure_path.exists():
        return {}
    try:
        value = json.loads(failure_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def load_traces(root: str | Path) -> list[Trace]:
    root = Path(root)
    traces: list[Trace] = []
    for response_path in sorted(root.rglob("all_responses.json")):
        trial_dir = response_path.parent
        blocks = _load_blocks(trial_dir)
        if blocks:
            traces.append(Trace(trial_dir, _success_from_path(trial_dir), blocks, _load_failure(trial_dir)))
    return traces


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return default


def _final_record(path: Path) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        if line.strip():
            value = _read_json_line(line)
            if value:
                return value
    return {}


def _read_json_line(line: str) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def compile_failure_candidates(
    root: str | Path, trial_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Compile failed terminal traces into non-executable candidate patterns."""
    root = Path(root)
    grouped: dict[str, list[dict[str, Any]]] = {
        exception_id: [] for exception_id in _CANDIDATE_ACTIONS
    }
    evidence_by_trial: dict[int, dict[str, Any]] = {}
    selected = set(trial_ids or ())
    for metrics_path in sorted(root.rglob("trial_metrics.json")):
        metrics = _read_json(metrics_path, {})
        trial = int(metrics.get("trial", 0))
        if not trial or (selected and trial not in selected) or metrics.get("task_completed"):
            continue
        responses = _read_json(metrics_path.parent / "all_responses.json", [])
        record = _final_record(
            root / "world_state" / f"trial_{trial:03d}" / "trajectory.jsonl"
        )
        world = dict(record.get("world_state", {}))
        state = dict(world.get("task_state", {}))
        summary = (metrics_path.parent / "summary.txt").read_text(
            encoding="utf-8", errors="replace"
        ) if (metrics_path.parent / "summary.txt").exists() else ""
        preflight_grasp = any(
            item.get("decision") == "code_preflight_rejected"
            and item.get("offline_action_id") == "stage_002"
            for item in responses if isinstance(item, dict)
        )
        aspect = float(state.get("max_horizontal_aspect_ratio", 0.0) or 0.0)
        distance = float(state.get("distance_to_target", float("inf")) or float("inf"))
        patterns: list[str] = []
        if "Exception: CLEAR_RECOVERY_FAILED" in summary:
            patterns.append("CLEAR_RECOVERY_FAILED")
        if aspect >= 2.0 and preflight_grasp:
            patterns.append("SLENDER_GRASP_POSE_UNSAFE")
        if state.get("object_stable") is False:
            patterns.append("UNSTABLE_RELEASE")
        if distance <= 0.12 and not state.get("task_completed", False):
            patterns.append("PLACE_ALIGNMENT_FAILED")
        if (
            "executing action in terminated episode" in summary
            or int(world.get("simulator_step", 0) or 0) >= 1500
        ):
            patterns.append("RECOVERY_BUDGET_LOW")
        evidence = {
            "trial": trial,
            "source_trace": str(metrics_path.parent),
            "reward": float(metrics.get("reward", 0.0)),
            "model_requests": int(metrics.get("model_request_count", 0)),
            "simulator_step": int(world.get("simulator_step", 0) or 0),
            "final_state": {
                key: state.get(key) for key in (
                    "target_grasped", "object_lifted", "object_stable",
                    "gripper_open", "distance_to_target",
                    "max_horizontal_aspect_ratio",
                )
            },
        }
        evidence_by_trial[trial] = evidence
        for exception_id in patterns:
            grouped[exception_id].append(json.loads(json.dumps(evidence)))

    return {
        "metadata": {
            "status": "candidate",
            "source_root": str(root),
            "promotion_gate": {
                "required_trials": sorted(selected),
                "minimum_successes": 4,
                "required_model_requests": 0,
            },
        },
        "evidence": [evidence_by_trial[trial] for trial in sorted(evidence_by_trial)],
        "candidates": [
            {
                "exception_id": exception_id,
                "candidate_action": _CANDIDATE_ACTIONS[exception_id],
                "status": "validation_required",
                "promote": False,
                "source_trials": [item["trial"] for item in evidence],
            }
            for exception_id, evidence in grouped.items() if evidence
        ],
    }


def write_failure_candidates(
    root: str | Path, output_path: str | Path, trial_ids: set[int] | None = None,
) -> dict[str, Any]:
    value = compile_failure_candidates(root, trial_ids)
    Path(output_path).write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return value


def _call_names(code: str) -> tuple[str, ...]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ()
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
    return tuple(names)


def _fingerprint(blocks: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(_call_names(block) for block in blocks)


def _primitive(code: str) -> str:
    return (_call_names(code) or ("code_block",))[0]


def _build_nominal_graph(traces: list[Trace]) -> dict[str, Any]:
    successful = [trace for trace in traces if trace.success and trace.blocks]
    if not successful:
        raise ValueError("no successful traces with code blocks found")
    representative = Counter(_fingerprint(trace.blocks) for trace in successful).most_common(1)[0][0]
    trace = next(trace for trace in successful if _fingerprint(trace.blocks) == representative)
    actions: list[dict[str, Any]] = []
    for index, code in enumerate(trace.blocks):
        action_id = f"stage_{index + 1:03d}"
        item: dict[str, Any] = {
            "id": action_id,
            "primitive": _primitive(code),
            "code": code,
            "metadata": {"calls": list(_call_names(code)), "source_trace": str(trace.path)},
        }
        if index + 1 < len(trace.blocks):
            item["next"] = f"stage_{index + 2:03d}"
        else:
            item["success_conditions"] = {"task_completed": True}
        actions.append(item)
    return {
        "metadata": {
            "status": "candidate",
            "source_trace": str(trace.path),
            "successful_trace_count": len(successful),
        },
        "initial": ["stage_001"],
        "actions": actions,
    }


def _build_recovery_library(traces: list[Trace], graph: dict[str, Any]) -> dict[str, Any]:
    """Compile explicitly labelled recoveries; leave implicit ones for review."""
    routes: dict[str, list[str]] = {}
    candidates: list[dict[str, Any]] = []
    nominal_count = len(graph["actions"])
    next_index = nominal_count + 1
    for trace in traces:
        if not trace.failure.get("exception_id"):
            continue
        exception_id = str(trace.failure["exception_id"])
        failed_block = int(trace.failure.get("failed_block_index", 0))
        if "recovery_code_blocks" in trace.failure:
            recovery_blocks = trace.failure["recovery_code_blocks"]
        else:
            recovery_blocks = trace.blocks[failed_block:]
        recovery_blocks = [str(code) for code in recovery_blocks]
        if not trace.failure.get("promote", True):
            candidates.append({
                "exception_id": exception_id,
                "failed_block_index": failed_block,
                "source_trace": str(trace.path),
                "trace_success": trace.success,
                "recovery_code_blocks": recovery_blocks[:3],
                "evidence": trace.failure.get("evidence"),
                "status": "review_required",
            })
            continue
        route: list[str] = []
        for code in recovery_blocks[:3]:
            action_id = f"recover_{exception_id.lower()}_{next_index:03d}"
            next_index += 1
            action = {
                "id": action_id,
                "primitive": _primitive(str(code)),
                "code": str(code),
                "metadata": {"source_trace": str(trace.path), "exception_id": exception_id},
            }
            graph["actions"].append(action)
            route.append(action_id)
        if route:
            if failed_block + 1 < nominal_count:
                graph["actions"][-len(route)]["next"] = f"stage_{failed_block + 2:03d}"
            routes.setdefault(exception_id, route)
    return {"recovery_library": routes, "candidates": candidates}


def compile_traces(root: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    traces = load_traces(root)
    if not traces:
        raise ValueError(f"no trial traces found under {root}")
    graph = _build_nominal_graph(traces)
    library = _build_recovery_library(traces, graph)
    stats = {
        "traces": len(traces),
        "successful": sum(trace.success for trace in traces),
        "failed": sum(not trace.success for trace in traces),
        "labelled_failures": sum(
            not trace.success and bool(trace.failure.get("exception_id")) for trace in traces
        ),
        "labelled_recoveries": sum(
            bool(trace.failure.get("exception_id")) for trace in traces
        ),
        "graph_actions": len(graph["actions"]),
        "recovery_routes": len(library["recovery_library"]),
        "recovery_candidates": len(library["candidates"]),
    }
    return graph, library, stats


def write_compilation(root: str | Path, graph_path: str | Path, library_path: str | Path) -> dict[str, int]:
    graph, library, stats = compile_traces(root)
    Path(graph_path).write_text(yaml.safe_dump(graph, sort_keys=False, allow_unicode=True), encoding="utf-8")
    Path(library_path).write_text(yaml.safe_dump(library, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile CAPX traces into offline recovery artifacts")
    parser.add_argument("input", type=Path)
    parser.add_argument("--graph", type=Path, default=Path("task_graph.yaml"))
    parser.add_argument("--library", type=Path, default=Path("recovery_library.yaml"))
    parser.add_argument("--failure-candidates", type=Path)
    parser.add_argument("--trials", type=int, nargs="*")
    args = parser.parse_args()
    if args.failure_candidates:
        result = write_failure_candidates(
            args.input, args.failure_candidates, set(args.trials or ()),
        )
        print(json.dumps({
            "evidence": len(result["evidence"]),
            "candidates": len(result["candidates"]),
            "output": str(args.failure_candidates),
        }, indent=2))
        return
    print(json.dumps(write_compilation(args.input, args.graph, args.library), indent=2))


if __name__ == "__main__":
    main()
