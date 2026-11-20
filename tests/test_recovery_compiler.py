import json

from capx.recovery.compiler import (
    compile_failure_candidates,
    compile_traces,
    write_compilation,
)


def _trial(path, completed, blocks, failure=None):
    path.mkdir(parents=True)
    (path / "all_responses.json").write_text(json.dumps([{
        "decision": "initial", "code_blocks": blocks,
    }]), encoding="utf-8")
    if failure is not None:
        (path / "failure.json").write_text(json.dumps(failure), encoding="utf-8")


def test_compiler_builds_nominal_graph_and_explicit_recovery(tmp_path):
    blocks = ["prepare()", "finish()"]
    _trial(tmp_path / "trial_01_sandboxrc_0_reward_1_taskcompleted_1", True, blocks)
    _trial(
        tmp_path / "trial_02_sandboxrc_1_reward_0_taskcompleted_0",
        False,
        blocks,
        {"exception_id": "GRASP_MISS", "failed_block_index": 0,
         "recovery_code_blocks": ["recover_grasp()"]},
    )

    graph, library, stats = compile_traces(tmp_path)

    assert stats == {
        "traces": 2, "successful": 1, "failed": 1,
        "labelled_failures": 1, "labelled_recoveries": 1,
        "graph_actions": 3, "recovery_routes": 1, "recovery_candidates": 0,
    }
    assert graph["actions"][0]["id"] == "stage_001"
    assert graph["actions"][0]["next"] == "stage_002"
    assert library["recovery_library"]["GRASP_MISS"]
    recovery_id = library["recovery_library"]["GRASP_MISS"][0]
    recovery = next(item for item in graph["actions"] if item["id"] == recovery_id)
    assert recovery["next"] == "stage_002"


def test_compiler_writes_yaml(tmp_path):
    _trial(tmp_path / "trial_01_sandboxrc_0_reward_1_taskcompleted_1", True, ["finish()"])
    stats = write_compilation(tmp_path, tmp_path / "graph.yaml", tmp_path / "library.yaml")
    assert stats["successful"] == 1
    assert (tmp_path / "graph.yaml").exists()
    assert (tmp_path / "library.yaml").exists()


def test_successful_trace_can_contribute_a_verified_recovery(tmp_path):
    _trial(
        tmp_path / "trial_01_sandboxrc_0_reward_1_taskcompleted_1",
        True,
        ["prepare()", "finish()"],
        {
            "exception_id": "GRASP_MISS",
            "failed_block_index": 0,
            "recovery_code_blocks": ["recover_grasp()"],
        },
    )

    graph, library, _stats = compile_traces(tmp_path)

    recovery_id = library["recovery_library"]["GRASP_MISS"][0]
    recovery = next(item for item in graph["actions"] if item["id"] == recovery_id)
    assert recovery["code"] == "recover_grasp()"


def test_unapproved_recovery_stays_non_executable(tmp_path):
    _trial(
        tmp_path / "trial_01_sandboxrc_0_reward_1_taskcompleted_1",
        True,
        ["prepare()", "finish()"],
        {
            "exception_id": "GRASP_MISS",
            "failed_block_index": 0,
            "recovery_code_blocks": ["seed_specific_recovery()"],
            "promote": False,
        },
    )

    graph, library, _stats = compile_traces(tmp_path)

    assert library["recovery_library"] == {}
    assert library["candidates"][0]["status"] == "review_required"
    assert len(graph["actions"]) == 2


def test_failed_terminal_trace_compiles_four_non_promoted_patterns(tmp_path):
    trial = tmp_path / "trial_08_sandboxrc_0_reward_0.749_taskcompleted_0"
    _trial(trial, False, ["attempt()"])
    (trial / "trial_metrics.json").write_text(json.dumps({
        "trial": 8, "task_completed": False, "reward": 0.749,
        "model_request_count": 3,
    }), encoding="utf-8")
    (trial / "all_responses.json").write_text(json.dumps([{
        "decision": "code_preflight_rejected", "offline_action_id": "stage_002",
    }]), encoding="utf-8")
    world = tmp_path / "world_state" / "trial_008"
    world.mkdir(parents=True)
    (world / "trajectory.jsonl").write_text(json.dumps({
        "world_state": {
            "simulator_step": 1500,
            "task_state": {
                "target_grasped": False, "object_lifted": False,
                "object_stable": False, "gripper_open": True,
                "distance_to_target": 0.01,
                "max_horizontal_aspect_ratio": 2.67,
                "task_completed": False,
            },
        },
    }) + "\n", encoding="utf-8")

    result = compile_failure_candidates(tmp_path, {8})

    assert {item["exception_id"] for item in result["candidates"]} == {
        "SLENDER_GRASP_POSE_UNSAFE", "UNSTABLE_RELEASE",
        "PLACE_ALIGNMENT_FAILED", "RECOVERY_BUDGET_LOW",
    }
    assert all(item["promote"] is False for item in result["candidates"])


def test_clear_recovery_failure_is_compiled_as_candidate(tmp_path):
    trial = tmp_path / "trial_01_sandboxrc_0_reward_0_taskcompleted_0"
    _trial(trial, False, ["attempt()"])
    (trial / "trial_metrics.json").write_text(json.dumps({
        "trial": 1, "task_completed": False, "reward": 0.0,
        "model_request_count": 0,
    }), encoding="utf-8")
    (trial / "summary.txt").write_text(
        "Exception: CLEAR_RECOVERY_FAILED\n", encoding="utf-8"
    )

    result = compile_failure_candidates(tmp_path, {1})

    assert result["candidates"] == [{
        "exception_id": "CLEAR_RECOVERY_FAILED",
        "candidate_action": "recover_clear_recovery_radial",
        "status": "validation_required",
        "promote": False,
        "source_trials": [1],
    }]
