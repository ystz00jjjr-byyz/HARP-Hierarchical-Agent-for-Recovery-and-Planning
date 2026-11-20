from pathlib import Path

import numpy as np
import yaml

from capx.envs.simulators.robosuite_bin_place import (
    FrankaRobosuiteBinPlaceLowLevel,
    box_inside_target_region,
    infeasible_region_place_scenario,
    region_place_scenario,
)


def test_region_place_supports_bounded_multi_grasp_miss_injection():
    import inspect

    parameters = inspect.signature(FrankaRobosuiteBinPlaceLowLevel).parameters
    config = yaml.safe_load(Path(
        "env_configs/region_place/physical_obstacle_safe_stop_5seed.yaml"
    ).read_text())

    assert "inject_grasp_miss_attempts" in parameters
    assert config["env"]["cfg"]["low_level_kwargs"][
        "inject_grasp_miss_attempts"
    ] == "${oc.env:REGION_PLACE_GRASP_MISS_ATTEMPTS,null}"
from capx.recovery.offline import (
    OfflineExecution,
    OfflineTaskGraph,
    OfflineWindowController,
    RecoveryLibrary,
)
from capx.recovery.execution_guard import ExecutionGuardConfig, validate_code_block
from capx.envs.launch import LaunchArgs
from capx.utils.launch_utils import _load_config


def test_region_place_formal_policy_uses_generic_target_region_exception():
    graph = OfflineTaskGraph.from_mapping(yaml.safe_load(
        Path("env_configs/region_place/offline_task_graph.yaml").read_text()
    ))
    library = RecoveryLibrary.from_mapping(yaml.safe_load(
        Path("env_configs/region_place/recovery_library.yaml").read_text()
    ))

    assert graph.actions["stage_002"].failure_routes["POSTCONDITION_FAILED"] == (
        "OUTSIDE_TARGET_REGION"
    )
    assert library.lookup("OUTSIDE_TARGET_REGION") == (
        "candidate_tight_clearance_regrasp",
    )
    assert library.lookup(
        "OUTSIDE_TARGET_REGION",
        {"target_geometry_type": "planar_region"},
    ) == ("candidate_tight_clearance_regrasp",)
    assert graph.actions["recover_target_region_regrasp"].next_action == (
        "candidate_tight_clearance_replace"
    )
    assert "container_entry_clear" not in graph.actions[
        "recover_target_region_regrasp"
    ].preconditions
    assert graph.actions["recover_target_region_regrasp"].metadata["status"] == (
        "formal"
    )
    assert "np.max(primary_extent[:2]) / np.min(primary_extent[:2])" in graph.actions[
        "recover_target_region_regrasp"
    ].code
    assert graph.actions["recover_target_region_regrasp"].metadata[
        "failure_next_action"
    ] == "recover_target_region_regrasp_retry"
    assert library.lookup("GRASP_RECOVERY_RETRY_REQUIRED") == (
        "recover_target_region_regrasp_retry",
    )
    retry = graph.actions["recover_target_region_regrasp_retry"]
    assert retry.next_action == "candidate_tight_clearance_replace"
    assert retry.metadata["max_attempts"] == 1
    assert retry.metadata["failure_next_action"] == (
        "recover_post_recovery_reobserve"
    )


def test_structural_regrasp_replace_candidate_is_role_based_bounded_and_safe():
    graph = OfflineTaskGraph.from_mapping(yaml.safe_load(
        Path("env_configs/region_place/offline_task_graph.yaml").read_text()
    ))
    candidate = graph.actions["candidate_structurally_distinct_regrasp_replace"]

    assert candidate.metadata["status"] == "formal"
    assert candidate.metadata["max_attempts"] == 1
    assert candidate.metadata["precondition_failure_next_action"] == (
        "recover_budget_safe_stop"
    )
    assert 'get_object_pose("primary", return_bbox_extent=True)' in candidate.code
    assert 'sample_grasp_pose("primary")' in candidate.code
    assert 'get_target_region("target")' in candidate.code
    assert "red cube" not in candidate.code
    assert "primary_extent" in candidate.code
    assert candidate.metadata["validation_successes"] == 10
    assert candidate.metadata["validation_trials"] == 10
    assert candidate.metadata["direct_successes"] == 10

    guard = validate_code_block(candidate.code, ExecutionGuardConfig.from_dict({
        "enabled": True,
        "max_robot_api_calls_per_block": 12,
        "max_expensive_loop_iterations": 0,
    }))
    assert guard.ok, guard.reason


def test_tight_clearance_candidate_is_split_target_aligned_and_bounded():
    graph = OfflineTaskGraph.from_mapping(yaml.safe_load(
        Path("env_configs/region_place/offline_task_graph.yaml").read_text()
    ))
    library = RecoveryLibrary.from_mapping(yaml.safe_load(
        Path(
            "env_configs/region_place/tight_clearance_candidate_recovery.yaml"
        ).read_text()
    ))
    regrasp = graph.actions["candidate_tight_clearance_regrasp"]
    replace = graph.actions["candidate_tight_clearance_replace"]
    guard_config = ExecutionGuardConfig.from_dict({
        "enabled": True,
        "max_robot_api_calls_per_block": 5,
        "max_expensive_loop_iterations": 0,
    })

    assert library.lookup("OUTSIDE_TARGET_REGION") == (
        "candidate_tight_clearance_regrasp",
    )
    assert regrasp.next_action == "candidate_tight_clearance_replace"
    assert regrasp.metadata["strategy_id"] == "TIGHT_CLEARANCE_PLACE"
    assert replace.metadata["strategy_id"] == "TIGHT_CLEARANCE_PLACE"
    assert regrasp.metadata["status"] == "formal"
    assert replace.metadata["status"] == "formal"
    assert regrasp.metadata["holdout_task_successes"] == 8
    assert regrasp.metadata["holdout_task_trials"] == 10
    assert replace.args["target_quaternion"] == (
        "target_region_quaternion_wxyz"
    )
    assert "target_yaw" in replace.code
    assert "transit = np.array" in replace.code
    assert "goto_pose(transit, grasp_quat)" in replace.code
    assert validate_code_block(regrasp.code, guard_config).ok
    assert validate_code_block(replace.code, guard_config).ok


def test_region_place_reuses_interruptible_workspace_edge_recovery_chain():
    graph = OfflineTaskGraph.from_mapping(yaml.safe_load(
        Path("env_configs/region_place/offline_task_graph.yaml").read_text()
    ))
    library = RecoveryLibrary.from_mapping(yaml.safe_load(
        Path("env_configs/region_place/recovery_library.yaml").read_text()
    ))
    action_ids = (
        "recover_workspace_edge_reobserve",
        "recover_workspace_edge_check",
        "recover_workspace_edge_move",
        "recover_workspace_edge_post_move_reobserve",
        "recover_workspace_edge_regrasp",
    )
    guard_config = ExecutionGuardConfig.from_dict({
        "enabled": True,
        "max_robot_api_calls_per_block": 6,
        "max_expensive_loop_iterations": 0,
    })

    assert library.lookup("WORKSPACE_EDGE_GRASP_MISS") == (
        "recover_workspace_edge_reobserve",
    )
    assert library.lookup("WORKSPACE_EDGE_MOVE_RETRY_REQUIRED") == (
        "recover_workspace_edge_move_retry",
    )
    assert library.lookup("GRASP_MISS") == (
        "recover_target_region_regrasp",
    )
    assert library.lookup("UNSTABLE_RELEASE") == (
        "candidate_tight_clearance_regrasp",
    )
    assert graph.actions["recover_workspace_edge_move"].metadata[
        "failure_next_action"
    ] == "recover_workspace_edge_move_retry"
    assert graph.actions["recover_workspace_edge_move_retry"].metadata[
        "max_attempts"
    ] == 1
    assert graph.actions["recover_workspace_edge_move_retry"].metadata[
        "failure_next_action"
    ] == "recover_budget_safe_stop"
    assert graph.actions["recover_workspace_edge_move"].next_action == (
        "recover_workspace_edge_post_move_reobserve"
    )
    assert graph.actions["recover_workspace_edge_move_retry"].next_action == (
        "recover_workspace_edge_post_move_reobserve"
    )
    post_move_reobserve = graph.actions[
        "recover_workspace_edge_post_move_reobserve"
    ]
    assert post_move_reobserve.metadata["observation_refresh"] is True
    assert post_move_reobserve.next_action == "recover_workspace_edge_regrasp"
    post_recovery_reobserve = graph.actions["recover_post_recovery_reobserve"]
    assert post_recovery_reobserve.metadata["observation_refresh"] is True
    assert post_recovery_reobserve.next_action == (
        "candidate_structurally_distinct_regrasp_replace"
    )
    assert graph.actions["recover_target_region_regrasp_retry"].metadata[
        "failure_next_action"
    ] == "recover_post_recovery_reobserve"
    assert graph.actions["candidate_tight_clearance_regrasp"].metadata[
        "failure_next_action"
    ] == "recover_post_recovery_reobserve"
    assert graph.actions["recover_workspace_edge_regrasp"].metadata[
        "failure_next_action"
    ] == "recover_post_recovery_reobserve"
    assert graph.actions["candidate_tight_clearance_replace"].metadata[
        "failure_next_action"
    ] == "recover_tight_clearance_nudge"
    nudge = graph.actions["recover_tight_clearance_nudge"]
    assert nudge.metadata["max_attempts"] == 1
    assert nudge.metadata["interrupt_on_failure"] is True
    assert nudge.metadata["failure_next_action"] == "recover_budget_safe_stop"
    assert 'sample_grasp_pose("primary")' in nudge.code
    assert nudge.next_action == "recover_tight_clearance_replace_retry"
    assert "push_end" not in nudge.code
    assert validate_code_block(nudge.code, guard_config).ok
    replace_retry = graph.actions["recover_tight_clearance_replace_retry"]
    assert "goto_pose(transit, grasp_quat)" in replace_retry.code
    assert 'get_target_region("target")' in replace_retry.code
    assert replace_retry.metadata["failure_next_action"] == (
        "recover_budget_safe_stop"
    )
    assert validate_code_block(replace_retry.code, guard_config).ok
    assert graph.actions["candidate_structurally_distinct_regrasp_replace"].metadata[
        "failure_next_action"
    ] == "recover_budget_safe_stop"
    assert [graph.actions[action_id].next_action for action_id in action_ids] == [
        "recover_workspace_edge_check",
        "recover_workspace_edge_move",
        "recover_workspace_edge_post_move_reobserve",
        "recover_workspace_edge_regrasp",
        "candidate_tight_clearance_replace",
    ]
    for action_id in action_ids:
        action = graph.actions[action_id]
        assert action.metadata["interrupt_on_failure"] is True
        assert validate_code_block(action.code, guard_config).ok


def test_structural_candidate_validation_routes_directly_without_model():
    library = RecoveryLibrary.from_mapping(yaml.safe_load(
        Path(
            "env_configs/region_place/candidate_structural_recovery_library.yaml"
        ).read_text()
    ))

    assert library.lookup("OUTSIDE_TARGET_REGION") == (
        "candidate_structurally_distinct_regrasp_replace",
    )
    assert library.lookup("RECOVERY_BUDGET_LOW") == ("recover_budget_safe_stop",)


def test_formal_ab_improved_route_is_bounded_and_keeps_baseline_configs():
    graph = OfflineTaskGraph.from_mapping(yaml.safe_load(
        Path("env_configs/region_place/offline_task_graph.yaml").read_text()
    ))
    raw_library = yaml.safe_load(Path(
        "env_configs/region_place/formal_ab_improved_recovery_library.yaml"
    ).read_text())
    library = RecoveryLibrary.from_mapping(raw_library)

    validation = raw_library["metadata"]["validation"]
    assert raw_library["metadata"]["status"] == "formal"
    assert validation["formal_promotion"] is True
    assert validation["unseen_successes"] == 9
    assert validation["unseen_trials"] == 10
    assert validation["max_structured_l3_requests_per_trial"] == 1
    assert validation["arbitrary_recovery_code_executed"] == 0
    assert validation["failed_seed_terminal_action"] == (
        "model_fallback_budget_safe_stop"
    )
    assert library.lookup("OUTSIDE_TARGET_REGION") == (
        "recover_target_region_regrasp",
        "recover_target_region_replace_then_structural",
    )
    assert library.lookup("WORKSPACE_EDGE_GRASP_MISS") == (
        "recover_workspace_edge_reobserve",
    )
    final_replace = graph.actions["recover_target_region_replace_then_structural"]
    assert final_replace.metadata["failure_next_action"] == (
        "recover_post_recovery_reobserve"
    )
    assert final_replace.metadata["max_attempts"] == 1


def test_formal_ab_improved_configs_preserve_one_request_budget(monkeypatch):
    monkeypatch.setenv("REGION_PLACE_OUTPUT_DIR", "outputs/test-formal-improved")
    loaded = {}
    for level in ("l1", "l2"):
        path = f"env_configs/region_place/formal_ab_{level}_improved.yaml"
        _, config, _ = _load_config(LaunchArgs(config_path=path))
        loaded[level] = config
        assert config["hybrid_recovery"]["recovery_library"].endswith(
            "formal_ab_improved_recovery_library.yaml"
        )

    assert loaded["l1"]["hybrid_recovery"]["max_model_fallbacks_total"] == 0
    assert loaded["l2"]["hybrid_recovery"]["max_model_fallbacks_total"] == 1
    assert loaded["l2"]["hybrid_recovery"][
        "model_recovery_candidate_limit"
    ] == 2


def test_structural_candidate_infeasible_state_executes_only_safe_stop():
    loaded = OfflineTaskGraph.from_mapping(yaml.safe_load(
        Path("env_configs/region_place/offline_task_graph.yaml").read_text()
    ))
    graph = OfflineTaskGraph(
        loaded.actions, ("candidate_structurally_distinct_regrasp_replace",)
    )
    library = RecoveryLibrary.from_mapping(yaml.safe_load(
        Path(
            "env_configs/region_place/candidate_structural_recovery_library.yaml"
        ).read_text()
    ))
    executed = []

    def execute(action):
        executed.append(action.action_id)
        state = {"gripper_open": True, "task_completed": False}
        return OfflineExecution(state=state, state_window=(state, state, state))

    result = OfflineWindowController(graph, library).run(execute, initial_state={
        "object_observation_ready": True,
        "target_observation_ready": True,
        "workspace_reachable": False,
        "target_region_policy_eligible": True,
        "target_region_clear": True,
    })

    assert executed == ["recover_budget_safe_stop"]
    assert result.status == "blocked"


def test_infeasible_region_scenarios_are_deterministic_and_non_feasible():
    failures = (
        "TARGET_REGION_TOO_SMALL",
        "OBJECT_UNREACHABLE",
        "OBSERVATION_FAILURE",
        "TARGET_REGION_BLOCKED",
    )
    scenarios = [infeasible_region_place_scenario("cube", 301, item) for item in failures]

    assert [item["infeasible_scenario"] for item in scenarios] == list(failures)
    assert all(item["expected_geometry_feasible"] is False for item in scenarios)
    assert scenarios[0]["target_inner_extent"] == (0.03, 0.03, 0.0)


def test_nominal_region_grasp_precondition_failure_routes_to_safe_stop():
    graph = OfflineTaskGraph.from_mapping(yaml.safe_load(
        Path("env_configs/region_place/offline_task_graph.yaml").read_text()
    ))

    assert graph.actions["stage_001"].metadata[
        "precondition_failure_next_action"
    ] == "recover_budget_safe_stop"


def test_infeasible_task_safe_stop_is_formal_and_model_free():
    graph_data = yaml.safe_load(
        Path("env_configs/region_place/offline_task_graph.yaml").read_text()
    )
    graph = OfflineTaskGraph.from_mapping(graph_data)
    library = RecoveryLibrary.from_mapping(yaml.safe_load(
        Path("env_configs/region_place/recovery_library.yaml").read_text()
    ))
    safe_stop = graph.actions["recover_budget_safe_stop"]

    assert safe_stop.metadata["status"] == "formal"
    assert safe_stop.metadata["strategy_id"] == "INFEASIBLE_TASK_SAFE_STOP"
    assert safe_stop.metadata["scope"] == "placement_task_preflight"
    assert safe_stop.metadata["validation"]["classification_correct"] == 55
    assert safe_stop.metadata["validation"]["classification_trials"] == 55
    assert safe_stop.metadata["validation"]["model_requests"] == 0
    assert library.lookup("INFEASIBLE_TASK") == ("recover_budget_safe_stop",)


def test_infeasible_safe_stop_config_disables_model_fallbacks(monkeypatch):
    monkeypatch.setenv("REGION_PLACE_INFEASIBLE_SCENARIO", "TARGET_REGION_TOO_SMALL")
    monkeypatch.setenv(
        "REGION_PLACE_OUTPUT_DIR", "/tmp/region-place-infeasible-test"
    )
    _, config, _ = _load_config(LaunchArgs(
        config_path="env_configs/region_place/infeasible_safe_stop_3seed.yaml",
    ))

    assert config["hybrid_recovery"]["max_model_fallbacks_total"] == 0
    assert config["hybrid_recovery"]["stage_actions"][0] == "stage_001"


def test_nominal_precondition_safe_stop_precedes_generic_recovery_dispatch():
    source = Path("capx/envs/trial.py").read_text()

    assert "and policy_failure_next_action is None" in source
    assert '"hybrid_terminal_safe_stop": bool(' in source
    assert 'policy_action.metadata.get("terminal") == "safe_stop"' in source


def test_target_region_completion_requires_full_footprint_and_surface_contact():
    upright = np.array([1.0, 0.0, 0.0, 0.0])
    half_sizes = np.array([0.02, 0.02, 0.02])
    center = np.zeros(3)
    extent = np.array([0.12, 0.10, 0.0])

    assert box_inside_target_region(
        np.array([0.0, 0.0, 0.02]), upright, half_sizes, center, extent
    )
    assert not box_inside_target_region(
        np.array([0.05, 0.0, 0.02]), upright, half_sizes, center, extent
    )
    assert not box_inside_target_region(
        np.array([0.0, 0.0, 0.05]), upright, half_sizes, center, extent
    )


def test_region_scenarios_remove_container_height_and_rename_fault():
    scenario = region_place_scenario("cube", 4)

    assert scenario["target_inner_extent"][2] == 0.0
    assert scenario["rim_height"] == 0.0
    assert scenario["injected_exception"] == "OUTSIDE_TARGET_REGION"
    assert scenario["scenario_id"].startswith("region_cube")


def test_region_place_l2_fallback_is_bounded_after_two_policy_grasps(monkeypatch):
    monkeypatch.setenv("REGION_PLACE_OUTPUT_DIR", "outputs/test-region-l2")
    _, config, _ = _load_config(LaunchArgs(
        config_path="env_configs/region_place/l2_fallback_proof.yaml"
    ))
    hybrid = config["hybrid_recovery"]

    assert hybrid["policy_first"] is True
    assert hybrid["max_model_fallbacks_per_exception"] == 1
    assert hybrid["max_model_fallbacks_total"] == 1
    assert hybrid["model_recovery_template"] == "bin_place_regrasp_replace_v1"


def test_region_place_l3_replay_keeps_one_model_request_budget(monkeypatch):
    monkeypatch.setenv("REGION_PLACE_OUTPUT_DIR", "outputs/test-region-l3")
    _, config, _ = _load_config(LaunchArgs(
        config_path="env_configs/region_place/l3_fallback_replay_seed23.yaml"
    ))
    hybrid = config["hybrid_recovery"]

    assert hybrid["policy_first"] is True
    assert hybrid["max_model_fallbacks_per_exception"] == 1
    assert hybrid["max_model_fallbacks_total"] == 1
    assert "model_recovery_template" not in hybrid


def test_l3_model_led_mode_uses_model_plan_and_policy_only_for_recovery(monkeypatch):
    monkeypatch.setenv("REGION_PLACE_OUTPUT_DIR", "outputs/test-region-l3-model-led")
    _, config, _ = _load_config(LaunchArgs(
        config_path=(
            "env_configs/region_place/l3_model_led_policy_recovery.yaml"
        )
    ))
    hybrid = config["hybrid_recovery"]

    assert config["use_oracle_code"] is False
    assert hybrid["policy_first"] is False
    assert hybrid["model_led_policy_recovery"] is True
    assert hybrid["max_model_fallbacks_per_exception"] == 1
    assert hybrid["max_model_fallbacks_total"] == 1
    assert hybrid["model_recovery_stage_indices"] == [0, 1]
    assert hybrid["model_recovery_template"] == "bin_place_regrasp_replace_v1"
    assert hybrid["model_recovery_candidate_limit"] == 1
    assert hybrid["recovery_library"].endswith(
        "formal_ab_improved_recovery_library.yaml"
    )
    raw = yaml.safe_load(Path(
        "env_configs/region_place/l3_model_led_policy_recovery.yaml"
    ).read_text())
    assert "separate structured-object" in raw["env"]["cfg"][
        "multi_turn_prompt"
    ]
    assert raw["metadata"]["frozen_validation_seeds"] == [23, 26, 27, 29, 31]


def test_structural_failure_proof_configs_are_paired_and_single_request(monkeypatch):
    monkeypatch.setenv("REGION_PLACE_OUTPUT_DIR", "outputs/test-structural-fallback")
    loaded = {}
    for level in ("l2", "l3"):
        config_path = Path(
            f"env_configs/region_place/{level}_structural_failure_proof_5seed.yaml"
        )
        raw = yaml.safe_load(config_path.read_text())
        _, config, _ = _load_config(LaunchArgs(config_path=str(config_path)))
        loaded[level] = config
        low_level = raw["env"]["cfg"]["low_level_kwargs"]
        hybrid = config["hybrid_recovery"]
        assert low_level["inject_primary_release_failures"] == 2
        assert hybrid["max_model_fallbacks_per_exception"] == 1
        assert hybrid["max_model_fallbacks_total"] == 1
        assert hybrid["recovery_library"].endswith(
            "candidate_structural_recovery_library.yaml"
        )
        assert raw["metadata"]["seeds"] == [201, 202, 203, 204, 205]

    assert loaded["l2"]["hybrid_recovery"]["model_recovery_template"] == (
        "bin_place_regrasp_replace_v1"
    )
    assert "model_recovery_template" not in loaded["l3"]["hybrid_recovery"]
