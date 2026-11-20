from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from capx.envs.simulators.robosuite_bin_place import (
    FrankaRobosuiteBinPlaceLowLevel,
    _as_bool,
    bin_place_scenario,
    box_inside_open_bin,
    fit_planar_rectangle,
    largest_connected_component,
    planar_footprint_extent,
    workspace_edge_scenario,
)
from capx.integrations.franka.control import FrankaControlApi
from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi
from capx.integrations.vision.sam3 import select_sam3_result
from capx.recovery.offline import OfflineTaskGraph, RecoveryLibrary, attach_action_library


def test_planar_policy_footprint_ignores_temporary_grasp_tilt():
    half_sizes = np.array([0.02, 0.018, 0.018])
    target_quat = np.array([0.9961946981, 0.0, 0.0, 0.0871557427])
    flat = np.array([0.9455185756, 0.0, 0.0, -0.3255681545])
    tilted = np.array([0.9571146729, 0.0007266331, -0.0008006688, -0.2897073246])

    max_yaw_extent = 2.0 * np.hypot(*half_sizes[:2])
    assert np.all(
        planar_footprint_extent(flat, target_quat, half_sizes)
        <= max_yaw_extent + 1e-9
    )
    assert np.all(
        planar_footprint_extent(tilted, target_quat, half_sizes)
        <= max_yaw_extent + 1e-9
    )


def test_planar_rectangle_recovers_center_extent_and_yaw():
    yaw = np.deg2rad(27.0)
    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)],
    ])
    local = np.array(np.meshgrid(
        np.linspace(-0.09, 0.09, 30),
        np.linspace(-0.055, 0.055, 24),
    )).reshape(2, -1).T
    points = local @ rotation.T + np.array([0.51, 0.24])

    center, extent, measured_yaw = fit_planar_rectangle(points)

    assert np.allclose(center, [0.51, 0.24], atol=0.002)
    assert np.allclose(np.sort(extent), [0.11, 0.18], atol=0.01)
    assert abs(np.sin(measured_yaw - yaw)) < 0.07


def test_largest_connected_component_rejects_color_noise():
    mask = np.zeros((12, 12), dtype=bool)
    mask[2:8, 3:10] = True
    mask[0, 0] = True
    mask[10:12, 0:2] = True

    selected = largest_connected_component(mask)

    assert int(selected.sum()) == 42
    assert selected[4, 5]
    assert not selected[0, 0]


def test_workspace_edge_scenarios_span_sides_positions_orientations_and_sizes():
    scenarios = [workspace_edge_scenario(seed) for seed in range(1, 11)]

    assert len({item["scenario_id"] for item in scenarios}) == 10
    assert {np.sign(item["object_xy"][1]) for item in scenarios} == {-1, 1}
    assert len({item["object_xy"][0] for item in scenarios}) >= 5
    assert len({item["object_yaw_degrees"] for item in scenarios}) == 10
    assert len({item["object_half_sizes"] for item in scenarios}) >= 5
    assert all(item["expected_geometry_feasible"] for item in scenarios)


def test_workspace_edge_holdouts_change_object_and_target_geometry():
    baseline = [workspace_edge_scenario(seed) for seed in range(1, 11)]
    holdouts = [workspace_edge_scenario(seed) for seed in range(11, 21)]

    assert not ({item["scenario_id"] for item in baseline} & {
        item["scenario_id"] for item in holdouts
    })
    assert all(
        holdout["object_xy"] != base["object_xy"]
        and holdout["object_yaw_degrees"] != base["object_yaw_degrees"]
        and holdout["bin_offset_xy"] != base["bin_offset_xy"]
        and holdout["bin_yaw_degrees"] != base["bin_yaw_degrees"]
        for base, holdout in zip(baseline, holdouts)
    )
    assert all(-0.145 <= item["object_xy"][1] <= -0.130 for item in holdouts)


def test_rotated_tight_region_uses_projected_object_footprint():
    scenario = workspace_edge_scenario(17)
    half_x, half_y = scenario["object_half_sizes"][:2]
    angle = np.deg2rad(
        scenario["object_yaw_degrees"] - scenario["bin_yaw_degrees"]
    )
    projected_x = 2.0 * (
        abs(np.cos(angle)) * half_x + abs(np.sin(angle)) * half_y
    )
    projected_y = 2.0 * (
        abs(np.sin(angle)) * half_x + abs(np.cos(angle)) * half_y
    )

    assert projected_x > 2.0 * max(half_x, half_y) or projected_y > 2.0 * max(
        half_x, half_y
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("false", False), ("0", False), ("true", True), ("1", True), (False, False)],
)
def test_bin_place_boolean_environment_values_are_parsed(value, expected):
    assert _as_bool(value) is expected


def test_bin_occlusion_mask_changes_only_target_region():
    frame = np.full((60, 100, 3), 255, dtype=np.uint8)
    masked = FrankaRobosuiteBinPlaceLowLevel._mask_bin_view(frame)

    assert np.array_equal(masked[:20, :50], frame[:20, :50])
    assert np.all(masked[20:50, 50:90] == 12)
    assert FrankaRobosuiteBinPlaceLowLevel._TARGET_CHANGE_SHIFT > 0.12


def test_box_inside_open_bin_requires_full_containment():
    floor_center = np.array([0.0, 0.0, 0.0])
    inner_extent = np.array([0.18, 0.22, 0.08])
    half_sizes = np.array([0.02, 0.02, 0.02])
    upright = np.array([1.0, 0.0, 0.0, 0.0])

    assert box_inside_open_bin(
        np.array([0.0, 0.0, 0.02]), upright, half_sizes,
        floor_center, inner_extent,
    )
    assert not box_inside_open_bin(
        np.array([0.08, 0.0, 0.02]), upright, half_sizes,
        floor_center, inner_extent,
    )
    assert not box_inside_open_bin(
        np.array([0.0, 0.0, 0.075]), upright, half_sizes,
        floor_center, inner_extent,
    )


def test_box_inside_rotated_bin_uses_container_frame():
    yaw_90 = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    world_outside = np.array([0.0, 0.06, 0.02])

    assert not box_inside_open_bin(
        world_outside,
        yaw_90,
        np.array([0.02, 0.02, 0.02]),
        np.zeros(3),
        np.array([0.10, 0.20, 0.08]),
        container_quaternion_wxyz=yaw_90,
        margin=0.0,
        tolerance=0.0,
    )


def test_geometry_sweep_defines_30_combinations_and_known_faults():
    scenarios = [
        bin_place_scenario(shape, seed)
        for shape in ("cube", "cuboid", "cylinder")
        for seed in range(1, 11)
    ]

    assert len({item["scenario_id"] for item in scenarios}) == 30
    assert sum(item["expected_geometry_feasible"] for item in scenarios) == 27
    assert sum(item["injected_exception"] == "OUTSIDE_CONTAINER" for item in scenarios) == 3
    assert sum(item["injected_exception"] == "UNSTABLE_RELEASE" for item in scenarios) == 3
    assert FrankaRobosuiteBinPlaceLowLevel._SWEEP_CAMERA_FOVY == 70.0


def test_geometry_holdout_seeds_are_deterministic_new_and_feasible():
    holdout = [bin_place_scenario("cube", seed) for seed in range(11, 16)]

    assert holdout == [bin_place_scenario("cube", seed) for seed in range(11, 16)]
    assert len({item["scenario_id"] for item in holdout}) == 5
    assert all("holdout" in item["scenario_id"] for item in holdout)
    assert len({item["object_xy"] for item in holdout}) == 5
    assert len({item["object_half_sizes"] for item in holdout}) == 5
    assert len({item["bin_yaw_degrees"] for item in holdout}) == 5
    assert all(item["expected_geometry_feasible"] for item in holdout)


def test_bin_place_graph_and_recovery_are_wired():
    graph = OfflineTaskGraph.from_mapping(yaml.safe_load(
        Path("env_configs/bin_place/offline_task_graph.yaml").read_text()
    ))
    graph = attach_action_library(graph, yaml.safe_load(
        Path("env_configs/recovery/grasp_miss_actions.yaml").read_text()
    ))
    library_value = yaml.safe_load(
        Path("env_configs/bin_place/recovery_library.yaml").read_text()
    )
    library = RecoveryLibrary.from_mapping(library_value)

    assert graph.initial == ("stage_001",)
    assert graph.actions["recover_bin_center_place"].metadata["status"] == "formal"
    assert graph.actions["stage_002"].success_conditions["object_inside_container"] is True
    assert library.lookup("GRASP_MISS") == ("recover_bin_grasp_replan",)
    assert library.lookup("WORKSPACE_EDGE_GRASP_MISS") == (
        "reobserve_workspace_edge_primary",
    )
    assert graph.actions["reobserve_workspace_edge_primary"].next_action == (
        "recenter_workspace_edge_primary"
    )
    assert graph.actions["recenter_workspace_edge_primary"].next_action == (
        "recover_bin_center_place"
    )
    assert graph.actions["reobserve_workspace_edge_primary"].metadata["interrupt_on_failure"] is True
    assert graph.actions["reobserve_workspace_edge_primary"].metadata["status"] == "formal"
    assert graph.actions["recenter_workspace_edge_primary"].metadata["status"] == "formal"
    assert library_value["metadata"]["validation"]["workspace_edge_successes"] == 10
    assert graph.actions["recenter_workspace_edge_primary"].metadata["interrupt_on_failure"] is True
    assert library.lookup("TARGET_POSITION_CHANGED") == (
        "candidate_reobserve_changed_target",
    )
    assert library.lookup("OBJECT_OUT_OF_WORKSPACE") == (
        "candidate_reobserve_out_of_workspace",
    )
    assert graph.actions["candidate_reobserve_changed_target"].metadata["status"] == "formal"
    assert graph.actions["candidate_move_primary_into_workspace"].metadata["status"] == "formal"
    assert graph.actions["candidate_check_workspace_recovery"].preconditions == {
        "workspace_recovery_feasible": True,
        "object_stable": True,
    }
    assert graph.actions["candidate_move_primary_into_workspace"].success_conditions[
        "object_inside_nominal_workspace"
    ] is True
    assert graph.actions["candidate_move_primary_into_workspace"].next_action == (
        "candidate_regrasp_workspace_primary"
    )
    assert "push_distance" in graph.actions["candidate_move_primary_into_workspace"].code
    assert "safe_grasp_pos" not in graph.actions["candidate_move_primary_into_workspace"].code
    assert graph.actions["candidate_regrasp_workspace_primary"].next_action == (
        "recover_bin_center_replace"
    )
    assert "workspace_was_pushed" in graph.actions["candidate_regrasp_workspace_primary"].code
    assert graph.actions["recover_grasp_miss"].metadata["status"] == "cross_task_validated"
    assert library.lookup("OUTSIDE_CONTAINER") == ("recover_bin_center_place",)
    assert library.lookup("OUTSIDE_TARGET_REGION") == ("recover_bin_center_place",)
    assert library.lookup(
        "OUTSIDE_CONTAINER", {"target_geometry_type": "container"}
    ) == ("recover_bin_center_place",)
    assert graph.actions["stage_002"].failure_routes["POSTCONDITION_FAILED"] == (
        "OUTSIDE_TARGET_REGION"
    )
    assert library.lookup("GRASP_RECOVERY_FAILED") == ("recover_bin_grasp_replan",)
    assert library.lookup("BIN_PLACE_RECOVERY_FAILED") == ("recover_bin_grasp_replan",)
    assert "sample_grasp_pose" in graph.actions["recover_bin_grasp_replan"].code
    assert "target_pose" not in graph.actions["stage_002"].code
    assert "sample_grasp_pose" not in graph.actions["stage_001"].code
    assert "sample_grasp_pose" not in graph.actions["recover_bin_center_place"].code
    assert graph.actions["recover_bin_center_place"].next_action == "recover_bin_center_replace"
    assert graph.actions["recover_bin_center_place"].metadata["interrupt_on_failure"] is True
    assert "get_target_region" not in graph.actions["recover_bin_center_place"].code
    assert graph.actions["recover_bin_center_place"].code.count("open_gripper()") == 1
    assert graph.actions["recover_bin_center_replace"].code.count("open_gripper()") == 1
    assert graph.actions["recover_bin_center_replace"].code.rstrip().endswith("open_gripper()")
    assert graph.actions["stage_002"].metadata["accept_semantic_stability"] is True
    assert graph.actions["stage_002"].code.count("open_gripper()") == 2
    assert graph.actions["stage_001"].metadata["accept_semantic_success"] is True
    assert graph.actions["recover_bin_grasp_replan"].metadata["accept_semantic_success"] is True
    assert graph.actions["stage_001"].args["primary_pose"] == "cube_poses_robot_base.primary"
    assert graph.actions["recover_bin_grasp_replan"].args["primary_extent"] == "cube_extents_robot_base.primary"
    assert library.lookup("BIN_ENTRY_BLOCKED") is None
    assert yaml.safe_load(
        Path("env_configs/bin_place/recovery_library.yaml").read_text()
    )["metadata"]["validation"]["formal_promotion"] is True


def test_privileged_api_exposes_home_pose_needed_by_shared_recovery():
    api = object.__new__(FrankaControlPrivilegedApi)

    assert api.functions()["home_pose"].__func__ is FrankaControlPrivilegedApi.home_pose


def test_target_change_is_armed_after_stale_region_query_and_applied_before_motion():
    env = SimpleNamespace(
        _active_exception_id="TARGET_POSITION_CHANGED",
        _unknown_anomaly_applied=False,
        _target_change_armed=False,
        _blocked_stale_goto_attempts=0,
    )
    env._primary_is_grasped = lambda: True
    applied = []
    env._apply_unknown_anomaly = lambda: applied.append(True)

    FrankaRobosuiteBinPlaceLowLevel.after_target_region_query(env)
    assert FrankaRobosuiteBinPlaceLowLevel.before_goto_pose(env) is True

    assert applied == [True]
    assert env._target_change_armed is False


def test_target_change_cancels_only_the_stale_motion_block():
    env = SimpleNamespace(_target_change_armed=False, _blocked_stale_goto_attempts=2)
    env.record_world_state = lambda *args, **kwargs: None

    assert FrankaRobosuiteBinPlaceLowLevel.before_goto_pose(env) is False
    assert FrankaRobosuiteBinPlaceLowLevel.before_goto_pose(env) is False
    assert FrankaRobosuiteBinPlaceLowLevel.before_goto_pose(env) is True


def test_camera_world_state_uses_only_visual_object_and_target_measurements():
    env = object.__new__(FrankaRobosuiteBinPlaceLowLevel)
    env.cube_half_sizes = (0.02, 0.02, 0.02)
    env._calibrated_target_center = np.array([0.50, 0.25, -0.10])
    env._calibrated_target_extent = np.array([0.18, 0.22, 0.08])
    env.world_state_source = "camera_rgbd"
    env.target_mode = "region"
    env._camera_state_positions = []
    env._camera_state_last_measurement = None
    env._camera_target_last_measurement = None
    env._gripper_fraction = 1.0
    env._active_exception_id = None
    env._unknown_anomaly_applied = False
    env._measure_primary_from_camera = lambda: (
        np.array([0.50, 0.25, -0.08]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.04, 0.04, 0.04]),
        0.91,
    )
    env._measure_target_from_camera = lambda: (
        np.array([0.50, 0.25, -0.10]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.18, 0.22, 0.0]),
        0.89,
    )
    env._primary_pose_robot_base = lambda: pytest.fail("simulator pose leaked")
    env._primary_is_grasped = lambda: pytest.fail("simulator grasp leaked")
    env._object_inside_container = lambda: pytest.fail("simulator containment leaked")
    env._object_stable = lambda: pytest.fail("simulator velocity leaked")

    state = env._get_camera_text_world_state()

    assert state["world_state_source"] == "camera_rgbd"
    assert state["uses_simulator_truth"] is False
    assert state["state_provenance"]["object_pose"] == "sam3_rgbd"
    assert state["state_provenance"]["target_region"] == "sam3_rgbd"
    assert state["object_inside_container"] is True
    assert state["task_completed"] is True
    assert state["target_observation_ready"] is True
    assert state["workspace_recovery_feasible"] is True

    env._active_exception_id = "TARGET_POSITION_CHANGED"
    env._unknown_anomaly_applied = True
    changed_target_state = env._get_camera_text_world_state()
    assert changed_target_state["active_exception_id"] == "TARGET_POSITION_CHANGED"
    assert changed_target_state["target_observation_ready"] is True

    env._measure_primary_from_camera = lambda: (
        np.array([0.51, 0.25, -0.08]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.04, 0.04, 0.04]),
        0.91,
    )
    moving_state = env._get_camera_text_world_state()

    assert moving_state["object_stable"] is False
    assert moving_state["task_completed"] is False


def test_camera_world_state_uses_stale_visual_cache_fail_closed():
    env = object.__new__(FrankaRobosuiteBinPlaceLowLevel)
    env.cube_half_sizes = (0.02, 0.02, 0.02)
    env._calibrated_target_center = np.array([0.50, 0.25, -0.10])
    env._calibrated_target_extent = np.array([0.18, 0.22, 0.08])
    env.world_state_source = "camera_rgbd"
    env.target_mode = "region"
    env._camera_state_positions = []
    env._camera_state_last_measurement = (
        np.array([0.50, 0.25, -0.08]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.04, 0.04, 0.04]),
        0.91,
    )
    env._camera_target_last_measurement = (
        np.array([0.50, 0.25, -0.10]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.18, 0.22, 0.0]),
        0.89,
    )
    env._gripper_fraction = 1.0
    env._measure_primary_from_camera = lambda: (_ for _ in ()).throw(
        RuntimeError("temporarily occluded")
    )
    env._measure_target_from_camera = lambda: (
        np.array([0.50, 0.25, -0.10]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.18, 0.22, 0.0]),
        0.89,
    )

    state = env._get_camera_text_world_state()

    assert state["measurement_fresh"] is False
    assert state["object_stable"] is False
    assert state["task_completed"] is False
    assert state["confidence"] == 0.0


def test_camera_world_state_tracks_grasped_object_from_robot_proprioception():
    env = object.__new__(FrankaRobosuiteBinPlaceLowLevel)
    env.cube_half_sizes = (0.02, 0.02, 0.02)
    env.world_state_source = "camera_rgbd"
    env.target_mode = "region"
    env._camera_state_positions = []
    env._camera_state_last_measurement = (
        np.array([0.60, -0.10, -0.08]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.04, 0.04, 0.04]),
        0.91,
    )
    env._camera_target_last_measurement = None
    env._gripper_fraction = 0.0
    env._active_exception_id = None
    env._unknown_anomaly_applied = False
    env._measure_primary_from_camera = lambda: (_ for _ in ()).throw(
        RuntimeError("object occluded by gripper")
    )
    env._measure_target_from_camera = lambda: (
        np.array([0.62, 0.40, -0.10]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.16, 0.22, 0.0]),
        0.89,
    )
    env._gripper_tool_position_robot_base = lambda: np.array([0.60, -0.10, 0.12])
    env._actual_gripper_opening_fraction = lambda: 0.55

    state = env._get_camera_text_world_state()

    assert state["state_provenance"]["object_pose"] == (
        "robot_proprioceptive_grasp_track"
    )
    assert state["object_observation_ready"] is True
    assert state["target_grasped"] is True
    assert state["object_lifted"] is True
    assert state["task_completed"] is False


def test_cylinder_stability_uses_surface_speed_while_box_uses_angular_speed():
    env = object.__new__(FrankaRobosuiteBinPlaceLowLevel)
    env.primary_body_id = 0
    env.cube_half_sizes = np.array([0.02, 0.02, 0.025])
    env._unstable_release_fault_active = False
    env.robosuite_env = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(
        cvel=np.array([[0.42, 0.0, 0.0, 0.01, 0.0, 0.0]])
    )))

    env.object_shape = "cylinder"
    assert env._object_stable() is True

    env.robosuite_env.sim.data.cvel[0, 0] = 1.3
    assert env._object_stable() is False

    env.object_shape = "cube"
    assert env._object_stable() is False


def test_sam3_selection_rejects_scene_sized_false_positive():
    scene = {"mask": np.ones((100, 100), dtype=bool), "score": 0.99}
    object_mask = np.zeros((100, 100), dtype=bool)
    object_mask[40:50, 40:50] = True
    expected = {"mask": object_mask, "score": 0.80}

    assert select_sam3_result([scene, expected], (100, 100)) is expected


def test_franka_sam3_selection_rejects_rgbd_extent_outlier():
    api = object.__new__(FrankaControlApi)
    api._env = SimpleNamespace(cube_half_sizes=np.array([0.02, 0.02, 0.02]))
    depth = np.ones((100, 100), dtype=float)
    intrinsics = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    wide_mask = np.zeros((100, 100), dtype=bool)
    wide_mask[40:45, 30:70] = True
    object_mask = np.zeros((100, 100), dtype=bool)
    object_mask[40:45, 48:53] = True
    expected = {"mask": object_mask, "score": 0.8}

    selected = api._select_plausible_sam3_result(
        [{"mask": wide_mask, "score": 0.99}, expected],
        depth.shape,
        depth,
        intrinsics,
    )

    assert selected is expected
    graph = OfflineTaskGraph.from_mapping(yaml.safe_load(
        Path("env_configs/bin_place/offline_task_graph.yaml").read_text()
    ))
    assert graph.actions["recover_bin_center_place"].code.count("open_gripper()") == 1
    assert graph.actions["recover_bin_center_replace"].code.count("open_gripper()") == 1


def test_visual_control_reads_calibrated_target_region_not_recorder_truth():
    api = object.__new__(FrankaControlApi)
    api._env = SimpleNamespace(
        get_target_region_state=lambda: {
            "target_object_center_robot_base": [0.5, 0.25, -0.08],
            "target_region_extent_robot_base": [0.18, 0.22, 0.08],
            "container_rim_z_robot_base": -0.02,
        },
        get_task_recorder_state=lambda: pytest.fail("recorder truth leaked"),
    )

    center, extent, rim_z = api.get_target_region("target")

    assert np.allclose(center, [0.5, 0.25, -0.08])
    assert np.allclose(extent, [0.18, 0.22, 0.08])
    assert rim_z == pytest.approx(-0.02)


def test_bin_release_fault_offsets_primary_and_marks_only_first_unstable(monkeypatch):
    env = object.__new__(FrankaRobosuiteBinPlaceLowLevel)
    env.inject_primary_release_failures = 2
    env.inject_first_primary_release_unstable = True
    env.primary_release_fault_offset = 0.10
    env._primary_release_faults_injected = 0
    env._pending_primary_release_fault_steps = None
    env._unstable_release_fault_active = False
    env._gripper_fraction = 0.0
    grasped = [True]
    env._primary_is_grasped = lambda: grasped[0]
    env._primary_over_target_xy = lambda: True
    events = []
    env.record_world_state = lambda event, **kwargs: events.append((event, kwargs))

    class _Data:
        qpos = np.array([0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])

        def get_joint_qpos(self, _name):
            return self.qpos.copy()

        def set_joint_qpos(self, _name, value):
            self.qpos = np.asarray(value)

    data = _Data()
    env.primary_object = SimpleNamespace(joints=["primary_joint"])
    env.robosuite_env = SimpleNamespace(
        sim=SimpleNamespace(data=data, forward=lambda: None)
    )
    monkeypatch.setattr(
        RobosuiteBaseEnv, "_set_gripper",
        lambda self, value: setattr(self, "_gripper_fraction", value),
    )
    monkeypatch.setattr(RobosuiteBaseEnv, "_step_once", lambda _self: None)

    env._set_gripper(1.0)
    for _ in range(10):
        env._step_once()

    assert data.qpos[0] == pytest.approx(0.60)
    assert env._unstable_release_fault_active is True
    assert events[-1][1]["metadata"]["exception_id"] == "UNSTABLE_RELEASE"

    grasped[0] = False
    env._set_gripper(1.0)
    assert env._unstable_release_fault_active is True
    env._set_gripper(0.0)
    env._set_gripper(1.0)
    assert env._unstable_release_fault_active is False

    grasped[0] = True
    env._set_gripper(0.0)
    env._set_gripper(1.0)
    for _ in range(10):
        env._step_once()

    assert data.qpos[0] == pytest.approx(0.70)
    assert env._unstable_release_fault_active is False
    assert events[-1][1]["metadata"]["exception_id"] == "OUTSIDE_CONTAINER"
