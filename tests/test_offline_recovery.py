from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from robosuite.models.objects import BoxObject, CylinderObject

from capx.envs.simulators.robosuite_cubes_restack import (
    FrankaRobosuiteCubesRestackLowLevel,
    cuboid_policy_geometry,
)
from capx.envs.simulators.robosuite_cube_lift import FrankaRobosuiteCubeLiftLowLevel
from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from capx.envs.tasks.franka.franka_cube_restack import ORACLE_CODE, PROMPT
from capx.envs.trial import (
    _completed_before_termination,
    _is_completed_terminal_result,
    _policy_task_completed,
)
from capx.integrations.franka.control import FrankaControlApi
from capx.recovery.offline import (
    OfflineAction,
    OfflineExecution,
    OfflineTaskGraph,
    OfflineWindowController,
    RecoveryLibrary,
    StateParameterError,
    attach_action_library,
    bind_state_parameters,
    prepend_failure_trigger,
    select_hybrid_recovery,
)


def _graph():
    return OfflineTaskGraph.from_mapping({
        "initial": ["a1"],
        "actions": [
            {
                "id": "a1", "primitive": "prepare", "code": "prepare()",
                "success_conditions": {"prepared": True}, "next": "a2",
                "failure_routes": {"POSTCONDITION_FAILED": "PREPARE_FAILED"},
            },
            {
                "id": "a2", "primitive": "finish", "code": "finish()",
                "preconditions": {"prepared": True},
                "success_conditions": {"task_completed": True},
            },
            {
                "id": "r1", "primitive": "reset", "code": "reset()",
                "success_conditions": {"prepared": True}, "next": "a2",
            },
        ],
    })


def test_nonprivileged_policy_completion_ignores_evaluator_truth():
    assert _policy_task_completed(
        {"uses_simulator_truth": False, "task_completed": False}, True
    ) is False
    assert _policy_task_completed(
        {"uses_simulator_truth": False, "task_completed": True}, False
    ) is True
    assert _policy_task_completed({"task_completed": False}, True) is True


def test_completion_is_reconciled_only_for_terminated_episode():
    env = SimpleNamespace(low_level_env=SimpleNamespace(task_completed=lambda: True))

    assert _completed_before_termination(
        env, ValueError("executing action in terminated episode"), False
    ) is True
    assert _completed_before_termination(env, ValueError("other failure"), True) is False
    assert _is_completed_terminal_result(
        True, False, "ValueError: executing action in terminated episode"
    ) is True
    assert _is_completed_terminal_result(False, True, "") is False


def test_offline_controller_slides_and_completes_without_model():
    calls = []

    def execute(action):
        calls.append(action.action_id)
        if action.action_id == "a1":
            state = {"prepared": True, "task_completed": False}
        else:
            state = {"prepared": True, "task_completed": True}
        return OfflineExecution(state=state, state_window=(state, state, state), task_completed=state["task_completed"])

    result = OfflineWindowController(
        _graph(), RecoveryLibrary.from_mapping({})
    ).run(execute)

    assert result.status == "success"
    assert calls == ["a1", "a2"]
    assert result.executed_actions == ("a1", "a2")


def test_offline_controller_switches_to_predeclared_recovery_route():
    calls = []

    def execute(action):
        calls.append(action.action_id)
        if action.action_id == "a1":
            state = {"prepared": False, "task_completed": False}
        elif action.action_id == "r1":
            state = {"prepared": True, "task_completed": False}
        else:
            state = {"prepared": True, "task_completed": True}
        return OfflineExecution(state=state, state_window=(state, state, state), task_completed=state["task_completed"])

    # A1 has no precondition, so its postcondition failure selects the named route.
    result = OfflineWindowController(
        _graph(), RecoveryLibrary.from_mapping({"PREPARE_FAILED": ["r1"]})
    ).run(execute)

    assert result.status == "success"
    assert calls == ["a1", "r1", "a2"]
    assert any(event.event_type == "recovery.selected" for event in result.events)


def test_replay_trigger_keeps_declared_exception_when_state_window_is_uncertain():
    graph = OfflineTaskGraph.from_mapping({
        "initial": ["replay"],
        "actions": [
            {
                "id": "replay", "primitive": "replay_failure_trigger",
                "code": "pass", "success_conditions": {"grasped": True},
                "metadata": {"exception_id": "GRASP_MISS"},
            },
            {
                "id": "recover", "primitive": "grasp", "code": "recover()",
                "success_conditions": {"task_completed": True},
            },
        ],
    })
    calls = []

    def execute(action):
        calls.append(action.action_id)
        if action.action_id == "replay":
            frames = ({"grasped": False, "signal": 0.0},
                      {"grasped": False, "signal": 10.0},
                      {"grasped": False, "signal": 20.0})
            return OfflineExecution(state=frames[-1], state_window=frames, task_completed=False)
        state = {"task_completed": True}
        return OfflineExecution(state=state, state_window=(state,) * 3, task_completed=True)

    result = OfflineWindowController(
        graph, RecoveryLibrary.from_mapping({"GRASP_MISS": ["recover"]})
    ).run(execute)

    assert result.status == "success"
    assert calls == ["replay", "recover"]


def test_directed_failure_trigger_applies_state_overrides():
    graph = prepend_failure_trigger(
        _graph(), "RECOVERY_BUDGET_LOW", {"recovery_budget_low": True}
    )
    graph = OfflineTaskGraph(
        actions={
            **graph.actions,
            "safe": OfflineAction(
                action_id="safe", primitive="safe_stop", code="open_gripper()",
                preconditions={"recovery_budget_low": True},
                success_conditions={"gripper_open": True},
            ),
        },
        initial=graph.initial,
    )
    calls = []

    def execute(action):
        calls.append(action.action_id)
        state = {"gripper_open": action.action_id == "safe"}
        return OfflineExecution(state=state, state_window=(state,) * 3, task_completed=False)

    result = OfflineWindowController(
        graph, RecoveryLibrary.from_mapping({"RECOVERY_BUDGET_LOW": ["safe"]})
    ).run(execute)

    assert calls == ["injected_failure_trigger", "safe"]
    assert result.executed_actions == ("safe",)


def test_directed_failure_trigger_can_follow_preparation_actions():
    graph = prepend_failure_trigger(
        _graph(), "PLACE_ALIGNMENT_FAILED", prepare_actions=("a1", "r1")
    )

    assert graph.initial == ("a1", "r1", "injected_failure_trigger")


def test_recovery_accepts_declared_conditions_when_object_itself_is_stable():
    graph = OfflineTaskGraph.from_mapping({
        "initial": ["recover"],
        "actions": [{
            "id": "recover",
            "code": "recover()",
            "success_conditions": {"target_grasped": True, "object_lifted": True},
            "metadata": {"accept_semantic_stability": True},
        }],
    })
    frames = (
        {"target_grasped": True, "object_lifted": True, "object_stable": True, "signal": 0.0},
        {"target_grasped": True, "object_lifted": True, "object_stable": True, "signal": 10.0},
        {"target_grasped": True, "object_lifted": True, "object_stable": True, "signal": 20.0},
    )

    result = OfflineWindowController(
        graph, RecoveryLibrary.from_mapping({})
    ).run(lambda _action: OfflineExecution(
        state=frames[-1], state_window=frames, task_completed=True
    ))

    assert result.status == "success"
    assert result.executed_actions == ("recover",)


def test_grasp_accepts_fresh_semantic_success_while_lift_is_moving():
    graph = OfflineTaskGraph.from_mapping({
        "initial": ["grasp"],
        "actions": [{
            "id": "grasp",
            "code": "grasp()",
            "success_conditions": {"target_grasped": True, "object_lifted": True},
            "metadata": {"accept_semantic_success": True},
        }],
    })
    frames = tuple({
        "target_grasped": True,
        "object_lifted": True,
        "object_stable": False,
        "measurement_fresh": True,
        "height": float(index * 0.1),
    } for index in range(3))

    result = OfflineWindowController(
        graph, RecoveryLibrary.from_mapping({})
    ).run(lambda _action: OfflineExecution(
        state=frames[-1], state_window=frames, task_completed=True
    ))

    assert result.status == "success"
    assert result.executed_actions == ("grasp",)


def test_environment_termination_safe_stops_without_recovery():
    graph = OfflineTaskGraph.from_mapping({
        "initial": ["grasp"],
        "actions": [{
            "id": "grasp",
            "code": "grasp()",
            "success_conditions": {"target_grasped": True},
            "failure_routes": {"POSTCONDITION_FAILED": "GRASP_MISS"},
        }, {
            "id": "recover",
            "code": "recover()",
            "success_conditions": {"target_grasped": True},
        }],
    })
    calls = []

    def execute(action):
        calls.append(action.action_id)
        state = {"target_grasped": False, "object_stable": True}
        return OfflineExecution(
            state=state, state_window=(state,) * 3,
            task_completed=False, terminated=True,
        )

    result = OfflineWindowController(
        graph, RecoveryLibrary.from_mapping({"GRASP_MISS": ["recover"]})
    ).run(execute)

    assert calls == ["grasp"]
    assert result.status == "aborted"
    assert result.exception_id == "ENVIRONMENT_TERMINATED"


def test_completed_before_termination_finishes_without_recovery():
    graph = OfflineTaskGraph.from_mapping({
        "initial": ["place"],
        "actions": [{
            "id": "place",
            "code": "place()",
            "success_conditions": {"object_inside_container": True},
        }],
    })

    result = OfflineWindowController(
        graph, RecoveryLibrary.from_mapping({})
    ).run(lambda _action: OfflineExecution(
        state={"object_inside_container": False, "task_completed": False},
        task_completed=True,
        terminated=True,
        sandbox_rc=1,
        exception_id="COMPLETED_BEFORE_TERMINATION",
    ))

    assert result.status == "success"
    assert result.task_completed is True
    assert result.exception_id == "COMPLETED_BEFORE_TERMINATION"
    assert result.executed_actions == ("place",)


def test_declared_fresh_semantic_failure_routes_despite_numeric_jitter():
    graph = OfflineTaskGraph.from_mapping({
        "initial": ["grasp"],
        "actions": [{
            "id": "grasp",
            "code": "grasp()",
            "success_conditions": {"target_grasped": True},
            "failure_routes": {"POSTCONDITION_FAILED": "GRASP_MISS"},
            "metadata": {"accept_semantic_failure": True},
        }, {
            "id": "recover",
            "code": "recover()",
            "success_conditions": {"target_grasped": True},
        }],
    })
    calls = []

    def execute(action):
        calls.append(action.action_id)
        if action.action_id == "grasp":
            frames = tuple({
                "target_grasped": False,
                "object_stable": True,
                "measurement_fresh": True,
                "signal": float(index * 10),
            } for index in range(3))
            return OfflineExecution(state=frames[-1], state_window=frames)
        state = {"target_grasped": True, "object_stable": True}
        return OfflineExecution(state=state, state_window=(state,) * 3, task_completed=True)

    result = OfflineWindowController(
        graph, RecoveryLibrary.from_mapping({"GRASP_MISS": ["recover"]})
    ).run(execute)

    assert result.status == "success"
    assert calls == ["grasp", "recover"]


def test_sandbox_failure_route_wins_over_unstable_state_window():
    graph = OfflineTaskGraph.from_mapping({
        "initial": ["grasp"],
        "actions": [{
            "id": "grasp",
            "code": "grasp()",
            "success_conditions": {"target_grasped": True},
            "failure_routes": {"SANDBOX_ERROR": "GRASP_MISS"},
        }, {
            "id": "recover",
            "code": "recover()",
            "success_conditions": {"target_grasped": True},
        }],
    })
    calls = []

    def execute(action):
        calls.append(action.action_id)
        if action.action_id == "grasp":
            frames = tuple({"target_grasped": False, "signal": index * 10.0} for index in range(3))
            return OfflineExecution(state=frames[-1], state_window=frames, sandbox_rc=1)
        state = {"target_grasped": True}
        return OfflineExecution(state=state, state_window=(state,) * 3, task_completed=True)

    result = OfflineWindowController(
        graph, RecoveryLibrary.from_mapping({"GRASP_MISS": ["recover"]})
    ).run(execute)

    assert result.status == "success"
    assert calls == ["grasp", "recover"]


def test_unknown_offline_failure_safe_stops():
    def execute(_action):
        state = {"prepared": False, "task_completed": False}
        return OfflineExecution(state=state, state_window=(state, state, state), task_completed=False)

    result = OfflineWindowController(
        _graph(), RecoveryLibrary.from_mapping({})
    ).run(execute)

    assert result.status == "aborted"
    assert result.exception_id == "PREPARE_FAILED"
    assert result.failed_action == "a1"


def test_recovery_library_reads_compiler_output_wrapper():
    library = RecoveryLibrary.from_mapping({
        "recovery_library": {"GRASP_MISS": ["r1"]},
        "candidates": [{"status": "review_required"}],
    })

    assert library.lookup("GRASP_MISS") == ("r1",)


def test_shared_action_library_attaches_without_changing_task_initial_action():
    graph = _graph()
    merged = attach_action_library(graph, {
        "actions": [{"id": "recover_grasp", "code": "recover()"}],
    })

    assert merged.initial == graph.initial
    assert merged.actions["recover_grasp"].code == "recover()"


def test_offline_action_binds_current_state_without_fixed_coordinates():
    action = OfflineAction.from_mapping({
        "id": "recover_grasp",
        "primitive": "grasp",
        "parameters": {"red_pose": "cube_poses_robot_base.primary"},
        "code": "red_pose = _offline_params['red_pose']",
    })

    code, parameters = bind_state_parameters(action, {
        "cube_poses_robot_base": {"primary": [0.51, -0.02, -0.10, 1, 0, 0, 0]},
    })

    assert parameters["red_pose"][:3] == [0.51, -0.02, -0.10]
    assert "0.4583" not in code
    namespace = {}
    exec(code, namespace)
    assert namespace["red_pose"][:3] == [0.51, -0.02, -0.10]


def test_offline_action_rejects_missing_state_parameter():
    action = OfflineAction.from_mapping({
        "id": "recover_grasp",
        "code": "move(_offline_params['red_pose'])",
        "parameters": {"red_pose": "cube_poses_robot_base.primary"},
    })

    with pytest.raises(StateParameterError, match="missing state path"):
        bind_state_parameters(action, {})


def test_task_roles_resolve_to_current_visual_names():
    api = object.__new__(FrankaControlApi)
    api._env = SimpleNamespace(
        role_object_names={"primary": "blue cube", "secondary": "yellow cube"}
    )

    assert api._resolve_object_name("primary") == "blue cube"
    assert api._resolve_object_name("secondary") == "yellow cube"
    assert api._resolve_object_name("handle") == "handle"


def test_restack_prompt_and_oracle_use_roles_and_dynamic_geometry():
    policy_text = PROMPT + ORACLE_CODE

    assert "red cube" not in policy_text
    assert "green cube" not in policy_text
    assert 'get_object_pose("primary"' in ORACLE_CODE
    assert 'get_object_pose("secondary"' in ORACLE_CODE
    assert "primary_ext" in ORACLE_CODE
    assert "secondary_ext" in ORACLE_CODE
    assert "sample_grasp_pose" in ORACLE_CODE


def test_cube_pose_state_converts_robosuite_xyzw_to_wxyz():
    env = object.__new__(FrankaRobosuiteCubesRestackLowLevel)
    env.base_link_idx = 0
    env.robosuite_env = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(
        xquat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        xpos=np.array([[0.0, 0.0, 0.0]]),
    )))
    sin_cos = np.sqrt(0.5)

    poses = env._cube_pose_dict({
        "cubeA_pos": np.array([0.5, 0.1, -0.1]),
        "cubeA_quat": np.array([0.0, 0.0, sin_cos, sin_cos]),
        "cubeB_pos": np.array([0.4, -0.1, -0.1]),
        "cubeB_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    })

    assert np.allclose(poses["primary"][3:], [sin_cos, 0.0, 0.0, sin_cos])
    assert np.allclose(poses["secondary"][3:], [1.0, 0.0, 0.0, 0.0])


def test_cube_lift_pose_state_converts_robosuite_xyzw_to_wxyz():
    env = object.__new__(FrankaRobosuiteCubeLiftLowLevel)
    env.base_link_idx = 0
    env.robosuite_env = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(
        xquat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        xpos=np.array([[0.0, 0.0, 0.0]]),
    )))
    sin_cos = np.sqrt(0.5)

    poses = env._cube_pose_dict({
        "cube_pos": np.array([0.5, 0.1, -0.1]),
        "cube_quat": np.array([0.0, 0.0, sin_cos, sin_cos]),
    })

    assert np.allclose(poses["primary"][3:], [sin_cos, 0.0, 0.0, sin_cos])


def test_cube_lift_validation_fault_skips_only_first_gripper_close(monkeypatch):
    env = object.__new__(FrankaRobosuiteCubeLiftLowLevel)
    env.inject_first_grasp_miss = True
    env._grasp_miss_injected = False
    commands = []
    events = []
    monkeypatch.setattr(
        RobosuiteBaseEnv, "_set_gripper", lambda _self, value: commands.append(value)
    )
    env.record_world_state = lambda event, **kwargs: events.append((event, kwargs))

    env._set_gripper(0.0)
    env._set_gripper(0.0)

    assert commands == [1.0, 0.0]
    assert events[0][0] == "fault_injected"
    assert events[0][1]["metadata"]["exception_id"] == "GRASP_MISS"


def test_restack_release_fault_offsets_primary_and_marks_only_first_unstable(monkeypatch):
    env = object.__new__(FrankaRobosuiteCubesRestackLowLevel)
    env.inject_primary_release_failures = 2
    env.inject_first_primary_release_unstable = True
    env.primary_release_fault_offset = 0.06
    env._primary_release_faults_injected = 0
    env._pending_primary_release_fault_steps = None
    env._unstable_release_fault_active = False
    env._gripper_fraction = 0.0
    grasped = [True]
    env._primary_is_grasped = lambda: grasped[0]
    events = []
    env.record_world_state = lambda event, **kwargs: events.append((event, kwargs))

    class _Data:
        qpos = np.array([0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])

        def get_joint_qpos(self, _name):
            return self.qpos.copy()

        def set_joint_qpos(self, _name, value):
            self.qpos = np.asarray(value)

    data = _Data()
    env.robosuite_env = SimpleNamespace(
        cubeA=SimpleNamespace(joints=["cubeA_joint"]),
        sim=SimpleNamespace(data=data, forward=lambda: None),
    )
    monkeypatch.setattr(
        RobosuiteBaseEnv, "_set_gripper",
        lambda self, value: setattr(self, "_gripper_fraction", value),
    )
    monkeypatch.setattr(RobosuiteBaseEnv, "_step_once", lambda _self: None)

    env._set_gripper(1.0)
    for _ in range(10):
        env._step_once()

    assert data.qpos[0] == pytest.approx(0.56)
    assert env._unstable_release_fault_active is True
    assert events[-1][1]["metadata"]["exception_id"] == "UNSTABLE_RELEASE"

    grasped[0] = False
    env._set_gripper(1.0)
    assert env._unstable_release_fault_active is False

    grasped[0] = True
    env._set_gripper(1.0)
    for _ in range(10):
        env._step_once()

    assert data.qpos[0] == pytest.approx(0.62)
    assert env._unstable_release_fault_active is False
    assert events[-1][1]["metadata"]["exception_id"] == "PLACE_FAILED"


def test_restack_environment_builds_real_cylinder_task_objects():
    env = object.__new__(FrankaRobosuiteCubesRestackLowLevel)
    env.object_shape = "cylinder"

    obj = env._make_task_object("cubeA", 0.02, "red")

    assert isinstance(obj, CylinderObject)
    assert np.allclose(obj.size, [0.02, 0.02])


def test_cuboid_profiles_pass_geometry_gate_and_unsafe_shapes_do_not():
    env = object.__new__(FrankaRobosuiteCubesRestackLowLevel)
    env.object_shape = "cuboid"
    env.cube_A_half_sizes = (0.04, 0.015, 0.02)
    env.cube_B_half_sizes = (0.04, 0.015, 0.02)

    obj = env._make_task_object("cubeA", 0.02, "red")
    assert isinstance(obj, BoxObject)
    assert np.allclose(obj.size, [0.04, 0.015, 0.02])

    for half_sizes in ((0.025, 0.02, 0.02), (0.03, 0.02, 0.02), (0.04, 0.015, 0.02)):
        assert cuboid_policy_geometry("cuboid", half_sizes, half_sizes)[
            "geometry_policy_eligible"
        ] is True

    assert cuboid_policy_geometry(
        "cuboid", (0.08, 0.04, 0.02), (0.08, 0.04, 0.02)
    )["geometry_policy_eligible"] is False
    assert cuboid_policy_geometry(
        "cylinder", (0.02, 0.02, 0.02), (0.02, 0.02, 0.02)
    )["geometry_policy_eligible"] is False


def test_formal_recovery_binds_sizes_and_uses_current_object_orientation():
    graph = OfflineTaskGraph.from_mapping(yaml.safe_load(
        Path("env_configs/cube_restack/offline_task_graph.yaml").read_text()
    ))
    graph = attach_action_library(graph, yaml.safe_load(
        Path("env_configs/recovery/grasp_miss_actions.yaml").read_text()
    ))
    strategy_code = "\n".join(action.code for action in graph.actions.values())
    assert '"red cube"' not in strategy_code
    assert '"green cube"' not in strategy_code
    assert "np.array([0.0, 0.0, 1.0, 0.0])" not in strategy_code

    state = {
        "cube_poses_robot_base": {
            "primary": [0.50, 0.02, -0.09, 0.70710678, 0, 0, 0.70710678],
            "secondary": [0.40, -0.08, -0.10, 1, 0, 0, 0],
        },
        "cube_extents_robot_base": {
            "primary": [0.06, 0.06, 0.06],
            "secondary": [0.04, 0.04, 0.04],
        },
        "target_pose_robot_base": [
            0.50, 0.02, -0.09, 0.70710678, 0, 0, 0.70710678
        ],
        "target_extent_robot_base": [0.06, 0.06, 0.06],
    }
    calls = []
    namespace = {
        "goto_pose": lambda pos, quat, **_kwargs: calls.append((np.asarray(pos), np.asarray(quat))),
        "home_pose": lambda: None,
        "open_gripper": lambda: None,
        "close_gripper": lambda: None,
    }

    recovery_code, _ = bind_state_parameters(graph.actions["recover_grasp_miss"], state)
    exec(recovery_code, namespace)
    assert np.allclose(
        namespace["target_grasp_quat"],
        [0.0, -0.70710678, 0.70710678, 0.0],
    )
    assert namespace["approach_height"] == pytest.approx(0.12)
    assert namespace["lift_height"] == pytest.approx(0.18)

    state["cube_poses_robot_base"]["primary"] = [*namespace["target_lift_pos"], 1, 0, 0, 0]
    placement_code, _ = bind_state_parameters(
        graph.actions["stage_003_after_recovery"], state
    )
    exec(placement_code, namespace)
    expected_center = np.array([0.40, -0.08, -0.05])
    assert namespace["center_spacing"] == pytest.approx(0.05)
    assert np.allclose(calls[-2][0], expected_center)


def test_restack_place_failures_select_parameterized_recovery_routes():
    graph = OfflineTaskGraph.from_mapping(yaml.safe_load(
        Path("env_configs/cube_restack/offline_task_graph.yaml").read_text()
    ))
    graph = attach_action_library(graph, yaml.safe_load(
        Path("env_configs/recovery/grasp_miss_actions.yaml").read_text()
    ))
    library = RecoveryLibrary.from_mapping(yaml.safe_load(
        Path("env_configs/cube_restack/offline_recovery_library.yaml").read_text()
    ))

    exception_id, route = select_hybrid_recovery(
        graph, library, ("stage_001", "stage_002", "stage_003"), 2,
        state={"object_stable": True},
    )
    assert exception_id == "PLACE_FAILED"
    assert route == ("recover_place_failed",)

    exception_id, route = select_hybrid_recovery(
        graph, library, ("stage_001", "stage_002", "stage_003"), 2,
        state={"object_stable": False},
    )
    assert exception_id == "UNSTABLE_RELEASE"
    assert route == ("recover_unstable_release", "recover_place_failed")

    action = graph.actions["recover_place_failed"]
    assert action.args == {
        "primary_pose": "cube_poses_robot_base.primary",
        "secondary_pose": "cube_poses_robot_base.secondary",
        "primary_extent": "cube_extents_robot_base.primary",
        "secondary_extent": "cube_extents_robot_base.secondary",
    }
    assert "sample_grasp_pose" not in action.code
    assert "primary_pose[3:7]" in action.code
    clear_action = graph.actions["recover_clear_precondition"]
    assert "sample_grasp_pose" not in clear_action.code
    assert "secondary_pose[3:7]" in clear_action.code
    assert graph.actions["stage_002"].metadata["accept_semantic_stability"] is True
    assert library.lookup("PLACE_RECOVERY_FAILED") == (
        "recover_grasp_miss_rotated", "stage_003_after_recovery"
    )
    assert library.lookup("GRASP_RECOVERY_FAILED") == (
        "recover_grasp_miss_rotated", "stage_003_after_recovery"
    )


def test_promoted_anomalies_select_interruptible_formal_recovery_routes():
    graph = OfflineTaskGraph.from_mapping(yaml.safe_load(
        Path("env_configs/bin_place/offline_task_graph.yaml").read_text()
    ))
    library = RecoveryLibrary.from_mapping(yaml.safe_load(
        Path("env_configs/bin_place/recovery_library.yaml").read_text()
    ))

    exception_id, route = select_hybrid_recovery(
        graph,
        library,
        ("stage_001", "stage_002"),
        1,
        state={
            "active_exception_id": "TARGET_POSITION_CHANGED",
            "object_stable": True,
        },
    )

    assert exception_id == "TARGET_POSITION_CHANGED"
    assert route == (
        "candidate_reobserve_changed_target",
        "candidate_rebind_changed_target",
        "stage_002",
    )
    assert all(
        graph.actions[action_id].metadata["status"] == "formal"
        for action_id in route[:2]
    )

    exception_id, route = select_hybrid_recovery(
        graph,
        library,
        ("stage_001", "stage_002"),
        1,
        state={
            "active_exception_id": "OBJECT_OUT_OF_WORKSPACE",
            "object_stable": True,
        },
    )

    assert exception_id == "OBJECT_OUT_OF_WORKSPACE"
    assert route == (
        "candidate_reobserve_out_of_workspace",
        "candidate_check_workspace_recovery",
        "candidate_move_primary_into_workspace",
    )
    assert all(
        graph.actions[action_id].metadata["status"] == "formal"
        for action_id in route
    )


def test_workspace_edge_grasp_miss_selects_interruptible_recenter_chain():
    graph = OfflineTaskGraph.from_mapping(yaml.safe_load(
        Path("env_configs/bin_place/offline_task_graph.yaml").read_text()
    ))
    library = RecoveryLibrary.from_mapping(yaml.safe_load(
        Path("env_configs/bin_place/recovery_library.yaml").read_text()
    ))

    exception_id, route = select_hybrid_recovery(
        graph,
        library,
        ("stage_001", "stage_002"),
        0,
        state={
            "target_grasped": False,
            "object_stable": True,
            "object_near_workspace_boundary": True,
        },
    )

    assert exception_id == "WORKSPACE_EDGE_GRASP_MISS"
    assert route == (
        "reobserve_workspace_edge_primary",
        "recenter_workspace_edge_primary",
        "recover_bin_center_place",
    )
    assert graph.actions["recover_bin_center_place"].next_action == (
        "recover_bin_center_replace"
    )


@pytest.mark.parametrize(
    ("initial_action", "state", "failed_action", "execute_calls"),
    [
        (
            "reobserve_workspace_edge_primary",
            {"workspace_reachable": False, "object_stable": True},
            "reobserve_workspace_edge_primary",
            0,
        ),
        (
            "recenter_workspace_edge_primary",
            {
                "object_near_workspace_boundary": True,
                "workspace_reachable": True,
                "object_stable": False,
            },
            "recenter_workspace_edge_primary",
            0,
        ),
    ],
)
def test_workspace_edge_safety_preconditions_stop_before_regrasp(
    initial_action, state, failed_action, execute_calls
):
    graph = OfflineTaskGraph.from_mapping(yaml.safe_load(
        Path("env_configs/bin_place/offline_task_graph.yaml").read_text()
    ))
    graph = OfflineTaskGraph(graph.actions, (initial_action,))
    calls = []

    result = OfflineWindowController(
        graph, RecoveryLibrary.from_mapping({})
    ).run(lambda action: calls.append(action.action_id), initial_state=state)

    assert result.status == "aborted"
    assert result.failed_action == failed_action
    assert len(calls) == execute_calls
    assert "recover_bin_center_place" not in calls
    assert result.events[-1].event_type == "safe_stop"


def test_workspace_edge_observation_failure_stops_before_recenter_and_regrasp():
    graph = OfflineTaskGraph.from_mapping(yaml.safe_load(
        Path("env_configs/bin_place/offline_task_graph.yaml").read_text()
    ))
    graph = OfflineTaskGraph(graph.actions, ("reobserve_workspace_edge_primary",))
    calls = []

    def execute(action):
        calls.append(action.action_id)
        state = {
            "object_observation_ready": False,
            "object_stable": True,
            "task_completed": False,
        }
        return OfflineExecution(state=state, state_window=(state,) * 3)

    result = OfflineWindowController(
        graph, RecoveryLibrary.from_mapping({})
    ).run(
        execute,
        initial_state={"workspace_reachable": True, "object_stable": True},
    )

    assert result.status == "aborted"
    assert calls == ["reobserve_workspace_edge_primary"]
    assert result.exception_id == "WORKSPACE_EDGE_REOBSERVE_FAILED"
    assert result.events[-1].event_type == "safe_stop"


def test_slender_candidate_routes_use_geometry_and_budget_state():
    action_value = yaml.safe_load(Path(
        "env_configs/cube_restack/cuboid_slender_candidate_actions.yaml"
    ).read_text())
    recovery_value = yaml.safe_load(Path(
        "env_configs/cube_restack/cuboid_slender_candidate_recovery.yaml"
    ).read_text())
    assert action_value["metadata"]["status"] == "formal"
    assert recovery_value["metadata"]["status"] == "formal"
    graph = OfflineTaskGraph.from_mapping(yaml.safe_load(
        Path("env_configs/cube_restack/offline_task_graph.yaml").read_text()
    ))
    graph = attach_action_library(graph, action_value)
    library = RecoveryLibrary.from_mapping(recovery_value)
    stages = ("stage_001", "stage_002", "stage_003")

    assert library.lookup("CLEAR_RECOVERY_FAILED") == (
        "recover_clear_recovery_radial",
    )

    exception_id, route = select_hybrid_recovery(
        graph, library, stages, 1,
        state={
            "object_stable": True, "target_grasped": False,
            "max_horizontal_aspect_ratio": 2.67,
            "recovery_budget_low": False,
        },
    )
    assert exception_id == "SLENDER_GRASP_POSE_UNSAFE"
    assert route == ("recover_slender_grasp_pose", "stage_003_after_recovery")

    exception_id, route = select_hybrid_recovery(
        graph, library, stages, 2,
        state={
            "object_stable": True, "task_completed": False,
            "distance_to_target": 0.02,
            "max_horizontal_aspect_ratio": 2.67,
            "recovery_budget_low": False,
        },
    )
    assert exception_id == "PLACE_ALIGNMENT_FAILED"
    assert route == ("recover_slender_place_alignment",)

    exception_id, route = select_hybrid_recovery(
        graph, library, stages, 2,
        state={"object_stable": True, "recovery_budget_low": True},
    )
    assert exception_id == "RECOVERY_BUDGET_LOW"
    assert route == ("recover_budget_safe_stop",)
