from __future__ import annotations

import inspect
from pathlib import Path
from types import MethodType

import numpy as np
import yaml

from capx.envs.simulators.robosuite_bin_place import (
    FrankaRobosuiteBinPlaceLowLevel,
)
from capx.recovery.offline import OfflineTaskGraph, RecoveryLibrary
from capx.recovery.structural_unknown_ab import (
    ANOMALY_CLASSES,
    anomaly_for_seed,
    checkpoint_path,
)
from capx.recovery.structural_unknown_benchmark import collect_benchmark


def test_structural_unknown_seed_matrix_is_five_by_ten():
    observed = [anomaly_for_seed(seed) for seed in range(301, 351)]
    assert len(ANOMALY_CLASSES) == 5
    for index, anomaly_class in enumerate(ANOMALY_CLASSES):
        assert observed[index * 10:(index + 1) * 10] == [anomaly_class] * 10


def test_checkpoint_loader_is_an_explicit_low_level_option():
    parameters = inspect.signature(FrankaRobosuiteBinPlaceLowLevel).parameters
    assert "recovery_checkpoint_dir" in parameters
    assert "recovery_checkpoint_required" in parameters
    assert checkpoint_path("checkpoints", 301) == Path("checkpoints/seed_0301.npz")


def test_region_place_exposes_primary_through_shared_rgbd_api():
    env = object.__new__(FrankaRobosuiteBinPlaceLowLevel)
    env.role_object_names = {"primary": "red rectangular block"}
    measurement = (
        np.array([0.5, 0.0, 0.02]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.06, 0.04, 0.03]),
        0.9,
    )
    env._measure_primary_from_camera = MethodType(lambda self: measurement, env)

    assert env.get_rgbd_object_state("primary") is measurement
    assert env.get_rgbd_object_state("red rectangular block") is measurement


def test_region_place_rebinds_single_object_from_rgbd():
    env = object.__new__(FrankaRobosuiteBinPlaceLowLevel)
    primary = ("primary",)
    target = ("target",)
    env._measure_primary_from_camera = MethodType(lambda self: primary, env)
    env._measure_target_from_camera = MethodType(lambda self: target, env)
    events = []
    env.record_world_state = MethodType(
        lambda self, event, **kwargs: events.append((event, kwargs)), env
    )

    assert env.rebind_roles_from_rgbd() is True
    assert env._camera_state_last_measurement is primary
    assert env._camera_target_last_measurement is target
    assert events == [("roles_rebound", {"metadata": {"source": "camera_rgbd"}})]


def test_l3_and_native_configs_share_state_and_budget_but_not_code_authority():
    l3 = yaml.safe_load(Path(
        "env_configs/region_place/structural_unknown_l3.yaml"
    ).read_text())
    native = yaml.safe_load(Path(
        "env_configs/region_place/structural_unknown_native.yaml"
    ).read_text())
    l3_kwargs = l3["env"]["cfg"]["low_level_kwargs"]
    native_kwargs = native["env"]["cfg"]["low_level_kwargs"]

    assert l3_kwargs == native_kwargs
    assert l3["env"]["cfg"]["apis"] == ["FrankaMultiObjectSortRgbdApi"]
    assert native["env"]["cfg"]["apis"] == ["FrankaMultiObjectSortRgbdApi"]
    assert l3["max_model_requests_per_trial"] == 5
    assert native["max_model_requests_per_trial"] == 5
    assert l3["text_world_state_feedback"] == native["text_world_state_feedback"]
    assert l3["hybrid_recovery"]["model_recovery_template"] == (
        "bin_place_regrasp_replace_v1"
    )
    assert "hybrid_recovery" not in native
    assert l3["metadata"]["model_authority"] == (
        "structured_template_and_bounded_parameters_only"
    )
    assert native["metadata"]["model_authority"] == (
        "free_bounded_recovery_code"
    )


def test_l4_same_checkpoint_config_uses_graph_without_model_code_authority():
    l4 = yaml.safe_load(Path(
        "env_configs/region_place/structural_unknown_l4.yaml"
    ).read_text())
    l3 = yaml.safe_load(Path(
        "env_configs/region_place/structural_unknown_l3.yaml"
    ).read_text())

    assert l4["env"]["cfg"]["low_level_kwargs"] == l3["env"]["cfg"]["low_level_kwargs"]
    assert l4["max_model_requests_per_trial"] == l3["max_model_requests_per_trial"] == 5
    assert l4["hybrid_recovery"]["model_initial_plan_template"] == "recovery_graph_v1"
    assert l4["hybrid_recovery"]["model_recovery_template"] == "recovery_graph_v1"
    assert l4["metadata"]["arbitrary_model_robot_code"] is False


def test_checkpoint_graph_forces_postcondition_before_structured_fallback():
    graph = OfflineTaskGraph.from_mapping(yaml.safe_load(Path(
        "env_configs/region_place/structural_unknown_checkpoint_graph.yaml"
    ).read_text()))
    library = RecoveryLibrary.from_mapping(yaml.safe_load(Path(
        "env_configs/region_place/structural_unknown_checkpoint_library.yaml"
    ).read_text()))

    marker = graph.actions["checkpoint_failure_marker"]
    assert marker.success_conditions == {"task_completed": True}
    assert marker.metadata["failure_next_action"] == "recover_budget_safe_stop"
    assert graph.actions["candidate_grasp_primary"].success_conditions[
        "target_grasped"
    ] is True
    assert graph.actions["candidate_replace_primary"].success_conditions[
        "task_completed"
    ] is True
    assert library.lookup("RECOVERY_BUDGET_LOW") == ("recover_budget_safe_stop",)


def test_report_trial_count_must_be_positive(tmp_path):
    try:
        collect_benchmark(tmp_path, model="test/model", trial_count=0)
    except ValueError as error:
        assert str(error) == "trial_count must be positive"
    else:
        raise AssertionError("zero-sized report should be rejected")
