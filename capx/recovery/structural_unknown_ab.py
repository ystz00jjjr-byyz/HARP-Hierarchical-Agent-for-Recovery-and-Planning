"""Same-checkpoint structural-unknown recovery benchmark utilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


CHECKPOINT_SCHEMA_VERSION = 1
ANOMALY_CLASSES = (
    "TARGET_POSITION_CHANGED",
    "OBJECT_OUT_OF_WORKSPACE",
    "UNSTABLE_RELEASE",
    "TIGHT_CLEARANCE_PLACE",
    "COMBINED_TARGET_SHIFT_WORKSPACE_EDGE",
)
DEFAULT_SEED_START = 301
DEFAULT_SEEDS_PER_CLASS = 10


def anomaly_for_seed(
    seed: int,
    *,
    seed_start: int = DEFAULT_SEED_START,
    seeds_per_class: int = DEFAULT_SEEDS_PER_CLASS,
) -> str:
    index = (int(seed) - int(seed_start)) // int(seeds_per_class)
    if index < 0 or index >= len(ANOMALY_CLASSES):
        raise ValueError(f"seed {seed} is outside the structural-unknown matrix")
    return ANOMALY_CLASSES[index]


def checkpoint_path(directory: str | Path, seed: int) -> Path:
    return Path(directory) / f"seed_{int(seed):04d}.npz"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rotation_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=float)
    w, x, y, z = np.array([w, x, y, z]) / np.linalg.norm([w, x, y, z])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(left, dtype=float)
    w2, x2, y2, z2 = np.asarray(right, dtype=float)
    result = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    return result / np.linalg.norm(result)


def _set_primary_pose_robot_base(
    env: Any, position: np.ndarray, yaw_degrees: float
) -> None:
    sim = env.robosuite_env.sim
    base_position = np.asarray(sim.data.xpos[env.base_link_idx], dtype=float)
    base_quaternion = np.asarray(sim.data.xquat[env.base_link_idx], dtype=float)
    yaw = np.deg2rad(float(yaw_degrees))
    object_quaternion_base = np.array([
        np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)
    ])
    world_position = base_position + _rotation_matrix_wxyz(base_quaternion) @ np.asarray(
        position, dtype=float
    )
    world_quaternion = _quaternion_multiply(
        base_quaternion, object_quaternion_base
    )
    joint_name = env.primary_object.joints[0]
    sim.data.set_joint_qpos(
        joint_name, np.concatenate([world_position, world_quaternion])
    )
    set_joint_qvel = getattr(sim.data, "set_joint_qvel", None)
    if callable(set_joint_qvel):
        set_joint_qvel(joint_name, np.zeros(6))
    sim.forward()


def _move_target(env: Any, shift_xy: np.ndarray) -> dict[str, Any]:
    stale = env.get_target_region_state()
    sim = env.robosuite_env.sim
    body_id = sim.model.body_name2id("bin2")
    shift = np.array([float(shift_xy[0]), float(shift_xy[1]), 0.0])
    sim.model.body_pos[body_id] += shift
    env.robosuite_env.bin2_pos = np.asarray(env.robosuite_env.bin2_pos) + shift
    sim.forward()
    floor_world, inner_extent, target_quaternion_world = env._target_geometry_world()
    env._calibrated_target_center = env._world_point_to_robot_base(floor_world)
    env._calibrated_target_extent = np.asarray(inner_extent, dtype=float)
    return {
        "center": stale["target_region_center_robot_base"],
        "quaternion": stale["target_region_quaternion_wxyz"],
        "extent": stale["target_region_extent_robot_base"],
        "score": 1.0,
    }


def prepare_structural_failure_state(env: Any, anomaly_class: str, seed: int) -> dict[str, Any]:
    """Apply one deterministic, physically feasible post-failure state."""
    if anomaly_class not in ANOMALY_CLASSES:
        raise ValueError(f"unsupported anomaly class: {anomaly_class}")
    rng = np.random.default_rng(int(seed) * 104729)
    stale_target = None

    if anomaly_class in {
        "TARGET_POSITION_CHANGED", "COMBINED_TARGET_SHIFT_WORKSPACE_EDGE"
    }:
        angle = rng.uniform(-0.55, 0.55)
        distance = rng.uniform(0.065, 0.09)
        stale_target = _move_target(
            env, distance * np.array([np.cos(angle), np.sin(angle)])
        )
        env._blocked_stale_goto_attempts = 1
        env._blocked_release_attempts = 1
        env._clear_target_cache_after_blocked_stale_motion = True

    if anomaly_class in {
        "OBJECT_OUT_OF_WORKSPACE", "COMBINED_TARGET_SHIFT_WORKSPACE_EDGE"
    }:
        side = -1.0 if int(seed) % 2 else 1.0
        target_state = env.get_target_region_state()
        surface_z = float(target_state["target_region_center_robot_base"][2])
        half_height = float(env.cube_half_sizes[2])
        _set_primary_pose_robot_base(
            env,
            np.array([
                rng.uniform(0.76, 0.82),
                side * rng.uniform(0.29, 0.36),
                surface_z + half_height,
            ]),
            rng.uniform(-80.0, 80.0),
        )

    if anomaly_class == "UNSTABLE_RELEASE":
        floor_world, inner_extent, target_quaternion = env._target_geometry_world()
        local = np.array([
            0.5 * float(inner_extent[0]) + float(env.cube_half_sizes[0]) + 0.018,
            rng.uniform(-0.02, 0.02),
            float(env.cube_half_sizes[2]),
        ])
        world_position = (
            np.asarray(floor_world)
            + _rotation_matrix_wxyz(np.asarray(target_quaternion)) @ local
        )
        base_position = np.asarray(
            env._world_point_to_robot_base(world_position), dtype=float
        )
        _set_primary_pose_robot_base(env, base_position, rng.uniform(-75.0, 75.0))
        env._unstable_release_fault_active = True

    if anomaly_class == "TIGHT_CLEARANCE_PLACE":
        scenario = dict(env._active_scenario or {})
        half_x, half_y = np.asarray(env.cube_half_sizes, dtype=float)[:2]
        orientation_invariant_extent = 2.0 * float(np.hypot(half_x, half_y))
        margins = rng.uniform(0.0105, 0.016, size=2)
        scenario["target_inner_extent"] = (
            orientation_invariant_extent + float(margins[0]),
            orientation_invariant_extent + float(margins[1]),
            0.0,
        )
        scenario["scenario_id"] = f"structural_tight_{int(seed):04d}"
        scenario["expected_geometry_feasible"] = True
        env._active_scenario = scenario
        env._apply_active_scenario()

    env._active_exception_id = anomaly_class
    env._unknown_anomaly_applied = True
    env._target_change_armed = False
    env._camera_state_positions = []
    env._camera_state_last_measurement = None
    env._camera_rgbd_cache = None
    if stale_target is None:
        env._camera_target_last_measurement = None
    else:
        env._camera_target_last_measurement = (
            np.asarray(stale_target["center"], dtype=float),
            np.asarray(stale_target["quaternion"], dtype=float),
            np.asarray(stale_target["extent"], dtype=float),
            float(stale_target["score"]),
        )
    env.robosuite_env.sim.forward()

    truth = env.get_task_recorder_state()
    footprint = np.asarray(env.cube_half_sizes[:2], dtype=float) * 2.0
    target_extent = np.asarray(truth["target_region_extent_robot_base"][:2], dtype=float)
    physically_feasible = bool(
        np.all(target_extent > footprint)
        and truth.get("workspace_recovery_feasible", False)
        and truth.get("target_region_clear", False)
        and truth.get("object_observation_ready", False)
        and not truth.get("task_completed", False)
    )
    return {
        "anomaly_class": anomaly_class,
        "seed": int(seed),
        "physical_feasibility_precheck": physically_feasible,
        "physical_feasibility_basis": {
            "target_footprint_fit": bool(np.all(target_extent > footprint)),
            "workspace_recovery_feasible": bool(
                truth.get("workspace_recovery_feasible", False)
            ),
            "target_region_clear": bool(truth.get("target_region_clear", False)),
            "observation_ready": bool(truth.get("object_observation_ready", False)),
            "not_already_complete": not bool(truth.get("task_completed", False)),
        },
        "inherited_policy_status": "postcondition_failed",
        "single_policy_direct_completion_allowed": False,
        "stale_target_measurement": stale_target,
    }


def save_region_place_checkpoint(
    env: Any, path: str | Path, *, seed: int, case_metadata: dict[str, Any]
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sim = env.robosuite_env.sim
    runtime = {
        "step_count": int(env._step_count),
        "sim_step_count": int(env._sim_step_count),
        "gripper_fraction": float(env._gripper_fraction),
        "primary_release_faults_injected": int(env._primary_release_faults_injected),
        "pending_primary_release_fault_steps": env._pending_primary_release_fault_steps,
        "unstable_release_fault_active": bool(env._unstable_release_fault_active),
        "active_exception_id": env._active_exception_id,
        "unknown_anomaly_applied": bool(env._unknown_anomaly_applied),
        "target_change_armed": bool(env._target_change_armed),
        "blocked_stale_goto_attempts": int(env._blocked_stale_goto_attempts),
        "blocked_release_attempts": int(env._blocked_release_attempts),
        "visual_occlusion_steps": int(env._visual_occlusion_steps),
        "clear_target_cache_after_blocked_stale_motion": bool(
            getattr(env, "_clear_target_cache_after_blocked_stale_motion", False)
        ),
        "active_scenario": env._active_scenario,
        "stale_target_measurement": case_metadata.get("stale_target_measurement"),
    }
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "seed": int(seed),
        "object_shape": str(env.object_shape),
        "cube_half_sizes": [float(value) for value in env.cube_half_sizes],
        "runtime": runtime,
        **case_metadata,
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            sim_state=np.asarray(sim.get_state().flatten()),
            model_body_pos=np.asarray(sim.model.body_pos),
            model_body_quat=np.asarray(sim.model.body_quat),
            model_geom_pos=np.asarray(sim.model.geom_pos),
            model_geom_quat=np.asarray(sim.model.geom_quat),
            model_geom_size=np.asarray(sim.model.geom_size),
            model_geom_contype=np.asarray(sim.model.geom_contype),
            model_geom_conaffinity=np.asarray(sim.model.geom_conaffinity),
            model_geom_rgba=np.asarray(sim.model.geom_rgba),
            bin2_pos=np.asarray(env.robosuite_env.bin2_pos),
            bin_size=np.asarray(env.robosuite_env.bin_size),
            calibrated_target_center=np.asarray(env._calibrated_target_center),
            calibrated_target_extent=np.asarray(env._calibrated_target_extent),
            active_target_quaternion=np.asarray(env._active_target_quaternion),
            active_rim_height=np.asarray(env._active_rim_height),
        )
    os.replace(temp_path, path)
    result = dict(metadata)
    result["checkpoint_path"] = str(path)
    result["checkpoint_sha256"] = sha256_file(path)
    return result


def load_region_place_checkpoint(
    env: Any, path: str | Path, *, expected_seed: int | None = None
) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported recovery checkpoint schema")
        if expected_seed is not None and int(metadata["seed"]) != int(expected_seed):
            raise ValueError("recovery checkpoint seed mismatch")
        if str(metadata["object_shape"]) != str(env.object_shape):
            raise ValueError("recovery checkpoint object shape mismatch")
        if not np.allclose(
            np.asarray(metadata["cube_half_sizes"], dtype=float),
            np.asarray(env.cube_half_sizes, dtype=float),
        ):
            raise ValueError("recovery checkpoint object size mismatch")
        sim = env.robosuite_env.sim
        for name in (
            "body_pos", "body_quat", "geom_pos", "geom_quat", "geom_size",
            "geom_contype", "geom_conaffinity", "geom_rgba",
        ):
            target = getattr(sim.model, name)
            source = np.asarray(payload[f"model_{name}"])
            if target.shape != source.shape:
                raise ValueError(f"recovery checkpoint model shape mismatch: {name}")
            target[...] = source
        env.robosuite_env.bin2_pos = np.asarray(payload["bin2_pos"]).copy()
        env.robosuite_env.bin_size[...] = np.asarray(payload["bin_size"])
        env._calibrated_target_center = np.asarray(
            payload["calibrated_target_center"]
        ).copy()
        env._calibrated_target_extent = np.asarray(
            payload["calibrated_target_extent"]
        ).copy()
        env._active_target_quaternion = np.asarray(
            payload["active_target_quaternion"]
        ).copy()
        env._active_rim_height = float(payload["active_rim_height"].item())
        sim.set_state_from_flattened(np.asarray(payload["sim_state"]))
        sim.forward()

    runtime = dict(metadata["runtime"])
    env._step_count = int(runtime["step_count"])
    env._sim_step_count = int(runtime["sim_step_count"])
    env._gripper_fraction = float(runtime["gripper_fraction"])
    env._primary_release_faults_injected = int(
        runtime["primary_release_faults_injected"]
    )
    env._pending_primary_release_fault_steps = runtime[
        "pending_primary_release_fault_steps"
    ]
    env._unstable_release_fault_active = bool(
        runtime["unstable_release_fault_active"]
    )
    env._active_exception_id = runtime["active_exception_id"]
    env._unknown_anomaly_applied = bool(runtime["unknown_anomaly_applied"])
    env._target_change_armed = bool(runtime["target_change_armed"])
    env._blocked_stale_goto_attempts = int(runtime["blocked_stale_goto_attempts"])
    env._blocked_release_attempts = int(runtime["blocked_release_attempts"])
    env._visual_occlusion_steps = int(runtime["visual_occlusion_steps"])
    env._clear_target_cache_after_blocked_stale_motion = bool(
        runtime.get("clear_target_cache_after_blocked_stale_motion", False)
    )
    env._active_scenario = runtime.get("active_scenario")
    stale = runtime.get("stale_target_measurement")
    env._camera_target_last_measurement = (
        None
        if stale is None
        else (
            np.asarray(stale["center"], dtype=float),
            np.asarray(stale["quaternion"], dtype=float),
            np.asarray(stale["extent"], dtype=float),
            float(stale["score"]),
        )
    )
    env._camera_state_positions = []
    env._camera_state_last_measurement = None
    env._camera_rgbd_cache = None
    env._camera_measurement_calls = 0
    observations = env.robosuite_env._get_observations()
    env._current_joints = np.asarray(
        observations["robot0_joint_pos"], dtype=np.float64
    )
    env._current_joints[6] -= np.pi
    result = dict(metadata)
    result["checkpoint_path"] = str(path)
    result["checkpoint_sha256"] = sha256_file(path)
    env._loaded_recovery_checkpoint = result
    return result


def generate_checkpoints(
    output_dir: str | Path,
    *,
    seed_start: int = DEFAULT_SEED_START,
    seeds_per_class: int = DEFAULT_SEEDS_PER_CLASS,
    object_shape: str = "cuboid",
) -> dict[str, Any]:
    from capx.envs.simulators.robosuite_bin_place import (
        FrankaRobosuiteBinPlaceLowLevel,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = FrankaRobosuiteBinPlaceLowLevel(
        max_steps=3200,
        privileged=True,
        enable_render=False,
        object_shape=object_shape,
        target_mode="region",
        scenario_sweep=True,
        world_state_source="simulator",
    )
    cases = []
    try:
        for offset in range(len(ANOMALY_CLASSES) * seeds_per_class):
            seed = seed_start + offset
            anomaly_class = anomaly_for_seed(
                seed, seed_start=seed_start, seeds_per_class=seeds_per_class
            )
            env.reset(seed=seed, options={"trial": seed})
            case = prepare_structural_failure_state(env, anomaly_class, seed)
            if not case["physical_feasibility_precheck"]:
                raise RuntimeError(
                    f"seed {seed} failed physical feasibility precheck: "
                    f"{case['physical_feasibility_basis']}"
                )
            cases.append(save_region_place_checkpoint(
                env,
                checkpoint_path(output_dir, seed),
                seed=seed,
                case_metadata=case,
            ))
    finally:
        close = getattr(env.robosuite_env, "close", None)
        if callable(close):
            close()
    manifest = {
        "experiment": "structural_unknown_region_place_same_checkpoint_ab",
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "seed_start": seed_start,
        "seeds_per_class": seeds_per_class,
        "anomaly_classes": list(ANOMALY_CLASSES),
        "case_count": len(cases),
        "all_physically_feasible": all(
            case["physical_feasibility_precheck"] for case in cases
        ),
        "cases": cases,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seeds-per-class", type=int, default=DEFAULT_SEEDS_PER_CLASS)
    parser.add_argument("--object-shape", default="cuboid")
    args = parser.parse_args()
    manifest = generate_checkpoints(
        args.output_dir,
        seed_start=args.seed_start,
        seeds_per_class=args.seeds_per_class,
        object_shape=args.object_shape,
    )
    print(json.dumps({
        "case_count": manifest["case_count"],
        "all_physically_feasible": manifest["all_physically_feasible"],
        "manifest": str(args.output_dir / "manifest.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
