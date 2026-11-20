"""Single-cube placement into a Robosuite bin compartment."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.utils.camera_utils import get_real_depth_map

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from capx.utils.depth_utils import depth_to_pointcloud


def _as_bool(value: bool | str) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _rotation_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def bin_place_scenario(object_shape: str, seed: int) -> dict[str, Any]:
    """Return a fixed validation case or a deterministic holdout geometry."""
    if object_shape not in {"cube", "cuboid", "cylinder"}:
        raise ValueError("object_shape must be cube, cuboid, or cylinder")
    if int(seed) > 10:
        rng = np.random.default_rng(int(seed))
        half_sizes = rng.uniform(0.018, 0.026, size=3)
        if object_shape == "cuboid":
            half_sizes = np.array([
                rng.uniform(0.026, 0.034), rng.uniform(0.014, 0.020),
                rng.uniform(0.018, 0.026),
            ])
        elif object_shape == "cylinder":
            radius = rng.uniform(0.018, 0.025)
            half_sizes = np.array([radius, radius, rng.uniform(0.022, 0.035)])
        return {
            "scenario_id": f"{object_shape}_holdout_{int(seed):03d}",
            "object_shape": object_shape,
            "object_half_sizes": tuple(np.round(half_sizes, 4)),
            "object_xy": tuple(np.round(rng.uniform(-0.06, 0.06, size=2), 4)),
            "object_yaw_degrees": float(np.round(rng.uniform(-75.0, 75.0), 1)),
            "bin_offset_xy": tuple(np.round(rng.uniform(-0.035, 0.035, size=2), 4)),
            "bin_yaw_degrees": float(np.round(rng.uniform(-30.0, 30.0), 1)),
            "target_inner_extent": tuple(np.round([
                rng.uniform(0.14, 0.20), rng.uniform(0.16, 0.23),
                rng.uniform(0.07, 0.11),
            ], 4)),
            "rim_height": float(np.round(rng.uniform(0.07, 0.11), 4)),
            "injected_exception": None,
            "expected_geometry_feasible": True,
        }
    index = (int(seed) - 1) % 10
    layouts = (
        ((0.000, 0.000), 0, (0.175, 0.225, 0.080), 0.080, (-0.03, 0.02), 0),
        ((0.030, -0.020), 15, (0.150, 0.180, 0.070), 0.080, (0.05, -0.06), 20),
        ((-0.030, 0.020), -20, (0.200, 0.160, 0.100), 0.070, (-0.06, 0.04), -35),
        ((0.020, 0.030), 20, (0.140, 0.200, 0.080), 0.090, (0.04, 0.04), 30),
        ((-0.020, -0.030), -15, (0.110, 0.140, 0.070), 0.100, (-0.05, -0.05), -45),
        ((0.040, 0.000), 25, (0.180, 0.130, 0.090), 0.090, (0.07, -0.03), 60),
        ((-0.040, 0.010), -25, (0.130, 0.220, 0.120), 0.120, (-0.05, 0.05), -60),
        ((0.010, -0.040), 10, (0.160, 0.160, 0.080), 0.080, (0.03, 0.05), 45),
        ((0.030, 0.030), 25, (0.120, 0.180, 0.060), 0.080, (-0.03, -0.06), -30),
        ((0.000, 0.000), 0, (0.070, 0.080, 0.050), 0.070, (0.00, 0.00), 0),
    )
    sizes = {
        "cube": (
            (0.020, 0.020, 0.020), (0.018, 0.018, 0.018),
            (0.025, 0.025, 0.025), (0.020, 0.020, 0.020),
            (0.024, 0.024, 0.020), (0.022, 0.022, 0.025),
            (0.018, 0.018, 0.025), (0.020, 0.020, 0.020),
            (0.022, 0.022, 0.020), (0.050, 0.050, 0.030),
        ),
        "cuboid": (
            (0.025, 0.015, 0.020), (0.030, 0.015, 0.018),
            (0.030, 0.015, 0.020), (0.030, 0.017, 0.020),
            (0.028, 0.018, 0.022), (0.025, 0.015, 0.025),
            (0.030, 0.017, 0.020), (0.030, 0.016, 0.018),
            (0.028, 0.016, 0.022), (0.060, 0.040, 0.030),
        ),
        "cylinder": (
            (0.020, 0.020, 0.025), (0.018, 0.018, 0.030),
            (0.025, 0.025, 0.020), (0.022, 0.022, 0.030),
            (0.024, 0.024, 0.025), (0.018, 0.018, 0.035),
            (0.020, 0.020, 0.030), (0.023, 0.023, 0.025),
            (0.021, 0.021, 0.028), (0.050, 0.050, 0.035),
        ),
    }
    offset, bin_yaw, extent, rim_height, object_xy, object_yaw = layouts[index]
    fault = "OUTSIDE_CONTAINER" if index == 3 else "UNSTABLE_RELEASE" if index == 7 else None
    return {
        "scenario_id": f"{object_shape}_{index + 1:02d}",
        "object_shape": object_shape,
        "object_half_sizes": sizes[object_shape][index],
        "object_xy": object_xy,
        "object_yaw_degrees": object_yaw,
        "bin_offset_xy": offset,
        "bin_yaw_degrees": bin_yaw,
        "target_inner_extent": extent,
        "rim_height": rim_height,
        "injected_exception": fault,
        "expected_geometry_feasible": index != 9,
    }


def region_place_scenario(object_shape: str, seed: int) -> dict[str, Any]:
    """Reuse the geometry sweep without container walls or height limits."""
    scenario = dict(bin_place_scenario(object_shape, seed))
    scenario["scenario_id"] = scenario["scenario_id"].replace(
        object_shape, f"region_{object_shape}", 1
    )
    scenario["target_inner_extent"] = (
        *scenario["target_inner_extent"][:2], 0.0
    )
    scenario["rim_height"] = 0.0
    if scenario["injected_exception"] == "OUTSIDE_CONTAINER":
        scenario["injected_exception"] = "OUTSIDE_TARGET_REGION"
    scenario["expected_geometry_feasible"] = bool(
        2.0 * max(scenario["object_half_sizes"][:2]) + 0.02
        <= min(scenario["target_inner_extent"][:2])
    )
    return scenario


def infeasible_region_place_scenario(
    object_shape: str, seed: int, failure: str
) -> dict[str, Any]:
    """Return a deterministic region-place scene that must safe-stop."""
    allowed = {
        "TARGET_REGION_TOO_SMALL",
        "OBJECT_UNREACHABLE",
        "OBSERVATION_FAILURE",
        "TARGET_REGION_BLOCKED",
    }
    if failure not in allowed:
        raise ValueError(f"infeasible_scenario must be one of {sorted(allowed)}")
    scenario = region_place_scenario(object_shape, seed)
    scenario["scenario_id"] = f"infeasible_{failure.lower()}_{int(seed):03d}"
    scenario["infeasible_scenario"] = failure
    scenario["injected_exception"] = failure
    scenario["expected_geometry_feasible"] = False
    if failure == "TARGET_REGION_TOO_SMALL":
        scenario["target_inner_extent"] = (0.03, 0.03, 0.0)
    return scenario


def workspace_edge_scenario(seed: int) -> dict[str, Any]:
    """Return ten feasible edge-grasp cases spanning both lateral sides."""
    seed = int(seed)
    index = (seed - 1) % 10
    generation = (seed - 1) // 10
    if generation >= 2:
        rng = np.random.default_rng(seed)
        half_sizes = rng.uniform(0.018, 0.025, size=3)
        infeasible = seed % 10 == 0
        target_extent = (
            np.array([0.04, 0.04])
            if infeasible
            else rng.uniform([0.13, 0.15], [0.20, 0.23])
        )
        return {
            "scenario_id": f"workspace_edge_random_{seed:03d}",
            "object_shape": "cube",
            "object_half_sizes": tuple(np.round(half_sizes, 4)),
            "object_xy": tuple(np.round([
                rng.uniform(-0.10, 0.08), rng.uniform(-0.17, -0.13)
            ], 4)),
            "object_yaw_degrees": float(np.round(rng.uniform(-80.0, 80.0), 1)),
            "bin_offset_xy": tuple(np.round(rng.uniform(-0.025, 0.025, 2), 4)),
            "bin_yaw_degrees": float(np.round(rng.uniform(-30.0, 30.0), 1)),
            "target_inner_extent": (
                float(np.round(target_extent[0], 4)),
                float(np.round(target_extent[1], 4)),
                0.08,
            ),
            "rim_height": 0.08,
            "injected_exception": "INFEASIBLE_TASK" if infeasible else None,
            "expected_geometry_feasible": not infeasible,
        }
    object_x, object_y, object_yaw, half_sizes = (
        (-0.10, -0.14, -45, (0.018, 0.018, 0.018)),
        (-0.05, 0.42, -30, (0.020, 0.020, 0.020)),
        (0.00, -0.15, -15, (0.022, 0.018, 0.020)),
        (0.05, 0.43, 0, (0.018, 0.022, 0.020)),
        (0.02, -0.16, 15, (0.024, 0.020, 0.018)),
        (-0.08, 0.44, 30, (0.020, 0.024, 0.018)),
        (-0.03, -0.17, 45, (0.025, 0.018, 0.022)),
        (0.03, 0.43, 60, (0.018, 0.025, 0.022)),
        (0.08, -0.15, 75, (0.022, 0.022, 0.025)),
        (-0.06, 0.46, 90, (0.020, 0.020, 0.025)),
    )[index]
    bin_offset_xy = (0.0, 0.0)
    bin_yaw_degrees = 0
    if generation:
        # Preserve the validated first ten cases while producing genuinely new
        # deterministic holdouts for later seed decades.
        direction = -1.0 if index % 2 else 1.0
        object_x += direction * (0.006 + 0.002 * (generation % 3))
        # The simulated table only supports the negative lateral workspace
        # edge.  The corresponding positive-side coordinates are already past
        # the tabletop, so reserve them for infeasibility tests rather than
        # counting inevitable falls as recovery failures.
        object_y = -(0.130 + 0.0015 * float(index))
        object_yaw += direction * (5 + 2 * (generation % 4))
        size_delta = 0.001 * (1 + generation % 2)
        half_sizes = (
            half_sizes[0] + (size_delta if index % 3 == 0 else 0.0),
            half_sizes[1] + (size_delta if index % 3 == 1 else 0.0),
            half_sizes[2] + (size_delta if index % 3 == 2 else 0.0),
        )
        bin_offset_xy = (
            direction * (0.008 + 0.001 * generation),
            (float(index % 3) - 1.0) * 0.008,
        )
        bin_yaw_degrees = direction * (8 + 2 * (generation % 5))
    return {
        "scenario_id": (
            f"workspace_edge_{index + 1:02d}"
            if generation == 0
            else f"workspace_edge_{index + 1:02d}_g{generation:02d}"
        ),
        "object_shape": "cube",
        "object_half_sizes": half_sizes,
        "object_xy": (object_x, object_y),
        "object_yaw_degrees": object_yaw,
        "bin_offset_xy": bin_offset_xy,
        "bin_yaw_degrees": bin_yaw_degrees,
        "target_inner_extent": (0.175, 0.225, 0.080),
        "rim_height": 0.080,
        "injected_exception": None,
        "expected_geometry_feasible": True,
    }


def box_inside_open_bin(
    object_position: np.ndarray,
    object_quaternion_wxyz: np.ndarray,
    object_half_sizes: np.ndarray,
    floor_center: np.ndarray,
    inner_extent: np.ndarray,
    *,
    container_quaternion_wxyz: np.ndarray | None = None,
    margin: float = 0.005,
    tolerance: float = 0.008,
) -> bool:
    """Return whether the complete oriented box is inside the open bin."""
    position = np.asarray(object_position, dtype=float)
    half_sizes = np.asarray(object_half_sizes, dtype=float)
    floor_center = np.asarray(floor_center, dtype=float)
    inner_extent = np.asarray(inner_extent, dtype=float)
    container_rotation = _rotation_matrix_wxyz(
        np.asarray(container_quaternion_wxyz, dtype=float)
        if container_quaternion_wxyz is not None
        else np.array([1.0, 0.0, 0.0, 0.0])
    )
    relative_position = container_rotation.T @ (position - floor_center)
    relative_rotation = container_rotation.T @ _rotation_matrix_wxyz(
        object_quaternion_wxyz
    )
    projected_half = np.abs(relative_rotation) @ half_sizes
    horizontal_limit = 0.5 * inner_extent[:2] - projected_half[:2] - margin
    if np.any(horizontal_limit < 0.0):
        return False
    horizontal_inside = np.all(
        np.abs(relative_position[:2]) <= horizontal_limit + tolerance
    )
    bottom = relative_position[2] - projected_half[2]
    top = relative_position[2] + projected_half[2]
    vertical_inside = (
        bottom >= -tolerance
        and top <= inner_extent[2] + tolerance
    )
    return bool(horizontal_inside and vertical_inside)


def box_inside_target_region(
    object_position: np.ndarray,
    object_quaternion_wxyz: np.ndarray,
    object_half_sizes: np.ndarray,
    surface_center: np.ndarray,
    region_extent: np.ndarray,
    *,
    region_quaternion_wxyz: np.ndarray | None = None,
    margin: float = 0.005,
    horizontal_tolerance: float = 0.008,
    surface_tolerance: float = 0.012,
) -> bool:
    """Return whether the complete object footprint rests in a target region."""
    region_rotation = _rotation_matrix_wxyz(
        np.asarray(region_quaternion_wxyz, dtype=float)
        if region_quaternion_wxyz is not None
        else np.array([1.0, 0.0, 0.0, 0.0])
    )
    relative_position = region_rotation.T @ (
        np.asarray(object_position, dtype=float) - np.asarray(surface_center, dtype=float)
    )
    relative_rotation = region_rotation.T @ _rotation_matrix_wxyz(
        object_quaternion_wxyz
    )
    projected_half = np.abs(relative_rotation) @ np.asarray(
        object_half_sizes, dtype=float
    )
    horizontal_limit = (
        0.5 * np.asarray(region_extent, dtype=float)[:2]
        - projected_half[:2]
        - margin
    )
    bottom = relative_position[2] - projected_half[2]
    return bool(
        np.all(horizontal_limit >= 0.0)
        and np.all(
            np.abs(relative_position[:2])
            <= horizontal_limit + horizontal_tolerance
        )
        and abs(float(bottom)) <= surface_tolerance
    )


def planar_footprint_extent(
    object_quaternion_wxyz: np.ndarray,
    target_quaternion_wxyz: np.ndarray,
    object_half_sizes: np.ndarray,
) -> np.ndarray:
    """Project the settled object's XY footprint into the target yaw frame."""
    relative = (
        _rotation_matrix_wxyz(target_quaternion_wxyz).T
        @ _rotation_matrix_wxyz(object_quaternion_wxyz)
    )
    yaw = float(np.arctan2(relative[1, 0], relative[0, 0]))
    c, s = abs(float(np.cos(yaw))), abs(float(np.sin(yaw)))
    half_x, half_y = np.asarray(object_half_sizes, dtype=float)[:2]
    return 2.0 * np.array([
        c * half_x + s * half_y,
        s * half_x + c * half_y,
    ])


def fit_planar_rectangle(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a robust minimum-area XY rectangle without simulator geometry."""
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 20:
        raise ValueError("at least 20 finite XY points are required")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 20:
        raise ValueError("at least 20 finite XY points are required")
    points = points - np.median(points, axis=0)
    best: tuple[float, np.ndarray, np.ndarray, float] | None = None
    for yaw in np.linspace(0.0, np.pi / 2.0, 91, endpoint=False):
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        rotation = np.array([[c, s], [-s, c]])
        local = points @ rotation.T
        lower, upper = np.percentile(local, [3.0, 97.0], axis=0)
        extent = upper - lower
        candidate = (float(np.prod(extent)), lower, upper, yaw)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    _, lower, upper, yaw = best
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    local_to_world = np.array([[c, -s], [s, c]])
    center = np.median(np.asarray(points_xy, dtype=float), axis=0) + (
        0.5 * (lower + upper) @ local_to_world.T
    )
    return center, np.maximum(upper - lower, 0.008), yaw


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 4-connected foreground component in a binary image."""
    source = np.asarray(mask, dtype=bool)
    remaining = {tuple(item) for item in np.argwhere(source)}
    largest: list[tuple[int, int]] = []
    height, width = source.shape
    while remaining:
        start = remaining.pop()
        component = [start]
        queue = deque([start])
        while queue:
            row, col = queue.popleft()
            for neighbor in (
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1),
            ):
                if (
                    0 <= neighbor[0] < height
                    and 0 <= neighbor[1] < width
                    and neighbor in remaining
                ):
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    queue.append(neighbor)
        if len(component) > len(largest):
            largest = component
    result = np.zeros_like(source)
    if largest:
        rows, cols = zip(*largest)
        result[np.asarray(rows), np.asarray(cols)] = True
    return result


class _CubeBinPlaceTask(PickPlace):
    """Robosuite's native bin arena with one parameterized cube."""

    def __init__(
        self,
        cube_half_sizes: tuple[float, float, float],
        object_shape: str = "cube",
        target_obstacle_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.cube_half_sizes = np.asarray(cube_half_sizes, dtype=float)
        self.object_shape = object_shape
        self.target_obstacle_mode = target_obstacle_mode
        super().__init__(single_object_mode=0, **kwargs)
        self.object_to_id = {"primary": 0, **({"obstacle": 1} if target_obstacle_mode else {})}

    def _construct_visual_objects(self) -> None:
        self.visual_objects = []

    def _construct_objects(self) -> None:
        object_class = CylinderObject if self.object_shape == "cylinder" else BoxObject
        size = (
            [self.cube_half_sizes[0], self.cube_half_sizes[2]]
            if self.object_shape == "cylinder"
            else self.cube_half_sizes
        )
        self.objects = [
            object_class(
                name="Primary",
                size=size,
                rgba=[1.0, 0.0, 0.0, 1.0],
            )
        ]
        if self.target_obstacle_mode:
            obstacle_sizes = {
                "center": [0.04, 0.04, 0.04],
                "edge": [0.03, 0.06, 0.03],
                "entry": [0.07, 0.025, 0.05],
                "removed": [0.04, 0.04, 0.04],
            }
            self.objects.append(BoxObject(
                name="TargetObstacle",
                size=obstacle_sizes[self.target_obstacle_mode],
                rgba=[0.15, 0.15, 0.15, 1.0],
            ))


class FrankaRobosuiteBinPlaceLowLevel(RobosuiteBaseEnv):
    """Place one cube fully inside the lower-left target-bin compartment."""

    _SUBSAMPLE_RATE = 5
    _SWEEP_CAMERA_FOVY = 70.0
    _BIN_WALL_THICKNESS = 0.02
    _BIN_FLOOR_TOP_OFFSET = 0.02
    _BIN_INTERIOR_HEIGHT = 0.08
    _CAMERA_STABLE_POSITION_TOLERANCE = 0.005
    _CYLINDER_SURFACE_SPEED_TOLERANCE = 0.025
    _TARGET_CHANGE_SHIFT = 0.14
    _UNKNOWN_ANOMALIES = (
        "TEMPORARY_BIN_OCCLUSION",
        "TARGET_POSITION_CHANGED",
        "OBJECT_OUT_OF_WORKSPACE",
    )

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1500,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = False,
        enable_render: bool = False,
        cube_half_sizes: str | list[float] | tuple[float, float, float] = (0.02, 0.02, 0.02),
        inject_primary_release_failures: int = 0,
        inject_first_primary_release_unstable: bool = False,
        inject_first_grasp_miss: bool = False,
        inject_grasp_miss_attempts: int | None = None,
        primary_release_fault_offset: float = 0.10,
        object_shape: str = "cube",
        target_mode: str = "bin",
        scenario_sweep: bool = False,
        workspace_edge_sweep: bool = False,
        scenario_seed: int | str | None = None,
        unknown_anomaly_sweep: bool = False,
        unknown_anomaly_id: str | None = None,
        infeasible_scenario: str | None = None,
        camera_failure_mode: str | None = None,
        target_obstacle_mode: str | None = None,
        boundary_just_fit: bool = False,
        recovery_checkpoint_dir: str | None = None,
        recovery_checkpoint_required: bool = False,
        object_out_of_workspace_offset: str | list[float] | tuple[float, float, float] = (
            0.65, -0.45, 0.0,
        ),
        world_state_source: str = "simulator",
        target_region_center_robot_base: str | list[float] | tuple[float, float, float] = (
            0.5025, 0.2575, -0.102,
        ),
        target_region_extent_robot_base: str | list[float] | tuple[float, float, float] = (
            0.175, 0.225, 0.08,
        ),
    ) -> None:
        super().__init__(
            controller_cfg=controller_cfg,
            max_steps=max_steps,
            seed=seed,
            viser_debug=False,
            privileged=privileged,
            enable_render=enable_render,
        )
        if isinstance(cube_half_sizes, str):
            cube_half_sizes = cube_half_sizes.split(",")
        sizes = np.asarray(cube_half_sizes, dtype=float)
        if sizes.shape != (3,) or not np.all(np.isfinite(sizes)) or np.any(sizes <= 0):
            raise ValueError("cube_half_sizes must contain three positive finite values")
        inject_primary_release_failures = int(inject_primary_release_failures)
        if inject_grasp_miss_attempts is None:
            inject_grasp_miss_attempts = int(_as_bool(inject_first_grasp_miss))
        inject_grasp_miss_attempts = int(inject_grasp_miss_attempts)
        primary_release_fault_offset = float(primary_release_fault_offset)
        if inject_primary_release_failures < 0:
            raise ValueError("inject_primary_release_failures must be non-negative")
        if inject_grasp_miss_attempts < 0:
            raise ValueError("inject_grasp_miss_attempts must be non-negative")
        if primary_release_fault_offset <= 0:
            raise ValueError("primary_release_fault_offset must be positive")
        if world_state_source not in {"simulator", "camera_rgbd"}:
            raise ValueError("world_state_source must be 'simulator' or 'camera_rgbd'")
        if object_shape not in {"cube", "cuboid", "cylinder"}:
            raise ValueError("object_shape must be cube, cuboid, or cylinder")
        if target_mode not in {"bin", "region"}:
            raise ValueError("target_mode must be bin or region")
        target_center = self._vector3(
            target_region_center_robot_base, "target_region_center_robot_base"
        )
        target_extent = self._vector3(
            target_region_extent_robot_base, "target_region_extent_robot_base"
        )
        if np.any(target_extent <= 0):
            raise ValueError("target_region_extent_robot_base must be positive")
        self.cube_half_sizes = tuple(float(value) for value in sizes)
        self.object_shape = object_shape
        self.target_mode = target_mode
        self.scenario_sweep = _as_bool(scenario_sweep)
        self.workspace_edge_sweep = _as_bool(workspace_edge_sweep)
        self.unknown_anomaly_sweep = _as_bool(unknown_anomaly_sweep)
        self.unknown_anomaly_id = (
            str(unknown_anomaly_id).strip() if unknown_anomaly_id else None
        )
        if (
            self.unknown_anomaly_id is not None
            and self.unknown_anomaly_id not in self._UNKNOWN_ANOMALIES
        ):
            raise ValueError(
                f"unknown_anomaly_id must be one of {self._UNKNOWN_ANOMALIES}"
            )
        self.infeasible_scenario = (
            str(infeasible_scenario).strip().upper() if infeasible_scenario else None
        )
        self.camera_failure_mode = (
            str(camera_failure_mode).strip().lower() if camera_failure_mode else None
        )
        if self.camera_failure_mode not in {
            None, "black_frame", "occlusion", "sam_no_detection", "stale_observation",
            "transient_drop",
        }:
            raise ValueError("unsupported camera_failure_mode")
        self.target_obstacle_mode = (
            str(target_obstacle_mode).strip().lower() if target_obstacle_mode else None
        )
        if self.target_obstacle_mode not in {None, "center", "edge", "entry", "removed"}:
            raise ValueError("target_obstacle_mode must be center, edge, entry, or removed")
        self.boundary_just_fit = _as_bool(boundary_just_fit)
        self.recovery_checkpoint_dir = (
            Path(recovery_checkpoint_dir) if recovery_checkpoint_dir else None
        )
        self.recovery_checkpoint_required = _as_bool(recovery_checkpoint_required)
        self._loaded_recovery_checkpoint: dict[str, Any] | None = None
        self.object_out_of_workspace_offset = self._vector3(
            object_out_of_workspace_offset, "object_out_of_workspace_offset"
        )
        self.scenario_seed = (
            int(scenario_seed)
            if scenario_seed is not None and str(scenario_seed).strip()
            else None
        )
        if self.scenario_seed is not None and not 1 <= self.scenario_seed <= 10:
            raise ValueError("scenario_seed must be between 1 and 10")
        if self.infeasible_scenario is not None:
            infeasible_region_place_scenario(
                self.object_shape, self.scenario_seed or seed or 1,
                self.infeasible_scenario,
            )
        self._active_scenario: dict[str, Any] | None = None
        self._active_target_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self._active_rim_height = float(target_extent[2])
        self.inject_primary_release_failures = int(inject_primary_release_failures)
        self._configured_primary_release_failures = int(inject_primary_release_failures)
        self.inject_first_primary_release_unstable = _as_bool(
            inject_first_primary_release_unstable
        )
        self._configured_first_primary_release_unstable = _as_bool(
            inject_first_primary_release_unstable
        )
        self.inject_first_grasp_miss = _as_bool(inject_first_grasp_miss)
        self.inject_grasp_miss_attempts = inject_grasp_miss_attempts
        self._grasp_misses_injected = 0
        self._grasp_miss_injected = False
        self.primary_release_fault_offset = float(primary_release_fault_offset)
        self._primary_release_faults_injected = 0
        self._grasp_misses_injected = 0
        self._grasp_miss_injected = False
        self._grasp_miss_injected = False
        self._pending_primary_release_fault_steps: int | None = None
        self._unstable_release_fault_active = False
        self._active_exception_id: str | None = None
        self._unknown_anomaly_applied = False
        self._target_change_armed = False
        self._blocked_stale_goto_attempts = 0
        self._blocked_release_attempts = 0
        self._visual_occlusion_steps = 0
        self._clear_target_cache_after_blocked_stale_motion = False
        self.world_state_source = world_state_source
        self._calibrated_target_center = target_center
        self._calibrated_target_extent = target_extent
        self._camera_state_positions: list[np.ndarray] = []
        self._camera_state_last_measurement = None
        self._camera_target_last_measurement = None
        self._camera_rgbd_cache = None
        self._camera_measurement_calls = 0
        self._camera_state_segment_fn = None
        if self.world_state_source == "camera_rgbd":
            from capx.integrations.vision.sam3 import init_sam3

            self._camera_state_segment_fn = init_sam3()
            self.get_text_world_state = self._get_camera_text_world_state
            # The wrist-adjacent robot view cannot see both the source table and
            # the target region. A fixed overhead RGB-D view observes both.
            self.save_camera_name = "birdview"
            self.render_camera_names = [self.save_camera_name]
        visual_name = {
            "cube": "red cube",
            "cuboid": "red rectangular block",
            "cylinder": "red cylinder",
        }[self.object_shape]
        self.role_object_names = {
            "primary": visual_name,
            "target": "target bin" if target_mode == "bin" else "green target region",
        }

        camera_kwargs = {
            "robots": ["Panda"],
            "controller_configs": load_composite_controller_config(controller=self.controller_cfg),
            "horizon": max_steps,
            "reward_shaping": False,
            "cube_half_sizes": self.cube_half_sizes,
            "object_shape": self.object_shape,
            "target_obstacle_mode": self.target_obstacle_mode,
            "seed": seed,
        }
        if privileged and not enable_render:
            camera_kwargs.update(
                use_camera_obs=False,
                has_renderer=False,
                has_offscreen_renderer=False,
                camera_names=[],
                renderer="mujoco",
            )
            self.render_camera_names = []
        else:
            camera_kwargs.update(
                has_renderer=not privileged,
                has_offscreen_renderer=True,
                camera_names=self.render_camera_names,
                camera_depths=True,
                camera_heights=self._render_height,
                camera_widths=self._render_width,
                renderer="mujoco",
            )
        self._robosuite_task_kwargs = dict(camera_kwargs)
        self.robosuite_env = _CubeBinPlaceTask(**camera_kwargs)
        self._compiled_object_half_sizes = self.cube_half_sizes
        self.primary_object = self.robosuite_env.objects[0]
        self.primary_body_id = self.robosuite_env.obj_body_id[self.primary_object.name]
        self.target_obstacle = (
            self.robosuite_env.objects[1] if self.target_obstacle_mode else None
        )

        self.register_world_body(
            entity_id="cube:primary",
            semantic_name="red cube",
            body_name=self.primary_object.root_body,
            attributes={"task_role": "primary"},
        )
        self.register_world_body(
            entity_id="container:target",
            semantic_name="target bin" if target_mode == "bin" else "target region",
            body_name="bin2",
            entity_type="container",
            attributes={"task_role": "target"},
        )
        self._init_robot_links()
        self._init_viser_debug(viser_debug)

    @staticmethod
    def _vector3(value: str | list[float] | tuple[float, float, float], name: str) -> np.ndarray:
        if isinstance(value, str):
            value = value.replace(",", " ").split()
        result = np.asarray(value, dtype=float)
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain three finite values")
        return result

    def _target_geometry_world(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._active_scenario is not None:
            body_position = np.asarray(self.robosuite_env.bin2_pos, dtype=float)
            yaw = np.deg2rad(float(self._active_scenario["bin_yaw_degrees"]))
            body_quaternion = np.array([
                np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)
            ])
            rotation = _rotation_matrix_wxyz(body_quaternion)
            inner_extent = np.asarray(
                self._active_scenario["target_inner_extent"], dtype=float
            )
            outer_wall_center = inner_extent[:2] + self._BIN_WALL_THICKNESS
            local_center = np.array([
                0.0 if self.target_mode == "region" else -0.5 * outer_wall_center[0],
                0.0 if self.target_mode == "region" else -0.5 * outer_wall_center[1],
                self._BIN_FLOOR_TOP_OFFSET,
            ])
            center = body_position + rotation @ local_center
            return center, inner_extent, body_quaternion
        bin_size = np.asarray(self.robosuite_env.bin_size, dtype=float)
        center = np.asarray(self.robosuite_env.bin2_pos, dtype=float).copy()
        center[:2] -= bin_size[:2] / 4.0
        center[2] += self._BIN_FLOOR_TOP_OFFSET
        inner_extent = np.array([
            bin_size[0] / 2.0 - self._BIN_WALL_THICKNESS,
            bin_size[1] / 2.0 - self._BIN_WALL_THICKNESS,
            self._BIN_INTERIOR_HEIGHT,
        ])
        sim = self.robosuite_env.sim
        body_id = sim.model.body_name2id("bin2")
        return center, inner_extent, np.asarray(sim.model.body_quat[body_id], dtype=float)

    def _world_point_to_robot_base(self, point: np.ndarray) -> np.ndarray:
        base = vtf.SE3(wxyz_xyz=np.concatenate([
            self.robosuite_env.sim.data.xquat[self.base_link_idx],
            self.robosuite_env.sim.data.xpos[self.base_link_idx],
        ]))
        return (base.inverse() @ vtf.SE3.from_translation(point)).translation()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._active_scenario = (
            workspace_edge_scenario(self.scenario_seed or seed or 1)
            if self.workspace_edge_sweep
            else (
                region_place_scenario(
                    self.object_shape, self.scenario_seed or seed or 1
                )
                if self.target_mode == "region"
                else bin_place_scenario(
                    self.object_shape, self.scenario_seed or seed or 1
                )
            )
            if self.scenario_sweep
            else None
        )
        if self.infeasible_scenario is not None:
            self._active_scenario = infeasible_region_place_scenario(
                self.object_shape, seed or self.scenario_seed or 1,
                self.infeasible_scenario,
            )
        if self._active_scenario is not None and self.boundary_just_fit:
            self._active_scenario = dict(self._active_scenario)
            half_x, half_y = self._active_scenario["object_half_sizes"][:2]
            orientation_invariant_extent = 2.0 * float(np.hypot(half_x, half_y))
            self._active_scenario["target_inner_extent"] = (
                orientation_invariant_extent + 0.02,
                orientation_invariant_extent + 0.02,
                0.0,
            )
            self._active_scenario["scenario_id"] += "_just_fit"
            self._active_scenario["expected_geometry_feasible"] = True
        if self._active_scenario is not None:
            desired_sizes = tuple(
                float(value) for value in self._active_scenario["object_half_sizes"]
            )
            if desired_sizes != self._compiled_object_half_sizes:
                self._rebuild_robosuite_task(desired_sizes)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.robosuite_env.seed = seed
            self.robosuite_env.rng = np.random.default_rng(seed)
            initializer = self.robosuite_env.placement_initializer
            initializer.rng = self.robosuite_env.rng
            for sampler in getattr(initializer, "samplers", {}).values():
                sampler.rng = self.robosuite_env.rng

        self.robosuite_env.reset()
        if self._active_scenario is not None:
            self._apply_active_scenario()
        if self.infeasible_scenario == "OBJECT_UNREACHABLE":
            qpos = np.array(
                self.robosuite_env.sim.data.get_joint_qpos(
                    self.primary_object.joints[0]
                ),
                dtype=float,
                copy=True,
            )
            qpos[:3] += self.object_out_of_workspace_offset
            self.robosuite_env.sim.data.set_joint_qpos(
                self.primary_object.joints[0], qpos
            )
            self.robosuite_env.sim.forward()
        self.robosuite_env.sim.data.qpos[6] -= np.pi
        self._step_count = 0
        self._sim_step_count = 0
        self._primary_release_faults_injected = 0
        self._pending_primary_release_fault_steps = None
        self._unstable_release_fault_active = False
        self._active_exception_id = None
        self._unknown_anomaly_applied = False
        self._target_change_armed = False
        self._blocked_stale_goto_attempts = 0
        self._blocked_release_attempts = 0
        self._visual_occlusion_steps = 0
        self._clear_target_cache_after_blocked_stale_motion = False
        self._loaded_recovery_checkpoint = None
        if self.unknown_anomaly_id is not None:
            self._active_exception_id = self.unknown_anomaly_id
        elif self.unknown_anomaly_sweep:
            trial_seed = int(seed or 1)
            self._active_exception_id = self._UNKNOWN_ANOMALIES[(trial_seed - 1) % 3]
        self._camera_state_positions = []
        self._camera_state_last_measurement = None
        self._camera_target_last_measurement = None
        self._camera_rgbd_cache = None
        self._camera_measurement_calls = 0
        if self._active_scenario is not None:
            exception_id = self._active_scenario["injected_exception"]
            self.inject_primary_release_failures = (
                int(exception_id is not None)
                if exception_id is not None
                else self._configured_primary_release_failures
            )
            self.inject_first_primary_release_unstable = (
                exception_id == "UNSTABLE_RELEASE"
                if exception_id is not None
                else self._configured_first_primary_release_unstable
            )
        for _ in range(50):
            self.robosuite_env.sim.forward()
            self.robosuite_env.sim.step()
            self._set_gripper(1.0)

        if self.recovery_checkpoint_dir is not None:
            from capx.recovery.structural_unknown_ab import (
                checkpoint_path,
                load_region_place_checkpoint,
            )

            recovery_path = checkpoint_path(
                self.recovery_checkpoint_dir, int(seed or self.scenario_seed or 1)
            )
            if recovery_path.exists():
                load_region_place_checkpoint(
                    self, recovery_path, expected_seed=int(seed or 1)
                )
            elif self.recovery_checkpoint_required:
                raise FileNotFoundError(
                    f"required recovery checkpoint is missing: {recovery_path}"
                )

        robosuite_obs = self.robosuite_env._get_observations()
        self._current_joints = np.asarray(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
        self._current_joints[6] -= np.pi
        obs = self.get_observation()
        self.gripper_link_wxyz_xyz = np.concatenate([
            self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
            self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
        ])
        info = {
            "task_prompt": (
                "Place the primary object fully inside the target bin and open the gripper. Quaternions are WXYZ."
                if self.target_mode == "bin"
                else "Place the primary object fully inside the marked target region and open the gripper. Quaternions are WXYZ."
            ),
            "scenario": dict(self._active_scenario or {}),
        }
        reset_metadata: dict[str, Any] = {"seed": seed}
        if self._loaded_recovery_checkpoint is not None:
            reset_metadata["recovery_checkpoint"] = {
                "checkpoint_sha256": self._loaded_recovery_checkpoint[
                    "checkpoint_sha256"
                ],
                "anomaly_class": self._loaded_recovery_checkpoint["anomaly_class"],
                "physical_feasibility_precheck": self._loaded_recovery_checkpoint[
                    "physical_feasibility_precheck"
                ],
            }
        self.record_world_state("reset", observation=obs, metadata=reset_metadata)
        return obs, info

    def _rebuild_robosuite_task(
        self, half_sizes: tuple[float, float, float]
    ) -> None:
        """Recompile MuJoCo so per-trial object dimensions affect collisions."""
        self.robosuite_env.close()
        self.cube_half_sizes = half_sizes
        kwargs = dict(self._robosuite_task_kwargs)
        kwargs["cube_half_sizes"] = half_sizes
        self.robosuite_env = _CubeBinPlaceTask(**kwargs)
        self._compiled_object_half_sizes = half_sizes
        self.primary_object = self.robosuite_env.objects[0]
        self.primary_body_id = self.robosuite_env.obj_body_id[self.primary_object.name]
        self.target_obstacle = (
            self.robosuite_env.objects[1] if self.target_obstacle_mode else None
        )
        self._init_robot_links()

    def _apply_active_scenario(self) -> None:
        """Apply one deterministic sweep condition to compiled MuJoCo geometry."""
        scenario = self._active_scenario
        assert scenario is not None
        sim = self.robosuite_env.sim
        inner_extent = np.asarray(scenario["target_inner_extent"], dtype=float)
        rim_height = float(scenario["rim_height"])
        wall_half = 0.5 * self._BIN_WALL_THICKNESS
        outer_wall_center = inner_extent[:2] + self._BIN_WALL_THICKNESS
        region_local_center = np.zeros(2)
        wall_vertical_half = 0.5 * (self._BIN_FLOOR_TOP_OFFSET + rim_height)

        body_id = sim.model.body_name2id("bin2")
        bin_position = np.array([
            0.1, 0.24 if self.target_mode == "region" else 0.28, 0.8
        ], dtype=float)
        bin_position[:2] += np.asarray(scenario["bin_offset_xy"], dtype=float)
        yaw = np.deg2rad(float(scenario["bin_yaw_degrees"]))
        bin_quaternion = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])
        sim.model.body_pos[body_id] = bin_position
        sim.model.body_quat[body_id] = bin_quaternion
        self.robosuite_env.bin2_pos = bin_position.copy()
        self.robosuite_env.bin_size[:2] = 2.0 * outer_wall_center

        geom_start = int(sim.model.body_geomadr[body_id])
        geom_stop = geom_start + int(sim.model.body_geomnum[body_id])
        for geom_id in range(geom_start, geom_stop):
            position = sim.model.geom_pos[geom_id]
            size = sim.model.geom_size[geom_id]
            if position[2] < -0.1:
                continue
            if abs(position[2]) < 1e-6 and size[2] <= 0.021:
                if self.target_mode == "region":
                    position[:2] = region_local_center
                    size[:3] = [0.5 * inner_extent[0], 0.5 * inner_extent[1], 0.02]
                    if int(sim.model.geom_group[geom_id]) == 1:
                        sim.model.geom_matid[geom_id] = -1
                        sim.model.geom_rgba[geom_id] = [0.05, 0.85, 0.20, 1.0]
                else:
                    size[:3] = [outer_wall_center[0] + wall_half,
                                outer_wall_center[1] + wall_half, 0.02]
                continue
            if position[2] <= 0.0:
                continue
            if self.target_mode == "region":
                sim.model.geom_contype[geom_id] = 0
                sim.model.geom_conaffinity[geom_id] = 0
                sim.model.geom_rgba[geom_id, 3] = 0.0
                continue
            position[2] = wall_vertical_half
            size[2] = wall_vertical_half
            if size[0] < size[1]:
                size[:2] = [wall_half, outer_wall_center[1] + wall_half]
                position[0] = np.sign(position[0]) * outer_wall_center[0]
            else:
                size[:2] = [outer_wall_center[0] + wall_half, wall_half]
                position[1] = np.sign(position[1]) * outer_wall_center[1]

        half_sizes = np.asarray(self.cube_half_sizes, dtype=float)

        object_position = np.asarray(self.robosuite_env.bin1_pos, dtype=float).copy()
        object_position[:2] += np.asarray(scenario["object_xy"], dtype=float)
        object_position[2] += self._BIN_FLOOR_TOP_OFFSET + half_sizes[2]
        object_yaw = np.deg2rad(float(scenario["object_yaw_degrees"]))
        object_quaternion = np.array([
            np.cos(0.5 * object_yaw), 0.0, 0.0, np.sin(0.5 * object_yaw)
        ])
        sim.data.set_joint_qpos(
            self.primary_object.joints[0],
            np.concatenate([object_position, object_quaternion]),
        )
        if self.target_obstacle is not None:
            floor_world, _, target_quaternion_world = self._target_geometry_world()
            obstacle_half_z = {
                "center": 0.04, "edge": 0.03, "entry": 0.05, "removed": 0.04,
            }[self.target_obstacle_mode]
            local_offset = {
                "center": np.array([0.0, 0.0, obstacle_half_z]),
                "edge": np.array([
                    0.5 * inner_extent[0] - 0.025, 0.0, obstacle_half_z
                ]),
                "entry": np.array([
                    0.0, -0.5 * inner_extent[1] - 0.03, obstacle_half_z
                ]),
                "removed": np.array([0.0, -1.0, obstacle_half_z]),
            }[self.target_obstacle_mode]
            obstacle_position = (
                floor_world
                + _rotation_matrix_wxyz(target_quaternion_world) @ local_offset
            )
            sim.data.set_joint_qpos(
                self.target_obstacle.joints[0],
                np.concatenate([obstacle_position, target_quaternion_world]),
            )
        sim.forward()

        floor_world, _, target_quaternion_world = self._target_geometry_world()
        self._calibrated_target_center = self._world_point_to_robot_base(floor_world)
        base_rotation = vtf.SO3(wxyz=self.base_link_wxyz_xyz[:4])
        self._active_target_quaternion = (
            base_rotation.inverse() @ vtf.SO3(wxyz=target_quaternion_world)
        ).wxyz
        self._calibrated_target_extent = inner_extent
        self._active_rim_height = rim_height

    def get_target_region_state(self, target_name: str = "target") -> dict[str, Any]:
        """Return visual geometry in camera mode, simulator geometry otherwise."""
        if str(target_name).strip().lower() not in {"target", "target bin"}:
            raise ValueError(f"Target region is unavailable: {target_name}")
        if self.world_state_source == "camera_rgbd":
            measurement = getattr(self, "_camera_target_last_measurement", None)
            if measurement is None:
                measurement = self._measure_target_from_camera()
                self._camera_target_last_measurement = measurement
            floor_center, target_quaternion, inner_extent, _ = measurement
        else:
            floor_world, inner_extent, target_quaternion = self._target_geometry_world()
            floor_center = self._world_point_to_robot_base(floor_world)
            base_rotation = vtf.SO3(wxyz=self.base_link_wxyz_xyz[:4])
            target_quaternion = (
                base_rotation.inverse() @ vtf.SO3(wxyz=target_quaternion)
            ).wxyz
        observed_object = getattr(self, "_camera_state_last_measurement", None)
        object_half_height = (
            0.5 * float(observed_object[2][2])
            if self.world_state_source == "camera_rgbd" and observed_object is not None
            else float(self.cube_half_sizes[2])
        )
        object_center = floor_center + np.array([0.0, 0.0, object_half_height])
        rim_height = (
            0.0
            if self.world_state_source == "camera_rgbd" and self.target_mode == "region"
            else float(inner_extent[2])
        )
        return {
            "target_region_center_robot_base": floor_center.tolist(),
            "target_region_extent_robot_base": np.asarray(inner_extent).tolist(),
            "target_object_center_robot_base": object_center.tolist(),
            "target_region_quaternion_wxyz": np.asarray(target_quaternion).tolist(),
            "container_rim_z_robot_base": float(floor_center[2] + rim_height),
            "scenario_id": (
                self._active_scenario["scenario_id"]
                if getattr(self, "_active_scenario", None) is not None
                else "nominal"
            ),
        }

    def _capture_camera_rgbd(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Capture one RGB-D frame and calibrated camera transform per sim step."""
        sim_time = float(self.robosuite_env.sim.data.time)
        cached = getattr(self, "_camera_rgbd_cache", None)
        if cached is not None and cached[0] == sim_time:
            return cached[1]
        camera = self.get_observation()[self.save_camera_name]
        rgb, depth_buffer = self.robosuite_env.sim.render(
            camera_name=self.save_camera_name,
            width=self._render_width,
            height=self._render_height,
            depth=True,
        )
        rgb = np.asarray(rgb[::-1]).copy()
        failure_mode = getattr(self, "camera_failure_mode", None)
        if failure_mode == "black_frame":
            rgb.fill(0)
        elif failure_mode == "occlusion":
            rgb[:, : int(0.72 * rgb.shape[1])] = 8
        low_information_fraction = float(
            np.mean(np.max(rgb, axis=2) - np.min(rgb, axis=2) < 2)
        )
        if float(np.std(rgb)) < 2.0 or low_information_fraction > 0.70:
            raise RuntimeError("camera frame failed the visual-information gate")
        depth = np.asarray(get_real_depth_map(self.robosuite_env.sim, depth_buffer[::-1]))
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        camera_points = depth_to_pointcloud(
            depth, np.asarray(camera["intrinsics"], dtype=float), filter_invalid=False
        )
        captured = (rgb, depth, camera_points, np.asarray(camera["pose_mat"], dtype=float))
        self._camera_rgbd_cache = (sim_time, captured)
        return captured

    def _gripper_tool_position_robot_base(self) -> np.ndarray:
        """Return the tool-center position from robot proprioception."""
        tool = (
            vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz).inverse()
            @ vtf.SE3(wxyz_xyz=self.gripper_link_wxyz_xyz)
            @ vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2.0),
                translation=np.array([0.0, 0.0, -0.107]),
            )
        )
        return np.asarray(tool.translation(), dtype=float)

    def _actual_gripper_opening_fraction(self) -> float:
        """Read finger separation from robot proprioception, not command state."""
        observations = self.robosuite_env._get_observations()
        finger_qpos = np.asarray(observations["robot0_gripper_qpos"], dtype=float)
        return float(np.mean(np.abs(finger_qpos)) / self.gripper_metric_length)

    def _measure_primary_from_camera(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Estimate the primary pose from rendered RGB-D, never object-body state."""
        if self._camera_state_segment_fn is None:
            raise RuntimeError("camera world-state estimator is not initialized")
        failure_mode = getattr(self, "camera_failure_mode", None)
        self._camera_measurement_calls = getattr(self, "_camera_measurement_calls", 0) + 1
        if (
            failure_mode == "stale_observation"
            and self._camera_measurement_calls > 1
        ):
            raise RuntimeError("injected stale camera observation")
        if failure_mode == "transient_drop" and self._camera_measurement_calls == 2:
            raise RuntimeError("injected transient camera dropout")
        rgb, depth, camera_points, pose_mat = self._capture_camera_rgbd()
        results = self._camera_state_segment_fn(
            rgb, text_prompt=self.role_object_names["primary"]
        )
        if failure_mode == "sam_no_detection":
            results = []
        if not results:
            raise RuntimeError("camera world-state estimator found no primary object")
        measurements = []
        max_mask_pixels = depth.size * 0.08
        visual_name = self.role_object_names["primary"].lower()
        color_channel = next(
            (index for name, index in (("red", 0), ("green", 1), ("blue", 2))
             if name in visual_name),
            None,
        )
        color_mask = None
        if color_channel is not None:
            other_channels = [index for index in range(3) if index != color_channel]
            primary_channel = rgb[:, :, color_channel].astype(int)
            secondary_channels = rgb[:, :, other_channels].astype(int)
            color_mask = (
                (primary_channel > 120)
                & (primary_channel >= 1.5 * np.max(secondary_channels, axis=2))
            )
        for result in results:
            mask = np.asarray(result.get("mask"), dtype=bool)
            if mask.shape == depth.shape and color_mask is not None:
                if int(color_mask.sum()) < 20:
                    continue
                mask = color_mask
            mask_pixels = int(mask.sum()) if mask.shape == depth.shape else 0
            if not 20 <= mask_pixels <= max_mask_pixels:
                continue
            valid = (
                mask.reshape(-1)
                & np.isfinite(camera_points).all(axis=1)
                & (camera_points[:, 2] > 0)
            )
            if int(valid.sum()) < 20:
                continue
            selected = camera_points[valid]
            points_robot = (pose_mat[:3, :3] @ selected.T).T + pose_mat[:3, 3]
            lower, upper = np.percentile(points_robot, [5.0, 95.0], axis=0)
            observed_extent = upper - lower
            if np.any(observed_extent > 0.12):
                continue
            measurements.append((float(result.get("score", 0.0)), points_robot, lower, upper))
        if not measurements:
            raise RuntimeError("camera world-state estimator found no size-plausible object")
        score, points_robot, lower, upper = max(measurements, key=lambda item: item[0])
        center_xy, extent_xy, yaw = fit_planar_rectangle(points_robot[:, :2])
        observed_height = max(float(upper[2] - lower[2]), 0.008)
        if getattr(self, "object_shape", "cube") == "cube":
            observed_height = float(np.median(extent_xy))
        position = np.array([
            center_xy[0], center_xy[1], upper[2] - 0.5 * observed_height,
        ])
        observed_extent = np.array([extent_xy[0], extent_xy[1], observed_height])
        return (
            position,
            np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)]),
            observed_extent,
            score,
        )

    def get_rgbd_object_state(
        self, role: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Expose the single primary object through the shared RGB-D API."""
        requested = str(role).strip().lower()
        primary_name = str(self.role_object_names["primary"]).lower()
        if requested not in {"primary", primary_name}:
            raise ValueError(f"Unknown object role: {role}")
        return self._measure_primary_from_camera()

    def rebind_roles_from_rgbd(self) -> bool:
        """Re-observe the fixed single-object role and its target."""
        self._camera_state_last_measurement = self._measure_primary_from_camera()
        self._camera_target_last_measurement = self._measure_target_from_camera()
        self.record_world_state("roles_rebound", metadata={"source": "camera_rgbd"})
        return True

    def _measure_target_from_camera(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Estimate planar target center, yaw, and extent from segmented RGB-D."""
        if self._camera_state_segment_fn is None:
            raise RuntimeError("camera world-state estimator is not initialized")
        rgb, depth, camera_points, pose_mat = self._capture_camera_rgbd()
        results = self._camera_state_segment_fn(
            rgb, text_prompt=self.role_object_names["target"]
        )
        if getattr(self, "camera_failure_mode", None) == "sam_no_detection":
            results = []
        measurements = []
        green = rgb[:, :, 1].astype(int)
        green_mask = largest_connected_component(
            (green > 90)
            & (green >= 1.25 * rgb[:, :, 0].astype(int))
            & (green >= 1.25 * rgb[:, :, 2].astype(int))
        )
        for result in results:
            mask = np.asarray(result.get("mask"), dtype=bool)
            if mask.shape == depth.shape and int(green_mask.sum()) >= 80:
                mask = green_mask
            pixels = int(mask.sum()) if mask.shape == depth.shape else 0
            if not 80 <= pixels <= depth.size * 0.35:
                continue
            valid = (
                mask.reshape(-1)
                & np.isfinite(camera_points).all(axis=1)
                & (camera_points[:, 2] > 0)
            )
            if int(valid.sum()) < 80:
                continue
            selected = camera_points[valid]
            points_robot = (pose_mat[:3, :3] @ selected.T).T + pose_mat[:3, 3]
            median_z = float(np.median(points_robot[:, 2]))
            points_robot = points_robot[
                np.abs(points_robot[:, 2] - median_z) <= 0.015
            ]
            if len(points_robot) < 80:
                continue
            center_xy, extent_xy, yaw = fit_planar_rectangle(points_robot[:, :2])
            if np.any(extent_xy < 0.04) or np.any(extent_xy > 0.50):
                continue
            surface_z = float(np.median(points_robot[:, 2]))
            measurements.append((
                float(result.get("score", 0.0)), center_xy, extent_xy, yaw, surface_z
            ))
        if not measurements:
            raise RuntimeError("camera world-state estimator found no planar target region")
        score, center_xy, extent_xy, yaw, surface_z = max(
            measurements, key=lambda item: item[0]
        )
        return (
            np.array([center_xy[0], center_xy[1], surface_z]),
            np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)]),
            np.array([extent_xy[0], extent_xy[1], 0.0]),
            score,
        )

    def _get_camera_text_world_state(self) -> dict[str, Any]:
        object_fresh = True
        try:
            measurement = self._measure_primary_from_camera()
            self._camera_state_last_measurement = measurement
        except (RuntimeError, ValueError):
            measurement = getattr(self, "_camera_state_last_measurement", None)
            object_fresh = False
        target_measurement = getattr(self, "_camera_target_last_measurement", None)
        target_fresh = target_measurement is not None
        if target_measurement is None:
            try:
                target_measurement = self._measure_target_from_camera()
                self._camera_target_last_measurement = target_measurement
                target_fresh = True
            except (RuntimeError, ValueError):
                target_measurement = None
                target_fresh = False
        measurement_fresh = bool(object_fresh and target_fresh)
        if measurement is None or target_measurement is None:
            target = {
                    "target_region_center_robot_base": [0.0, 0.0, 0.0],
                    "target_region_extent_robot_base": [0.0, 0.0, 0.0],
                    "target_object_center_robot_base": [0.0, 0.0, 0.0],
                    "target_region_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "container_rim_z_robot_base": 0.0,
                    "scenario_id": (
                        self._active_scenario["scenario_id"]
                        if getattr(self, "_active_scenario", None) is not None
                        else "nominal"
                    ),
                }
            target_clear = getattr(self, "target_obstacle_mode", None) in {None, "removed"}
            return {
                    "world_state_source": "camera_rgbd",
                    "uses_simulator_truth": False,
                    "target_geometry_type": (
                        "planar_region" if self.target_mode == "region" else "container"
                    ),
                    "measurement_fresh": False,
                    "state_provenance": {
                        "object_pose": (
                            "sam3_rgbd" if measurement is not None else "sam3_rgbd_unavailable"
                        ),
                        "gripper": "robot_proprioception",
                        "target_region": (
                            "sam3_rgbd" if target_measurement is not None
                            else "sam3_rgbd_unavailable"
                        ),
                        "completion": "fail_closed",
                    },
                    **target,
                    "target_grasped": False,
                    "wrong_object_grasped": False,
                    "gripper_contact": False,
                    "gripper_open": bool(self._gripper_fraction >= 0.8),
                    "gripper_closed": bool(self._gripper_fraction <= 0.2),
                    "object_lifted": False,
                    "object_at_target": False,
                    "object_inside_container": False,
                    "object_inside_target_region": False,
                    "object_stable": False,
                    "container_entry_clear": False,
                    "bin_policy_eligible": False,
                    "target_region_policy_eligible": False,
                    "target_region_clear": target_clear,
                    "geometry_feasible": False,
                    "gripper_feasible": False,
                    "workspace_reachable": False,
                    "workspace_recovery_feasible": False,
                    "object_observation_ready": False,
                    "target_observation_ready": False,
                    "task_completed": False,
                    "reward": 0.0,
                    "recoverable_environment": False,
                    "confidence": 0.0,
            }
        position, quaternion, extent, detection_score = measurement
        proprioceptive_grasp_track = bool(
            not object_fresh
            and self._gripper_fraction <= 0.2
            and self._actual_gripper_opening_fraction() >= 0.15
        )
        if proprioceptive_grasp_track:
            position = self._gripper_tool_position_robot_base() - np.array([
                0.0, 0.0, max(0.004, 0.175 * float(extent[2]))
            ])
        measurement_fresh = bool(
            target_fresh and (object_fresh or proprioceptive_grasp_track)
        )
        if not object_fresh:
            detection_score = 0.0
        target = self.get_target_region_state()
        floor_center = np.asarray(target["target_region_center_robot_base"])
        inner_extent = np.asarray(target["target_region_extent_robot_base"])
        target_quaternion = np.asarray(target["target_region_quaternion_wxyz"])
        object_center = floor_center + np.array([0.0, 0.0, 0.5 * extent[2]])
        if self._gripper_fraction >= 0.8:
            position[2] = max(
                float(position[2]),
                float(floor_center[2] + 0.5 * extent[2]),
            )
        self._camera_state_positions.append(position.copy())
        self._camera_state_positions = self._camera_state_positions[-3:]
        stable = bool(
            object_fresh
            and (
                len(self._camera_state_positions) < 2
                or max(
                    np.linalg.norm(item - self._camera_state_positions[-1])
                    for item in self._camera_state_positions[:-1]
                ) <= self._CAMERA_STABLE_POSITION_TOLERANCE
            )
        )
        gripper_open = bool(self._gripper_fraction >= 0.8)
        gripper_closed = bool(self._gripper_fraction <= 0.2)
        lifted = bool(position[2] > floor_center[2] + 0.5 * extent[2] + 0.04)
        grasped = bool(
            gripper_closed and lifted and (object_fresh or proprioceptive_grasp_track)
        )
        inside = (
            box_inside_target_region(
                position,
                quaternion,
                0.5 * extent,
                floor_center,
                inner_extent,
                region_quaternion_wxyz=target_quaternion,
                horizontal_tolerance=0.015,
            )
            if getattr(self, "target_mode", "bin") == "region"
            else box_inside_open_bin(
                position,
                quaternion,
                0.5 * extent,
                floor_center,
                inner_extent,
                container_quaternion_wxyz=target_quaternion,
                tolerance=0.015,
            )
        )
        relative_rotation = (
            _rotation_matrix_wxyz(target_quaternion).T
            @ _rotation_matrix_wxyz(quaternion)
        )
        projected_extent = 2.0 * (
            np.abs(relative_rotation) @ (0.5 * np.asarray(extent))
        )
        footprint_extent = planar_footprint_extent(
            quaternion, target_quaternion, 0.5 * np.asarray(extent)
        )
        fit_margin = inner_extent[:2] - footprint_extent
        geometry_feasible = bool(
            np.all(fit_margin >= 0.01)
            and (
                getattr(self, "target_mode", "bin") == "region"
                or projected_extent[2] <= inner_extent[2]
            )
        )
        gripper_feasible = bool(float(np.min(extent[:2])) <= 0.075)
        workspace_reachable = bool(
            0.35 <= floor_center[0] <= 0.72
            and 0.08 <= floor_center[1] <= 0.42
        )
        near_workspace_boundary = bool(abs(float(position[1])) >= 0.27)
        object_inside_nominal_workspace = bool(
            position[0] <= 0.85 and abs(float(position[1])) < 0.27
        )
        workspace_recovery_feasible = bool(
            0.20 <= position[0] <= 0.85
            and abs(float(position[1])) <= 0.42
        )
        active_exception_id = (
            self._active_exception_id
            if getattr(self, "_unknown_anomaly_applied", False)
            else None
        )
        target_observation_ready = target_fresh
        target_clear = getattr(self, "target_obstacle_mode", None) in {None, "removed"}
        policy_eligible = bool(
            geometry_feasible and gripper_feasible and workspace_reachable
        )
        completed = bool(inside and stable and not grasped and gripper_open)
        return {
            "world_state_source": "camera_rgbd",
            "uses_simulator_truth": False,
            "target_geometry_type": (
                "planar_region" if self.target_mode == "region" else "container"
            ),
            "measurement_fresh": measurement_fresh,
            "state_provenance": {
                "object_pose": (
                    "sam3_rgbd" if object_fresh else "robot_proprioceptive_grasp_track"
                ),
                "gripper": "robot_proprioception",
                "target_region": "sam3_rgbd" if target_fresh else "sam3_rgbd_stale",
                "completion": "camera_semantic_estimate",
            },
            "cube_poses_robot_base": {
                "primary": [float(value) for value in np.concatenate([position, quaternion])]
            },
            "cube_extents_robot_base": {"primary": extent.tolist()},
            "target_pose_robot_base": [
                float(value) for value in np.concatenate([position, quaternion])
            ],
            "target_extent_robot_base": extent.tolist(),
            "object_shapes": {"primary": getattr(self, "object_shape", "cube")},
            **target,
            "target_object_center_robot_base": object_center.tolist(),
            "target_grasped": grasped,
            "wrong_object_grasped": False,
            "gripper_contact": grasped,
            "gripper_open": gripper_open,
            "gripper_closed": gripper_closed,
            "object_lifted": lifted,
            "object_at_target": inside,
            "object_inside_container": inside,
            "object_stable": stable,
            "container_entry_clear": policy_eligible and target_clear,
            "bin_policy_eligible": policy_eligible,
            "target_region_policy_eligible": policy_eligible,
            "target_region_clear": target_clear,
            "geometry_feasible": geometry_feasible,
            "gripper_feasible": gripper_feasible,
            "workspace_reachable": workspace_reachable,
            "object_inside_nominal_workspace": object_inside_nominal_workspace,
            "workspace_recovery_feasible": workspace_recovery_feasible,
            "object_near_workspace_boundary": near_workspace_boundary,
            "object_observation_ready": bool(object_fresh or proprioceptive_grasp_track),
            "target_observation_ready": target_observation_ready,
            "active_exception_id": active_exception_id,
            "unknown_anomaly_active": bool(active_exception_id),
            "entry_margin_xy": fit_margin.tolist(),
            "injected_exception": (
                self._active_scenario["injected_exception"]
                if getattr(self, "_active_scenario", None) is not None
                else None
            ),
            "distance_to_target": float(np.linalg.norm(position - object_center)),
            "task_completed": completed,
            "reward": float(completed),
            "recoverable_environment": bool(
                workspace_recovery_feasible
                and target_clear
                and target_fresh
                and (object_fresh or proprioceptive_grasp_track)
            ),
            "confidence": round(detection_score, 2),
        }

    def _primary_pose_robot_base(self) -> list[float]:
        sim = self.robosuite_env.sim
        world = vtf.SE3(wxyz_xyz=np.concatenate([
            sim.data.xquat[self.primary_body_id],
            sim.data.xpos[self.primary_body_id],
        ]))
        base = vtf.SE3(wxyz_xyz=np.concatenate([
            sim.data.xquat[self.base_link_idx],
            sim.data.xpos[self.base_link_idx],
        ]))
        pose = base.inverse() @ world
        return [float(value) for value in np.concatenate([
            pose.translation(), pose.rotation().wxyz
        ])]

    def _object_stable(self) -> bool:
        velocity = np.asarray(self.robosuite_env.sim.data.cvel[self.primary_body_id])
        angular_speed = float(np.linalg.norm(velocity[:3]))
        angular_stable = (
            float(self.cube_half_sizes[0]) * angular_speed
            <= self._CYLINDER_SURFACE_SPEED_TOLERANCE
            if self.object_shape == "cylinder"
            else angular_speed <= 0.20
        )
        return bool(
            np.linalg.norm(velocity[3:]) <= 0.05
            and angular_stable
            and not self._unstable_release_fault_active
        )

    def _object_inside_container(self) -> bool:
        sim = self.robosuite_env.sim
        floor_center, inner_extent, target_quaternion = self._target_geometry_world()
        return box_inside_open_bin(
            sim.data.xpos[self.primary_body_id],
            sim.data.xquat[self.primary_body_id],
            np.asarray(self.cube_half_sizes),
            floor_center,
            inner_extent,
            container_quaternion_wxyz=target_quaternion,
        )

    def _object_inside_target_region(self) -> bool:
        if self.target_mode == "bin":
            return self._object_inside_container()
        sim = self.robosuite_env.sim
        surface_center, region_extent, region_quaternion = self._target_geometry_world()
        return box_inside_target_region(
            sim.data.xpos[self.primary_body_id],
            sim.data.xquat[self.primary_body_id],
            np.asarray(self.cube_half_sizes),
            surface_center,
            region_extent,
            region_quaternion_wxyz=region_quaternion,
        )

    def _primary_is_grasped(self) -> bool:
        return bool(self.robosuite_env._check_grasp(
            gripper=self.robosuite_env.robots[0].gripper,
            object_geoms=self.primary_object,
        ))

    def _set_gripper(self, fraction: float) -> None:
        if (
            getattr(self, "_grasp_misses_injected", 0)
            < getattr(self, "inject_grasp_miss_attempts", 0)
            and fraction <= 0.2
        ):
            self._grasp_misses_injected += 1
            self._grasp_miss_injected = True
            self.record_world_state(
                "fault_injected",
                metadata={
                    "exception_id": "GRASP_MISS",
                    "fault_index": self._grasp_misses_injected,
                    "command": "close_gripper",
                },
            )
            fraction = 1.0
        opening = fraction >= 0.8
        opening_transition = opening and self._gripper_fraction < 0.8
        if (
            opening_transition
            and getattr(self, "_active_exception_id", None)
            and not getattr(self, "_unknown_anomaly_applied", False)
            and self._primary_is_grasped()
            and self._primary_over_target_xy()
        ):
            self._apply_unknown_anomaly()
        if opening and getattr(self, "_blocked_release_attempts", 0) > 0:
            self._blocked_release_attempts -= 1
            self.record_world_state(
                "release_blocked_after_target_change",
                metadata={
                    "exception_id": "TARGET_POSITION_CHANGED",
                    "remaining_blocked_releases": self._blocked_release_attempts,
                },
            )
            return
        if opening and getattr(self, "_visual_occlusion_steps", 0) > 0:
            return
        if opening_transition and self._unstable_release_fault_active:
            self._unstable_release_fault_active = False
            self.record_world_state(
                "fault_recovery_started",
                metadata={"exception_id": "UNSTABLE_RELEASE"},
            )
        if (
            opening_transition
            and self._primary_release_faults_injected
            < self.inject_primary_release_failures
            and self._primary_is_grasped()
            and self._primary_over_target_xy()
        ):
            self._primary_release_faults_injected += 1
            self._pending_primary_release_fault_steps = 10
            fault_index = self._primary_release_faults_injected
            exception_id = (
                "UNSTABLE_RELEASE"
                if self.inject_first_primary_release_unstable and fault_index == 1
                else (
                    "OUTSIDE_TARGET_REGION"
                    if getattr(self, "target_mode", "bin") == "region"
                    else "OUTSIDE_CONTAINER"
                )
            )
            self.record_world_state(
                "fault_injected",
                metadata={
                    "exception_id": exception_id,
                    "fault_index": fault_index,
                    "command": "open_gripper",
                },
            )
        if opening_transition:
            # A release starts a new visual stability phase.  Do not compare
            # the placed object with positions cached while it was grasped.
            self._camera_state_positions = []
            self._camera_state_last_measurement = None
            self._camera_rgbd_cache = None
        super()._set_gripper(fraction)

    def after_target_region_query(self) -> None:
        """Arm a target move after returning the now-stale target geometry."""
        self._target_change_armed = bool(
            self._active_exception_id == "TARGET_POSITION_CHANGED"
            and not self._unknown_anomaly_applied
            and self._primary_is_grasped()
        )

    def before_goto_pose(self) -> bool:
        """Move an armed target before executing the stale placement motion."""
        if self._target_change_armed:
            self._target_change_armed = False
            self._apply_unknown_anomaly()
        if self._blocked_stale_goto_attempts > 0:
            self._blocked_stale_goto_attempts -= 1
            if (
                self._blocked_stale_goto_attempts == 0
                and getattr(
                    self, "_clear_target_cache_after_blocked_stale_motion", False
                )
            ):
                self._camera_target_last_measurement = None
                self._camera_rgbd_cache = None
                self._clear_target_cache_after_blocked_stale_motion = False
            self.record_world_state(
                "stale_target_motion_blocked",
                metadata={
                    "exception_id": "TARGET_POSITION_CHANGED",
                    "remaining_blocked_motions": self._blocked_stale_goto_attempts,
                },
            )
            return False
        return True

    def _apply_unknown_anomaly(self) -> None:
        """Apply one held-out anomaly exactly once at the first target release."""
        exception_id = self._active_exception_id
        if exception_id is None:
            return
        self._unknown_anomaly_applied = True
        sim = self.robosuite_env.sim
        if exception_id == "TEMPORARY_BIN_OCCLUSION":
            # Both nominal open commands are blocked while the camera view is
            # occluded; the mask clears before the model receives control.
            self._visual_occlusion_steps = 70
        elif exception_id == "TARGET_POSITION_CHANGED":
            body_id = sim.model.body_name2id("bin2")
            shift = np.array([self._TARGET_CHANGE_SHIFT, 0.0, 0.0])
            sim.model.body_pos[body_id] += shift
            self.robosuite_env.bin2_pos = (
                np.asarray(self.robosuite_env.bin2_pos, dtype=float) + shift
            )
            # The target moves before release. Block both release commands in
            # the current nominal block and its three stale target motions so
            # recovery starts while still grasped.
            self._blocked_stale_goto_attempts = 3
            self._blocked_release_attempts = 2
            self._camera_target_last_measurement = None
            self._camera_rgbd_cache = None
            sim.forward()
        elif exception_id == "OBJECT_OUT_OF_WORKSPACE":
            joint_name = self.primary_object.joints[0]
            qpos = np.array(sim.data.get_joint_qpos(joint_name), dtype=float, copy=True)
            qpos[:3] += self.object_out_of_workspace_offset
            sim.data.set_joint_qpos(joint_name, qpos)
            sim.forward()
        self.record_world_state(
            "unknown_anomaly_injected",
            metadata={"exception_id": exception_id},
        )

    def _primary_over_target_xy(self) -> bool:
        """Use simulator truth only to localize deterministic fault injection."""
        floor_center, inner_extent, target_quaternion = self._target_geometry_world()
        rotation = _rotation_matrix_wxyz(target_quaternion)
        relative = rotation.T @ (
            self.robosuite_env.sim.data.xpos[self.primary_body_id] - floor_center
        )
        return bool(np.all(np.abs(relative[:2]) <= 0.5 * inner_extent[:2]))

    def _step_once(self) -> None:
        super()._step_once()
        if getattr(self, "_visual_occlusion_steps", 0) > 0:
            self._visual_occlusion_steps -= 1
        if self._pending_primary_release_fault_steps is None:
            return
        self._pending_primary_release_fault_steps -= 1
        if self._pending_primary_release_fault_steps > 0:
            return
        self._pending_primary_release_fault_steps = None
        sim = self.robosuite_env.sim
        joint_name = self.primary_object.joints[0]
        qpos = np.array(sim.data.get_joint_qpos(joint_name), dtype=float, copy=True)
        if getattr(self, "_active_scenario", None) is not None:
            qpos[:3] = np.asarray(self.robosuite_env.bin1_pos, dtype=float)
            qpos[0] += self.primary_release_fault_offset
            qpos[2] += self._BIN_FLOOR_TOP_OFFSET + self.cube_half_sizes[2]
        else:
            qpos[0] += self.primary_release_fault_offset
        sim.data.set_joint_qpos(joint_name, qpos)
        set_joint_qvel = getattr(sim.data, "set_joint_qvel", None)
        if callable(set_joint_qvel):
            set_joint_qvel(joint_name, np.zeros(6))
        sim.forward()
        fault_index = self._primary_release_faults_injected
        self._unstable_release_fault_active = bool(
            self.inject_first_primary_release_unstable and fault_index == 1
        )
        self.record_world_state(
            "fault_applied",
            metadata={
                "exception_id": (
                    "UNSTABLE_RELEASE"
                    if self._unstable_release_fault_active
                    else (
                        "OUTSIDE_TARGET_REGION"
                        if getattr(self, "target_mode", "bin") == "region"
                        else "OUTSIDE_CONTAINER"
                    )
                ),
                "fault_index": fault_index,
                "primary_x_offset": self.primary_release_fault_offset,
            },
        )

    def get_task_recorder_state(
        self, observation: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        pose = self._primary_pose_robot_base()
        floor_world, inner_extent, target_quaternion_world = self._target_geometry_world()
        floor_base = self._world_point_to_robot_base(floor_world)
        base_rotation = vtf.SO3(wxyz=self.base_link_wxyz_xyz[:4])
        target_quaternion = (
            base_rotation.inverse() @ vtf.SO3(wxyz=target_quaternion_world)
        ).wxyz
        extent = np.asarray(self.cube_half_sizes) * 2.0
        target_center = floor_base + np.array([0.0, 0.0, self.cube_half_sizes[2]])
        grasped = self._primary_is_grasped()
        inside = self._object_inside_target_region()
        stable = self._object_stable()
        relative_rotation = (
            _rotation_matrix_wxyz(target_quaternion).T
            @ _rotation_matrix_wxyz(np.asarray(pose)[3:7])
        )
        projected_extent = 2.0 * (
            np.abs(relative_rotation) @ np.asarray(self.cube_half_sizes)
        )
        footprint_extent = planar_footprint_extent(
            np.asarray(pose)[3:7], target_quaternion, self.cube_half_sizes
        )
        policy_eligible = bool(
            np.all(footprint_extent + 0.02 <= inner_extent[:2] + 1e-9)
            and (
                self.target_mode == "region"
                or projected_extent[2] <= inner_extent[2]
            )
        )
        pose_position = np.asarray(pose)[:3]
        workspace_reachable = bool(pose_position[0] <= 0.85)
        object_inside_nominal_workspace = bool(
            workspace_reachable and abs(float(pose_position[1])) < 0.27
        )
        workspace_recovery_feasible = bool(
            0.20 <= pose_position[0] <= 0.85
            and abs(float(pose_position[1])) <= 0.42
        )
        observation_ready = self.infeasible_scenario != "OBSERVATION_FAILURE"
        target_region_clear = (
            self.infeasible_scenario != "TARGET_REGION_BLOCKED"
            and self.target_obstacle_mode in {None, "removed"}
        )
        if self.infeasible_scenario == "OBJECT_UNREACHABLE":
            workspace_reachable = False
            workspace_recovery_feasible = False
        near_workspace_boundary = bool(abs(float(pose[1])) >= 0.27)
        completed = bool(inside and stable and not grasped and self._gripper_fraction >= 0.8)
        return {
            "target_geometry_type": (
                "planar_region" if self.target_mode == "region" else "container"
            ),
            "cube_poses_robot_base": {"primary": pose},
            "cube_extents_robot_base": {"primary": extent.tolist()},
            "target_pose_robot_base": pose,
            "target_extent_robot_base": extent.tolist(),
            "target_region_center_robot_base": floor_base.tolist(),
            "target_region_extent_robot_base": inner_extent.tolist(),
            "target_region_quaternion_wxyz": target_quaternion.tolist(),
            "target_object_center_robot_base": target_center.tolist(),
            "container_rim_z_robot_base": float(
                floor_base[2]
                + (
                    self._active_rim_height
                    if self._active_scenario is not None
                    else inner_extent[2]
                )
            ),
            "target_grasped": grasped,
            "wrong_object_grasped": False,
            "gripper_contact": grasped,
            "gripper_open": bool(self._gripper_fraction >= 0.8),
            "gripper_closed": bool(self._gripper_fraction <= 0.2),
            "object_lifted": bool(np.asarray(pose)[2] > target_center[2] + 0.04),
            "object_at_target": inside,
            "object_inside_container": inside,
            "object_inside_target_region": inside,
            "object_stable": stable,
            "container_entry_clear": policy_eligible and target_region_clear,
            "bin_policy_eligible": policy_eligible,
            "target_region_policy_eligible": policy_eligible,
            "target_region_clear": target_region_clear,
            "distance_to_target": float(np.linalg.norm(np.asarray(pose[:3]) - target_center)),
            "task_completed": completed,
            "reward": float(completed),
            "recoverable_environment": bool(
                workspace_reachable and observation_ready and target_region_clear
            ),
            "active_exception_id": (
                self._active_exception_id if self._unknown_anomaly_applied else None
            ),
            "unknown_anomaly_active": bool(self._unknown_anomaly_applied),
            "visual_occlusion_active": bool(self._visual_occlusion_steps > 0),
            "workspace_reachable": workspace_reachable,
            "object_inside_nominal_workspace": object_inside_nominal_workspace,
            "workspace_recovery_feasible": workspace_recovery_feasible,
            "object_near_workspace_boundary": near_workspace_boundary,
            "object_observation_ready": observation_ready,
            "target_observation_ready": observation_ready,
            "confidence": 1.0 if observation_ready else 0.0,
        }

    @staticmethod
    def _mask_bin_view(frame: np.ndarray) -> np.ndarray:
        masked = np.asarray(frame).copy()
        height, width = masked.shape[:2]
        masked[height // 3: 5 * height // 6, width // 2: 9 * width // 10] = 12
        return masked

    def _record_frame(self) -> None:
        super()._record_frame()
        if getattr(self, "_visual_occlusion_steps", 0) > 0 and self._frame_buffer:
            self._frame_buffer[-1] = self._mask_bin_view(self._frame_buffer[-1])

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        frame = super().render(mode)
        if getattr(self, "_visual_occlusion_steps", 0) > 0:
            return self._mask_bin_view(frame)
        return frame

    def compute_reward(self) -> float:
        return float(self.task_completed())

    def task_completed(self) -> bool:
        return bool(
            self._object_inside_target_region()
            and self._object_stable()
            and not self._primary_is_grasped()
            and self._gripper_fraction >= 0.8
        )

    def get_observation(self) -> dict[str, Any]:
        obs = self.robosuite_env._get_observations()
        obs["cube_poses"] = {
            "primary": np.asarray(self._primary_pose_robot_base(), dtype=np.float32)
        }
        self._process_camera_observations(obs)
        self._compute_gripper_obs(obs)
        return obs


__all__ = [
    "FrankaRobosuiteBinPlaceLowLevel",
    "box_inside_open_bin",
    "box_inside_target_region",
    "fit_planar_rectangle",
    "infeasible_region_place_scenario",
    "largest_connected_component",
    "planar_footprint_extent",
    "region_place_scenario",
]
