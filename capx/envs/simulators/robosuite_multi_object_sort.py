"""Two-object sorting onto two tabletop target regions."""

from __future__ import annotations

from typing import Any

import numpy as np
import viser.transforms as vtf
from robosuite.environments.manipulation.stack import Stack
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import new_body, new_geom

from capx.envs.simulators.robosuite_bin_place import (
    FrankaRobosuiteBinPlaceLowLevel,
    box_inside_target_region,
    fit_planar_rectangle,
    largest_connected_component,
)
from capx.envs.simulators.robosuite_cubes_restack import (
    FrankaRobosuiteCubesRestackLowLevel,
)


def assess_primary_workspace_recovery(
    position: np.ndarray, extent: np.ndarray,
) -> dict[str, Any]:
    """Classify an RGB-D primary observation for bounded edge recovery."""
    position = np.asarray(position, dtype=float).reshape(3)
    extent = np.asarray(extent, dtype=float).reshape(3)
    inside_regrasp_workspace = bool(
        0.44 <= position[0] <= 0.74 and abs(position[1]) <= 0.28
    )
    finite_geometry = bool(
        np.all(np.isfinite(position))
        and np.all(np.isfinite(extent))
        and np.all(extent >= 0.015)
        and np.all(extent <= 0.10)
    )
    within_bounded_move_region = bool(
        0.36 <= position[0] <= 0.83 and abs(position[1]) <= 0.28
    )
    return {
        "object_inside_regrasp_workspace": inside_regrasp_workspace,
        "workspace_edge_recovery_required": not inside_regrasp_workspace,
        "workspace_edge_recovery_feasible": bool(
            finite_geometry and within_bounded_move_region
        ),
    }


class _MultiObjectSortTask(Stack):
    """Stack scene with two visual-only tabletop target regions."""

    def _load_model(self) -> None:
        super()._load_model()
        for name, color in (
            ("target_primary_marker", [0.0, 0.25, 1.0, 1.0]),
            ("target_secondary_marker", [1.0, 0.9, 0.0, 1.0]),
        ):
            body = new_body(name=name, pos=[0.0, 0.0, 0.801])
            body.append(new_geom(
                name=f"{name}_geom",
                type="box",
                size=[0.06, 0.06, 0.001],
                rgba=color,
                contype=0,
                conaffinity=0,
                group=1,
            ))
            self.model.worldbody.append(body)


class _ThreeObjectSortTask(_MultiObjectSortTask):
    """Add one magenta cuboid and one cyan planar target to the sort scene."""

    TERTIARY_HALF_SIZES = (0.018, 0.028, 0.020)

    def _load_model(self) -> None:
        super()._load_model()
        self.cubeC = BoxObject(
            name="cubeC",
            size=self.TERTIARY_HALF_SIZES,
            rgba=[1.0, 0.0, 1.0, 1.0],
        )
        self.model.merge_assets(self.cubeC)
        self.model.worldbody.append(self.cubeC.get_obj())
        marker = new_body(name="target_tertiary_marker", pos=[0.0, 0.0, 0.801])
        marker.append(new_geom(
            name="target_tertiary_marker_geom",
            type="box",
            size=[0.055, 0.055, 0.001],
            rgba=[0.0, 1.0, 1.0, 1.0],
            contype=0,
            conaffinity=0,
            group=1,
        ))
        self.model.worldbody.append(marker)

    def _setup_references(self) -> None:
        super()._setup_references()
        self.cubeC_body_id = self.sim.model.body_name2id(self.cubeC.root_body)


class _FourObjectSortTask(_ThreeObjectSortTask):
    """Add an orange fourth cuboid and a violet target region."""

    QUATERNARY_HALF_SIZES = (0.020, 0.024, 0.020)

    def _load_model(self) -> None:
        super()._load_model()
        self.cubeD = BoxObject(
            name="cubeD",
            size=self.QUATERNARY_HALF_SIZES,
            rgba=[1.0, 0.15, 0.0, 1.0],
        )
        self.model.merge_assets(self.cubeD)
        self.model.worldbody.append(self.cubeD.get_obj())
        marker = new_body(name="target_quaternary_marker", pos=[0.0, 0.0, 0.801])
        marker.append(new_geom(
            name="target_quaternary_marker_geom",
            type="box",
            size=[0.05, 0.05, 0.001],
            rgba=[0.15, 0.0, 1.0, 1.0],
            contype=0,
            conaffinity=0,
            group=1,
        ))
        self.model.worldbody.append(marker)

    def _setup_references(self) -> None:
        super()._setup_references()
        self.cubeD_body_id = self.sim.model.body_name2id(self.cubeD.root_body)


def multi_object_sort_scenario(
    seed: int, primary_source_x: float | None = None
) -> dict[str, Any]:
    """Return deterministic source and target locations for one smoke seed."""
    rng = np.random.default_rng(int(seed))
    jitter = rng.uniform(-0.018, 0.018, size=(4, 2))
    base = np.array([
        [-0.12, -0.14],
        [-0.12, 0.14],
        [0.14, -0.14],
        [0.14, 0.14],
    ])
    points = base + jitter
    if primary_source_x is not None:
        points[0, 0] = float(primary_source_x)
    return {
        "scenario_id": f"multi_object_sort_{int(seed):03d}",
        "source_primary_xy": points[0].tolist(),
        "source_secondary_xy": points[1].tolist(),
        "target_primary_xy": points[2].tolist(),
        "target_secondary_xy": points[3].tolist(),
        "target_extent": [0.12, 0.12, 0.0],
        "primary_yaw": float(rng.uniform(-np.pi, np.pi)),
        "secondary_yaw": float(rng.uniform(-np.pi, np.pi)),
    }


def three_object_sort_scenario(seed: int) -> dict[str, Any]:
    """Return six separated source/target poses for a three-role sort task."""
    rng = np.random.default_rng(int(seed))
    jitter = rng.uniform(-0.012, 0.012, size=(6, 2))
    base = np.array([
        [-0.16, -0.20],
        [-0.16, 0.20],
        [-0.16, 0.00],
        [0.16, -0.20],
        [0.16, 0.20],
        [0.16, 0.00],
    ])
    points = base + jitter
    roles = ("primary", "secondary", "tertiary")
    result: dict[str, Any] = {
        "scenario_id": f"three_object_sort_{int(seed):03d}",
        "target_extent": [0.11, 0.11, 0.0],
    }
    for index, role in enumerate(roles):
        result[f"source_{role}_xy"] = points[index].tolist()
        result[f"target_{role}_xy"] = points[index + len(roles)].tolist()
        result[f"{role}_yaw"] = float(rng.uniform(-np.pi, np.pi))
    return result


def four_object_sort_scenario(seed: int) -> dict[str, Any]:
    """Return eight separated source/target poses for a four-role sort task."""
    rng = np.random.default_rng(int(seed))
    jitter = rng.uniform(-0.010, 0.010, size=(8, 2))
    lanes = (-0.24, -0.08, 0.08, 0.24)
    base = np.array([
        *[[-0.16, lane] for lane in lanes],
        *[[0.16, lane] for lane in lanes],
    ])
    points = base + jitter
    roles = ("primary", "secondary", "tertiary", "quaternary")
    result: dict[str, Any] = {
        "scenario_id": f"four_object_sort_{int(seed):03d}",
        "target_extent": [0.10, 0.10, 0.0],
    }
    for index, role in enumerate(roles):
        result[f"source_{role}_xy"] = points[index].tolist()
        result[f"target_{role}_xy"] = points[index + len(roles)].tolist()
        result[f"{role}_yaw"] = float(rng.uniform(-np.pi, np.pi))
    return result


class FrankaRobosuiteMultiObjectSortLowLevel(
    FrankaRobosuiteCubesRestackLowLevel
):
    """Sort two to four objects into role-bound planar regions."""

    sort_roles = ("primary", "secondary")

    def __init__(
        self,
        *args: Any,
        target_extent: float = 0.12,
        world_state_source: str = "simulator",
        inject_role_binding_error: bool = False,
        inject_grasp_miss_attempts: int = 0,
        force_primary_release_faults: bool = False,
        directed_primary_source_x: float | str | None = None,
        include_tertiary: bool = False,
        include_quaternary: bool = False,
        **kwargs: Any,
    ) -> None:
        inject_role_binding_error = str(inject_role_binding_error).strip().lower() in {
            "1", "true", "yes", "on"
        }
        inject_grasp_miss_attempts = int(inject_grasp_miss_attempts)
        kwargs["inject_primary_release_failures"] = int(
            kwargs.get("inject_primary_release_failures", 0)
        )
        kwargs["primary_release_fault_offset"] = float(
            kwargs.get("primary_release_fault_offset", 0.06)
        )
        if world_state_source not in {"simulator", "camera_rgbd"}:
            raise ValueError("world_state_source must be 'simulator' or 'camera_rgbd'")
        if inject_grasp_miss_attempts < 0:
            raise ValueError("inject_grasp_miss_attempts must be non-negative")
        self.include_quaternary = str(include_quaternary).strip().lower() in {
            "1", "true", "yes", "on"
        }
        self.include_tertiary = self.include_quaternary or (
            str(include_tertiary).strip().lower() in {"1", "true", "yes", "on"}
        )
        super().__init__(
            *args,
            robosuite_env_class=(
                _FourObjectSortTask if self.include_quaternary
                else _ThreeObjectSortTask if self.include_tertiary
                else _MultiObjectSortTask
            ),
            render_camera_names=["birdview"] if world_state_source == "camera_rgbd" else None,
            **kwargs,
        )
        if target_extent <= 0:
            raise ValueError("target_extent must be positive")
        self.target_extent = float(target_extent)
        self.world_state_source = world_state_source
        self.inject_role_binding_error = inject_role_binding_error
        self.inject_grasp_miss_attempts = inject_grasp_miss_attempts
        self.force_primary_release_faults = str(
            force_primary_release_faults
        ).strip().lower() in {"1", "true", "yes", "on"}
        directed_value = str(directed_primary_source_x).strip().lower()
        self.directed_primary_source_x = (
            None if directed_value in {"none", "null", ""}
            else float(directed_primary_source_x)
        )
        self._grasp_misses_injected = 0
        self._active_exception_id = (
            "ROLE_BINDING_ERROR" if self.inject_role_binding_error else None
        )
        self.sort_roles = (
            ("primary", "secondary", "tertiary", "quaternary")
            if self.include_quaternary
            else ("primary", "secondary", "tertiary")
            if self.include_tertiary else ("primary", "secondary")
        )
        self._sort_scenario = (
            four_object_sort_scenario(1)
            if self.include_quaternary
            else three_object_sort_scenario(1)
            if self.include_tertiary else multi_object_sort_scenario(1)
        )
        if self.include_tertiary:
            self.cube_C_half_sizes = _ThreeObjectSortTask.TERTIARY_HALF_SIZES
            self.robosuite_env.cubeC = BoxObject(
                name="cubeC",
                size=self.cube_C_half_sizes,
                rgba=[1.0, 0.0, 1.0, 1.0],
            )
            self.role_object_names["tertiary"] = "magenta cuboid"
            self.register_world_body(
                entity_id="cube:tertiary",
                semantic_name=self.role_object_names["tertiary"],
                body_name=self.robosuite_env.cubeC.root_body,
                attributes={"task_role": "tertiary"},
            )
        if self.include_quaternary:
            self.cube_D_half_sizes = _FourObjectSortTask.QUATERNARY_HALF_SIZES
            self.robosuite_env.cubeD = BoxObject(
                name="cubeD",
                size=self.cube_D_half_sizes,
                rgba=[1.0, 0.15, 0.0, 1.0],
            )
            self.role_object_names["quaternary"] = "orange cuboid"
            self.register_world_body(
                entity_id="cube:quaternary",
                semantic_name=self.role_object_names["quaternary"],
                body_name=self.robosuite_env.cubeD.root_body,
                attributes={"task_role": "quaternary"},
            )
        self._visual_role_names = dict(self.role_object_names)
        self._configure_role_binding()
        self.role_target_names = {
            "target_primary": "blue primary target region",
            "target_secondary": "yellow secondary target region",
        }
        if self.include_tertiary:
            self.role_target_names["target_tertiary"] = (
                "cyan tertiary target region"
            )
        if self.include_quaternary:
            self.role_target_names["target_quaternary"] = (
                "violet quaternary target region"
            )
        self._camera_rgbd_cache = None
        self._camera_measurements: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
        self._camera_positions: dict[str, list[np.ndarray]] = {
            role: [] for role in self.sort_roles
        }
        self._grasp_baseline_positions: dict[str, np.ndarray] = {}
        self._expected_grasp_role: str | None = None
        self._last_released_role: str | None = None
        if self.world_state_source == "camera_rgbd":
            self.get_text_world_state = self._get_camera_text_world_state

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._sort_scenario = (
            four_object_sort_scenario(seed or 1)
            if self.include_quaternary
            else three_object_sort_scenario(seed or 1)
            if self.include_tertiary
            else multi_object_sort_scenario(seed or 1, self.directed_primary_source_x)
        )
        self._grasp_misses_injected = 0
        self._active_exception_id = (
            "ROLE_BINDING_ERROR" if self.inject_role_binding_error else None
        )
        self._configure_role_binding()
        self._camera_rgbd_cache = None
        self._camera_measurements = {}
        self._camera_positions = {role: [] for role in self.sort_roles}
        self._grasp_baseline_positions = {}
        self._expected_grasp_role = None
        self._last_released_role = None
        table = np.asarray(self.robosuite_env.table_offset, dtype=float)
        for role in self.sort_roles:
            obj = self._role_object(role)
            half_sizes = self._role_half_sizes(role)
            xy = np.asarray(self._sort_scenario[f"source_{role}_xy"], dtype=float)
            yaw = float(self._sort_scenario[f"{role}_yaw"])
            position = table + np.array([xy[0], xy[1], float(half_sizes[2]) + 0.01])
            quaternion = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])
            self.robosuite_env.sim.data.set_joint_qpos(
                obj.joints[0], np.concatenate([position, quaternion])
            )
        for role in self.sort_roles:
            marker_id = self.robosuite_env.sim.model.body_name2id(
                f"target_{role}_marker"
            )
            xy = np.asarray(self._sort_scenario[f"target_{role}_xy"], dtype=float)
            self.robosuite_env.sim.model.body_pos[marker_id] = np.array([
                xy[0], xy[1], float(table[2]) + 0.001
            ])
        for _ in range(20):
            self.robosuite_env.sim.forward()
            self.robosuite_env.sim.step()
            self._set_gripper(1.0)
        if self.world_state_source == "camera_rgbd":
            for role in self.sort_roles:
                target_name = f"target_{role}"
                try:
                    self._camera_measurements[target_name] = (
                        self._measure_colored_entity(target_name)
                    )
                except RuntimeError:
                    continue
        observation = self.get_observation()
        info = {
            "task_prompt": self._task_prompt(),
            "scenario": dict(self._sort_scenario),
        }
        self.record_world_state(
            "multi_object_sort_reset", observation=observation, metadata={"seed": seed}
        )
        if self.inject_role_binding_error:
            self.record_world_state(
                "fault_injected",
                metadata={"exception_id": "ROLE_BINDING_ERROR"},
            )
        return observation, info

    def _configure_role_binding(self) -> None:
        self.role_object_names = dict(self._visual_role_names)
        if self.inject_role_binding_error:
            self.role_object_names["primary"] = self._visual_role_names["secondary"]
            self.role_object_names["secondary"] = self._visual_role_names["primary"]

    def _task_prompt(self) -> str:
        assignments = " and ".join(
            f"{role} in target_{role}" for role in self.sort_roles
        )
        return f"Place {assignments}, then open the gripper."

    def _role_object(self, role: str):
        return {
            "primary": self.robosuite_env.cubeA,
            "secondary": self.robosuite_env.cubeB,
            "tertiary": getattr(self.robosuite_env, "cubeC", None),
            "quaternary": getattr(self.robosuite_env, "cubeD", None),
        }[role]

    def _role_half_sizes(self, role: str) -> tuple[float, float, float]:
        return {
            "primary": self.cube_A_half_sizes,
            "secondary": self.cube_B_half_sizes,
            "tertiary": getattr(self, "cube_C_half_sizes", (0.018, 0.028, 0.020)),
            "quaternary": getattr(self, "cube_D_half_sizes", (0.020, 0.024, 0.020)),
        }[role]

    def _world_to_base(self, point: np.ndarray) -> np.ndarray:
        base = vtf.SE3(wxyz_xyz=np.concatenate([
            self.robosuite_env.sim.data.xquat[self.base_link_idx],
            self.robosuite_env.sim.data.xpos[self.base_link_idx],
        ]))
        return (base.inverse() @ vtf.SE3.from_translation(point)).translation()

    def _target_state(self, target_name: str) -> dict[str, Any]:
        aliases = {
            "target": "target_primary",
            "target_primary": "target_primary",
            "primary target": "target_primary",
            "target_secondary": "target_secondary",
            "secondary target": "target_secondary",
            "target_tertiary": "target_tertiary",
            "tertiary target": "target_tertiary",
            "target_quaternary": "target_quaternary",
            "quaternary target": "target_quaternary",
        }
        target = aliases.get(str(target_name).strip().lower())
        if target is None:
            raise ValueError(f"Target region is unavailable: {target_name}")
        role = target.removeprefix("target_")
        table = np.asarray(self.robosuite_env.table_offset, dtype=float)
        xy = np.asarray(self._sort_scenario[f"target_{role}_xy"], dtype=float)
        floor_world = table + np.array([xy[0], xy[1], 0.0])
        floor_base = self._world_to_base(floor_world)
        half_height = self._role_half_sizes(role)[2]
        object_center = floor_base + np.array([0.0, 0.0, float(half_height)])
        return {
            "target_name": target,
            "target_region_center_robot_base": floor_base.tolist(),
            "target_object_center_robot_base": object_center.tolist(),
            "target_region_extent_robot_base": [
                self.target_extent, self.target_extent, 0.0
            ],
            "target_region_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "container_rim_z_robot_base": float(floor_base[2]),
        }

    def get_target_region_state(self, target_name: str = "target") -> dict[str, Any]:
        if self.world_state_source == "camera_rgbd":
            target = self._target_alias(target_name)
            try:
                measurement = self._measure_colored_entity(target)
                self._camera_measurements[target] = measurement
            except RuntimeError:
                measurement = self._camera_measurements.get(target)
                if measurement is None:
                    raise
            position, quaternion, extent, _ = measurement
            role = target.removeprefix("target_")
            half_height = self._role_half_sizes(role)[2]
            return {
                "target_name": target,
                "target_region_center_robot_base": position.tolist(),
                "target_object_center_robot_base": (
                    position + np.array([0.0, 0.0, half_height])
                ).tolist(),
                "target_region_extent_robot_base": extent.tolist(),
                "target_region_quaternion_wxyz": quaternion.tolist(),
                "container_rim_z_robot_base": float(position[2]),
            }
        return self._target_state(target_name)

    @staticmethod
    def _target_alias(target_name: str) -> str:
        aliases = {
            "target": "target_primary",
            "target_primary": "target_primary",
            "primary target": "target_primary",
            "target_secondary": "target_secondary",
            "secondary target": "target_secondary",
            "target_tertiary": "target_tertiary",
            "tertiary target": "target_tertiary",
            "target_quaternary": "target_quaternary",
            "quaternary target": "target_quaternary",
        }
        target = aliases.get(str(target_name).strip().lower())
        if target is None:
            raise ValueError(f"Target region is unavailable: {target_name}")
        return target

    def _capture_camera_rgbd(self):
        return FrankaRobosuiteBinPlaceLowLevel._capture_camera_rgbd(self)

    @staticmethod
    def _required_rgbd_points(name: str, mask_pixels: int) -> int:
        if name.startswith("target_"):
            return 30
        return min(20, max(8, int(np.ceil(0.10 * mask_pixels))))

    @staticmethod
    def _object_extent_is_plausible(
        measured_xy: np.ndarray, expected_xy: np.ndarray
    ) -> bool:
        measured_xy = np.sort(np.asarray(measured_xy, dtype=float))
        expected_xy = np.sort(np.asarray(expected_xy, dtype=float))
        return bool(
            np.all(measured_xy >= 0.35 * expected_xy)
            and np.all(measured_xy <= 2.0 * expected_xy)
        )

    def _configured_object_height(self, name: str) -> float:
        half_sizes = self._role_half_sizes(name)
        return 2.0 * float(half_sizes[2])

    def _measure_colored_entity(
        self, name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        rgb, depth, camera_points, pose_mat = self._capture_camera_rgbd()
        channels = rgb.astype(int)
        red, green, blue = (channels[:, :, index] for index in range(3))
        masks = {
            "primary": (
                (red > 120) & (red >= 1.45 * green) & (red >= 1.45 * blue)
                & (green <= 0.42 * red) & (blue >= 1.05 * green)
            ),
            "secondary": (green > 90) & (green >= 1.30 * red) & (green >= 1.30 * blue),
            "tertiary": (
                (red > 110) & (blue > 110)
                & (red >= 0.75 * blue) & (blue >= 0.75 * red)
                & (green < 0.70 * np.minimum(red, blue))
            ),
            "quaternary": (
                (red > 120) & (green > 15)
                & (green >= 0.08 * red) & (green <= 0.55 * red)
                & (blue < 0.80 * green)
            ),
            "target_primary": (blue > 100) & (blue >= 1.30 * red) & (blue >= 1.20 * green),
            "target_secondary": (red > 120) & (green > 105) & (blue < 0.70 * np.minimum(red, green)),
            "target_tertiary": (
                (green > 100) & (blue > 100)
                & (green >= 0.75 * blue) & (blue >= 0.75 * green)
                & (red < 0.70 * np.minimum(green, blue))
            ),
            "target_quaternary": (
                (blue > 140) & (red > 20)
                & (blue >= 1.60 * red)
                & (green < 0.80 * red)
            ),
        }
        mask = largest_connected_component(masks[name])
        valid = (
            mask.reshape(-1)
            & np.isfinite(camera_points).all(axis=1)
            & (camera_points[:, 2] > 0)
        )
        mask_pixels = int(mask.sum())
        minimum = self._required_rgbd_points(name, mask_pixels)
        if int(valid.sum()) < minimum:
            raise RuntimeError(
                f"RGB-D found no size-plausible {name}: "
                f"mask_pixels={mask_pixels}, valid_depth={int(valid.sum())}, "
                f"required={minimum}"
            )
        selected = camera_points[valid]
        points = (pose_mat[:3, :3] @ selected.T).T + pose_mat[:3, 3]
        if name.startswith("target_"):
            median_z = float(np.median(points[:, 2]))
            points = points[np.abs(points[:, 2] - median_z) <= 0.012]
        lower, upper = np.percentile(points, [5.0, 95.0], axis=0)
        try:
            center_xy, extent_xy, yaw = fit_planar_rectangle(points[:, :2])
        except ValueError as error:
            raise RuntimeError(
                f"RGB-D found too few stable {name} points for pose fitting: "
                f"{len(points)}"
            ) from error
        height = max(float(upper[2] - lower[2]), 0.008)
        if not name.startswith("target_"):
            height = self._configured_object_height(name)
        position = np.array([
            center_xy[0], center_xy[1],
            float(np.median(points[:, 2])) if name.startswith("target_")
            else float(upper[2] - 0.5 * height),
        ])
        extent = np.array([
            extent_xy[0],
            extent_xy[1],
            0.0 if name.startswith("target_") else height,
        ])
        if not name.startswith("target_"):
            half_sizes = self._role_half_sizes(name)
            expected_xy = 2.0 * np.asarray(half_sizes[:2], dtype=float)
            if not self._object_extent_is_plausible(extent[:2], expected_xy):
                raise RuntimeError(
                    f"RGB-D rejected implausible {name} extent: "
                    f"measured={extent[:2].round(4).tolist()}, "
                    f"expected={expected_xy.round(4).tolist()}"
                )
        quaternion = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])
        return position, quaternion, extent, 1.0

    def get_rgbd_object_state(
        self, role: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Resolve the configured role binding, then measure it from RGB-D."""
        requested = str(role).strip().lower()
        if requested not in self.role_object_names:
            requested = next(
                (key for key, value in self.role_object_names.items()
                 if requested == str(value).lower()),
                requested,
            )
        if requested not in self.role_object_names:
            raise ValueError(f"Unknown object role: {role}")
        visual_name = self.role_object_names[requested]
        observed_role = next(
            key for key, value in self._visual_role_names.items()
            if value == visual_name
        )
        self._expected_grasp_role = observed_role
        measurement = self._measure_colored_entity(observed_role)
        self._grasp_baseline_positions[observed_role] = measurement[0].copy()
        return measurement

    def _gripper_tool_position_robot_base(self) -> np.ndarray:
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

    @staticmethod
    def _has_confirmed_lift(
        position: np.ndarray,
        extent: np.ndarray,
        baseline_position: np.ndarray | None,
        gripper_position: np.ndarray,
        gripper_open: bool,
    ) -> bool:
        """Require visible lift and gripper proximity; occlusion is not evidence."""
        if gripper_open or baseline_position is None:
            return False
        min_lift = max(0.025, 0.5 * float(extent[2]))
        max_gripper_distance = max(0.08, 2.5 * float(np.max(extent)))
        return bool(
            float(position[2] - baseline_position[2]) >= min_lift
            and np.linalg.norm(position - gripper_position) <= max_gripper_distance
        )

    def rebind_roles_from_rgbd(self) -> bool:
        """Restore the only bindings that role-error injection can swap."""
        for role in ("primary", "secondary"):
            self._measure_colored_entity(role)
        self.role_object_names = dict(self._visual_role_names)
        self._active_exception_id = None
        self.record_world_state("roles_rebound", metadata={"source": "camera_rgbd"})
        return True

    def _get_camera_text_world_state(self) -> dict[str, Any]:
        measurements = dict(self._camera_measurements)
        fresh_names: set[str] = set()
        target_names = tuple(f"target_{role}" for role in self.sort_roles)
        for name in (*self.sort_roles, *target_names):
            try:
                measurements[name] = self._measure_colored_entity(name)
                fresh_names.add(name)
            except (RuntimeError, ValueError):
                # A placed object can cover its static target marker. Keep that
                # target's last RGB-D geometry without discarding fresh object
                # measurements from the same frame.
                continue
        self._camera_measurements = measurements
        required = set(self.sort_roles) | set(target_names)
        if set(measurements) != required:
            return {
                "world_state_source": "camera_rgbd",
                "uses_simulator_truth": False,
                "measurement_fresh": False,
                "object_observation_ready": False,
                "target_observation_ready": False,
                "task_completed": False,
                "reward": 0.0,
                "recoverable_environment": False,
                "confidence": 0.0,
            }
        target_observation_ready = set(target_names).issubset(measurements)
        poses = {}
        extents = {}
        targets = {}
        inside = {}
        stable = {}
        for role in self.sort_roles:
            position, quaternion, extent, _ = measurements[role]
            target_position, target_quaternion, target_extent, _ = measurements[f"target_{role}"]
            poses[role] = np.concatenate([position, quaternion]).tolist()
            extents[role] = extent.tolist()
            targets[role] = {
                "target_name": f"target_{role}",
                "target_region_center_robot_base": target_position.tolist(),
                "target_region_extent_robot_base": target_extent.tolist(),
                "target_region_quaternion_wxyz": target_quaternion.tolist(),
            }
            inside[role] = bool(box_inside_target_region(
                position, quaternion, 0.5 * extent,
                target_position, target_extent,
                region_quaternion_wxyz=target_quaternion,
                horizontal_tolerance=0.015,
                surface_tolerance=0.015,
            ))
            history = self._camera_positions[role]
            history.append(position.copy())
            del history[:-3]
            stable[role] = bool(len(history) < 2 or max(
                np.linalg.norm(item - history[-1]) for item in history[:-1]
            ) <= 0.006)
        gripper_open = bool(self._gripper_fraction >= 0.8)
        gripper_closed = bool(self._gripper_fraction <= 0.2)
        gripper_position = (
            np.zeros(3)
            if gripper_open
            else self._gripper_tool_position_robot_base()
        )
        proprioceptive_grasp_role = None
        expected_role = getattr(self, "_expected_grasp_role", None)
        if (
            gripper_closed
            and expected_role in set(self.sort_roles)
            and expected_role not in fresh_names
            and self._actual_gripper_opening_fraction() >= 0.15
        ):
            baseline = self._grasp_baseline_positions.get(expected_role)
            extent = np.asarray(extents[expected_role], dtype=float)
            min_lift = max(0.025, 0.5 * float(extent[2]))
            if baseline is not None and gripper_position[2] - baseline[2] >= min_lift:
                proprioceptive_grasp_role = expected_role
                tracked_position = gripper_position - np.array([
                    0.0, 0.0, max(0.004, 0.175 * float(extent[2]))
                ])
                poses[expected_role][:3] = tracked_position.tolist()
        observed_roles = fresh_names | (
            {proprioceptive_grasp_role}
            if proprioceptive_grasp_role is not None
            else set()
        )
        object_observation_ready = set(self.sort_roles).issubset(observed_roles)
        lifted_roles = [
            role for role in self.sort_roles
            if (
                role == proprioceptive_grasp_role
                or (
                    role in fresh_names
                    and self._has_confirmed_lift(
                        np.asarray(poses[role][:3], dtype=float),
                        extents[role],
                        getattr(self, "_grasp_baseline_positions", {}).get(role),
                        gripper_position,
                        gripper_open,
                    )
                )
            )
        ]
        active_grasp_role = lifted_roles[0] if len(lifted_roles) == 1 else None
        grasp_confirmation_source = (
            "rgbd_lift_and_gripper_motion" if active_grasp_role is not None else None
        )
        role_error = self.role_object_names != self._visual_role_names
        completed = bool(
            object_observation_ready
            and target_observation_ready
            and all(inside.values())
            and all(stable.values())
            and gripper_open
            and not role_error
        )
        active_role = next((role for role in self.sort_roles if not inside[role]), None)
        workspace_recovery = assess_primary_workspace_recovery(
            np.asarray(poses["primary"][:3], dtype=float),
            np.asarray(extents["primary"], dtype=float),
        )
        primary_horizontal = np.sort(np.asarray(extents["primary"][:2], dtype=float))
        release_state = self._released_role_state(inside)
        cuboid_graspability = {
            "short_side_m": float(primary_horizontal[0]),
            "long_side_m": float(primary_horizontal[1]),
            "max_grasp_width_m": 0.075,
            "short_side_graspable": bool(primary_horizontal[0] <= 0.075),
            "orthogonal_side_graspable": bool(primary_horizontal[1] <= 0.075),
        }
        return {
            "world_state_source": "camera_rgbd",
            "uses_simulator_truth": False,
            "measurement_fresh": object_observation_ready,
            "state_provenance": {
                "object_poses": (
                    "rgbd_plus_robot_proprioceptive_grasp_track"
                    if proprioceptive_grasp_role is not None
                    else "rgbd_color_segmentation"
                ),
                "target_regions": (
                    "rgbd_color_segmentation"
                    if set(target_names).issubset(fresh_names)
                    else "rgbd_color_segmentation_cached_static_geometry"
                ),
                "gripper": "robot_proprioception",
                "completion": "camera_semantic_estimate",
                "object_dimensions": "rgbd_xy_with_configured_height_prior",
            },
            "cube_poses_robot_base": poses,
            "cube_extents_robot_base": extents,
            "object_shapes": {
                role: getattr(self, "object_shape", "cube")
                for role in self.sort_roles
            },
            "cuboid_graspability": cuboid_graspability,
            "ik_collision_margin": getattr(
                self, "_last_ik_collision_margin_report", {"safe": False, "reason": "not_checked"}
            ),
            "target_regions": targets,
            "objects_at_targets": inside,
            "objects_stable": stable,
            "object_stable": bool(all(stable.values())),
            "active_object_role": active_role,
            "wrong_role_binding": role_error,
            "active_grasp_role": active_grasp_role,
            "grasp_confirmation_source": grasp_confirmation_source,
            "target_grasped": bool(active_grasp_role == active_role),
            "object_lifted": bool(active_grasp_role is not None),
            "wrong_object_grasped": bool(
                active_grasp_role is not None
                and active_role is not None
                and active_grasp_role != active_role
            ),
            **release_state,
            "active_exception_id": self._active_exception_id,
            "object_observation_ready": object_observation_ready,
            "target_observation_ready": target_observation_ready,
            **workspace_recovery,
            "gripper_open": gripper_open,
            "gripper_closed": gripper_closed,
            "task_completed": completed,
            "reward": float(completed),
            "recoverable_environment": True,
            "confidence": 1.0,
        }

    def _should_force_primary_release_fault(
        self, *, opening_transition: bool, primary_grasped: bool
    ) -> bool:
        return bool(
            self.force_primary_release_faults
            and opening_transition
            and not primary_grasped
            and self._expected_grasp_role == "primary"
            and self._primary_release_faults_injected >= 1
            and self._primary_release_faults_injected
            < self.inject_primary_release_failures
        )

    def _set_gripper(self, fraction: float) -> None:
        opening_transition = bool(fraction >= 0.8 and self._gripper_fraction < 0.8)
        expected_role = getattr(self, "_expected_grasp_role", None)
        if opening_transition and expected_role in set(getattr(self, "sort_roles", ())):
            self._last_released_role = expected_role
        if fraction <= 0.2 and self._grasp_misses_injected < self.inject_grasp_miss_attempts:
            self._grasp_misses_injected += 1
            self._active_exception_id = "GRASP_MISS"
            self.record_world_state(
                "fault_injected",
                metadata={
                    "exception_id": "GRASP_MISS",
                    "fault_index": self._grasp_misses_injected,
                    "command": "close_gripper",
                },
            )
            fraction = 1.0
        elif (
            fraction >= 0.8
            and self._primary_release_faults_injected < self.inject_primary_release_failures
            and self._primary_is_grasped()
        ):
            self._active_exception_id = "PLACE_FAILED"
        elif self._should_force_primary_release_fault(
            opening_transition=opening_transition,
            primary_grasped=False,
        ):
            self._primary_release_faults_injected += 1
            self._pending_primary_release_fault_steps = 10
            self._active_exception_id = "PLACE_FAILED"
            self.record_world_state(
                "fault_injected",
                metadata={
                    "exception_id": "PLACE_FAILED",
                    "fault_index": self._primary_release_faults_injected,
                    "command": "open_gripper",
                    "forced_for_strict_pairing": True,
                },
            )
        if opening_transition:
            self._camera_positions = {role: [] for role in self.sort_roles}
            self._expected_grasp_role = None
        super()._set_gripper(fraction)

    def _released_role_state(self, inside: dict[str, bool]) -> dict[str, Any]:
        role = getattr(self, "_last_released_role", None)
        return {
            "last_released_role": role,
            "last_released_role_at_target": bool(role and inside.get(role, False)),
        }

    def _sort_pose_dict(self) -> dict[str, list[float]]:
        poses = self._cube_pose_dict(self.robosuite_env._get_observations())
        if not self.include_tertiary:
            return poses
        sim = self.robosuite_env.sim
        base = vtf.SE3(wxyz_xyz=np.concatenate([
            sim.data.xquat[self.base_link_idx], sim.data.xpos[self.base_link_idx]
        ])).inverse()
        for role, obj in (
            ("tertiary", self.robosuite_env.cubeC),
            ("quaternary", getattr(self.robosuite_env, "cubeD", None)),
        ):
            if role not in self.sort_roles or obj is None:
                continue
            body_id = sim.model.body_name2id(obj.root_body)
            world = vtf.SE3(wxyz_xyz=np.concatenate([
                sim.data.xquat[body_id], sim.data.xpos[body_id]
            ]))
            pose = base @ world
            poses[role] = np.concatenate([
                pose.translation(), pose.rotation().wxyz
            ]).astype(float).tolist()
        return poses

    def _inside_assigned_target(self, role: str) -> bool:
        pose = self._sort_pose_dict()[role]
        target = self._target_state(f"target_{role}")
        half_sizes = self._role_half_sizes(role)
        return box_inside_target_region(
            np.asarray(pose[:3]),
            np.asarray(pose[3:]),
            np.asarray(half_sizes),
            np.asarray(target["target_region_center_robot_base"]),
            np.asarray(target["target_region_extent_robot_base"]),
        )

    def task_completed(self) -> bool:
        gripper = self.robosuite_env.robots[0].gripper
        holding = any(
            self.robosuite_env._check_grasp(
                gripper=gripper, object_geoms=self._role_object(role)
            )
            for role in self.sort_roles
        )
        return bool(
            all(self._inside_assigned_target(role) for role in self.sort_roles)
            and not holding
            and self._gripper_fraction >= 0.8
        )

    def compute_reward(self) -> float:
        return float(self.task_completed())

    def get_task_recorder_state(
        self, observation: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        poses = self._sort_pose_dict()
        inside = {
            role: self._inside_assigned_target(role) for role in self.sort_roles
        }
        completed = self.task_completed()
        return {
            "cube_poses_robot_base": poses,
            "cube_extents_robot_base": {
                role: (2.0 * np.asarray(self._role_half_sizes(role))).tolist()
                for role in self.sort_roles
            },
            "role_object_names": dict(self.role_object_names),
            "role_target_names": dict(self.role_target_names),
            "target_regions": {
                role: self._target_state(f"target_{role}")
                for role in self.sort_roles
            },
            "objects_at_targets": inside,
            **self._released_role_state(inside),
            "active_object_role": next(
                (role for role in self.sort_roles if not inside[role]), None
            ),
            "gripper_open": bool(self._gripper_fraction >= 0.8),
            "task_completed": completed,
            "reward": float(completed),
            "world_state_source": "simulator",
            "uses_simulator_truth": True,
            "recoverable_environment": True,
            "confidence": 1.0,
            "scenario_id": self._sort_scenario["scenario_id"],
        }


__all__ = [
    "FrankaRobosuiteMultiObjectSortLowLevel",
    "multi_object_sort_scenario",
    "three_object_sort_scenario",
    "four_object_sort_scenario",
]
