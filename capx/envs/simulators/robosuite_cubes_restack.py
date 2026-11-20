"""Low-level Robosuite Franka environment compatible with FrankaControlApi.

This module provides a thin wrapper around Robosuite's Stack environment
that implements the same interface as FrankaPickPlaceLowLevel, making it
hot-swappable for code execution environments.
"""

from __future__ import annotations

import collections.abc
from copy import copy
from typing import Any

import numpy as np
import robosuite as suite
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.utils import RandomizationError
from robosuite.utils.placement_samplers import ObjectPositionSampler
from robosuite.utils.transform_utils import quat_multiply

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv


_CUBE_COLORS = {
    "red": (1.0, 0.0, 0.0, 1.0),
    "green": (0.0, 1.0, 0.0, 1.0),
    "blue": (0.0, 0.25, 1.0, 1.0),
    "yellow": (1.0, 0.9, 0.0, 1.0),
}


def cuboid_policy_geometry(
    object_shape: str,
    primary_half_sizes: tuple[float, float, float] | np.ndarray,
    secondary_half_sizes: tuple[float, float, float] | np.ndarray,
) -> dict[str, float | bool]:
    """Return the deterministic gripper and support gate used by policy-first."""
    primary_xy = np.sort(2.0 * np.asarray(primary_half_sizes, dtype=float)[:2])
    secondary_xy = np.sort(2.0 * np.asarray(secondary_half_sizes, dtype=float)[:2])
    primary_width = float(primary_xy[0])
    secondary_width = float(secondary_xy[0])
    gripper_limit = 0.075
    support_feasible = bool(np.all(primary_xy <= secondary_xy + 0.004))
    aspect_ratio = max(
        float(primary_xy[-1] / primary_xy[0]),
        float(secondary_xy[-1] / secondary_xy[0]),
    )
    return {
        "geometry_policy_eligible": bool(
            object_shape in {"cube", "cuboid"}
            and primary_width <= gripper_limit
            and secondary_width <= gripper_limit
            and support_feasible
            and aspect_ratio <= 3.0
        ),
        "primary_gripper_feasible": primary_width <= gripper_limit,
        "secondary_gripper_feasible": secondary_width <= gripper_limit,
        "stack_support_feasible": support_feasible,
        "primary_gripper_width_required": primary_width,
        "secondary_gripper_width_required": secondary_width,
        "max_horizontal_aspect_ratio": aspect_ratio,
    }


class StackedObjectRandomSampler(ObjectPositionSampler):
    """
    Places all objects within the table uniformly random.

    Args:
        name (str): Name of this sampler.

        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models

        x_range (2-array of float): Specify the (min, max) relative x_range used to uniformly place objects

        y_range (2-array of float): Specify the (min, max) relative y_range used to uniformly place objects

        rotation (None or float or Iterable):
            :`None`: Add uniform random random rotation
            :`Iterable (a,b)`: Uniformly randomize rotation angle between a and b (in radians)
            :`value`: Add fixed angle rotation

        rotation_axis (str): Can be 'x', 'y', or 'z'. Axis about which to apply the requested rotation

        ensure_object_boundary_in_range (bool):
            :`True`: The center of object is at position:
                 [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
            :`False`:
                [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]

        ensure_valid_placement (bool): If True, will check for correct (valid) object placements

        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur

        z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
        x_range=(0, 0),
        y_range=(0, 0),
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.0,
        rng=None,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.rotation = rotation
        self.rotation_axis = rotation_axis

        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
            rng=rng,
        )

    def _sample_x(self, object_horizontal_radius):
        minimum, maximum = self.x_range
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return self.rng.uniform(high=maximum, low=minimum)

    def _sample_y(self, object_horizontal_radius):
        minimum, maximum = self.y_range
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return self.rng.uniform(high=maximum, low=minimum)

    def _sample_quat(self):
        if self.rotation is None:
            rot_angle = self.rng.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.abc.Iterable):
            rot_angle = self.rng.uniform(high=max(self.rotation), low=min(self.rotation))
        else:
            rot_angle = self.rotation

        if self.rotation_axis == "x":
            return np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0])
        elif self.rotation_axis == "y":
            return np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0])
        elif self.rotation_axis == "z":
            return np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)])
        else:
            raise ValueError(
                f"Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {self.rotation_axis}"
            )

    def sample(self, fixtures=None, reference=None, on_top=True):
        # Standardize inputs
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert reference in placed_objects, (
                f"Invalid reference received. Current options are: {placed_objects.keys()}, requested: {reference}"
            )
            ref_pos, _, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        else:
            base_offset = np.array(reference)
            assert base_offset.shape[0] == 3, (
                f"Invalid reference received. Should be (x,y,z) 3-tuple, but got: {base_offset}"
            )

        def _location_valid(object_x, object_y, object_z, horizontal_radius, bottom_offset):
            if not self.ensure_valid_placement:
                return True
            for (x, y, z), _, other_obj in placed_objects.values():
                if (
                    np.linalg.norm((object_x - x, object_y - y))
                    <= other_obj.horizontal_radius + horizontal_radius
                ) and (object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]):
                    return False
            return True

        mujoco_objects = list(self.mujoco_objects)

        # Special-case: place the second object directly on top of the first.
        if len(mujoco_objects) >= 2:
            first_obj = mujoco_objects[0]
            second_obj = mujoco_objects[1]

            assert first_obj.name not in placed_objects, (
                f"Object '{first_obj.name}' has already been sampled!"
            )
            assert second_obj.name not in placed_objects, (
                f"Object '{second_obj.name}' has already been sampled!"
            )

            first_hr = first_obj.horizontal_radius
            first_bo = first_obj.bottom_offset
            second_hr = second_obj.horizontal_radius
            second_bo = second_obj.bottom_offset

            success = False
            for _ in range(5000):
                object_x = self._sample_x(first_hr) + base_offset[0]
                object_y = self._sample_y(first_hr) + base_offset[1]
                object_z = self.z_offset + base_offset[2]
                if on_top:
                    object_z -= first_bo[-1]

                if not _location_valid(object_x, object_y, object_z, first_hr, first_bo):
                    continue

                quat1 = self._sample_quat()
                if hasattr(first_obj, "init_quat"):
                    quat1 = quat_multiply(quat1, first_obj.init_quat)
                pos1 = (object_x, object_y, object_z)
                placed_objects[first_obj.name] = (pos1, quat1, first_obj)

                z_gap = max(float(self.z_offset), 0.01)
                object2_x = object_x - 0.01
                object2_y = object_y
                object2_z = object_z + first_obj.top_offset[-1] - second_bo[-1] + z_gap

                if not _location_valid(object2_x, object2_y, object2_z, second_hr, second_bo):
                    placed_objects.pop(first_obj.name, None)
                    continue

                quat2 = self._sample_quat()
                if hasattr(second_obj, "init_quat"):
                    quat2 = quat_multiply(quat2, second_obj.init_quat)
                pos2 = (object2_x, object2_y, object2_z)
                placed_objects[second_obj.name] = (pos2, quat2, second_obj)

                success = True
                break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

            remaining_objects = mujoco_objects[2:]
        else:
            remaining_objects = mujoco_objects

        for obj in remaining_objects:
            assert obj.name not in placed_objects, f"Object '{obj.name}' has already been sampled!"

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False
            for _ in range(5000):
                object_x = self._sample_x(horizontal_radius) + base_offset[0]
                object_y = self._sample_y(horizontal_radius) + base_offset[1]
                object_z = self.z_offset + base_offset[2]
                if on_top:
                    object_z -= bottom_offset[-1]

                if _location_valid(object_x, object_y, object_z, horizontal_radius, bottom_offset):
                    quat = self._sample_quat()
                    if hasattr(obj, "init_quat"):
                        quat = quat_multiply(quat, obj.init_quat)
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj.name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

        return placed_objects


class FrankaRobosuiteCubesRestackLowLevel(RobosuiteBaseEnv):
    """Robosuite Franka Stack environment with FrankaPickPlaceLowLevel-compatible interface."""

    _SUBSAMPLE_RATE = 5

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1500,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = False,
        enable_render: bool = False,
        cube_A_length: float = 0.02,
        cube_B_length: float = 0.02,
        cube_A_half_sizes: str | list[float] | tuple[float, float, float] | None = None,
        cube_B_half_sizes: str | list[float] | tuple[float, float, float] | None = None,
        primary_color: str = "red",
        secondary_color: str = "green",
        object_shape: str = "cube",
        inject_primary_release_failures: int = 0,
        inject_first_primary_release_unstable: bool = False,
        primary_release_fault_offset: float = 0.06,
        recovery_step_reserve: int = 250,
        robosuite_env_class: type | None = None,
        render_camera_names: list[str] | None = None,
    ) -> None:
        super().__init__(
            controller_cfg=controller_cfg,
            max_steps=max_steps,
            seed=seed,
            viser_debug=False,
            privileged=privileged,
            enable_render=enable_render,
        )
        if render_camera_names is not None:
            self.render_camera_names = list(render_camera_names)
            self.save_camera_name = self.render_camera_names[0]
        env_class = (
            robosuite_env_class
            or suite.environments.manipulation.stack.Stack
        )

        if cube_A_length <= 0 or cube_B_length <= 0:
            raise ValueError("cube half-lengths must be positive")
        if primary_color not in _CUBE_COLORS or secondary_color not in _CUBE_COLORS:
            raise ValueError(f"supported cube colors: {sorted(_CUBE_COLORS)}")
        if object_shape not in {"cube", "cuboid", "cylinder"}:
            raise ValueError("object_shape must be 'cube', 'cuboid', or 'cylinder'")
        if inject_primary_release_failures < 0:
            raise ValueError("inject_primary_release_failures must be non-negative")
        if primary_release_fault_offset <= 0:
            raise ValueError("primary_release_fault_offset must be positive")
        if recovery_step_reserve < 1:
            raise ValueError("recovery_step_reserve must be positive")
        self.cube_A_length = float(cube_A_length)
        self.cube_B_length = float(cube_B_length)
        self.cube_A_half_sizes = self._normalize_half_sizes(
            cube_A_half_sizes, self.cube_A_length, "cube_A_half_sizes"
        )
        self.cube_B_half_sizes = self._normalize_half_sizes(
            cube_B_half_sizes, self.cube_B_length, "cube_B_half_sizes"
        )
        self.primary_color = primary_color
        self.secondary_color = secondary_color
        self.object_shape = object_shape
        self.inject_primary_release_failures = int(inject_primary_release_failures)
        self.inject_first_primary_release_unstable = bool(
            inject_first_primary_release_unstable
        )
        self.primary_release_fault_offset = float(primary_release_fault_offset)
        self.recovery_step_reserve = int(recovery_step_reserve)
        self._primary_release_faults_injected = 0
        self._pending_primary_release_fault_steps: int | None = None
        self._unstable_release_fault_active = False
        object_noun = {
            "cube": "cube", "cuboid": "cuboid", "cylinder": "cylinder"
        }[object_shape]
        self.role_object_names = {
            "primary": f"{primary_color} {object_noun}",
            "secondary": f"{secondary_color} {object_noun}",
        }

        # Initialize Robosuite environment
        if privileged:
            if not enable_render:
                self.render_camera_names = []
                self.robosuite_env = env_class(
                    robots=["Panda"],
                    use_camera_obs=False,
                    has_renderer=False,
                    has_offscreen_renderer=False,
                    camera_names=self.render_camera_names,
                    renderer="mujoco",
                    reward_shaping=True,
                    camera_heights=self._render_height,
                    camera_widths=self._render_width,
                    controller_configs=load_composite_controller_config(
                        controller=self.controller_cfg
                    ),
                    horizon=max_steps,
                    cube_A_length=self.cube_A_length,
                    cube_B_length=self.cube_B_length,
                    cube_A_half_sizes=self.cube_A_half_sizes,
                    cube_B_half_sizes=self.cube_B_half_sizes,
                    object_shape=self.object_shape,
                )
            else:
                self.robosuite_env = env_class(
                    robots=["Panda"],
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    camera_names=self.render_camera_names,
                    camera_depths=True,
                    renderer="mujoco",
                    camera_heights=self._render_height,
                    camera_widths=self._render_width,
                    controller_configs=load_composite_controller_config(
                        controller=self.controller_cfg
                    ),
                    horizon=max_steps,
                    reward_shaping=True,
                    cube_A_length=self.cube_A_length,
                    cube_B_length=self.cube_B_length,
                    cube_A_half_sizes=self.cube_A_half_sizes,
                    cube_B_half_sizes=self.cube_B_half_sizes,
                    object_shape=self.object_shape,
                )
        else:
            self.robosuite_env = env_class(
                robots=["Panda"],
                has_renderer=True,
                has_offscreen_renderer=True,
                camera_names=self.render_camera_names,
                # camera_segmentations=self.segmentation_level,
                camera_depths=True,
                renderer="mujoco",
                camera_heights=self._render_height,
                camera_widths=self._render_width,
                controller_configs=load_composite_controller_config(controller=self.controller_cfg),
                horizon=max_steps,
                reward_shaping=True,
                cube_A_length=self.cube_A_length,
                cube_B_length=self.cube_B_length,
                cube_A_half_sizes=self.cube_A_half_sizes,
                cube_B_half_sizes=self.cube_B_half_sizes,
                object_shape=self.object_shape,
            )

        for obj, rgba in (
            (self.robosuite_env.cubeA, _CUBE_COLORS[primary_color]),
            (self.robosuite_env.cubeB, _CUBE_COLORS[secondary_color]),
        ):
            for geom_name in set(obj.visual_geoms + obj.contact_geoms):
                geom_id = self.robosuite_env.sim.model.geom_name2id(geom_name)
                self.robosuite_env.sim.model.geom_matid[geom_id] = -1
                self.robosuite_env.sim.model.geom_rgba[geom_id] = rgba

        self.robosuite_env.cubeA = self._make_task_object(
            "cubeA", self.cube_A_length, self.primary_color
        )
        self.robosuite_env.cubeB = self._make_task_object(
            "cubeB", self.cube_B_length, self.secondary_color
        )
        cubes = [self.robosuite_env.cubeA, self.robosuite_env.cubeB]

        self.robosuite_env.placement_initializer = StackedObjectRandomSampler(
            name="ObjectSampler",
            mujoco_objects=cubes,
            x_range=[-0.12, 0.16],
            y_range=[-0.12, 0.12],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.robosuite_env.table_offset,
            z_offset=0.01,
            rng=self.robosuite_env.rng,
        )

        self.register_world_body(
            entity_id="cube:primary",
            semantic_name=self.role_object_names["primary"],
            body_name=self.robosuite_env.cubeA.root_body,
            attributes={"task_role": "primary"},
        )
        self.register_world_body(
            entity_id="cube:secondary",
            semantic_name=self.role_object_names["secondary"],
            body_name=self.robosuite_env.cubeB.root_body,
            attributes={"task_role": "secondary", "initially_stacked": True},
        )
        self._init_robot_links()
        self._init_viser_debug(viser_debug)

    @staticmethod
    def _normalize_half_sizes(
        value: str | list[float] | tuple[float, float, float] | None,
        fallback: float,
        name: str,
    ) -> tuple[float, float, float]:
        if isinstance(value, str):
            value = value.split(",")
        sizes = np.asarray(value if value is not None else [fallback] * 3, dtype=float)
        if sizes.shape != (3,) or not np.all(np.isfinite(sizes)) or np.any(sizes <= 0):
            raise ValueError(f"{name} must contain three positive finite half-sizes")
        return tuple(float(item) for item in sizes)

    def _make_task_object(self, name: str, half_size: float, color: str):
        object_class = CylinderObject if self.object_shape == "cylinder" else BoxObject
        if self.object_shape == "cuboid":
            size = self.cube_A_half_sizes if name == "cubeA" else self.cube_B_half_sizes
        else:
            size = [half_size] * (2 if self.object_shape == "cylinder" else 3)
        return object_class(name=name, size=size, rgba=list(_CUBE_COLORS[color]))

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.robosuite_env.seed = seed
            self.robosuite_env.rng = np.random.default_rng(seed)
            self.robosuite_env.placement_initializer.rng = self.robosuite_env.rng

        self.robosuite_env.reset()
        # Adjust initial orientation
        self.robosuite_env.sim.data.qpos[6] -= np.pi

        self._step_count = 0
        self._sim_step_count = 0
        self._primary_release_faults_injected = 0
        self._pending_primary_release_fault_steps = None
        self._unstable_release_fault_active = False

        robosuite_obs = self.robosuite_env._get_observations()
        self._current_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
        # We do this because for some reason modifying qpos does not update robot0_joint_pos
        self._current_joints[6] -= np.pi

        obs = self.get_observation()
        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )
        self.robosuite_env.cubeA = self._make_task_object(
            "cubeA", self.cube_A_length, self.primary_color
        )
        self.robosuite_env.cubeB = self._make_task_object(
            "cubeB", self.cube_B_length, self.secondary_color
        )

        # Settle the environment
        for _ in range(50):
            self.robosuite_env.sim.forward()
            self.robosuite_env.sim.step()
            self._set_gripper(1.0)

        info = {
            "task_prompt": "Place the primary cube on top of the secondary cube. Quaternions are WXYZ."
        }
        self.record_world_state("reset", observation=obs, metadata={"seed": seed})
        return obs, info

    def _primary_is_grasped(self) -> bool:
        return bool(self.robosuite_env._check_grasp(
            gripper=self.robosuite_env.robots[0].gripper,
            object_geoms=self.robosuite_env.cubeA,
        ))

    def _set_gripper(self, fraction: float) -> None:
        opening = fraction >= 0.8
        if opening and self._unstable_release_fault_active:
            self._unstable_release_fault_active = False
            self.record_world_state(
                "fault_recovery_started",
                metadata={"exception_id": "UNSTABLE_RELEASE"},
            )
        if (
            opening
            and self._primary_release_faults_injected
            < self.inject_primary_release_failures
            and self._primary_is_grasped()
        ):
            self._primary_release_faults_injected += 1
            self._pending_primary_release_fault_steps = 10
            fault_index = self._primary_release_faults_injected
            exception_id = (
                "UNSTABLE_RELEASE"
                if self.inject_first_primary_release_unstable and fault_index == 1
                else "PLACE_FAILED"
            )
            self.record_world_state(
                "fault_injected",
                metadata={
                    "exception_id": exception_id,
                    "fault_index": fault_index,
                    "command": "open_gripper",
                },
            )
        super()._set_gripper(fraction)

    def _step_once(self) -> None:
        super()._step_once()
        if self._pending_primary_release_fault_steps is None:
            return
        self._pending_primary_release_fault_steps -= 1
        if self._pending_primary_release_fault_steps > 0:
            return
        self._pending_primary_release_fault_steps = None
        sim = self.robosuite_env.sim
        joint_name = self.robosuite_env.cubeA.joints[0]
        qpos = np.array(sim.data.get_joint_qpos(joint_name), dtype=float, copy=True)
        qpos[0] += self.primary_release_fault_offset
        sim.data.set_joint_qpos(joint_name, qpos)
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
                    else "PLACE_FAILED"
                ),
                "fault_index": fault_index,
                "primary_x_offset": self.primary_release_fault_offset,
            },
        )

    def get_task_recorder_state(
        self, observation: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if observation is not None and "cube_poses" in observation:
            poses = observation["cube_poses"]
        else:
            poses = self._cube_pose_dict(self.robosuite_env._get_observations())
        sim = self.robosuite_env.sim
        primary_pos = np.asarray(sim.data.body_xpos[self.robosuite_env.cubeA_body_id])
        secondary_pos = np.asarray(sim.data.body_xpos[self.robosuite_env.cubeB_body_id])
        target_grasped = bool(self.robosuite_env._check_grasp(
            gripper=self.robosuite_env.robots[0].gripper,
            object_geoms=self.robosuite_env.cubeA,
        ))
        wrong_object_grasped = bool(self.robosuite_env._check_grasp(
            gripper=self.robosuite_env.robots[0].gripper,
            object_geoms=self.robosuite_env.cubeB,
        ))
        primary_velocity = np.asarray(sim.data.cvel[self.robosuite_env.cubeA_body_id])
        secondary_velocity = np.asarray(sim.data.cvel[self.robosuite_env.cubeB_body_id])
        primary_linear_speed = float(np.linalg.norm(primary_velocity[3:]))
        primary_angular_speed = float(np.linalg.norm(primary_velocity[:3]))
        secondary_linear_speed = float(np.linalg.norm(secondary_velocity[3:]))
        secondary_angular_speed = float(np.linalg.norm(secondary_velocity[:3]))
        primary_stable = (
            primary_linear_speed <= 0.05
            and primary_angular_speed <= 0.20
            and not self._unstable_release_fault_active
        )
        secondary_stable = secondary_linear_speed <= 0.05 and secondary_angular_speed <= 0.20
        horizontal_distance = float(
            np.linalg.norm(primary_pos[:2] - secondary_pos[:2])
        )
        primary_half_sizes = np.asarray(self.cube_A_half_sizes)
        secondary_half_sizes = np.asarray(self.cube_B_half_sizes)
        restack_clearance_threshold = float(
            np.linalg.norm(primary_half_sizes[:2])
            + np.linalg.norm(secondary_half_sizes[:2])
            + 0.005
        )
        restack_precondition_cleared = bool(
            horizontal_distance > restack_clearance_threshold
            and secondary_stable
            and not wrong_object_grasped
        )
        target_position = secondary_pos.copy()
        target_position[2] += primary_half_sizes[2] + secondary_half_sizes[2]
        distance_to_target = float(np.linalg.norm(primary_pos - target_position))
        table_height = float(self.robosuite_env.table_offset[2])
        object_lifted = bool(
            primary_pos[2] > table_height + primary_half_sizes[2] + 0.02
        )
        task_completed = bool(self.task_completed())
        geometry = cuboid_policy_geometry(
            self.object_shape, primary_half_sizes, secondary_half_sizes
        )
        simulator_steps_remaining = max(0, int(self.max_steps - self._sim_step_count))
        return {
            "cube_poses_robot_base": poses,
            "cube_extents_robot_base": {
                "primary": (2.0 * primary_half_sizes).tolist(),
                "secondary": (2.0 * secondary_half_sizes).tolist(),
            },
            "target_pose_robot_base": poses["primary"],
            "target_extent_robot_base": (2.0 * primary_half_sizes).tolist(),
            "role_object_names": dict(self.role_object_names),
            "object_shapes": {
                "primary": self.object_shape,
                "secondary": self.object_shape,
            },
            "target_grasped": target_grasped,
            "wrong_object_grasped": wrong_object_grasped,
            "gripper_contact": target_grasped or wrong_object_grasped,
            "gripper_open": bool(self._gripper_fraction >= 0.8),
            "gripper_closed": bool(self._gripper_fraction <= 0.2),
            "object_lifted": object_lifted,
            "object_at_target": task_completed,
            "object_stable": primary_stable,
            "secondary_object_stable": secondary_stable,
            "restack_precondition_cleared": restack_precondition_cleared,
            "restack_clearance_threshold": restack_clearance_threshold,
            "horizontal_distance_between_cubes": horizontal_distance,
            "distance_to_target": distance_to_target,
            "simulator_steps_remaining": simulator_steps_remaining,
            "recovery_step_reserve": self.recovery_step_reserve,
            "recovery_budget_low": simulator_steps_remaining < self.recovery_step_reserve,
            **geometry,
            "primary_cube_linear_speed": primary_linear_speed,
            "primary_cube_angular_speed": primary_angular_speed,
            "task_completed": task_completed,
            "reward": float(self.compute_reward()),
            "recoverable_environment": True,
            "confidence": 1.0,
        }

    def _cube_pose_dict(self, robosuite_obs: dict[str, Any]) -> dict[str, list[float]]:
        """Get cube poses in robot base frame."""
        base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx],
                self.robosuite_env.sim.data.xpos[self.base_link_idx],
            ]
        )

        cubeA_world = vtf.SE3(
            wxyz_xyz=np.concatenate([
                np.asarray(robosuite_obs["cubeA_quat"])[[3, 0, 1, 2]],
                robosuite_obs["cubeA_pos"],
            ])
        )
        cubeB_world = vtf.SE3(
            wxyz_xyz=np.concatenate([
                np.asarray(robosuite_obs["cubeB_quat"])[[3, 0, 1, 2]],
                robosuite_obs["cubeB_pos"],
            ])
        )

        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        cubeA_robot_base = base_transform @ cubeA_world
        cubeB_robot_base = base_transform @ cubeB_world

        return {
            "primary": [
                float(x)
                for x in np.concatenate(
                    [cubeA_robot_base.translation(), cubeA_robot_base.rotation().wxyz]
                )
            ],
            "secondary": [
                float(x)
                for x in np.concatenate(
                    [cubeB_robot_base.translation(), cubeB_robot_base.rotation().wxyz]
                )
            ],
        }

    def compute_reward(self) -> float:
        """Compute dense stacking reward."""
        if self.robosuite_env.reward(action=None) > 0.4:
            # Secondary check to make sure both cubes are not up in the air
            robosuite_obs = self.robosuite_env._get_observations()
            pose_dict = self._cube_pose_dict(robosuite_obs)
            cube_pose_array = np.stack(
                [
                    np.asarray(pose_dict["primary"], dtype=np.float32),
                    np.asarray(pose_dict["secondary"], dtype=np.float32),
                ],
                axis=0,
            )
            if cube_pose_array[0, 2] > 0.04 and cube_pose_array[1, 2] > 0.04:
                return 0.0
            else:
                return self.robosuite_env.reward(action=None)
        else:
            return self.robosuite_env.reward(action=None)

    def task_completed(self) -> bool:
        """Compute if the task is completed."""
        if self.robosuite_env._check_success():
            # Secondary check to make sure both cubes are not up in the air
            robosuite_obs = self.robosuite_env._get_observations()
            pose_dict = self._cube_pose_dict(robosuite_obs)
            cube_pose_array = np.stack(
                [
                    np.asarray(pose_dict["primary"], dtype=np.float32),
                    np.asarray(pose_dict["secondary"], dtype=np.float32),
                ],
                axis=0,
            )
            if cube_pose_array[0, 2] > 0.04 and cube_pose_array[1, 2] > 0.04:
                return False
            else:
                return True
        else:
            return False

    def get_observation(self) -> dict[str, Any]:
        """Get observation in FrankaPickPlaceLowLevel format."""
        robosuite_obs = self.robosuite_env._get_observations()
        pose_dict = self._cube_pose_dict(robosuite_obs)
        cube_pose_array = np.stack(
            [
                np.asarray(pose_dict["primary"], dtype=np.float32),
                np.asarray(pose_dict["secondary"], dtype=np.float32),
            ],
            axis=0,
        )

        robosuite_obs["cube_poses"] = {
            "primary": cube_pose_array[0],
            "secondary": cube_pose_array[1],
        }

        self._process_camera_observations(robosuite_obs)
        self._compute_gripper_obs(robosuite_obs)

        return robosuite_obs


__all__ = ["FrankaRobosuiteCubesRestackLowLevel", "cuboid_policy_geometry"]
