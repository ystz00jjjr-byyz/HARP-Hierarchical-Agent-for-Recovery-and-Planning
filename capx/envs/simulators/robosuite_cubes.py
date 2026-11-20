"""Low-level Robosuite Franka environment compatible with FrankaControlApi.

This module provides a thin wrapper around Robosuite's Stack environment
that implements the same interface as FrankaPickPlaceLowLevel, making it
hot-swappable for code execution environments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import robosuite as suite
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from robosuite.utils.placement_samplers import UniformRandomSampler

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv


class FrankaRobosuiteCubesLowLevel(RobosuiteBaseEnv):
    """Robosuite Franka Stack environment with FrankaPickPlaceLowLevel-compatible interface.

    This wrapper provides the same control interface as FrankaPickPlaceLowLevel
    (move_to_joints_blocking, _set_gripper, _step_once) but uses Robosuite's
    Stack environment as the backend.
    """

    _SUBSAMPLE_RATE = 5

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1500,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = False,
        enable_render: bool = False,
    ) -> None:
        super().__init__(
            controller_cfg=controller_cfg,
            max_steps=max_steps,
            seed=seed,
            viser_debug=False,
            privileged=privileged,
            enable_render=enable_render,
        )

        # Initialize Robosuite environment
        if privileged:
            if not enable_render:
                self.render_camera_names = []
                self.robosuite_env = suite.environments.manipulation.stack.Stack(
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
                )
            else:
                self.robosuite_env = suite.environments.manipulation.stack.Stack(
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
                )
        else:
            self.robosuite_env = suite.environments.manipulation.stack.Stack(
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
            )
        cubes = [self.robosuite_env.cubeA, self.robosuite_env.cubeB]

        self.robosuite_env.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=cubes,
            x_range=[-0.18, 0.18],
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
            semantic_name="red cube",
            body_name=self.robosuite_env.cubeA.root_body,
            attributes={"task_role": "primary"},
        )
        self.register_world_body(
            entity_id="cube:secondary",
            semantic_name="green cube",
            body_name=self.robosuite_env.cubeB.root_body,
            attributes={"task_role": "secondary"},
        )

        self._init_robot_links()
        self._init_viser_debug(viser_debug)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            # Robosuite owns the RNG used by UniformRandomSampler.  Updating
            # only this wrapper's RNG leaves object placement unseeded and
            # breaks paired comparisons across experiment groups.
            self.robosuite_env.seed = seed
            self.robosuite_env.rng = np.random.default_rng(seed)
            self.robosuite_env.placement_initializer.rng = self.robosuite_env.rng

        self.robosuite_env.reset()
        # Adjust initial orientation
        self.robosuite_env.sim.data.qpos[6] -= np.pi

        self._step_count = 0
        self._sim_step_count = 0

        # Settle the environment
        for _ in range(50):
            self.robosuite_env.sim.forward()
            self.robosuite_env.sim.step()
            self._set_gripper(1.0)

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

        info = {
            "task_prompt": "Place the primary cube on top of the secondary cube. Quaternions are WXYZ."
        }
        self.record_world_state("reset", observation=obs, metadata={"seed": seed})
        return obs, info

    def get_task_recorder_state(
        self, observation: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if observation is not None and "cube_poses" in observation:
            poses = observation["cube_poses"]
        else:
            poses = self._cube_pose_dict(self.robosuite_env._get_observations())
        sim = self.robosuite_env.sim
        cube_a_pos = np.asarray(sim.data.body_xpos[self.robosuite_env.cubeA_body_id])
        cube_b_pos = np.asarray(sim.data.body_xpos[self.robosuite_env.cubeB_body_id])
        target_grasped = bool(self.robosuite_env._check_grasp(
            gripper=self.robosuite_env.robots[0].gripper,
            object_geoms=self.robosuite_env.cubeA,
        ))
        wrong_object_grasped = bool(self.robosuite_env._check_grasp(
            gripper=self.robosuite_env.robots[0].gripper,
            object_geoms=self.robosuite_env.cubeB,
        ))
        cube_a_length = float(self.robosuite_env.cube_A_length)
        cube_b_length = float(self.robosuite_env.cube_B_length)
        target_position = cube_b_pos.copy()
        target_position[2] += (cube_a_length + cube_b_length) / 2.0
        distance_to_target = float(np.linalg.norm(cube_a_pos - target_position))
        horizontal_distance = float(np.linalg.norm(cube_a_pos[:2] - cube_b_pos[:2]))

        cvel = getattr(sim.data, "cvel", None)
        if cvel is None:
            linear_speed = angular_speed = 0.0
        else:
            body_velocity = np.asarray(cvel[self.robosuite_env.cubeA_body_id])
            angular_speed = float(np.linalg.norm(body_velocity[:3]))
            linear_speed = float(np.linalg.norm(body_velocity[3:]))
        # Robosuite's freshly settled cube can retain about 0.047 m/s of
        # reported body velocity, so 0.05 avoids labeling the reset state as
        # unstable while still rejecting visible post-release motion.
        object_stable = linear_speed <= 0.05 and angular_speed <= 0.20
        table_height = float(self.robosuite_env.table_offset[2])
        object_lifted = bool(cube_a_pos[2] > table_height + 0.04)
        task_completed = bool(self.task_completed())

        return {
            "cube_poses_robot_base": poses,
            "target_grasped": target_grasped,
            "wrong_object_grasped": wrong_object_grasped,
            "gripper_contact": target_grasped or wrong_object_grasped,
            "gripper_open": bool(self._gripper_fraction >= 0.8),
            "gripper_closed": bool(self._gripper_fraction <= 0.2),
            "object_lifted": object_lifted,
            "object_at_target": task_completed,
            "object_stable": object_stable,
            "distance_to_target": distance_to_target,
            "horizontal_distance_to_target": horizontal_distance,
            "primary_cube_linear_speed": linear_speed,
            "primary_cube_angular_speed": angular_speed,
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
            wxyz_xyz=np.concatenate([robosuite_obs["cubeA_quat"], robosuite_obs["cubeA_pos"]])
        )
        cubeB_world = vtf.SE3(
            wxyz_xyz=np.concatenate([robosuite_obs["cubeB_quat"], robosuite_obs["cubeB_pos"]])
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
        return self.robosuite_env.reward(action=None)

    def task_completed(self) -> bool:
        """Compute if the task is completed."""
        return self.robosuite_env._check_success()

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


__all__ = ["FrankaRobosuiteCubesLowLevel"]
