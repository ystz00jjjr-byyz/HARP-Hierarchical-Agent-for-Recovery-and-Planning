"""Low-level Robosuite Franka environment compatible with FrankaControlApi.

This module provides a thin wrapper around Robosuite's Lift environment
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

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv


class FrankaRobosuiteCubeLiftLowLevel(RobosuiteBaseEnv):
    """Robosuite Franka Cube Lift environment with FrankaPickPlaceLowLevel-compatible interface."""

    _SUBSAMPLE_RATE = 2

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1500,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = False,
        enable_render: bool = False,
        inject_first_grasp_miss: bool = False,
    ) -> None:
        super().__init__(
            controller_cfg=controller_cfg,
            max_steps=max_steps,
            seed=seed,
            viser_debug=False,
            privileged=privileged,
            enable_render=enable_render,
        )
        self.inject_first_grasp_miss = bool(inject_first_grasp_miss)
        self._grasp_miss_injected = False
        self.role_object_names = {"primary": "red cube"}

        # Initialize Robosuite environment
        if privileged:
            if not enable_render:
                self.render_camera_names = []
                self.robosuite_env = suite.environments.manipulation.lift.Lift(
                    robots=["Panda"],
                    use_camera_obs=False,
                    has_renderer=False,
                    has_offscreen_renderer=False,
                    camera_names=self.render_camera_names,
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
                self.robosuite_env = suite.environments.manipulation.lift.Lift(
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
            self.robosuite_env = suite.environments.manipulation.lift.Lift(
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

        self.register_world_body(
            entity_id="cube:primary",
            semantic_name="red cube",
            body_name=self.robosuite_env.cube.root_body,
            attributes={"task_role": "primary"},
        )
        self._init_robot_links()
        self._init_viser_debug(viser_debug)
        # Additional viser attributes specific to cube_lift
        if viser_debug:
            self.grasp_sample = None
            self.grasp_scores = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.robosuite_env.seed = seed
            self.robosuite_env.rng = np.random.default_rng(seed)
            if hasattr(self.robosuite_env, "placement_initializer"):
                self.robosuite_env.placement_initializer.rng = self.robosuite_env.rng

        self._grasp_miss_injected = False
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
            "task_prompt": "Pick up the red cube and lift it. Quaternions are WXYZ."
        }
        self.record_world_state("reset", observation=obs, metadata={"seed": seed})
        return obs, info

    def _set_gripper(self, fraction: float) -> None:
        """Inject one actuator-level close failure for hybrid recovery validation."""
        if (
            self.inject_first_grasp_miss
            and not self._grasp_miss_injected
            and fraction <= 0.2
        ):
            self._grasp_miss_injected = True
            self.record_world_state(
                "fault_injected",
                metadata={"exception_id": "GRASP_MISS", "command": "close_gripper"},
            )
            fraction = 1.0
        super()._set_gripper(fraction)

    def get_task_recorder_state(
        self, observation: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if observation is not None and "cube_poses" in observation:
            poses = observation["cube_poses"]
        else:
            poses = self._cube_pose_dict(self.robosuite_env._get_observations())
        sim = self.robosuite_env.sim
        cube_pos = np.asarray(sim.data.body_xpos[self.robosuite_env.cube_body_id])
        target_grasped = bool(self.robosuite_env._check_grasp(
            gripper=self.robosuite_env.robots[0].gripper,
            object_geoms=self.robosuite_env.cube,
        ))
        cvel = getattr(sim.data, "cvel", None)
        if cvel is None:
            linear_speed = angular_speed = 0.0
        else:
            body_velocity = np.asarray(cvel[self.robosuite_env.cube_body_id])
            angular_speed = float(np.linalg.norm(body_velocity[:3]))
            linear_speed = float(np.linalg.norm(body_velocity[3:]))
        table_height = float(self.robosuite_env.table_offset[2])
        lift_height = float(cube_pos[2] - table_height)
        object_lifted = bool(lift_height > 0.04)
        task_completed = bool(self.task_completed())
        cube_extent = [2.0 * float(value) for value in self.robosuite_env.cube.size]
        return {
            "cube_poses_robot_base": poses,
            "cube_extents_robot_base": {"primary": cube_extent},
            "target_pose_robot_base": poses["primary"],
            "target_extent_robot_base": cube_extent,
            "target_grasped": target_grasped,
            "wrong_object_grasped": False,
            "gripper_contact": target_grasped,
            "gripper_open": bool(self._gripper_fraction >= 0.8),
            "gripper_closed": bool(self._gripper_fraction <= 0.2),
            "object_lifted": object_lifted,
            "object_at_target": task_completed,
            "object_stable": linear_speed <= 0.05 and angular_speed <= 0.20,
            "distance_to_target": max(0.0, 0.04 - lift_height),
            "lift_height": lift_height,
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
            wxyz_xyz=np.concatenate([
                np.asarray(robosuite_obs["cube_quat"])[[3, 0, 1, 2]],
                robosuite_obs["cube_pos"],
            ])
        )

        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        cubeA_robot_base = base_transform @ cubeA_world

        return {
            "primary": [
                float(x)
                for x in np.concatenate(
                    [cubeA_robot_base.translation(), cubeA_robot_base.rotation().wxyz]
                )
            ],
        }

    def compute_reward(self) -> float:
        """Compute sparse stacking reward."""
        return self.robosuite_env.reward()

    def task_completed(self) -> bool:
        """Check if task is completed."""
        return self.robosuite_env._check_success()

    def get_observation(self) -> dict[str, Any]:
        """Get observation in FrankaPickPlaceLowLevel format."""
        robosuite_obs = self.robosuite_env._get_observations()
        pose_dict = self._cube_pose_dict(robosuite_obs)
        cube_pose_array = np.stack(
            [
                np.asarray(pose_dict["primary"], dtype=np.float32),
            ],
            axis=0,
        )

        robosuite_obs["cube_poses"] = {
            "primary": cube_pose_array[0],
        }

        self._process_camera_observations(robosuite_obs)
        self._compute_gripper_obs(robosuite_obs)

        return robosuite_obs

    # Viser debugging - uses the _update_viser_server from cube_lift's original
    # which checks hasattr(self, "viser_server") instead of self.viser_server is not None
    def _update_viser_server(self) -> None:
        obs = self.get_observation()
        if hasattr(self, "viser_server"):
            self._viser_init_check()

            obs_cartesian = obs["robot_cartesian_pos"][:-1]

            self.mjcf_ee_frame_handle.position = obs_cartesian[:3]
            self.mjcf_ee_frame_handle.wxyz = obs_cartesian[3:]

            from capx.utils.camera_utils import obs_get_rgb
            from capx.utils.depth_utils import depth_color_to_pointcloud
            import viser.transforms as vtf

            rbg_imgs = obs_get_rgb(obs)
            for image_key in rbg_imgs:
                self.viser_img_handle.image = rbg_imgs[image_key]

                if "pose" in obs[image_key]:
                    self.image_frustum_handle.position = obs[image_key]["pose"][:3]
                    self.image_frustum_handle.wxyz = obs[image_key]["pose"][3:]
                    self.image_frustum_handle.image = rbg_imgs[image_key]
                else:
                    self.image_frustum_handle.visible = False

                if "depth" in obs[image_key].get("images", {}):
                    points, colors = depth_color_to_pointcloud(
                        obs[image_key]["images"]["depth"][:, :, 0],
                        rbg_imgs[image_key],
                        obs[image_key]["intrinsics"],
                    )
                    self.viser_server.scene.add_point_cloud(
                        f"{image_key}/point_cloud",
                        points,
                        colors,
                        point_size=0.001,
                        point_shape="square",
                    )

            if self.cube_center is not None and self.cube_rot is not None:
                self.viser_server.scene.add_frame(
                    "robot0_robotview/cube_frame",
                    position=self.cube_center,
                    wxyz=vtf.SO3.from_matrix(self.cube_rot).wxyz,
                    axes_length=0.05,
                    axes_radius=0.005,
                )

            if self.cube_points is not None and self.cube_color is not None:
                self.viser_server.scene.add_point_cloud(
                    "robot0_robotview/cube_point_cloud",
                    self.cube_points,
                    self.cube_color,
                    point_size=0.001,
                    point_shape="square",
                )

            if self.grasp_sample is not None:
                grasp = self.grasp_sample[np.argmax(self.grasp_scores)]

                grasp_tf = vtf.SE3.from_matrix(grasp) @ vtf.SE3.from_translation(
                    np.array([0, 0, 0.1])
                )
                self.grasp_mesh_handle = self.viser_server.scene.add_frame(
                    "robot0_robotview/grasp",
                    position=grasp_tf.wxyz_xyz[-3:],
                    wxyz=grasp_tf.wxyz_xyz[:4],
                    axes_length=0.05,
                    axes_radius=0.0015,
                )


__all__ = ["FrankaRobosuiteCubeLiftLowLevel"]
