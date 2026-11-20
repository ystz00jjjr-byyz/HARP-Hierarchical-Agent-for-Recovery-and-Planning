"""RGB-D control adapter for the color-coded multi-object sort benchmark."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.integrations.franka.control import FrankaControlApi
from capx.integrations.motion.pyroki import check_ik_path_margin


class FrankaMultiObjectSortRgbdApi(FrankaControlApi):
    """Use calibrated RGB-D centers with a bounded top-down grasp template."""

    _GRASP_DEPTH_FRACTION = 0.20
    _GRASP_DEPTH_MIN = 0.004
    _GRASP_DEPTH_MAX = 0.008
    _ORTHOGONAL_GRASP_DEPTH_OFFSET = -0.002
    _CUBOID_MAX_GRASP_WIDTH = 0.075
    _PRE_CLOSE_SETTLE_STEPS = 5
    _EDGE_PUSH_MIN_CLEARANCE = 0.015
    _EDGE_PUSH_MAX_DISTANCE = 0.20

    def _observe_object(
        self, object_name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        try:
            return self._env.get_rgbd_object_state(object_name)
        except RuntimeError as error:
            if "RGB-D" not in str(error):
                raise
        role = str(object_name).strip().lower()
        visual_name = getattr(self._env, "role_object_names", {}).get(role)
        visual_roles = getattr(self._env, "_visual_role_names", {})
        observed_role = next(
            (key for key, value in visual_roles.items() if value == visual_name), role
        )
        clear_y = 0.30 if observed_role == "primary" else -0.30
        self.home_pose()
        self.open_gripper()
        self.goto_pose(
            np.array([0.50, clear_y, 0.18]),
            np.array([0.0, 0.0, 1.0, 0.0]),
        )
        return self._env.get_rgbd_object_state(object_name)

    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        position, quaternion, extent, _ = self._observe_object(object_name)
        return position, quaternion, extent if return_bbox_extent else None

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        position, _, extent, _ = self._observe_object(object_name)
        grasp = position.copy()
        grasp[2] -= float(np.clip(
            self._GRASP_DEPTH_FRACTION * extent[2],
            self._GRASP_DEPTH_MIN,
            self._GRASP_DEPTH_MAX,
        ))
        return grasp, np.array([0.0, 0.0, 1.0, 0.0])

    def sample_cuboid_grasp_pose(
        self, object_name: str, candidate_index: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the long-axis grasp first, then its orthogonal alternative."""
        grasp, quaternion, _ = self.sample_cuboid_grasp_plan(
            object_name, candidate_index
        )
        return grasp, quaternion

    def sample_cuboid_grasp_plan(
        self,
        object_name: str,
        candidate_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return grasp pose and extent from one atomic RGB-D observation."""
        if getattr(self._env, "object_shape", "cube") != "cuboid":
            raise ValueError("cuboid grasp candidates require object_shape=cuboid")
        if candidate_index not in (0, 1):
            raise ValueError("candidate_index must be 0 or 1")
        position, quaternion, extent, _ = self._observe_object(object_name)
        horizontal = np.asarray(extent[:2], dtype=float)
        required_width = float(np.sort(horizontal)[candidate_index])
        if required_width > self._CUBOID_MAX_GRASP_WIDTH:
            raise ValueError(
                f"cuboid grasp width {required_width:.3f} exceeds "
                f"{self._CUBOID_MAX_GRASP_WIDTH:.3f} m"
            )
        w, x, y, z = quaternion / np.linalg.norm(quaternion)
        yaw = np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        long_axis_yaw = yaw if horizontal[0] >= horizontal[1] else yaw + np.pi / 2.0
        grasp_yaw = long_axis_yaw + candidate_index * np.pi / 2.0
        grasp = position.copy()
        grasp_depth = (
            self._GRASP_DEPTH_FRACTION * extent[2]
            + candidate_index * self._ORTHOGONAL_GRASP_DEPTH_OFFSET
        )
        grasp[2] -= float(np.clip(
            grasp_depth,
            self._GRASP_DEPTH_MIN,
            self._GRASP_DEPTH_MAX,
        ))
        return (
            grasp,
            np.array([
                0.0, -np.sin(0.5 * grasp_yaw), np.cos(0.5 * grasp_yaw), 0.0,
            ]),
            extent.copy(),
        )

    def close_gripper(self) -> None:
        """Let the final RGB-D approach settle before closing."""
        self._env._set_gripper(1.0)
        for _ in range(self._PRE_CLOSE_SETTLE_STEPS):
            self._env._step_once()
        super().close_gripper()

    @staticmethod
    def _point_segment_distance_xy(
        point: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> float:
        segment = end[:2] - start[:2]
        denominator = float(np.dot(segment, segment))
        if denominator <= 1e-12:
            return float(np.linalg.norm(point[:2] - start[:2]))
        ratio = float(np.clip(np.dot(point[:2] - start[:2], segment) / denominator, 0, 1))
        closest = start[:2] + ratio * segment
        return float(np.linalg.norm(point[:2] - closest))

    @staticmethod
    def _edge_push_safe_center(
        position: np.ndarray, *, preserve_lane: bool
    ) -> np.ndarray:
        safe_center = np.asarray(position, dtype=float).copy()
        safe_center[0] = float(np.clip(position[0], 0.52, 0.66))
        y_limit = 0.24 if preserve_lane else 0.12
        safe_center[1] = float(np.clip(position[1], -y_limit, y_limit))
        return safe_center

    def sample_cuboid_edge_push_plan(
        self, object_name: str = "primary", preserve_lane: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preflight and return one bounded, collision-gated inward push."""
        if getattr(self._env, "object_shape", "cube") != "cuboid":
            raise ValueError("edge push candidate requires object_shape=cuboid")
        position, _, extent, _ = self._observe_object(object_name)
        safe_center = self._edge_push_safe_center(
            position, preserve_lane=preserve_lane
        )
        delta = safe_center[:2] - position[:2]
        push_distance = float(np.linalg.norm(delta))
        if not 0.03 <= push_distance <= self._EDGE_PUSH_MAX_DISTANCE:
            report = {"safe": False, "reason": "push_distance_out_of_bounds"}
            self._env._last_ik_collision_margin_report = report
            raise ValueError(report["reason"])

        direction = delta / push_distance
        half_extent_along_push = 0.5 * float(np.dot(np.abs(direction), extent[:2]))
        contact_offset = half_extent_along_push + 0.015
        contact = position.copy()
        contact[:2] -= direction * contact_offset
        push_end = safe_center.copy()
        push_end[:2] -= direction * contact_offset
        clearance = max(0.06, 2.0 * float(extent[2]))
        approach = contact + np.array([0.0, 0.0, clearance])
        retreat = push_end + np.array([0.0, 0.0, clearance])

        sort_roles = tuple(getattr(self._env, "sort_roles", ()))
        other_roles = tuple(role for role in sort_roles if role != object_name)
        if not other_roles:
            other_roles = (
                "secondary" if object_name == "primary" else "primary",
            )
        obstacle_clearance = float("inf")
        for other_role in other_roles:
            other_position, _, other_extent, _ = self._observe_object(other_role)
            obstacle_distance = self._point_segment_distance_xy(
                other_position, position, safe_center
            )
            clearance = obstacle_distance - 0.5 * (
                float(np.linalg.norm(extent[:2]))
                + float(np.linalg.norm(other_extent[:2]))
            )
            obstacle_clearance = min(obstacle_clearance, clearance)
        if obstacle_clearance < self._EDGE_PUSH_MIN_CLEARANCE:
            report = {
                "safe": False,
                "reason": "object_path_collision_margin",
                "object_path_clearance_m": obstacle_clearance,
            }
            self._env._last_ik_collision_margin_report = report
            raise ValueError(report["reason"])

        # Tilt the wrist inward along the measured push direction, so the
        # fingertip can reach farther than the wrist on either table edge.
        tilt = np.deg2rad(30.0)
        desired_offset_direction = np.array([
            np.sin(tilt) * direction[0],
            np.sin(tilt) * direction[1],
            np.cos(tilt),
        ])
        rotation, _ = SciRotation.align_vectors(
            [desired_offset_direction], [[0.0, 0.0, -1.0]]
        )
        xyzw = rotation.as_quat()
        quaternion = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
        tcp_offset = rotation.apply(self._TCP_OFFSET)
        path = np.asarray([
            np.concatenate([quaternion, pose + tcp_offset])
            for pose in (approach, contact, push_end, retreat)
        ])
        reports = []
        for restart_first_waypoint in (False, True):
            report = check_ik_path_margin(
                path,
                prev_cfg=self.cfg,
                min_self_collision_margin_m=0.005,
                min_joint_limit_margin_rad=0.05,
                max_joint_step_rad=0.55,
                max_position_error_m=0.02,
                restart_first_waypoint=restart_first_waypoint,
            )
            report["ik_seed_mode"] = (
                "global_first" if restart_first_waypoint else "continuous"
            )
            reports.append(report)
            if report.get("safe", False):
                break
        if not report.get("safe", False):
            report["candidate_rejections"] = [
                {"ik_seed_mode": item["ik_seed_mode"], "reason": item.get("reason")}
                for item in reports
            ]
        report["object_path_clearance_m"] = obstacle_clearance
        report["push_distance_m"] = push_distance
        checked_joint_waypoints = report.pop("joint_waypoints", [])
        self._env._last_ik_collision_margin_report = report
        if not report.get("safe", False):
            raise ValueError(f"unsafe edge push: {report.get('reason', 'unknown')}")
        self._edge_push_joint_waypoints = np.asarray(
            checked_joint_waypoints, dtype=float
        )
        return approach, contact, push_end, retreat, quaternion

    def execute_cuboid_edge_push(self) -> None:
        """Execute only the joint path accepted by the latest edge-push gate."""
        waypoints = getattr(self, "_edge_push_joint_waypoints", None)
        if waypoints is None or not len(waypoints):
            raise RuntimeError("no preflighted edge-push path")
        for cfg in waypoints:
            self._env.move_to_joints_blocking(cfg[:-1])
            self.cfg = cfg.copy()
        self._edge_push_joint_waypoints = None

    def rebind_roles(self) -> bool:
        """Re-observe both objects and restore their task-role bindings."""
        return bool(self._env.rebind_roles_from_rgbd())

    def functions(self):
        functions = super().functions()
        functions["rebind_roles"] = self.rebind_roles
        functions["sample_cuboid_grasp_pose"] = self.sample_cuboid_grasp_pose
        functions["sample_cuboid_grasp_plan"] = self.sample_cuboid_grasp_plan
        functions["sample_cuboid_edge_push_plan"] = self.sample_cuboid_edge_push_plan
        functions["execute_cuboid_edge_push"] = self.execute_cuboid_edge_push
        return functions


__all__ = ["FrankaMultiObjectSortRgbdApi"]
