from __future__ import annotations

import asyncio
import functools
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, List

import numpy as np
import pyroki as pk  # type: ignore
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.spatial.transform import Rotation, Slerp

import capx.integrations.motion.pyroki_snippets as pks


def slerp_quaternions(
    q_start: np.ndarray,  # wxyz format
    q_end: np.ndarray,  # wxyz format
    num_steps: int,
) -> np.ndarray:
    """SLERP interpolation between two quaternions.

    Args:
        q_start: Start quaternion in wxyz format
        q_end: End quaternion in wxyz format
        num_steps: Number of interpolation steps

    Returns:
        Array of shape (num_steps, 4) with interpolated quaternions in wxyz format
    """
    # scipy uses xyzw format, so convert
    r_start = Rotation.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]])
    r_end = Rotation.from_quat([q_end[1], q_end[2], q_end[3], q_end[0]])

    # Create slerp interpolator
    key_rots = Rotation.concatenate([r_start, r_end])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)

    # Interpolate
    times = np.linspace(0, 1, num_steps)
    interp_rots = slerp(times)

    # Convert back to wxyz format
    quats_xyzw = interp_rots.as_quat()
    quats_wxyz = np.column_stack(
        [quats_xyzw[:, 3], quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2]]
    )
    return quats_wxyz


def plan_trajectory_linear_ik(
    robot: pk.Robot,
    target_link_name: str,
    start_pos: np.ndarray,
    start_wxyz: np.ndarray,
    end_pos: np.ndarray,
    end_wxyz: np.ndarray,
    num_waypoints: int = 25,
    use_prev_cfg: bool = True,
    jump_threshold: float = 0.5,
) -> np.ndarray:
    """Plan a trajectory by linear interpolation + IK at each waypoint.

    Args:
        robot: PyRoKi robot model
        target_link_name: Name of the end-effector link
        start_pos: Start position (3,)
        start_wxyz: Start orientation quaternion in wxyz format (4,)
        end_pos: End position (3,)
        end_wxyz: End orientation quaternion in wxyz format (4,)
        num_waypoints: Number of waypoints in the trajectory
        use_prev_cfg: If True, use previous IK solution to bias next solve
        jump_threshold: Max allowed joint change (radians) between waypoints before warning

    Returns:
        Array of shape (num_waypoints, num_joints) with joint configurations
    """
    # Linear interpolation for positions
    positions = np.linspace(start_pos, end_pos, num_waypoints)

    # SLERP for orientations
    orientations = slerp_quaternions(start_wxyz, end_wxyz, num_waypoints)

    # Solve IK for each waypoint
    trajectory = []
    prev_cfg = None
    jump_warnings = []

    for i, (pos, wxyz) in enumerate(zip(positions, orientations)):
        if use_prev_cfg and prev_cfg is not None:
            # Use velocity-cost IK to stay close to previous solution
            for _ in range(15):
                cfg = pks.solve_ik_vel_cost(
                    robot=robot,
                    target_link_name=target_link_name,
                    target_wxyz=wxyz,
                    target_position=pos,
                    prev_cfg=prev_cfg,
                )
                print(
                    f"Error: {np.linalg.norm(cfg - prev_cfg)}",
                    np.allclose(cfg, prev_cfg, atol=1e-3),
                )
                if np.allclose(cfg, prev_cfg, atol=1e-3):
                    break
                else:
                    prev_cfg = cfg
        else:
            cfg = pks.solve_ik(
                robot=robot,
                target_link_name=target_link_name,
                target_wxyz=wxyz,
                target_position=pos,
            )
        cfg = np.array(cfg)

        # Check for large joint jumps
        if prev_cfg is not None:
            joint_diff = np.abs(cfg - prev_cfg)
            large_jumps = np.where(joint_diff > jump_threshold)[0]
            if len(large_jumps) > 0:
                for joint_idx in large_jumps:
                    jump_warnings.append(
                        f"  Waypoint {i}: joint {joint_idx} jumped {np.degrees(joint_diff[joint_idx]):.1f}° "
                        f"({joint_diff[joint_idx]:.3f} rad)"
                    )

        trajectory.append(cfg)
        prev_cfg = cfg

    # Print warnings summary
    if jump_warnings:
        print(
            f"\nWARNING: {len(jump_warnings)} large joint jump(s) detected (threshold: {np.degrees(jump_threshold):.1f}°):"
        )
        for warning in jump_warnings:
            print(warning)
        print()

    return np.array(trajectory)


# =====================================================
# Logging and FastAPI app
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyroki_server")

app = FastAPI()

_ROBOT = None
_ROBOT_COLL = None
_TARGET_LINK = None
_SELF_COLLISION_PAIR_MASK = None
_SELF_COLLISION_DECOMPOSITION_PADDING_M = 0.015


async def _run_in_thread(fn, *args, **kwargs):
    """Run a blocking CPU-bound function without blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


# =====================================================
# Pydantic Models
# =====================================================


class IkRequest(BaseModel):
    """IK request with optional prev_cfg for vel-cost IK."""

    target_pose_wxyz_xyz: list[float]  # length 7 (wxyz + xyz)
    prev_cfg: list[float] | None = None  # optional


class IkResponse(BaseModel):
    joint_positions: list[float]


class IkMarginRequest(BaseModel):
    """Preflight a short Cartesian path without moving the robot."""

    target_poses_wxyz_xyz: list[list[float]]
    prev_cfg: list[float] | None = None
    obstacles: list[dict] | None = None
    min_self_collision_margin_m: float = 0.005
    min_world_collision_margin_m: float = 0.01
    min_joint_limit_margin_rad: float = 0.10
    max_joint_step_rad: float = 0.50
    max_position_error_m: float = 0.02
    restart_first_waypoint: bool = False


class IkMarginResponse(BaseModel):
    safe: bool
    reason: str
    joint_waypoints: list[list[float]]
    min_self_collision_margin_m: float | None = None
    raw_min_self_collision_distance_m: float | None = None
    min_world_collision_margin_m: float | None = None
    min_joint_limit_margin_rad: float | None = None
    max_joint_step_rad: float | None = None
    max_position_error_m: float | None = None


class ObstacleEntry(BaseModel):
    type: str
    point: list[float] | None = None
    normal: list[float] | None = None
    center: list[float] | None = None
    radius: float | None = None
    position: list[float] | None = None
    height: float | None = None
    extent: list[float] | None = None  # for box obstacles


class PlanRequest(BaseModel):
    start_pose_wxyz_xyz: list[float]  # length 7: wxyz quaternion + xyz position
    end_pose_wxyz_xyz: list[float]  # length 7: wxyz quaternion + xyz position
    obstacles: list[dict] | None = None  # optional list of ObstacleEntry
    timesteps: int = 20
    dt: float = 0.02


class PlanResponse(BaseModel):
    waypoints: list[list[float]]
    dt: float


# =====================================================
# INTERNAL HELPERS
# =====================================================


def _build_world_coll(obstacles: list[dict[str, Any]] | None):
    if obstacles is None:
        return []
    world = []
    for obj in obstacles:
        t = obj.get("type")
        if t == "halfspace":
            p = np.array(obj["point"], dtype=np.float64)
            n = np.array(obj["normal"], dtype=np.float64)
            world.append(pk.collision.HalfSpace.from_point_and_normal(p, n))
        elif t == "sphere":
            c = np.array(obj["center"], dtype=np.float64)
            r = float(obj["radius"])
            world.append(pk.collision.Sphere.from_center_and_radius(c, np.array([r])))
        elif t == "capsule":
            pos = np.array(obj["position"], dtype=np.float64)
            rad = float(obj["radius"])
            h = float(obj["height"])
            world.append(
                pk.collision.Capsule.from_radius_height(
                    position=pos,
                    radius=np.array([rad]),
                    height=np.array([h]),
                )
            )
        elif t == "box":
            extent = np.array(obj["extent"], dtype=np.float64)
            pos = np.array(obj.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
            world.append(
                pk.collision.Box.from_extent(
                    extent=extent,
                    position=pos,
                )
            )
        else:
            logger.warning(f"Unknown obstacle type '{t}', ignoring")
    return world


# =====================================================
# SERVER INIT — Load Pyroki Only Once
# =====================================================
def set_min_distance_from_limits(urdf: yourdfpy.URDF, min_distance_from_limits: float = 0.15) -> yourdfpy.URDF:
    """
    Set the minimum distance from limits for the robot.
    min_distance_from_limits: float in radians
    """
    for joint in urdf.robot.joints:
        if joint.type == "revolute" and joint.limit is not None:
            if joint.limit.lower is not None and joint.limit.upper is not None:
                joint.limit.lower = joint.limit.lower + min_distance_from_limits
                joint.limit.upper = joint.limit.upper - min_distance_from_limits
    return urdf


def init_pyroki_server(
    robot_urdf_name: str = "panda_description", target_link_name: str = "panda_hand"
):
    global _ROBOT, _ROBOT_COLL, _TARGET_LINK, _SELF_COLLISION_PAIR_MASK

    logger.info(f"Loading robot URDF '{robot_urdf_name}' with Pyroki...")

    from robot_descriptions.loaders.yourdfpy import load_robot_description

    urdf = set_min_distance_from_limits(load_robot_description(robot_urdf_name))

    _ROBOT = pk.Robot.from_urdf(urdf)
    # _ROBOT_COLL = pk.collision.RobotCollision.from_urdf(urdf)
    sphere_decomposition = json.load(open(Path(__file__).parent / "assets" / "panda_spheres.json"))
    _ROBOT_COLL = pk.collision.RobotCollision.from_sphere_decomposition(
        sphere_decomposition=sphere_decomposition,
        urdf=urdf,
    )
    _TARGET_LINK = target_link_name
    reference_cfg = np.asarray(_ROBOT.joint_var_cls(0).default_factory(), dtype=float)
    reference_distances = np.asarray(
        _ROBOT_COLL.compute_self_collision_distance(_ROBOT, reference_cfg), dtype=float
    )
    # Sphere decompositions intentionally overlap on some adjacent links. Treat
    # those reference overlaps as allowed contacts instead of making every pose fail.
    _SELF_COLLISION_PAIR_MASK = reference_distances >= 0.0

    logger.info("PyRoki loaded and ready!")


# =====================================================
# ROUTES
# =====================================================


def _do_solve_ik(target_pose_wxyz_xyz: np.ndarray, prev_cfg: np.ndarray | None) -> list[float]:
    """Blocking IK solve (CPU-bound)."""
    if prev_cfg is None:
        q = pks.solve_ik(
            robot=_ROBOT,
            target_link_name=_TARGET_LINK,
            target_position=target_pose_wxyz_xyz[-3:],
            target_wxyz=target_pose_wxyz_xyz[:-3],
        )
    else:
        q = pks.solve_ik_vel_cost(
            robot=_ROBOT,
            target_link_name=_TARGET_LINK,
            target_position=target_pose_wxyz_xyz[-3:],
            target_wxyz=target_pose_wxyz_xyz[:-3],
            prev_cfg=prev_cfg,
        )
    return list(map(float, q))


@app.post("/ik", response_model=IkResponse)
async def solve_ik(req: IkRequest):
    if _ROBOT is None:
        raise HTTPException(503, "Pyroki not initialized")

    target_pose_wxyz_xyz = np.array(req.target_pose_wxyz_xyz, dtype=np.float64)
    prev_cfg = np.array(req.prev_cfg, dtype=np.float64) if req.prev_cfg is not None else None

    try:
        joints = await _run_in_thread(_do_solve_ik, target_pose_wxyz_xyz, prev_cfg)
    except Exception as e:
        logger.exception("IK failed")
        raise HTTPException(500, f"IK solve failed: {e}")

    return IkResponse(joint_positions=joints)


def _do_check_ik_margin(req: IkMarginRequest) -> IkMarginResponse:
    """Solve every waypoint and measure safety margins on the resulting path."""
    if not req.target_poses_wxyz_xyz:
        return IkMarginResponse(safe=False, reason="empty_path", joint_waypoints=[])

    lower = np.asarray(_ROBOT.joints.lower_limits, dtype=float)[:-1]
    upper = np.asarray(_ROBOT.joints.upper_limits, dtype=float)[:-1]
    target_link_index = _ROBOT.links.names.index(_TARGET_LINK)
    previous = np.asarray(req.prev_cfg, dtype=float) if req.prev_cfg is not None else None
    worlds = _build_world_coll(req.obstacles)
    waypoints: list[list[float]] = []
    self_margins: list[float] = []
    world_margins: list[float] = []
    joint_margins: list[float] = []
    joint_steps: list[float] = []
    position_errors: list[float] = []

    def record_checked_cfg(cfg: np.ndarray, prior: np.ndarray | None) -> None:
        arm_cfg = cfg[:-1]
        joint_margins.append(
            float(np.min(np.minimum(arm_cfg - lower, upper - arm_cfg)))
        )
        if prior is not None:
            joint_steps.append(float(np.max(np.abs(arm_cfg - prior[:-1]))))
        self_distance = np.asarray(
            _ROBOT_COLL.compute_self_collision_distance(_ROBOT, cfg), dtype=float
        )
        checked_self_distance = self_distance[_SELF_COLLISION_PAIR_MASK]
        if checked_self_distance.size == 0:
            raise RuntimeError("self collision model unavailable")
        self_margins.append(float(np.min(checked_self_distance)))
        for world in worlds:
            world_distance = np.asarray(
                _ROBOT_COLL.compute_world_collision_distance(_ROBOT, cfg, world),
                dtype=float,
            )
            world_margins.append(float(np.min(world_distance)))
        waypoints.append(cfg.tolist())

    try:
        for waypoint_index, raw_pose in enumerate(req.target_poses_wxyz_xyz):
            pose = np.asarray(raw_pose, dtype=float)
            if pose.shape != (7,) or not np.all(np.isfinite(pose)):
                return IkMarginResponse(
                    safe=False, reason="invalid_pose", joint_waypoints=waypoints
                )
            step_reference = previous
            solve_seed = (
                None
                if waypoint_index == 0 and req.restart_first_waypoint
                else previous
            )
            cfg = np.asarray(_do_solve_ik(pose, solve_seed), dtype=float)
            if cfg.shape != (_ROBOT.joints.num_actuated_joints,) or not np.all(
                np.isfinite(cfg)
            ):
                return IkMarginResponse(
                    safe=False, reason="invalid_ik_solution", joint_waypoints=waypoints
                )

            if step_reference is not None:
                segments = max(
                    1,
                    int(np.ceil(
                        np.max(np.abs(cfg[:-1] - step_reference[:-1]))
                        / max(req.max_joint_step_rad, 1e-6)
                    )),
                )
                prior = step_reference
                for segment in range(1, segments + 1):
                    checked_cfg = step_reference + (cfg - step_reference) * (
                        segment / segments
                    )
                    record_checked_cfg(checked_cfg, prior)
                    prior = checked_cfg
            else:
                record_checked_cfg(cfg, None)

            actual_pose = np.asarray(_ROBOT.forward_kinematics(cfg), dtype=float)[
                target_link_index
            ]
            position_errors.append(float(np.linalg.norm(actual_pose[4:] - pose[4:])))
            previous = cfg
    except Exception as error:
        logger.warning("IK margin preflight failed: %s", error)
        return IkMarginResponse(
            safe=False,
            reason="ik_or_collision_check_failed",
            joint_waypoints=waypoints,
        )

    raw_min_self_distance = min(self_margins)
    report = {
        "joint_waypoints": waypoints,
        "min_self_collision_margin_m": (
            raw_min_self_distance + _SELF_COLLISION_DECOMPOSITION_PADDING_M
        ),
        "raw_min_self_collision_distance_m": raw_min_self_distance,
        "min_world_collision_margin_m": min(world_margins) if world_margins else None,
        "min_joint_limit_margin_rad": min(joint_margins),
        "max_joint_step_rad": max(joint_steps, default=0.0),
        "max_position_error_m": max(position_errors),
    }
    checks = (
        (report["max_position_error_m"] <= req.max_position_error_m, "ik_position_error"),
        (
            report["min_joint_limit_margin_rad"] >= req.min_joint_limit_margin_rad,
            "joint_limit_margin",
        ),
        (report["max_joint_step_rad"] <= req.max_joint_step_rad, "joint_step_margin"),
        (
            report["min_self_collision_margin_m"]
            >= req.min_self_collision_margin_m,
            "self_collision_margin",
        ),
        (
            report["min_world_collision_margin_m"] is None
            or report["min_world_collision_margin_m"]
            >= req.min_world_collision_margin_m,
            "world_collision_margin",
        ),
    )
    reason = next((name for passed, name in checks if not passed), "ok")
    return IkMarginResponse(safe=reason == "ok", reason=reason, **report)


@app.post("/ik-margin", response_model=IkMarginResponse)
async def check_ik_margin(req: IkMarginRequest):
    if _ROBOT is None or _ROBOT_COLL is None:
        raise HTTPException(503, "Pyroki not initialized")
    return await _run_in_thread(_do_check_ik_margin, req)


def _do_plan_motion(req: PlanRequest) -> PlanResponse:
    """Blocking motion planning (CPU-bound)."""
    start_pose = np.array(req.start_pose_wxyz_xyz, dtype=np.float64)
    end_pose = np.array(req.end_pose_wxyz_xyz, dtype=np.float64)

    start_wxyz = start_pose[:4]
    start_position = start_pose[4:]
    end_wxyz = end_pose[:4]
    end_position = end_pose[4:]

    timesteps = req.timesteps
    dt = req.dt

    sol_traj = plan_trajectory_linear_ik(
        robot=_ROBOT,
        target_link_name=_TARGET_LINK,
        start_pos=start_position,
        start_wxyz=start_wxyz,
        end_pos=end_position,
        end_wxyz=end_wxyz,
        num_waypoints=timesteps,
    )
    sol_traj = np.asarray(sol_traj)

    return PlanResponse(
        waypoints=sol_traj.tolist(),
        dt=float(dt),
    )


@app.post("/plan", response_model=PlanResponse)
async def plan_motion(req: PlanRequest):
    if _ROBOT is None or _ROBOT_COLL is None:
        raise HTTPException(503, "Pyroki not initialized")

    try:
        return await _run_in_thread(_do_plan_motion, req)
    except Exception as e:
        logger.exception("Planning failure")
        raise HTTPException(500, f"Motion planning failed: {e}")


# =====================================================
# ENTRYPOINT
# =====================================================


def main(
    robot: str = "panda_description",
    target_link: str = "panda_hand",
    port: int = 8116,
    host: str = "127.0.0.1",
):
    init_pyroki_server(robot_urdf_name=robot, target_link_name=target_link)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
