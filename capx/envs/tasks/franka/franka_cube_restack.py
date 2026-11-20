from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a Franka Emika robot with API described below.
Goal: Gently place the primary cube on top of the secondary cube and then open the gripper. Nothing should be dropped from a height.
Use the extent of the cubes to calculate the exact height for placement.
You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
The functions (APIs) below are already imported to the environment. If you want to use numpy, scipy, torch, etc. you need to import them explicitly.
"""
ORACLE_CODE = """
import numpy as np

secondary_pos, _, secondary_ext = get_object_pose("secondary", return_bbox_extent=True)
primary_pos, _, primary_ext = get_object_pose("primary", return_bbox_extent=True)

# Move the secondary cube aside using its live grasp plan and current size.
pick_pos, pick_quat = sample_grasp_pose("secondary")
secondary_approach = max(0.05, 1.5 * float(secondary_ext[2]))
goto_pose(pick_pos, pick_quat, z_approach=secondary_approach)
close_gripper()
clearance = max(0.15, 0.5 * float(primary_ext[1] + secondary_ext[1]) + 0.10)
sideways_y = clearance if secondary_pos[1] < 0.10 else -clearance
secondary_center_z = primary_pos[2] + 0.5 * float(secondary_ext[2] - primary_ext[2])
grasp_offset = pick_pos - secondary_pos
secondary_place = np.array([
    secondary_pos[0], secondary_pos[1] + sideways_y, secondary_center_z
]) + grasp_offset
goto_pose(secondary_place + np.array([0.0, 0.0, secondary_approach]), pick_quat)
goto_pose(secondary_place, pick_quat)
open_gripper()
goto_pose(
    secondary_place + np.array([0.0, 0.0, max(0.10, 2.0 * float(secondary_ext[2]))]),
    pick_quat,
)

# Grasp the primary cube with a live planner result.
primary_pos, _, primary_ext = get_object_pose("primary", return_bbox_extent=True)
pick_pos, pick_quat = sample_grasp_pose("primary")
primary_approach = max(0.08, 2.0 * float(primary_ext[2]))
goto_pose(pick_pos, pick_quat, z_approach=primary_approach)
close_gripper()
lift_height = max(0.15, 3.0 * float(primary_ext[2]))
goto_pose(pick_pos + np.array([0.0, 0.0, lift_height]), pick_quat)

# Place the primary center one half-height sum above the secondary center.
secondary_pos, _, secondary_ext = get_object_pose("secondary", return_bbox_extent=True)
target_center = secondary_pos + np.array([
    0.0, 0.0, 0.5 * float(primary_ext[2] + secondary_ext[2])
])
target_gripper = target_center + (pick_pos - primary_pos)
place_approach = max(0.05, 1.5 * float(primary_ext[2]))
goto_pose(target_gripper + np.array([0.0, 0.0, place_approach]), pick_quat)
goto_pose(target_gripper, pick_quat)
open_gripper()
goto_pose(
    target_gripper + np.array([0.0, 0.0, max(0.10, 2.0 * float(primary_ext[2]))]),
    pick_quat,
)
"""


# ---------------------------- High-level Env -----------------------------
class FrankaRestackCodeEnv(CodeExecutionEnvBase):
    """High-level code environment for Franka cube restack using SimpleExecutor."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = [
    "FrankaRestackCodeEnv",
]
