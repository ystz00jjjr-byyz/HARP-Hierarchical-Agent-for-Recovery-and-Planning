"""Franka single-cube bin-placement task."""

from capx.envs.tasks.base import CodeExecutionEnvBase


PROMPT = """
You are controlling a Franka Emika robot.
Goal: place the primary cube fully inside the target bin, release it gently,
and open the gripper. Query the live target region and object extent. The whole
cube, not only its center, must fit inside the bin. Nothing may be dropped.
Write only executable Python code. Import numpy explicitly when needed.
"""

ORACLE_CODE = """
import numpy as np

primary_pos, _, primary_extent = get_object_pose("primary", return_bbox_extent=True)
grasp_pos, grasp_quat = sample_grasp_pose("primary")
approach = max(0.08, 2.0 * float(primary_extent[2]))
lift = max(0.15, 3.0 * float(primary_extent[2]))
goto_pose(grasp_pos, grasp_quat, z_approach=approach)
close_gripper()
goto_pose(grasp_pos + np.array([0.0, 0.0, lift]), grasp_quat)

target_center, _, rim_z = get_target_region("target")
target_gripper = target_center + (grasp_pos - primary_pos)
entry = target_gripper.copy()
entry[2] = max(entry[2] + approach, rim_z + 0.5 * float(primary_extent[2]) + 0.04)
goto_pose(entry, grasp_quat)
goto_pose(target_gripper, grasp_quat)
open_gripper()
goto_pose(entry, grasp_quat)
"""


class FrankaBinPlaceCodeEnv(CodeExecutionEnvBase):
    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = ["FrankaBinPlaceCodeEnv"]
