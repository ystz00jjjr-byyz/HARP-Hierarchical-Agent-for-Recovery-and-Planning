"""Franka two-object sorting task."""

from capx.envs.tasks.base import CodeExecutionEnvBase


PROMPT = """
Place the primary object fully inside target_primary, then place the secondary
object fully inside target_secondary. Use live object extents and target
geometry, release gently, and finish with the gripper open.
"""

ORACLE_CODE = """
import numpy as np

home_pose()
primary_pos, _, primary_extent = get_object_pose("primary", return_bbox_extent=True)
primary_grasp, primary_quat = sample_grasp_pose("primary")
primary_approach = max(0.08, 2.0 * float(primary_extent[2]))
goto_pose(primary_grasp, primary_quat, z_approach=primary_approach)
close_gripper()
goto_pose(primary_grasp + np.array([0.0, 0.0, 0.15]), primary_quat)
primary_target, _, _ = get_target_region("target_primary")
primary_place = primary_target + (primary_grasp - primary_pos)
goto_pose(primary_place + np.array([0.0, 0.0, primary_approach]), primary_quat)
goto_pose(primary_place, primary_quat)
open_gripper()
goto_pose(primary_place + np.array([0.0, 0.0, primary_approach]), primary_quat)

secondary_pos, _, secondary_extent = get_object_pose("secondary", return_bbox_extent=True)
secondary_grasp, secondary_quat = sample_grasp_pose("secondary")
secondary_approach = max(0.08, 2.0 * float(secondary_extent[2]))
goto_pose(secondary_grasp, secondary_quat, z_approach=secondary_approach)
close_gripper()
goto_pose(secondary_grasp + np.array([0.0, 0.0, 0.15]), secondary_quat)
secondary_target, _, _ = get_target_region("target_secondary")
secondary_place = secondary_target + (secondary_grasp - secondary_pos)
goto_pose(secondary_place + np.array([0.0, 0.0, secondary_approach]), secondary_quat)
goto_pose(secondary_place, secondary_quat)
open_gripper()
goto_pose(secondary_place + np.array([0.0, 0.0, secondary_approach]), secondary_quat)
home_pose()
"""

THREE_OBJECT_PROMPT = """
Place the primary object fully inside target_primary, the secondary object
fully inside target_secondary, and the tertiary object fully inside
target_tertiary. Use live RGB-D extents and target geometry, release gently,
and finish with the gripper open.
"""

THREE_OBJECT_ORACLE_CODE = ORACLE_CODE.removesuffix("home_pose()\n") + """
tertiary_pos, _, tertiary_extent = get_object_pose("tertiary", return_bbox_extent=True)
tertiary_grasp, tertiary_quat = sample_grasp_pose("tertiary")
tertiary_approach = max(0.08, 2.0 * float(tertiary_extent[2]))
goto_pose(tertiary_grasp, tertiary_quat, z_approach=tertiary_approach)
close_gripper()
goto_pose(tertiary_grasp + np.array([0.0, 0.0, 0.15]), tertiary_quat)
tertiary_target, _, _ = get_target_region("target_tertiary")
tertiary_place = tertiary_target + (tertiary_grasp - tertiary_pos)
goto_pose(tertiary_place + np.array([0.0, 0.0, tertiary_approach]), tertiary_quat)
goto_pose(tertiary_place, tertiary_quat)
open_gripper()
goto_pose(tertiary_place + np.array([0.0, 0.0, tertiary_approach]), tertiary_quat)
home_pose()
"""

FOUR_OBJECT_PROMPT = """
Place the primary, secondary, tertiary, and quaternary objects fully inside
their matching target regions. Use live RGB-D extents and target geometry,
release gently, and finish with the gripper open.
"""

FOUR_OBJECT_ORACLE_CODE = THREE_OBJECT_ORACLE_CODE.removesuffix("home_pose()\n") + """
quaternary_pos, _, quaternary_extent = get_object_pose(
    "quaternary", return_bbox_extent=True
)
quaternary_grasp, quaternary_quat = sample_grasp_pose("quaternary")
quaternary_approach = max(0.08, 2.0 * float(quaternary_extent[2]))
goto_pose(quaternary_grasp, quaternary_quat, z_approach=quaternary_approach)
close_gripper()
goto_pose(quaternary_grasp + np.array([0.0, 0.0, 0.15]), quaternary_quat)
quaternary_target, _, _ = get_target_region("target_quaternary")
quaternary_place = quaternary_target + (quaternary_grasp - quaternary_pos)
goto_pose(
    quaternary_place + np.array([0.0, 0.0, quaternary_approach]),
    quaternary_quat,
)
goto_pose(quaternary_place, quaternary_quat)
open_gripper()
goto_pose(
    quaternary_place + np.array([0.0, 0.0, quaternary_approach]),
    quaternary_quat,
)
home_pose()
"""


class FrankaMultiObjectSortCodeEnv(CodeExecutionEnvBase):
    prompt = PROMPT
    oracle_code = ORACLE_CODE


class FrankaThreeObjectSortCodeEnv(CodeExecutionEnvBase):
    prompt = THREE_OBJECT_PROMPT
    oracle_code = THREE_OBJECT_ORACLE_CODE


class FrankaFourObjectSortCodeEnv(CodeExecutionEnvBase):
    prompt = FOUR_OBJECT_PROMPT
    oracle_code = FOUR_OBJECT_ORACLE_CODE


__all__ = [
    "FrankaMultiObjectSortCodeEnv",
    "FrankaThreeObjectSortCodeEnv",
    "FrankaFourObjectSortCodeEnv",
]
