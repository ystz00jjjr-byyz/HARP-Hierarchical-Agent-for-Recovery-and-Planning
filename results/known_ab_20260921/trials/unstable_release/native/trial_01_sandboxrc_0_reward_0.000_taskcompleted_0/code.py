# Code block 0
# Stage 1: Grasp and lift the primary object
object_pos, _, object_extent = get_object_pose("red block", return_bbox_extent=True)
grasp_pos, grasp_quat = sample_grasp_pose("red block")

open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.25
goto_pose(lift_pos, grasp_quat)

# Code block 1
import numpy as np

# Stage 1: Grasp and lift the primary object
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("primary")
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
goto_pose(lift_pos, grasp_quat)

# Code block 2
# Stage 2: enter the bin, release, and retreat
target_pos, target_extent, _ = get_target_region("target")
quat = np.array([0.0, 0.0, 1.0, 0.0])

# Move safely above the target bin
above_bin = np.array([target_pos[0], target_pos[1], 0.15])
goto_pose(above_bin, quat)

# Lower into the bin below the rim
place_pos = np.array([target_pos[0], target_pos[1], target_pos[2] + 0.05])
goto_pose(place_pos, quat)

# Release gently inside the bin
open_gripper()

# Retreat back above the bin
goto_pose(above_bin, quat)

# Code block 3
home_pose()

# Code block 4
home_pose()

# Code block 5
home_pose()

# Code block 6
open_gripper()
home_pose()

# Code block 7
home_pose()

# Code block 8
home_pose()

# Code block 9
# Move to home pose to allow the object to settle and complete retreat
home_pose()

# Code block 10
# The object has been placed inside the target bin, but physics has not fully settled.
# Moving to home pose allows time for the object to settle and reach a stable state.
home_pose()

# Code block 11
home_pose()