# Code block 0
# Stage 1: Query object extent and grasp, then lift the primary object
obj_pos, _, obj_extent = get_object_pose('red block', return_bbox_extent=True)
grasp_pos, grasp_quat = sample_grasp_pose('red block')

open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.2
goto_pose(lift_pos, grasp_quat)

# Code block 1
import numpy as np

# Stage 1: Grasp and lift the primary object
obj_pos, obj_quat, obj_extent = get_object_pose("red block", return_bbox_extent=True)
target_pos, target_extent, target_rim_z = get_target_region("target")

open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("red block")
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.2
goto_pose(lift_pos, grasp_quat)

# Code block 2
# Stage 1: Grasp and lift the primary object
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("primary object")
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.2
goto_pose(lift_pos, grasp_quat)

# Code block 3
# Stage 1: Grasp and lift the primary object
home_pose()
open_gripper()

grasp_pos, grasp_quat = sample_grasp_pose("primary")

# Move to a high pre-grasp pose to clear the container rim safely
pre_grasp_pos = grasp_pos.copy()
pre_grasp_pos[2] += 0.15
goto_pose(pre_grasp_pos, grasp_quat)

# Descend to the grasp pose and close gripper
goto_pose(grasp_pos, grasp_quat)
close_gripper()

# Lift the object well above the rim
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.18
goto_pose(lift_pos, grasp_quat)

# Code block 4
# Stage 2: Enter the bin, release gently, and retreat
target_region = get_target_region("target")
target_center = target_region[0]

# Approach above target bin
above_bin_pos = target_center.copy()
above_bin_pos[2] += 0.18
goto_pose(above_bin_pos, grasp_quat)

# Lower into bin gently
place_pos = target_center.copy()
place_pos[2] += 0.03
goto_pose(place_pos, grasp_quat)

# Release object
open_gripper()

# Retreat upward and return to safe home
goto_pose(above_bin_pos, grasp_quat)
home_pose()