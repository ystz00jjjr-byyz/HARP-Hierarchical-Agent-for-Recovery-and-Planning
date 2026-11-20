# Code block 0
import numpy as np

# Stage 1: Query live object extent, grasp and lift the primary object
obj_pos, _, obj_extent = get_object_pose("red block", return_bbox_extent=True)
grasp_pos, grasp_quat = sample_grasp_pose("red block")

# Approach and grasp the primary object
open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()

# Lift the primary object safely above obstacles
lift_pos = grasp_pos + np.array([0.0, 0.0, 0.20])
goto_pose(lift_pos, grasp_quat)

# Code block 1
# Query live object extent and target region
obj_pos, _, obj_extent = get_object_pose('primary object', return_bbox_extent=True)
target_pos, target_quat, target_radius = get_target_region('target')

# Recovery stage: ACQUIRE_GRASP - grasp and lift primary object
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose('primary object')
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
goto_pose(lift_pos, grasp_quat)

# Code block 2
# Stage 1: Grasp and lift the primary object
obj_pos, obj_quat, obj_extent = get_object_pose('red block', return_bbox_extent=True)
target_pos, target_quat, target_radius = get_target_region('target')

open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose('red block')
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
goto_pose(lift_pos, grasp_quat)

# Code block 3
# Stage 1: Acquire grasp on primary object and lift safely
home_pose()
open_gripper()

obj_pos, _, obj_extent = get_object_pose('red block', return_bbox_extent=True)
grasp_pos, grasp_quat = sample_grasp_pose('red block')

# Approach vertically from well above the container rim to avoid collision
pre_grasp = grasp_pos.copy()
pre_grasp[2] = 0.12
goto_pose(pre_grasp, grasp_quat)

# Descend straight to grasp pose
goto_pose(grasp_pos, grasp_quat)
close_gripper()

# Lift the object clear of the container rim
lift_pos = grasp_pos.copy()
lift_pos[2] = 0.15
goto_pose(lift_pos, grasp_quat)

# Code block 4
# Stage 2: Move into target bin, release object gently, and retreat
target_pos, target_quat, _ = get_target_region('target')

# Position safely above the target bin
pre_place = target_pos.copy()
pre_place[2] = 0.15
goto_pose(pre_place, grasp_quat)

# Enter the bin gently
place_pos = target_pos.copy()
if place_pos[2] < -0.082:
    place_pos[2] = -0.082
goto_pose(place_pos, grasp_quat)

open_gripper()

# Retreat safely above bin and return to home
retreat_pos = target_pos.copy()
retreat_pos[2] = 0.15
goto_pose(retreat_pos, grasp_quat)
home_pose()