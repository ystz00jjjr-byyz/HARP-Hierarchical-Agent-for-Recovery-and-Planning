# Code block 0
import numpy as np

# Query the live object extent and target region
obj_pos, obj_quat, obj_extent = get_object_pose('red block', return_bbox_extent=True)
target_pos, target_extent, target_yaw = get_target_region('target')

# Compute target floor height and placement gripper z
obj_floor_z = obj_pos[2] - obj_extent[2] / 2.0
candidate_cavity_floor = target_pos[2] - target_extent[2] / 2.0 if (len(target_extent) == 3 and target_extent[2] > 0.02) else target_pos[2]
candidate_surface_floor = target_pos[2]

# Select floor estimate closest to current compartment floor
if abs(candidate_cavity_floor - obj_floor_z) < abs(candidate_surface_floor - obj_floor_z):
    target_floor_z = candidate_cavity_floor
else:
    target_floor_z = candidate_surface_floor

# Stage 1: Grasp and lift the primary object
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose('red block')
goto_pose(grasp_pos, grasp_quat, z_approach=0.10)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.20
goto_pose(lift_pos, grasp_quat)

# Code block 1
# Stage 2: Enter the bin, release, and retreat
above_target = np.array([target_pos[0], target_pos[1], lift_pos[2]])
goto_pose(above_target, grasp_quat)

# Calculate gripper z when the object gently rests on the target floor
gripper_from_bottom = grasp_pos[2] - obj_floor_z
target_gripper_z = target_floor_z + 0.005 + gripper_from_bottom
place_pos = np.array([target_pos[0], target_pos[1], target_gripper_z])
goto_pose(place_pos, grasp_quat)

open_gripper()

retreat_pos = np.array([target_pos[0], target_pos[1], lift_pos[2]])
goto_pose(retreat_pos, grasp_quat)