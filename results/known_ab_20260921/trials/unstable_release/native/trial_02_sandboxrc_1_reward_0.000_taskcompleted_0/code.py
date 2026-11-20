# Code block 0
import numpy as np

# Query live object extent and target region
object_pos, _, object_extent = get_object_pose("red block", return_bbox_extent=True)
target_pos, target_extent, target_yaw = get_target_region("target")

# Sample grasp pose for the red block
grasp_pos, grasp_quat = sample_grasp_pose("red block")

# Calculate lift height to safely clear bin walls
bin_top_z = target_pos[2] + (target_extent[2] / 2.0 if target_extent is not None else 0.1)
lift_z = max(grasp_pos[2] + 0.18, bin_top_z + 0.10)
lift_pos = np.array([grasp_pos[0], grasp_pos[1], lift_z])

# STAGE 1: Grasp and lift primary object
open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()
goto_pose(lift_pos, grasp_quat)

# Code block 1
# Stage 1: Grasp and lift the primary object
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose('red block')
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.2
goto_pose(lift_pos, grasp_quat)

# Code block 2
import numpy as np

# Stage 1: Grasp and lift the primary object
open_gripper()

try:
    grasp_pos, grasp_quat = sample_grasp_pose("primary")
except Exception:
    grasp_pos, _, _ = get_object_pose("primary", return_bbox_extent=True)
    grasp_quat = np.array([0.0, 0.0, 1.0, 0.0])

# Descend vertically to avoid colliding with or displacing the object
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()

# Lift the object well above the container rim
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
goto_pose(lift_pos, grasp_quat)

# Code block 3
# Stage 2: Enter the bin, release, and retreat
target_res = get_target_region("target")
target_pos = target_res[0]

# High entry pose above the bin to ensure clearing container rim
above_bin_pos = np.array([target_pos[0], target_pos[1], 0.10])
goto_pose(above_bin_pos, grasp_quat)

# Lower into the bin for gentle release
place_pos = np.array([target_pos[0], target_pos[1], target_pos[2] + 0.03])
goto_pose(place_pos, grasp_quat)

open_gripper()

# Retreat back above the bin
goto_pose(above_bin_pos, grasp_quat)