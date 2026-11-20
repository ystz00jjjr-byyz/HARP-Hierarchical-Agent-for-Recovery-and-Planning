# Code block 0
import numpy as np

# Query live object extent and target region
object_name = "red block"
obj_pos, obj_quat, obj_extent = get_object_pose(
    object_name, return_bbox_extent=True
)
target_pos, target_extent, target_yaw = get_target_region()

# Sample grasp pose for the primary object
grasp_pos, grasp_quat = sample_grasp_pose(object_name)
grasp_offset = grasp_pos - obj_pos

# Stage 1: Grasp and lift the primary object
open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.10)
close_gripper()

lift_z = max(grasp_pos[2] + 0.20, target_pos[2] + target_extent[2] / 2.0 + 0.10)
lift_pos = np.array([grasp_pos[0], grasp_pos[1], lift_z])
goto_pose(lift_pos, grasp_quat)

# Code block 1
import numpy as np

# Stage 1: Grasp and lift the primary object
open_gripper()
obj_pos, _, obj_extent = get_object_pose("primary object", return_bbox_extent=True)
grasp_pos, grasp_quat = sample_grasp_pose("primary object")

# Approach and grasp
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

# Lift the primary object
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.2
goto_pose(lift_pos, grasp_quat)

# Code block 2
# Query live object extent and target region
obj_pos, obj_quat, obj_extent = get_object_pose('primary', return_bbox_extent=True)
target_center, target_extent, target_yaw = get_target_region('target')

# Stage 1: grasp and lift the primary object
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose('primary')
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.18
goto_pose(lift_pos, grasp_quat)

# Code block 3
# Stage 2: enter the bin, release, and retreat
above_bin_pos = target_center.copy()
above_bin_pos[2] = lift_pos[2]
goto_pose(above_bin_pos, grasp_quat)

place_pos = target_center.copy()
place_pos[2] += 0.03
goto_pose(place_pos, grasp_quat)

open_gripper()
goto_pose(above_bin_pos, grasp_quat)