# Code block 0
import numpy as np

# Query the live object extent and target region
obj_pos, _, obj_extent = get_object_pose('red block', return_bbox_extent=True)
target_pos, target_quat, target_size = get_target_region('target')

# Sample grasp pose for the primary object
grasp_pos, grasp_quat = sample_grasp_pose('red block')

# Stage 1: Grasp and lift the primary object
open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()
lift_pos = grasp_pos + np.array([0.0, 0.0, 0.25])
goto_pose(lift_pos, grasp_quat)

# Code block 1
import numpy as np

# Stage 1: Grasp and lift the primary object
obj_pos, _, obj_extent = get_object_pose("primary", return_bbox_extent=True)
target_pos, _, _ = get_target_region("target")

open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("primary")
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.2
goto_pose(lift_pos, grasp_quat)

# Code block 2
# Stage 2: Enter the bin, release gently, and retreat
above_target = target_pos.copy()
above_target[2] = lift_pos[2]
goto_pose(above_target, grasp_quat)

place_pos = target_pos.copy()
place_pos[2] = grasp_pos[2] + 0.01
goto_pose(place_pos, grasp_quat)

open_gripper()

retreat_pos = place_pos.copy()
retreat_pos[2] += 0.2
goto_pose(retreat_pos, grasp_quat)