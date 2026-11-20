# Code block 0
import numpy as np

# Stage 1: Grasp and lift the primary object
# Query object pose, extent, and grasp pose
obj_pos, _, obj_extent = get_object_pose("primary object", return_bbox_extent=True)
grasp_pos, grasp_quat = sample_grasp_pose("primary object")

# Open gripper prior to approach
open_gripper()

# Approach and reach grasp pose
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)

# Securely grasp the object
close_gripper()

# Lift the object safely
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.2
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
# Stage 1: Grasp and lift the primary object
open_gripper()
obj_pos, _, obj_extent = get_object_pose("primary", return_bbox_extent=True)
target_center, target_extent, _ = get_target_region("target")

grasp_pos, grasp_quat = sample_grasp_pose("primary")
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
goto_pose(lift_pos, grasp_quat)

# Code block 3
# Stage 1: grasp and lift the primary object
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("primary")
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
goto_pose(lift_pos, grasp_quat)

# Code block 4
import numpy as np

# Stage 1: Reset to safe home pose, open gripper, and grasp object with explicit waypoint approach
home_pose()
open_gripper()

obj_pos, _, obj_extent = get_object_pose('primary', return_bbox_extent=True)
grasp_pos, grasp_quat = sample_grasp_pose('primary')

pre_grasp_pos = grasp_pos.copy()
pre_grasp_pos[2] += 0.12
goto_pose(pre_grasp_pos, grasp_quat)
goto_pose(grasp_pos, grasp_quat)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.20
goto_pose(lift_pos, grasp_quat)

# Code block 5
# Stage 2: Transfer above bin, descend inside, release, and retreat
target_pos, target_quat, _ = get_target_region('target')

above_target = np.array([target_pos[0], target_pos[1], 0.15])
goto_pose(above_target, grasp_quat)

place_pos = np.array([target_pos[0], target_pos[1], target_pos[2] + 0.04])
goto_pose(place_pos, grasp_quat)
open_gripper()

goto_pose(above_target, grasp_quat)