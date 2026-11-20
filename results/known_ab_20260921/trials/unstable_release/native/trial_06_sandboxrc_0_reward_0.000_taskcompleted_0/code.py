# Code block 0


# Code block 1
import numpy as np

open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose('primary')
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
goto_pose(lift_pos, grasp_quat)

# Code block 2
# Stage 1: Grasp and lift the primary object
obj_pos, _, obj_extent = get_object_pose('primary', return_bbox_extent=True)
grasp_pos, grasp_quat = sample_grasp_pose('primary')

open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] = 0.15
goto_pose(lift_pos, grasp_quat)

# Code block 3
import numpy as np

# Stage 1: Grasp and lift the primary object
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("primary")
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
goto_pose(lift_pos, grasp_quat)

# Code block 4
import numpy as np

# Stage 1: Query environment, grasp and lift the primary object
pos, _, extent = get_object_pose('primary', return_bbox_extent=True)
grasp_pos, grasp_quat = sample_grasp_pose('primary')
target_center, target_extent, _ = get_target_region('target')

# Structurally new sequence with explicit homing and pre-grasp approach
home_pose()
open_gripper()
goto_pose(grasp_pos + np.array([0.0, 0.0, 0.08]), grasp_quat)
goto_pose(grasp_pos, grasp_quat)
close_gripper()
goto_pose(grasp_pos + np.array([0.0, 0.0, 0.18]), grasp_quat)

# Code block 5
# Stage 2: Enter target bin, release gently, and retreat
place_z = target_center[2] + extent[2] / 2.0 + 0.02
place_pos = np.array([target_center[0], target_center[1], place_z])

goto_pose(place_pos + np.array([0.0, 0.0, 0.15]), grasp_quat)
goto_pose(place_pos, grasp_quat)
open_gripper()
goto_pose(place_pos + np.array([0.0, 0.0, 0.15]), grasp_quat)
home_pose()