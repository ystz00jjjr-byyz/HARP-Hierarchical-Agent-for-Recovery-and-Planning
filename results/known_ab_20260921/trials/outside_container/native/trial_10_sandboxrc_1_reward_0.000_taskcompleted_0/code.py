# Code block 0
import numpy as np

# Stage 1: Grasp and lift the primary object
obj_pos, _, obj_extent = get_object_pose("red block", return_bbox_extent=True)
target_pos, _, _ = get_target_region("target")
grasp_pos, grasp_quat = sample_grasp_pose("red block")

open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()
goto_pose(grasp_pos + np.array([0.0, 0.0, 0.2]), grasp_quat)

# Code block 1
import numpy as np

# Stage 1: Grasp and lift the primary object
try:
    obj_pos, obj_quat, obj_extent = get_object_pose('red block', return_bbox_extent=True)
except Exception:
    obj_pos, obj_quat, obj_extent = get_object_pose('red cube', return_bbox_extent=True)

try:
    grasp_pos, grasp_quat = sample_grasp_pose('red block')
except Exception:
    grasp_pos, grasp_quat = sample_grasp_pose('red cube')

open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.18
goto_pose(lift_pos, grasp_quat)

# Code block 2
# Stage 2: Enter the bin, release gently, and retreat
target_pos, target_quat, target_extent = get_target_region('target')

grasp_offset = grasp_pos - obj_pos

above_bin = np.array([target_pos[0], target_pos[1], lift_pos[2]])
goto_pose(above_bin, grasp_quat)

place_z = target_pos[2] + (obj_extent[2] / 2.0 if obj_extent is not None else 0.02) + grasp_offset[2] + 0.005
place_pos = np.array([target_pos[0], target_pos[1], place_z])
goto_pose(place_pos, grasp_quat)

open_gripper()
goto_pose(above_bin, grasp_quat)