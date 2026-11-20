# Code block 0
import numpy as np

# Stage 1: Grasp and lift the primary object
# Identify the primary object (red block / red cube)
obj_name = None
for name in ["red block", "red cube", "cube", "block"]:
    try:
        obj_pos, _, obj_extent = get_object_pose(name, return_bbox_extent=True)
        if obj_pos is not None:
            obj_name = name
            break
    except Exception:
        continue

# Sample grasp pose
grasp_pos, grasp_quat = sample_grasp_pose(obj_name)

# 1. Ensure gripper is open to prepare for grasp
open_gripper()

# 2. Approach and move to grasp pose
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)

# 3. Grasp the object securely
close_gripper()

# 4. Lift the object vertically
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.2
goto_pose(lift_pos, grasp_quat)

# Code block 1
import numpy as np

# Query live object extent and grasp pose
obj_name = 'red cube'
obj_pos, _, obj_extent = get_object_pose(obj_name, return_bbox_extent=True)
if obj_pos is None:
    obj_name = 'red block'
    obj_pos, _, obj_extent = get_object_pose(obj_name, return_bbox_extent=True)

grasp_pos, grasp_quat = sample_grasp_pose(obj_name)
target_center, target_extent, _ = get_target_region('target')

# Stage 1: grasp and lift the primary object
open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
goto_pose(lift_pos, grasp_quat)

# Code block 2
# Stage 2: enter the bin, release, and retreat
above_bin = np.array([target_center[0], target_center[1], lift_pos[2]])
goto_pose(above_bin, grasp_quat)

extent_z = obj_extent[2] if obj_extent is not None else 0.04
place_pos = np.array([target_center[0], target_center[1], grasp_pos[2] + 0.005])
goto_pose(place_pos, grasp_quat)

open_gripper()

retreat_pos = np.array([target_center[0], target_center[1], lift_pos[2]])
goto_pose(retreat_pos, grasp_quat)