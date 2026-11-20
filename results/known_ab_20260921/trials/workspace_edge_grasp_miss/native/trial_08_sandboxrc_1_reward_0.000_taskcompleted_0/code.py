# Code block 0
import numpy as np

# Query the live object extent and target region
object_name = "red block"
for name in ["red block", "red cube", "block", "cube"]:
    try:
        obj_pos, _, obj_extent = get_object_pose(name, return_bbox_extent=True)
        if obj_pos is not None and obj_extent is not None:
            grasp_pos, grasp_quat = sample_grasp_pose(name)
            object_name = name
            break
    except Exception:
        continue

try:
    target_pos, target_quat, target_size = get_target_region("target")
except Exception:
    try:
        target_pos, target_quat, target_size = get_target_region("target bin")
    except Exception:
        target_pos, target_quat, target_size = get_target_region()

# Determine lift height to safely clear bin walls
lift_z = max(grasp_pos[2] + 0.25, target_pos[2] + 0.25)
lift_pos = grasp_pos.copy()
lift_pos[2] = lift_z

# Stage 1: grasp and lift the primary object
open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()
goto_pose(lift_pos, grasp_quat)

# Code block 1
# Query live object extent and target region
_, _, obj_extent = get_object_pose('red cube', return_bbox_extent=True)
target_center, target_extent, _ = get_target_region()

# Stage 1: grasp and lift primary object
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose('red cube')
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.2
goto_pose(lift_pos, grasp_quat)

# Code block 2
# Stage 2: enter the bin, release, and retreat
pre_place_pos = np.array([target_center[0], target_center[1], lift_pos[2]])
goto_pose(pre_place_pos, grasp_quat)

place_pos = np.array([target_center[0], target_center[1], target_center[2] + obj_extent[2] / 2.0 + 0.01])
goto_pose(place_pos, grasp_quat)

open_gripper()
goto_pose(pre_place_pos, grasp_quat)