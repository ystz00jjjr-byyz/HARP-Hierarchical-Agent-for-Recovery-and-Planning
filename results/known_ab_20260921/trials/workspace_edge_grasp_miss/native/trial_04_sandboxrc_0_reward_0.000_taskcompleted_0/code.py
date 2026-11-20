# Code block 0
# Stage 1: Grasp and lift the primary object
obj_pos, _, obj_extent = get_object_pose("red block", return_bbox_extent=True)
target_pos, target_extent, target_yaw = get_target_region()

grasp_pos, grasp_quat = sample_grasp_pose("red block")

open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.25
goto_pose(lift_pos, grasp_quat)

# Code block 1
import numpy as np

# Stage 1: Grasp and lift the primary object
try:
    obj_pos, _, obj_extent = get_object_pose("primary", return_bbox_extent=True)
    grasp_pos, grasp_quat = sample_grasp_pose("primary")
except Exception:
    obj_pos, _, obj_extent = get_object_pose("primary object", return_bbox_extent=True)
    grasp_pos, grasp_quat = sample_grasp_pose("primary object")

target_pos, target_extent, _ = get_target_region("target")

open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.15)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.2
goto_pose(lift_pos, grasp_quat)

# Code block 2
import numpy as np

# Stage 1: Grasp and lift primary object
obj_pos, _, obj_extent = get_object_pose("primary", return_bbox_extent=True)
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("primary")
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
goto_pose(lift_pos, grasp_quat)

# Code block 3
import numpy as np

# Query live object extent and target region
try:
    obj_pos, obj_quat, obj_extent = get_object_pose('primary', return_bbox_extent=True)
except Exception:
    obj_pos, obj_quat, obj_extent = get_object_pose('primary object', return_bbox_extent=True)

target_pos, target_quat, _ = get_target_region()

# Stage 1: Grasp and lift the primary object
open_gripper()
try:
    grasp_pos, grasp_quat = sample_grasp_pose('primary')
except Exception:
    grasp_pos, grasp_quat = sample_grasp_pose('primary object')

goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
goto_pose(lift_pos, grasp_quat)

# Code block 4
import numpy as np

# Stage 1: Reset, grasp, and lift the primary object
home_pose()
open_gripper()

grasp_pos, grasp_quat = sample_grasp_pose("primary")
_, _, primary_extent = get_object_pose("primary", return_bbox_extent=True)

# Explicit two-step approach to avoid workspace boundary issues
pre_grasp_pos = grasp_pos.copy()
pre_grasp_pos[2] += 0.08
goto_pose(pre_grasp_pos, grasp_quat)

goto_pose(grasp_pos, grasp_quat)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
goto_pose(lift_pos, grasp_quat)

# Code block 5
# Stage 2: Move over the target bin, lower into bin, release, and retreat
target_pos, target_extent, _ = get_target_region("target")
gripper_down_quat = np.array([0.0, 0.0, 1.0, 0.0])

above_target = target_pos.copy()
above_target[2] += max(0.12, target_extent[2] / 2.0 + 0.10)
goto_pose(above_target, gripper_down_quat)

place_pos = target_pos.copy()
place_pos[2] = target_pos[2] + primary_extent[2] / 2.0 + 0.02
goto_pose(place_pos, gripper_down_quat)

open_gripper()

goto_pose(above_target, gripper_down_quat)
home_pose()