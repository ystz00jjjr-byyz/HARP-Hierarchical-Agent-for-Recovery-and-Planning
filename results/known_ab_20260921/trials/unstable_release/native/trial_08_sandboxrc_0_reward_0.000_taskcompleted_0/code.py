# Code block 0
import numpy as np

# Query live object extent and target region
object_name = "red block"
for name in ["red block", "red cube", "block"]:
    try:
        obj_pos, _, obj_extent = get_object_pose(name, return_bbox_extent=True)
        if obj_pos is not None:
            object_name = name
            break
    except Exception:
        continue

try:
    target_pos, target_extent, _ = get_target_region()
except Exception:
    target_pos, target_extent, _ = get_target_region("target bin")

# Sample grasp pose
grasp_pos, grasp_quat = sample_grasp_pose(object_name)

# Calculate safe lift height to clear all bin walls
bin_top_z = target_pos[2] + (target_extent[2] / 2.0 if len(target_extent) >= 3 else 0.1)
lift_z = max(grasp_pos[2] + 0.18, bin_top_z + 0.08)
lift_pos = np.array([grasp_pos[0], grasp_pos[1], lift_z])

# =========================================================================
# Stage 1: Grasp and lift the primary object
# =========================================================================
open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()
goto_pose(lift_pos, grasp_quat)

# Code block 1
import numpy as np

# Query live object extent and target region
obj_pos, _, obj_extent = get_object_pose("red block", return_bbox_extent=True)
target_pos, target_quat, _ = get_target_region("target")

# Stage 1: Grasp and lift the primary object
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("red block")
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] = 0.15
goto_pose(lift_pos, grasp_quat)

# Code block 2
import numpy as np

# Stage 1: Grasp and lift the primary object
open_gripper()

grasp_pos, grasp_quat = sample_grasp_pose('primary')
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
goto_pose(lift_pos, grasp_quat)

# Code block 3
import numpy as np

# Query target region geometry
target_center, target_extent, _ = get_target_region("target")
quat = np.array([0.0, 0.0, 1.0, 0.0])

# Lift higher at current grasp position for safe clearance
obj_pos, _, _ = get_object_pose("primary")
lift_pos = np.array([obj_pos[0], obj_pos[1], 0.15])
goto_pose(lift_pos, quat)

# Move directly above the target bin
above_target = np.array([target_center[0], target_center[1], 0.15])
goto_pose(above_target, quat)

# Lower fully inside the target bin below the rim (-0.022m)
place_pos = np.array([target_center[0], target_center[1], -0.06])
goto_pose(place_pos, quat)

# Gently release the object and retreat
open_gripper()
goto_pose(above_target, quat)

# Code block 4
home_pose()

# Code block 5
home_pose()

# Code block 6
home_pose()

# Code block 7
home_pose()

# Code block 8
home_pose()

# Code block 9
open_gripper()
home_pose()

# Code block 10
home_pose()

# Code block 11
open_gripper()
home_pose()