# Code block 0
import numpy as np

# Query live object extent and target region
obj_pos, _, obj_extent = get_object_pose("red block", return_bbox_extent=True)
grasp_pos, grasp_quat = sample_grasp_pose("red block")
target_pos, target_extent, _ = get_target_region("target")

# Stage 1: Grasp and lift the primary object
open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()
lift_pos = grasp_pos + np.array([0.0, 0.0, 0.18])
goto_pose(lift_pos, grasp_quat)

# Code block 1
import numpy as np

# Query live object extent and grasp pose
object_names = ['red block', 'primary', 'primary object', 'red cube']
grasp_pos, grasp_quat = None, None
obj_extent = None

for name in object_names:
    try:
        grasp_pos, grasp_quat = sample_grasp_pose(name)
        _, _, obj_extent = get_object_pose(name, return_bbox_extent=True)
        if grasp_pos is not None:
            break
    except Exception:
        continue

if grasp_pos is None:
    for name in object_names:
        try:
            pos, _, obj_extent = get_object_pose(name, return_bbox_extent=True)
            if pos is not None:
                grasp_pos = pos
                grasp_quat = np.array([0.0, 0.0, 1.0, 0.0])
                break
        except Exception:
            continue

target_center, target_extent, _ = get_target_region('target')

# --- Stage 1: Grasp and Lift ---
open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.18
goto_pose(lift_pos, grasp_quat)

# Code block 2
import numpy as np

# Stage 1: Grasp and lift the primary object
pos, quat, extent = get_object_pose('primary', return_bbox_extent=True)
grasp_pos, grasp_quat = sample_grasp_pose('primary')

open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.20
goto_pose(lift_pos, grasp_quat)

# Code block 3
import numpy as np

# Stage 1: Grasp and lift the primary object
open_gripper()
_, _, obj_extent = get_object_pose("primary", return_bbox_extent=True)
grasp_pos, grasp_quat = sample_grasp_pose("primary")
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()
lift_pos = grasp_pos + np.array([0.0, 0.0, 0.15])
goto_pose(lift_pos, grasp_quat)

# Code block 4
# Stage 1: Grasp and lift the primary object
open_gripper()
obj_pos, obj_quat, obj_extent = get_object_pose(
    "primary object", return_bbox_extent=True
)
target_pos, target_quat, target_radius = get_target_region("target")
grasp_pos, grasp_quat = sample_grasp_pose("primary object")

goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.2
goto_pose(lift_pos, grasp_quat)

# Code block 5
# Stage 1: Grasp and lift the primary object
open_gripper()
obj_pos, _, obj_extent = get_object_pose("primary", return_bbox_extent=True)
grasp_pos, grasp_quat = sample_grasp_pose("primary")

goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.2
goto_pose(lift_pos, grasp_quat)

# Code block 6
# Stage 2: Enter the bin, release, and retreat
target_pos, target_extent, _ = get_target_region("target")
offset = grasp_pos - obj_pos

above_bin = target_pos.copy()
above_bin[2] = 0.15
goto_pose(above_bin, grasp_quat)

place_pos = target_pos.copy()
place_pos[0] += offset[0]
place_pos[1] += offset[1]
place_pos[2] = target_pos[2] + 0.03 + offset[2]
goto_pose(place_pos, grasp_quat)

open_gripper()

goto_pose(above_bin, grasp_quat)