# Code block 0
import numpy as np

# Query live object extent and target region geometry
obj_pos, obj_quat, obj_extent = get_object_pose('red block', return_bbox_extent=True)
target_pos, target_extent, target_yaw = get_target_region()

# Stage 1: Grasp and lift the primary object
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose('red block')
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()

# Lift the object securely
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.2
goto_pose(lift_pos, grasp_quat)

# Code block 1
import numpy as np

# Stage 1: Grasp and lift the primary object
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose('primary')
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
goto_pose(lift_pos, grasp_quat)

# Code block 2
import numpy as np

# Query live object extent and target region
_, _, obj_extent = get_object_pose("primary", return_bbox_extent=True)
target_center, target_extent, _ = get_target_region("target")

# Placement orientation (gripper pointing downward)
quat = np.array([0.0, 0.0, 1.0, 0.0])

# Calculate placement position inside the bin for a gentle release
place_pos = np.array(
    [
        target_center[0],
        target_center[1],
        target_center[2] + obj_extent[2] / 2.0 + 0.015,
    ]
)

# Enter the bin using vertical approach
goto_pose(place_pos, quat, z_approach=0.15)

# Gently release the primary object
open_gripper()

# Retreat safely upward above the bin
retreat_pos = np.array([target_center[0], target_center[1], 0.15])
goto_pose(retreat_pos, quat)

# Code block 3
import numpy as np

# Stage 1: Grasp and lift the primary object
open_gripper()
obj_pos, _, bbox_extent = get_object_pose("primary", return_bbox_extent=True)
grasp_pos, grasp_quat = sample_grasp_pose("primary")

# Approach and grasp
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()

# Lift safely above container rims
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
goto_pose(lift_pos, grasp_quat)

# Code block 4
import numpy as np

# Retrieve target region geometry
target_center, _, _ = get_target_region('target')
down_quat = np.array([0.0, 0.0, 1.0, 0.0])

# Move above the bin
above_bin_pos = np.array([target_center[0], target_center[1], 0.10])
goto_pose(above_bin_pos, down_quat)

# Lower into the bin
place_pos = np.array([target_center[0], target_center[1], target_center[2] + 0.04])
goto_pose(place_pos, down_quat)

# Gently release the object
open_gripper()

# Retreat up safely
retreat_pos = np.array([target_center[0], target_center[1], 0.15])
goto_pose(retreat_pos, down_quat)