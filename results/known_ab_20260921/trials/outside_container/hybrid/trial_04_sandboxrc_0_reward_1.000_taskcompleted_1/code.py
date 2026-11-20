# Code block 0
import json
_offline_params = json.loads('{"primary_pose": [0.524024553745272, -0.02608510886650045, -0.08303964604807756, -0.13760794942302357, -3.33320098494465e-13, -1.0811304192714995e-12, 0.9904867754067141], "primary_extent": [0.04, 0.04, 0.04]}')

import numpy as np

primary_pose = np.asarray(_offline_params["primary_pose"], dtype=float)
primary_extent = np.asarray(_offline_params["primary_extent"], dtype=float)
primary_pos = primary_pose[:3]
primary_quat = primary_pose[3:7]
w, x, y, z = primary_quat / np.linalg.norm(primary_quat)
yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
primary_grasp_pos = primary_pos + np.array([
    0.0, 0.0, max(0.004, 0.175 * float(primary_extent[2]))
])
primary_grasp_quat = np.array([
    0.0, -np.sin(0.5 * yaw), np.cos(0.5 * yaw), 0.0
])
approach = max(0.08, 2.0 * float(primary_extent[2]))
lift = max(0.15, 3.0 * float(primary_extent[2]))
goto_pose(primary_grasp_pos, primary_grasp_quat, z_approach=approach)
close_gripper()
goto_pose(primary_grasp_pos + np.array([0.0, 0.0, lift]), primary_grasp_quat)

# Code block 1
if "target_grasp_pos" in globals():
    primary_grasp_pos = target_grasp_pos
    primary_grasp_quat = target_grasp_quat
    primary_pos = current_pos
target_center, target_extent, rim_z = get_target_region("target")
target_gripper = target_center + (primary_grasp_pos - primary_pos)
entry = target_gripper.copy()
entry[2] = max(
    entry[2] + approach,
    rim_z + 0.5 * float(primary_extent[2]) + 0.04,
)
goto_pose(entry, primary_grasp_quat)
goto_pose(target_gripper, primary_grasp_quat)
open_gripper()
goto_pose(entry, primary_grasp_quat)
open_gripper()

# Code block 2
import json
_offline_params = json.loads('{"primary_pose": [0.6327012876205169, 0.2562223078822035, -0.08210775542021675, -0.05564875356805497, -6.9176355040158005e-19, 1.4885695753323586e-17, 0.9984504074946949], "primary_extent": [0.04, 0.04, 0.04]}')

import numpy as np

primary_pose = np.asarray(_offline_params["primary_pose"], dtype=float)
primary_extent = np.asarray(_offline_params["primary_extent"], dtype=float)
current_pos = primary_pose[:3]
current_quat = primary_pose[3:7]
w, x, y, z = current_quat / np.linalg.norm(current_quat)
yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
grasp_pos = current_pos + np.array([
    0.0, 0.0, max(0.004, 0.175 * float(primary_extent[2]))
])
grasp_quat = np.array([
    0.0, -np.sin(0.5 * yaw), np.cos(0.5 * yaw), 0.0
])
approach = max(0.08, 2.0 * float(primary_extent[2]))
lift = max(0.15, 3.0 * float(primary_extent[2]))
open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=approach)
close_gripper()
goto_pose(grasp_pos + np.array([0.0, 0.0, lift]), grasp_quat)

# Code block 3
import numpy as np
current_pos, current_quat, primary_extent = get_object_pose("primary", return_bbox_extent=True)
grasp_pos, grasp_quat = sample_grasp_pose("primary")
approach = max(0.1, 2.0 * float(primary_extent[2]))
lift = max(0.18, 3.0 * float(primary_extent[2]))
home_pose()
open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=approach)
close_gripper()
goto_pose(grasp_pos + np.array([0.0, 0.0, lift]), grasp_quat)

# Code block 4
import json
_offline_params = json.loads('{"primary_pose": [0.632579525852188, 0.2569088023190653, -0.08210775542021675, -0.054200379536245916, -2.8380476364248613e-18, 1.2118142534387128e-17, 0.9985300790953302], "primary_extent": [0.04, 0.04, 0.04]}')

import numpy as np

primary_pose = np.asarray(_offline_params["primary_pose"], dtype=float)
primary_extent = np.asarray(_offline_params["primary_extent"], dtype=float)
current_pos = primary_pose[:3]
approach = max(0.08, 2.0 * float(primary_extent[2]))
lift = max(0.15, 3.0 * float(primary_extent[2]))
baseline_z = float(current_pos[2])
current_quat = primary_pose[3:7]
w, x, y, z = current_quat / np.linalg.norm(current_quat)
yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
topdown_quat = np.array([0.0, -np.sin(0.5 * yaw), np.cos(0.5 * yaw), 0.0])
for attempt in range(2):
    if attempt == 0:
        grasp_pos = current_pos + np.array([
            0.0, 0.0, max(0.004, 0.175 * float(primary_extent[2]))
        ])
        grasp_quat = topdown_quat
    else:
        grasp_pos, grasp_quat = sample_grasp_pose("primary")
    home_pose()
    open_gripper()
    goto_pose(grasp_pos, grasp_quat, z_approach=approach)
    close_gripper()
    goto_pose(grasp_pos + np.array([0.0, 0.0, lift]), grasp_quat)
    target_grasp_pos = grasp_pos
    target_grasp_quat = grasp_quat
    observed_pos, _, _ = get_object_pose("primary", return_bbox_extent=True)
    if float(observed_pos[2]) >= baseline_z + max(0.04, float(primary_extent[2])):
        break

# Code block 5
if "target_grasp_pos" in globals():
    primary_grasp_pos = target_grasp_pos
    primary_grasp_quat = target_grasp_quat
    primary_pos = current_pos
target_center, target_extent, rim_z = get_target_region("target")
target_gripper = target_center + (primary_grasp_pos - primary_pos)
entry = target_gripper.copy()
entry[2] = max(
    entry[2] + approach,
    rim_z + 0.5 * float(primary_extent[2]) + 0.04,
)
goto_pose(entry, primary_grasp_quat)
goto_pose(target_gripper, primary_grasp_quat)
open_gripper()
goto_pose(entry, primary_grasp_quat)
open_gripper()

# Code block 6
import numpy as np
current_pos, current_quat, primary_extent = get_object_pose("primary", return_bbox_extent=True)
grasp_pos, grasp_quat = sample_grasp_pose("primary")
approach = max(0.1, 2.0 * float(primary_extent[2]))
lift = max(0.18, 3.0 * float(primary_extent[2]))
home_pose()
open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=approach)
close_gripper()
goto_pose(grasp_pos + np.array([0.0, 0.0, lift]), grasp_quat)

# Code block 7
target_center, target_extent, rim_z = get_target_region("target")
target_object_center = target_center.copy()
target_object_center[2] += 0.5 * float(primary_extent[2])
target_gripper = target_object_center + (grasp_pos - current_pos)
entry = target_gripper.copy()
entry[2] = max(rim_z + 0.1, target_gripper[2] + 0.1)
goto_pose(entry, grasp_quat)
goto_pose(target_gripper, grasp_quat)
open_gripper()