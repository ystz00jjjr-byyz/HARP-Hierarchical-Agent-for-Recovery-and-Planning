# Code block 0
import json
_offline_params = json.loads('{"primary_pose": [0.5119898919505074, -0.2994675755363395, -0.08303964604807756, -0.8912792721860295, -4.2378884323612565e-13, 1.7431323813695317e-13, 0.45345480367015795], "primary_extent": [0.04, 0.04, 0.04]}')

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
_offline_params = json.loads('{"primary_pose": [0.6407759121662783, 0.2570581133990757, -0.08210775542021675, -0.8885293986512073, -2.1694004070568323e-17, 8.29902644584965e-18, 0.4588196897829512], "primary_extent": [0.04, 0.04, 0.04]}')

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
target_center, target_extent, rim_z = get_target_region("target")
target_object_center = target_center.copy()
target_object_center[2] += 0.5 * float(primary_extent[2])
target_gripper = target_object_center + (grasp_pos - current_pos)
entry = target_gripper.copy()
entry[2] = max(rim_z + 0.1, target_gripper[2] + 0.1)
goto_pose(entry, grasp_quat)
goto_pose(target_gripper, grasp_quat)
open_gripper()