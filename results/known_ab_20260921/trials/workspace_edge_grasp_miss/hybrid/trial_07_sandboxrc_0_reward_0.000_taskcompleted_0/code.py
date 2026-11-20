# Code block 0
import json
_offline_params = json.loads('{"primary_pose": [0.5700000000000001, -0.32000000000000006, -0.08011679094026625, 0.9238795325112867, 1.6818771063498547e-17, 4.329471000538139e-18, 0.3826834323650898], "primary_extent": [0.05, 0.036, 0.044]}')

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