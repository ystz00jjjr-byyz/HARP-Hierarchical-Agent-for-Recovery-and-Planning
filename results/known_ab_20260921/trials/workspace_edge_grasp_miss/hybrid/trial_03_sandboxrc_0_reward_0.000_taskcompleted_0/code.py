# Code block 0
import json
_offline_params = json.loads('{"primary_pose": [0.6, -0.30000000000000004, -0.08211679094024305, 0.9914448613738104, 1.0310483533190067e-17, -1.357400908478189e-18, -0.13052619222005157], "primary_extent": [0.044, 0.036, 0.04]}')

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
home_pose()
open_gripper()
observed_pos, observed_quat, observed_extent = get_object_pose(
    "primary", return_bbox_extent=True
)

# Code block 2
import json
_offline_params = json.loads('{"primary_pose": [0.5999624276689229, -0.3001261045523441, -0.08210775542021687, 0.9913634508408777, 1.1317949320605587e-17, -6.830388203033043e-18, -0.13114308345034828], "primary_extent": [0.044, 0.036, 0.04]}')

import numpy as np

primary_pose = np.asarray(_offline_params["primary_pose"], dtype=float)
primary_extent = np.asarray(_offline_params["primary_extent"], dtype=float)
current_pos = primary_pose[:3]
inward_y = -np.sign(current_pos[1]) if current_pos[1] != 0.0 else 1.0
push_quat = np.array([0.0, 0.0, 1.0, 0.0])
tool_z = current_pos[2] + max(0.004, 0.175 * float(primary_extent[2]))
push_start = current_pos + np.array([
    0.0,
    -inward_y * (0.5 * float(primary_extent[1]) + 0.025),
    tool_z - current_pos[2],
])
push_end = current_pos + np.array([0.0, inward_y * 0.07, tool_z - current_pos[2]])
home_pose()
close_gripper()
goto_pose(push_start, push_quat, z_approach=0.08)
goto_pose(push_end, push_quat)
home_pose()
open_gripper()

# Code block 3
import json
_offline_params = json.loads('{"primary_pose": [0.5945993952291146, -0.2546515442205479, -0.08410775542021687, 0.6996753912690833, -0.6996753912690812, 0.10224650044112074, -0.10224650044112067], "primary_extent": [0.044, 0.036, 0.04]}')

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

# Code block 4
target_center, target_extent, rim_z = get_target_region("target")
target_object_center = target_center.copy()
target_object_center[2] += 0.5 * float(primary_extent[2])
target_gripper = target_object_center + (grasp_pos - current_pos)
entry = target_gripper.copy()
entry[2] = max(rim_z + 0.08, target_gripper[2] + 0.08)
goto_pose(entry, grasp_quat)
goto_pose(target_gripper, grasp_quat)
open_gripper()