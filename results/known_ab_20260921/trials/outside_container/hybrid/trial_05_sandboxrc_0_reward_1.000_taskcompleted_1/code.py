# Code block 0
import json
_offline_params = json.loads('{"primary_pose": [0.49385292574893636, -0.3004583741998226, -0.08303964604807756, -0.9999966503253187, -4.4744133712250214e-13, 1.0854485071282075e-15, 0.0025883079689745115], "primary_extent": [0.04, 0.04, 0.04]}')

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
_offline_params = json.loads('{"primary_pose": [0.49381303179647656, -0.3007324333049508, -0.08210775542021687, -0.9999972301740534, -1.9639235002336026e-17, -2.378483681147151e-18, 0.00235364487975868], "primary_extent": [0.04, 0.04, 0.04]}')

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
_offline_params = json.loads('{"primary_pose": [0.48090109747710386, -0.2579080420395585, -0.08210775542021675, -0.7060881241252023, 0.7060881241252033, 0.03794154674432516, -0.037941546744325115], "primary_extent": [0.04, 0.04, 0.04]}')

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

# Code block 5
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

# Code block 6
target_center, target_extent, rim_z = get_target_region("target")
target_object_center = target_center.copy()
target_object_center[2] += 0.5 * float(primary_extent[2])
target_gripper = target_object_center + (grasp_pos - current_pos)
entry = target_gripper.copy()
entry[2] = max(rim_z + 0.1, target_gripper[2] + 0.1)
goto_pose(entry, grasp_quat)
goto_pose(target_gripper, grasp_quat)
open_gripper()