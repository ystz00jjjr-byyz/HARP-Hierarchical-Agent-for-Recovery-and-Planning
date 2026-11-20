# Code block 0
import json
_offline_params = json.loads('{"primary_pose": [0.52, 0.29000000000000004, -0.08411679094015156, 0.9659258262890683, 1.8560499882010634e-17, 4.973270954502697e-18, 0.25881904510252074], "primary_extent": [0.04, 0.048, 0.036]}')

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