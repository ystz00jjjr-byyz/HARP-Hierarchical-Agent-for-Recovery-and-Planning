# Code block 0


# Code block 1
import numpy as np

open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose('primary')
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
goto_pose(lift_pos, grasp_quat)