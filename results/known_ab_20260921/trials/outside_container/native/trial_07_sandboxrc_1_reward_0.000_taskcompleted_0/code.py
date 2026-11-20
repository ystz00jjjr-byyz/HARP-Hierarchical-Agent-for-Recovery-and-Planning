# Code block 0
# Stage 1: Grasp and lift the primary object
obj_pos, _, obj_extent = get_object_pose("red block", return_bbox_extent=True)
target_pos, target_extent, _ = get_target_region("target")

grasp_pos, grasp_quat = sample_grasp_pose("red block")

open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.20
goto_pose(lift_pos, grasp_quat)

# Code block 1
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("primary")
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
goto_pose(lift_pos, grasp_quat)