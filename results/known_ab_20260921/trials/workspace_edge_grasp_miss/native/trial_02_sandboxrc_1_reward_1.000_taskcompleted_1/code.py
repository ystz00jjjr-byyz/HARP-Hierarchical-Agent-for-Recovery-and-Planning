# Code block 0
# Query live object extent and target region
object_pos, _, object_extent = get_object_pose("red object", return_bbox_extent=True)
target_pos, _, _ = get_target_region()

# Sample grasp pose for the primary object
grasp_pos, grasp_quat = sample_grasp_pose("red object")

# Compute safe transit and gentle placement heights
lift_z = max(object_pos[2] + 0.2, target_pos[2] + 0.2)
lift_pos = np.array([grasp_pos[0], grasp_pos[1], lift_z])

target_above = np.array([target_pos[0], target_pos[1], lift_z])
target_place = np.array([target_pos[0], target_pos[1], target_pos[2] + object_extent[2] / 2.0 + 0.02])

# ==============================================================================
# Stage 1: Grasp and lift the primary object
# ==============================================================================
open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()
goto_pose(lift_pos, grasp_quat)

# Code block 1
# ==============================================================================
# Stage 2: Enter the bin, release gently, and retreat
# ==============================================================================
goto_pose(target_above, grasp_quat)
goto_pose(target_place, grasp_quat)
open_gripper()
goto_pose(target_above, grasp_quat)