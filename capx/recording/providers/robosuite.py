"""Generic MuJoCo/Robosuite state provider."""

from __future__ import annotations

from typing import Any

import numpy as np

from capx.recording.registry import EntityRegistry
from capx.recording.schema import CameraState, EntityState, Pose, WorldStateSnapshot


class RobosuiteWorldStateProvider:
    def __init__(self, env: Any, registry: EntityRegistry,
                 coordinate_frame: str = "mujoco_world") -> None:
        self.env = env
        self.registry = registry
        self.coordinate_frame = coordinate_frame

    def capture(self, *, episode_id: str, sequence_id: int,
                observation: dict[str, Any] | None = None,
                action: Any = None) -> WorldStateSnapshot:
        sim = self.env.robosuite_env.sim
        entities = self._robot_entities()
        entities.extend(self._registered_entities())
        task_state = self.env.get_task_recorder_state(observation)
        task_state["simulator_state"] = {
            "qpos": sim.data.qpos.copy(),
            "qvel": sim.data.qvel.copy(),
        }
        return WorldStateSnapshot(
            schema_version=2,
            episode_id=episode_id,
            sequence_id=sequence_id,
            simulator_step=self.env._sim_step_count,
            simulator_time=float(sim.data.time),
            coordinate_frame=self.coordinate_frame,
            entities=entities,
            cameras=self._cameras(observation),
            action=action,
            task_state=task_state,
        )

    def _robot_entities(self) -> list[EntityState]:
        robots = getattr(self.env.robosuite_env, "robots", None)
        if robots and hasattr(self.env, "gripper_link_idx_0"):
            return [self._indexed_robot_entity(index, robot) for index, robot in enumerate(robots)]
        return [self._robot_entity()]

    def _indexed_robot_entity(self, index: int, robot: Any) -> EntityState:
        sim = self.env.robosuite_env.sim
        link_idx = getattr(self.env, f"gripper_link_idx_{index}")
        joint_indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
        gripper_fraction = getattr(self.env, f"_gripper_fraction_{index}")
        return EntityState(
            entity_id=f"robot:robot{index}", entity_type="robot", semantic_name=f"robot{index}",
            pose=Pose(position=sim.data.xpos[link_idx].copy(),
                      quaternion_wxyz=sim.data.xquat[link_idx].copy()),
            joint_positions=sim.data.qpos[joint_indexes].copy(),
            joint_velocities=sim.data.qvel[joint_indexes].copy(),
            attributes={"gripper_fraction": gripper_fraction,
                        "pose_represents": "end_effector"},
        )

    def _robot_entity(self) -> EntityState:
        sim = self.env.robosuite_env.sim
        return EntityState(
            entity_id="robot:robot0", entity_type="robot", semantic_name="robot0",
            pose=Pose(position=sim.data.xpos[self.env.gripper_link_idx].copy(),
                      quaternion_wxyz=sim.data.xquat[self.env.gripper_link_idx].copy()),
            joint_positions=sim.data.qpos[:7].copy(), joint_velocities=sim.data.qvel[:7].copy(),
            attributes={"gripper_fraction": self.env._gripper_fraction,
                        "pose_represents": "end_effector"},
        )

    def _registered_entities(self) -> list[EntityState]:
        sim = self.env.robosuite_env.sim
        result = []
        for spec in self.registry.all():
            if spec.source_kind != "body":
                continue
            body_id = sim.model.body_name2id(spec.source_name)
            cvel = getattr(sim.data, "cvel", None)
            result.append(EntityState(
                entity_id=spec.entity_id, entity_type=spec.entity_type,
                semantic_name=spec.semantic_name,
                pose=Pose(position=sim.data.xpos[body_id].copy(),
                          quaternion_wxyz=sim.data.xquat[body_id].copy()),
                linear_velocity=None if cvel is None else cvel[body_id][3:].copy(),
                angular_velocity=None if cvel is None else cvel[body_id][:3].copy(),
                attributes={**spec.attributes, "source_name": spec.source_name,
                            "static": spec.static},
            ))
        return result

    def _cameras(self, observation: dict[str, Any] | None) -> list[CameraState]:
        if observation is None:
            return []
        result = []
        for camera_id, value in observation.items():
            if not isinstance(value, dict) or not isinstance(value.get("images"), dict):
                continue
            pose_arr = value.get("pose")
            pose = None if pose_arr is None else Pose(
                position=np.asarray(pose_arr)[:3], quaternion_wxyz=np.asarray(pose_arr)[3:7]
            )
            rgb = value["images"].get("rgb")
            depth = value["images"].get("depth")
            shape = np.asarray(rgb if rgb is not None else depth).shape if (rgb is not None or depth is not None) else ()
            result.append(CameraState(
                camera_id=str(camera_id), pose=pose,
                intrinsics=value.get("intrinsics"),
                width=int(shape[1]) if len(shape) >= 2 else None,
                height=int(shape[0]) if len(shape) >= 2 else None,
                rgb=rgb, depth=depth,
            ))
        return result
