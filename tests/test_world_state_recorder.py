import json

import numpy as np
import pytest
from types import SimpleNamespace

from capx.recording import (
    EntityRegistry,
    WorldStateEventBus,
    WorldStateRecorder,
    WorldStateRecorderConfig,
)
from capx.recording.providers import RobosuiteWorldStateProvider


def test_world_state_recorder_writes_jsonl_and_rgbd_sidecar(tmp_path):
    cfg = WorldStateRecorderConfig.from_dict({
        "enabled": True,
        "record_every_n_sim_steps": 3,
    })
    recorder = WorldStateRecorder(tmp_path / "trajectory", cfg)
    obs = {
        "robot0_robotview": {
            "intrinsics": np.eye(3),
            "pose": np.arange(7),
            "images": {
                "rgb": np.zeros((2, 3, 3), dtype=np.uint8),
                "depth": np.ones((2, 3, 1), dtype=np.float32),
            },
        },
        "cube_poses": {"primary": np.arange(7)},
    }
    recorder.record("reset", sim_step=0, sim_time=1.25,
                    state={"qpos": np.arange(4)}, observation=obs)
    recorder.close()

    manifest = json.loads((tmp_path / "trajectory" / "manifest.json").read_text())
    entry = json.loads((tmp_path / "trajectory" / "trajectory.jsonl").read_text())
    arrays = np.load(tmp_path / "trajectory" / entry["array_file"])

    assert manifest["privileged_side_channel"] is True
    assert manifest["agent_visible"] is False
    assert entry["event"] == "reset"
    assert entry["state"]["qpos"] == [0, 1, 2, 3]
    assert set(arrays.files) == {
        "robot0_robotview__rgb",
        "robot0_robotview__depth",
    }
    assert recorder.should_record_sim_step(6)
    assert not recorder.should_record_sim_step(7)


def test_world_state_recorder_rejects_invalid_config():
    with pytest.raises(ValueError, match="must be >= 1"):
        WorldStateRecorderConfig.from_dict({"record_every_n_sim_steps": 0})
    with pytest.raises(ValueError, match="Unknown"):
        WorldStateRecorderConfig.from_dict({"unexpected": True})


def test_robosuite_provider_emits_backend_neutral_entities_and_camera():
    registry = EntityRegistry()
    registry.register_body(entity_id="object:test", semantic_name="test object",
                           body_name="object_body")

    model = SimpleNamespace(body_name2id=lambda name: {"object_body": 1}[name])
    data = SimpleNamespace(
        time=2.5,
        xpos=np.array([[0, 0, 0], [1, 2, 3]], dtype=float),
        xquat=np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=float),
        qpos=np.arange(4, dtype=float),
        qvel=np.arange(4, dtype=float) / 10,
    )
    env = SimpleNamespace(
        robosuite_env=SimpleNamespace(sim=SimpleNamespace(model=model, data=data)),
        _sim_step_count=9,
        gripper_link_idx=0,
        _gripper_fraction=0.5,
        get_task_recorder_state=lambda obs: {"phase": "test"},
    )
    provider = RobosuiteWorldStateProvider(env, registry)
    observation = {"camera": {"pose": np.array([0, 0, 1, 1, 0, 0, 0]),
                               "intrinsics": np.eye(3),
                               "images": {"rgb": np.zeros((4, 5, 3), dtype=np.uint8),
                                          "depth": np.ones((4, 5, 1), dtype=np.float32)}}}
    snapshot = provider.capture(episode_id="trial_001", sequence_id=3,
                                observation=observation, action=np.array([1, 2]))

    assert snapshot.schema_version == 2
    assert snapshot.coordinate_frame == "mujoco_world"
    assert [entity.entity_id for entity in snapshot.entities] == [
        "robot:robot0", "object:test"
    ]
    assert snapshot.entities[1].pose.position.tolist() == [1.0, 2.0, 3.0]
    assert snapshot.cameras[0].width == 5
    assert snapshot.cameras[0].height == 4
    assert snapshot.task_state["phase"] == "test"
    assert snapshot.task_state["simulator_state"]["qpos"].tolist() == [0, 1, 2, 3]


def test_robosuite_provider_emits_two_robot_entities():
    data = SimpleNamespace(time=1.0, xpos=np.array([[1, 0, 0], [2, 0, 0.]]), xquat=np.array([[1, 0, 0, 0], [1, 0, 0, 0.]]), qpos=np.arange(16.), qvel=np.arange(16.) / 10)
    robots = [SimpleNamespace(_ref_joint_pos_indexes=np.arange(7)), SimpleNamespace(_ref_joint_pos_indexes=np.arange(8, 15))]
    env = SimpleNamespace(robosuite_env=SimpleNamespace(sim=SimpleNamespace(model=SimpleNamespace(), data=data), robots=robots), _sim_step_count=4, gripper_link_idx_0=0, gripper_link_idx_1=1, _gripper_fraction_0=0.25, _gripper_fraction_1=0.75, get_task_recorder_state=lambda obs: {})
    snapshot = RobosuiteWorldStateProvider(env, EntityRegistry()).capture(episode_id="trial_001", sequence_id=0)
    assert [entity.entity_id for entity in snapshot.entities] == ["robot:robot0", "robot:robot1"]
    assert snapshot.entities[1].joint_positions.tolist() == list(range(8, 15))
    assert snapshot.entities[1].attributes["gripper_fraction"] == 0.75


def test_event_bus_decouples_publishers_from_subscribers():
    received = []
    bus = WorldStateEventBus()
    bus.subscribe(received.append)
    bus.emit("control.gripper.close", {"arm": "robot0"})
    assert received[0].event_type == "control.gripper.close"
    assert received[0].payload == {"arm": "robot0"}
