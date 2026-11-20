from dataclasses import dataclass, field

from capx.recovery import (
    ExecutionOutcome,
    PlannedAction,
    StateWindowAggregator,
    StateWindowConfig,
    ThreeStepRecoveryHarness,
)
from capx.recovery.trial_feedback import (
    TextWorldStateFeedbackConfig,
    TrialWorldStateFeedback,
    append_text_world_state_feedback,
    capture_text_world_state,
    derive_recovery_stage,
    save_initial_state_fingerprint,
)
from capx.recovery.loop_guard import (
    RecoveryLoopGuard,
    RecoveryLoopGuardConfig,
    strategy_fingerprint,
)


def test_state_window_uses_boolean_majority_and_numeric_median():
    aggregator = StateWindowAggregator(StateWindowConfig(window_size=3))

    summary = aggregator.summarize([
        {"target_grasped": True, "alignment_error": 0.011},
        {"target_grasped": False, "alignment_error": 0.050},
        {"target_grasped": True, "alignment_error": 0.010},
    ])

    assert summary.state["target_grasped"] is True
    assert summary.state["alignment_error"] == 0.011
    assert summary.sample_count == 3
    assert "target_grasped" not in summary.unstable_signals
    assert "alignment_error" not in summary.unstable_signals
    assert summary.stable


def test_harness_rejects_one_frame_false_negative_after_stable_grasp():
    action = PlannedAction("grasp", "grasp", expected={"target_grasped": True})
    outcome = ExecutionOutcome(
        state={"target_grasped": True},
        state_window=(
            {"target_grasped": True},
            {"target_grasped": False},
            {"target_grasped": True},
        ),
    )
    harness = ThreeStepRecoveryHarness(lambda _action: outcome)

    result = harness.run([action], initial_state={"target_grasped": False})

    assert result.status == "success"
    assert result.scores[0].hard_pass
    assert result.scores[0].evidence["state_window_size"] == 3
    assert result.scores[0].evidence["state_window_stable"]


def test_missing_signal_across_window_lowers_confidence_and_marks_unstable():
    aggregator = StateWindowAggregator(StateWindowConfig(window_size=3))

    summary = aggregator.summarize([
        {"object_at_target": True},
        {},
        {},
    ])

    assert summary.state["object_at_target"] is True
    assert summary.confidence == 0.3333
    assert not summary.stable
    assert summary.unstable_signals == ("object_at_target",)


def test_numeric_pose_vectors_are_aggregated_coordinate_wise():
    aggregator = StateWindowAggregator(StateWindowConfig(window_size=3))

    summary = aggregator.summarize([
        {"position": [0.100, 0.200, 0.300]},
        {"position": [0.101, 0.199, 0.301]},
        {"position": [0.102, 0.201, 0.299]},
    ])

    assert summary.state["position"] == [0.101, 0.2, 0.3]
    assert summary.stable
    assert summary.confidence == 1.0


class _CustomTextStateEnv:
    def get_text_world_state(self):
        return {"target_grasped": True, "confidence": 0.9}


def test_trial_feedback_supports_custom_text_state_provider():
    config = TextWorldStateFeedbackConfig(enabled=True, window_size=3)
    feedback = TrialWorldStateFeedback(_CustomTextStateEnv(), config)

    summary = feedback.capture()
    prompt = feedback.prompt_text(summary)

    assert summary is not None
    assert summary.state["target_grasped"] is True
    assert '"target_grasped": true' in prompt
    assert "privileged recovery signal" in prompt


@dataclass
class _Pose:
    position: list[float]
    quaternion_wxyz: list[float]


@dataclass
class _Entity:
    entity_id: str
    entity_type: str
    semantic_name: str
    pose: _Pose
    attributes: dict = field(default_factory=dict)


@dataclass
class _Snapshot:
    task_state: dict
    simulator_step: int
    simulator_time: float
    entities: list[_Entity]


class _Provider:
    def capture(self, **_kwargs):
        return _Snapshot(
            task_state={"task_completed": False},
            simulator_step=12,
            simulator_time=0.6,
            entities=[_Entity("cube:a", "rigid_object", "cube", _Pose(
                [0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0]
            ))],
        )


class _ProviderEnv:
    world_state_provider = _Provider()


def test_provider_snapshot_is_converted_to_compact_text_state():
    state = capture_text_world_state(_ProviderEnv())

    assert state["task_completed"] is False
    assert state["entities"]["cube"]["position"] == [0.1, 0.2, 0.3]
    assert state["simulator_step"] == 12


def test_feedback_append_is_noop_when_feature_has_no_state():
    assert append_text_world_state_feedback("base", None) == "base"
    assert append_text_world_state_feedback("base", "state") == "base\n\nstate"


def test_feedback_config_rejects_unknown_options():
    try:
        TextWorldStateFeedbackConfig.from_dict({"enabled": True, "mystery": 1})
    except ValueError as exc:
        assert "mystery" in str(exc)
    else:
        raise AssertionError("unknown option should fail fast")


class _MutableRecoveryEnv:
    def __init__(self):
        self.state = {
            "task_completed": False,
            "target_grasped": False,
            "object_lifted": False,
            "object_at_target": False,
            "object_stable": True,
            "gripper_open": True,
            "gripper_closed": False,
            "distance_to_target": 0.20,
            "cube_poses_robot_base": {
                "primary": [0.5, -0.1, -0.09, 0, 0, 1, 0],
                "secondary": [0.6, 0.1, -0.09, 0, 0, 1, 0],
            },
        }

    def get_text_world_state(self):
        return dict(self.state)

    def _step_once(self):
        return None


def test_loop_guard_allows_two_retries_then_requires_switch_and_aborts():
    env = _MutableRecoveryEnv()
    feedback = TrialWorldStateFeedback(env, TextWorldStateFeedbackConfig(
        enabled=True,
        window_size=1,
        loop_guard_enabled=True,
        max_same_failure_retries=2,
    ))

    feedback.capture()
    assert feedback.capture() and feedback.last_directive.action == "retry"
    assert feedback.capture() and feedback.last_directive.action == "retry"
    summary = feedback.capture()

    assert feedback.last_directive.action == "switch_strategy"
    assert "HARD RECOVERY CONSTRAINT" in feedback.prompt_text(summary)
    repeated = "goto_pose(a, q)\nclose_gripper()\ngoto_pose(b, q)\nopen_gripper()"
    numeric_tweak = "goto_pose(a + 0.02, q)\nclose_gripper()\ngoto_pose(b, q)\nopen_gripper()"
    structural_switch = "home_pose()\ngoto_pose(a, q)\nclose_gripper()\ngoto_pose(b, q)\nopen_gripper()"
    assert not feedback.accepts_regeneration(numeric_tweak, repeated)
    assert feedback.accepts_regeneration(structural_switch, repeated)

    feedback.capture()
    assert feedback.should_abort


def test_loop_guard_resets_failure_count_when_semantic_progress_occurs():
    env = _MutableRecoveryEnv()
    feedback = TrialWorldStateFeedback(env, TextWorldStateFeedbackConfig(
        enabled=True, window_size=1, loop_guard_enabled=True
    ))
    feedback.capture()
    feedback.capture()
    env.state["target_grasped"] = True
    env.state["distance_to_target"] = 0.15

    feedback.capture()

    assert feedback.last_directive.progress_detected
    assert feedback.last_directive.repeated_failures == 0


def test_strategy_fingerprint_ignores_numeric_only_changes():
    first = "goto_pose(a, q, z_approach=0.1)\nclose_gripper()"
    second = "goto_pose(a, q, z_approach=0.2)\nclose_gripper()"
    assert strategy_fingerprint(first) == strategy_fingerprint(second)


def test_restack_clearance_counts_as_semantic_progress():
    guard = RecoveryLoopGuard(RecoveryLoopGuardConfig(enabled=True))
    guard.observe({"restack_precondition_cleared": False})

    directive = guard.observe({"restack_precondition_cleared": True})

    assert directive.progress_detected is True
    assert directive.action == "observe"


def test_initial_state_fingerprint_is_deterministic(tmp_path):
    env = _MutableRecoveryEnv()
    first = save_initial_state_fingerprint(env, tmp_path, 1)
    second = save_initial_state_fingerprint(env, tmp_path, 1)

    assert first is not None
    assert first["fingerprint"] == second["fingerprint"]
    assert (tmp_path / "initial_state_fingerprint_trial_001.json").exists()


def test_action_aligned_window_clears_pre_action_frames_and_collects_three():
    env = _MutableRecoveryEnv()
    feedback = TrialWorldStateFeedback(env, TextWorldStateFeedbackConfig(
        enabled=True,
        window_size=3,
        action_aligned_window=True,
        wait_for_stability=True,
        sample_sim_steps=1,
    ))
    feedback.capture()
    env.state["target_grasped"] = True
    env.state["object_lifted"] = True
    env.state["distance_to_target"] = 0.10

    summary = feedback.capture_action_window()

    assert summary.sample_count == 3
    assert summary.state["target_grasped"] is True
    assert feedback.history[-1]["capture_mode"] == "action_aligned"
    assert feedback.history[-1]["recovery_stage"] == "LIFTED_READY_TO_PLACE"


def test_recovery_stage_requires_release_before_completion():
    assert derive_recovery_stage({
        "object_at_target": True,
        "gripper_open": False,
        "task_completed": False,
    }) == "PLACED_NOT_RELEASED"
