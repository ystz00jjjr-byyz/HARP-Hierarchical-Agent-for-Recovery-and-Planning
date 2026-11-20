from capx.recovery import (
    ExecutionOutcome,
    PlannedAction,
    PrimitiveStepScorer,
    ThreeStepRecoveryHarness,
    ThreeStepWindowPlanner,
    build_recovery_prompt,
)


def test_three_step_window_slides_and_preserves_order():
    actions = [PlannedAction(str(i), "move_to") for i in range(4)]
    planner = ThreeStepWindowPlanner()
    planner.load(actions)

    assert [a.action_id for a in planner.current_window()] == ["0", "1", "2"]
    planner.advance()
    assert [a.action_id for a in planner.current_window()] == ["1", "2", "3"]
    planner.advance()
    planner.advance()
    assert [a.action_id for a in planner.current_window()] == ["3"]


def test_pick_place_text_world_state_runs_in_order():
    actions = [
        PlannedAction("align", "move_to", expected={"eef_aligned": True}),
        PlannedAction("grasp", "grasp", expected={"target_grasped": True}),
        PlannedAction("place", "place", expected={"object_at_target": True}),
        PlannedAction("release", "release", expected={"gripper_open": True}),
    ]
    states = iter([
        {"eef_aligned": True},
        {"target_grasped": True},
        {"object_at_target": True},
        {"gripper_open": True},
    ])
    events = []
    harness = ThreeStepRecoveryHarness(lambda _action: next(states), event_sink=events.append)

    result = harness.run(actions)

    assert result.status == "success"
    assert result.executed_actions == ("align", "grasp", "place", "release")
    windows = [e for e in events if e.event_type == "planning.window_ready"]
    assert [e.window_size for e in windows] == [3, 3, 2, 1]


def test_failed_alignment_can_retry_once():
    action = PlannedAction("align", "move_to", expected={"eef_aligned": True})
    states = iter([{"eef_aligned": False}, {"eef_aligned": True}])
    decisions = []

    def policy(_action, validation, _state):
        decisions.append(validation.ok)
        return "retry"

    harness = ThreeStepRecoveryHarness(lambda _action: next(states),
                                       recovery_policy=policy, max_retries=1)
    result = harness.run([action])

    assert result.status == "success"
    assert decisions == [False]


def test_recovery_prompt_contains_text_world_state_and_window():
    action = PlannedAction("grasp", "grasp", args={"object": "cube"})
    next_action = PlannedAction("place", "place")
    harness = ThreeStepRecoveryHarness(lambda _action: {"target_grasped": False})
    validation = harness.validator.validate(action, {"target_grasped": False})
    prompt = build_recovery_prompt(
        task="put cube in box",
        current_action=action,
        upcoming_actions=(action, next_action),
        state={"target_grasped": False, "robot": {"eef_aligned": False}},
        validation=validation,
    )

    assert "target_grasped: False" in prompt
    assert "robot:" in prompt
    assert "place" in prompt
    assert "retry, replan, abort" in prompt


def test_move_step_score_uses_pre_post_progress():
    action = PlannedAction(
        "align",
        "move_to",
        expected={"alignment_error": {"lte": 0.02}},
    )
    harness = ThreeStepRecoveryHarness(lambda _action: {"alignment_error": 0.01})
    validation = harness.validator.validate(action, {"alignment_error": 0.01})
    score = PrimitiveStepScorer().score(
        action,
        pre_state={"alignment_error": 0.08},
        outcome=ExecutionOutcome(state={"alignment_error": 0.01}),
        validation=validation,
    )

    assert score.hard_pass
    assert score.progress_score == 0.875
    assert score.total >= 85
    assert score.decision == "pass"


def test_failed_grasp_routes_to_retry_when_environment_is_recoverable():
    action = PlannedAction("grasp", "grasp", expected={"target_grasped": True})
    state = {"target_grasped": False, "recoverable_environment": True}
    harness = ThreeStepRecoveryHarness(lambda _action: state)
    validation = harness.validator.validate(action, state)
    score = PrimitiveStepScorer().score(
        action,
        pre_state={"target_grasped": False},
        outcome=ExecutionOutcome(state=state),
        validation=validation,
    )

    assert not score.hard_pass
    assert score.total <= 69
    assert score.decision == "retry"


def test_safety_violation_overrides_successful_post_condition():
    action = PlannedAction("align", "move_to", expected={"eef_aligned": True})
    state = {"eef_aligned": True, "collision": True}
    harness = ThreeStepRecoveryHarness(lambda _action: state)
    validation = harness.validator.validate(action, state)
    score = PrimitiveStepScorer().score(
        action,
        pre_state={"eef_aligned": False},
        outcome=ExecutionOutcome(state=state),
        validation=validation,
    )

    assert validation.ok
    assert not score.hard_pass
    assert score.total == 0
    assert score.decision == "abort"


def test_harness_emits_score_and_carries_it_in_result():
    action = PlannedAction(
        "align",
        "move_to",
        expected={"alignment_error": {"lte": 0.02}},
    )
    harness = ThreeStepRecoveryHarness(lambda _action: {"alignment_error": 0.01})

    result = harness.run([action], initial_state={"alignment_error": 0.08})

    assert result.status == "success"
    assert len(result.scores) == 1
    assert result.scores[0].decision == "pass"
    scoring_events = [event for event in result.events if event.event_type == "scoring.result"]
    assert scoring_events[0].payload["total"] == result.scores[0].total


def test_default_score_router_retries_recoverable_grasp():
    action = PlannedAction("grasp", "grasp", expected={"target_grasped": True})
    states = iter([
        {"target_grasped": False, "recoverable_environment": True},
        {"target_grasped": True, "recoverable_environment": True},
    ])
    harness = ThreeStepRecoveryHarness(lambda _action: next(states), max_retries=1)

    result = harness.run([action], initial_state={"target_grasped": False})

    assert result.status == "success"
    assert [score.decision for score in result.scores] == ["retry", "pass"]
    retry_events = [event for event in result.events if event.event_type == "recovery.retry"]
    assert retry_events[0].payload["attempt"] == 1
