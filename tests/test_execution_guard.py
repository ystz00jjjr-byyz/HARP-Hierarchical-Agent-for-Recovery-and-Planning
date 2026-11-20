import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from capx.recovery.execution_guard import (
    ExecutionGuardConfig,
    GuardResult,
    assign_stage_indices,
    task_completion_confirmed,
    validate_code_block,
    validate_decision,
    validate_stage,
)
from capx.recovery.stability import StateWindowSummary
from capx.envs import trial as trial_module
from capx.envs.launch import LaunchArgs
from capx.utils.launch_utils import _load_config


def _config(**overrides):
    return ExecutionGuardConfig.from_dict({"enabled": True, **overrides})


def test_l2_advisory_accepts_only_phase_whitelisted_json():
    accepted = trial_module._parse_l2_advisory(
        '{"decision":"CONTINUE","next_template":'
        '"bin_place_nominal_place_v1","confidence":0.8}',
        "stage",
    )

    assert accepted["decision"] == "CONTINUE"
    with pytest.raises(ValueError, match="fields must be exactly"):
        trial_module._parse_l2_advisory(
            '{"decision":"CONTINUE","next_template":'
            '"bin_place_nominal_place_v1","confidence":0.8,'
            '"code":"goto_pose(...)"}',
            "stage",
        )


def test_l2_advisory_reprompts_once_for_invalid_schema(monkeypatch):
    responses = iter([
        {
            "content": '{"decision":"FINISH","confidence":0.5}',
            "reasoning": None,
            "model_response_seconds": 1.0,
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 2,
                            "reasoning_tokens": 0, "total_tokens": 12},
        },
        {
            "content": '{"decision":"ACCEPT_COMPLETION","confidence":0.9}',
            "reasoning": None,
            "model_response_seconds": 2.0,
            "token_usage": {"prompt_tokens": 12, "completion_tokens": 2,
                            "reasoning_tokens": 0, "total_tokens": 14},
        },
    ])
    monkeypatch.setattr(trial_module, "_query_model", lambda *_: next(responses))

    selection, response, _ = trial_module._query_l2_advisory(
        SimpleNamespace(), phase="final", task_description="place object",
        world_state_feedback='{"task_completed":true}',
    )

    assert selection["decision"] == "ACCEPT_COMPLETION"
    assert response["model_request_count"] == 2
    assert response["token_usage"]["total_tokens"] == 26


def test_l2_final_advisory_uses_environment_authority_after_two_invalid_replies(
    monkeypatch,
):
    responses = iter([
        {"content": "FINISH", "token_usage": None},
        {"content": None, "token_usage": None},
    ])
    monkeypatch.setattr(trial_module, "_query_model", lambda *_: next(responses))

    selection, response, _ = trial_module._query_l2_advisory(
        SimpleNamespace(), phase="final", task_description="place object",
        world_state_feedback='{"task_completed":true}',
    )

    assert selection["decision"] == "ACCEPT_COMPLETION"
    assert response["model_request_count"] == 2
    assert response["l2_schema_fallback"] is True


def test_structured_l3_recovery_bypasses_generic_code_channel(monkeypatch):
    captured = []
    raw_selection = (
        '{"template":"bin_place_regrasp_replace_v1",'
        '"grasp_mode":"sample_grasp_pose","approach_clearance":0.1,'
        '"lift_clearance":0.15,"entry_clearance":0.1,'
        '"pregrasp_home":true}'
    )

    def query(_args, prompt):
        captured.append(prompt)
        return {
            "content": raw_selection,
            "reasoning": "bounded selection",
            "model_response_seconds": 1.25,
            "token_usage": {"total_tokens": 10},
        }

    monkeypatch.setattr(trial_module, "_query_model", query)
    config = {
        "use_wrist_camera": False,
        "use_visual_feedback": False,
        "use_img_differencing": False,
        "use_video_differencing": False,
        "use_parallel_ensemble": False,
        "candidate_reranking": {},
    }
    decision, selection, _, accounting, prompt, tokens = (
        trial_module._handle_multi_turn_step(
            SimpleNamespace(),
            {"full_prompt": []},
            SimpleNamespace(
                model="test-model", use_legacy_multi_turn_decision_prompt=False
            ),
            config,
            SimpleNamespace(model="test-model"),
            "GENERIC_REGENERATE_CHANNEL {executed_code}",
            ["old_code()"],
            1,
            {"stdout": "ok", "stderr": "failed stage"},
            "place primary",
            [],
            [],
            [],
            text_world_state_feedback='{"sample_count":3}',
            structured_recovery_schema=raw_selection,
            structured_recovery_context="GRASP_MISS: target_grasped=false",
        )
    )

    assert decision == "regenerate"
    assert selection == raw_selection
    assert accounting["model_response_seconds"] == 1.25
    assert tokens["total_tokens"] == 10
    assert prompt == captured[0]
    assert "GENERIC_REGENERATE_CHANNEL" not in repr(prompt)
    assert "Return exactly one raw JSON or Python dictionary" in repr(prompt)


def test_preflight_rejects_truncated_python_before_execution():
    result = validate_code_block(
        "target = np.array([\n    1.0,\n", _config()
    )

    assert not result.ok
    assert result.gate == "code_preflight"
    assert "invalid Python syntax" in result.reason


def test_preflight_accepts_short_atomic_robot_stage():
    code = """
pose, quat = sample_grasp_pose("green cube")
open_gripper()
goto_pose(pose, quat, z_approach=0.08)
close_gripper()
goto_pose(pose + [0, 0, 0.1], quat)
"""

    result = validate_code_block(code, _config())

    assert result.ok
    assert len(result.details["robot_action_calls"]) == 4


def test_preflight_rejects_monolithic_robot_stage():
    code = "\n".join("goto_pose(p, q)" for _ in range(9))

    result = validate_code_block(code, _config(max_robot_api_calls_per_block=8))

    assert not result.ok
    assert "too long for atomic verification" in result.reason


def test_preflight_rejects_expensive_api_in_unbounded_loop():
    code = "while not done:\n    goto_pose(p, q)"

    result = validate_code_block(code, _config())

    assert not result.ok
    assert "unbounded loop" in result.reason


def test_stage_gate_uses_world_state_postconditions():
    config = _config(stage_expectations=[{
        "name": "lift_target",
        "required": {"target_grasped": True, "height": {"gte": 0.1}},
    }])

    passed = validate_stage(
        0, {"target_grasped": True, "height": 0.12}, config
    )
    failed = validate_stage(
        0, {"target_grasped": False, "height": 0.12}, config
    )

    assert passed.ok
    assert not failed.ok
    assert "target_grasped" in failed.reason


def test_continue_is_rejected_when_stage_postcondition_failed():
    result = validate_decision(
        "continue",
        task_completed=False,
        stage_result=GuardResult(False, "stage_postcondition", "target not grasped"),
        config=_config(),
    )

    assert not result.ok
    assert result.gate == "stage_decision"


def test_finish_requires_authoritative_environment_completion():
    stage_passed = GuardResult(True, "stage_postcondition", "stage passed")

    rejected = validate_decision(
        "finish", task_completed=False, stage_result=stage_passed, config=_config()
    )
    accepted = validate_decision(
        "finish", task_completed=True, stage_result=stage_passed, config=_config()
    )

    assert not rejected.ok
    assert rejected.gate == "finish_completion"
    assert accepted.ok


def test_completion_gate_accepts_environment_or_stable_three_state_consensus():
    stable_completion = StateWindowSummary(
        state={"task_completed": True},
        confidence=1.0,
        stable=True,
        unstable_signals=(),
        sample_count=3,
    )

    assert task_completion_confirmed(task_completed=True)
    assert task_completion_confirmed(
        task_completed=False, state_summary=stable_completion
    )
    assert not task_completion_confirmed(
        task_completed=False,
        state_summary=StateWindowSummary(
            state={"task_completed": True},
            confidence=1.0,
            stable=True,
            unstable_signals=(),
            sample_count=2,
        ),
    )


def test_regeneration_remains_available_after_stage_failure():
    result = validate_decision(
        "regenerate",
        task_completed=False,
        stage_result=GuardResult(False, "stage_postcondition", "failed"),
        config=_config(),
    )

    assert result.ok


def test_regenerated_blocks_keep_recovery_stage_then_advance():
    metadata = assign_stage_indices(
        3, start=1, generation=2, regenerated=True, regenerated_at_idx=4
    )

    assert [item["stage_index"] for item in metadata] == [1, 2, 3]
    assert all(item["generation"] == 2 for item in metadata)
    assert all(item["regenerated_at_idx"] == 4 for item in metadata)


def test_execution_guard_config_rejects_unknown_options():
    with pytest.raises(ValueError, match="mystery"):
        ExecutionGuardConfig.from_dict({"enabled": True, "mystery": 1})


class _FakeTrialEnv:
    def __init__(self, oracle_code="x = 1"):
        self.oracle_code = oracle_code
        self.low_level_env = self
        self.executed = []
        self.completed = False
        self.prepared = False

    def reset(self, **_kwargs):
        self.completed = False
        self.prepared = False
        return ({"full_prompt": [{"role": "user", "content": [{"type": "text", "text": "task"}]}]}, {})

    def step(self, code):
        self.executed.append(code)
        if "recover()" in code:
            self.prepared = True
        if "complete()" in code:
            self.completed = True
        return (
            {"full_prompt": [{"role": "user", "content": [{"type": "text", "text": "task"}]}]},
            1.0 if self.completed else 0.0,
            self.completed,
            False,
            {"sandbox_rc": 0, "stdout": "", "stderr": "", "task_completed": self.completed},
        )

    def compute_reward(self):
        return 1.0 if self.completed else 0.0

    def task_completed(self):
        return self.completed

    def get_text_world_state(self):
        return {
            "prepared": self.prepared,
            "object_stable": True,
            "task_completed": self.completed,
        }

    def set_world_state_recorder(self, _recorder):
        return None

    def record_world_state_event(self, *_args, **_kwargs):
        return None


def _trial_config(tmp_path):
    return {
        "output_dir": str(tmp_path),
        "record_video": False,
        "use_video_differencing": False,
        "use_wrist_camera": False,
        "use_img_differencing": False,
        "use_visual_feedback": False,
        "use_oracle_code": True,
        "use_parallel_ensemble": False,
        "use_multimodel": False,
        "save_multiturn_prompts": True,
        "world_state_recorder": {"enabled": False},
        "text_world_state_feedback": {"enabled": False},
        "candidate_reranking": {"enabled": False},
        "execution_guard": {"enabled": True, "max_decision_reprompts": 2},
    }


def _trial_args():
    return SimpleNamespace(
        model="fake",
        visual_differencing_model="fake",
        visual_differencing_model_server_url="http://unused",
        visual_differencing_model_api_key=None,
        max_tokens=100,
        temperature=0.0,
        reasoning_effort="minimal",
        debug=False,
        use_legacy_multi_turn_decision_prompt=False,
    )


def test_trial_reprompts_premature_finish_without_executing_extra_code(
    tmp_path, monkeypatch
):
    env = _FakeTrialEnv()
    decisions = iter([
        ("finish", None, "looks done", None, [], {"total_tokens": 1}),
        ("regenerate", "```python\ncomplete()\n```", "recover", None, [], {"total_tokens": 1}),
        ("finish", None, "verified", None, [], {"total_tokens": 1}),
    ])
    monkeypatch.setattr(trial_module, "_handle_multi_turn_step", lambda *_a, **_k: next(decisions))

    summary = trial_module._run_single_trial(
        env, 1, _trial_args(), _trial_config(tmp_path), "{executed_code} {console_stdout} {console_stderr}"
    )

    assert env.executed == ["x = 1", "complete()"]
    assert summary.task_completed is True
    responses = json.loads((Path(summary.code_path).parent / "all_responses.json").read_text())
    assert [item["decision"] for item in responses].count("finish_rejected") == 1


def test_trial_stops_before_multiturn_query_after_environment_completion(
    tmp_path, monkeypatch
):
    env = _FakeTrialEnv(oracle_code="complete()")

    def fail_if_queried(*_args, **_kwargs):
        pytest.fail("multi-turn model must not be queried after deterministic completion")

    monkeypatch.setattr(trial_module, "_handle_multi_turn_step", fail_if_queried)

    summary = trial_module._run_single_trial(
        env,
        1,
        _trial_args(),
        _trial_config(tmp_path),
        "{executed_code} {console_stdout} {console_stderr}",
    )

    assert env.executed == ["complete()"]
    assert summary.task_completed is True


def test_trial_static_gate_prevents_invalid_code_from_reaching_environment(
    tmp_path, monkeypatch
):
    env = _FakeTrialEnv(oracle_code="target = [")
    decisions = iter([
        ("regenerate", "```python\ncomplete()\n```", "repair", None, [], {"total_tokens": 1}),
        ("finish", None, "verified", None, [], {"total_tokens": 1}),
    ])
    monkeypatch.setattr(trial_module, "_handle_multi_turn_step", lambda *_a, **_k: next(decisions))

    summary = trial_module._run_single_trial(
        env, 1, _trial_args(), _trial_config(tmp_path), "{executed_code} {console_stdout} {console_stderr}"
    )

    assert env.executed == ["complete()"]
    assert summary.task_completed is True
    responses = json.loads((Path(summary.code_path).parent / "all_responses.json").read_text())
    assert any(item["decision"] == "code_preflight_rejected" for item in responses)


def test_hybrid_online_contract_hit_miss_and_policy_failure(tmp_path, monkeypatch):
    graph = {
        "initial": ["stage_001"],
        "actions": [
            {
                "id": "stage_001",
                "code": "x = 1",
                "failure_routes": {"POSTCONDITION_FAILED": "PREPARE_FAILED"},
            },
            {
                "id": "recover",
                "code": "recover()",
                "success_conditions": {"prepared": True},
                "next": "finish",
            },
            {
                "id": "finish",
                "code": "complete()",
                "success_conditions": {"task_completed": True},
            },
            {
                "id": "bad_recover",
                "code": "x = 2",
                "success_conditions": {"prepared": True},
            },
        ],
    }

    def config(output_dir, library):
        Path(output_dir).mkdir(parents=True)
        value = _trial_config(output_dir)
        value["use_oracle_code"] = False
        value["text_world_state_feedback"] = {
            "enabled": True,
            "action_aligned_window": True,
            "wait_for_stability": False,
        }
        value["execution_guard"]["stage_expectations"] = [
            {"name": "prepared", "required": {"prepared": True}},
            {"name": "finished", "required": {"task_completed": True}},
        ]
        value["hybrid_recovery"] = {
            "enabled": True,
            "task_graph": graph,
            "recovery_library": library,
            "stage_actions": ["stage_001"],
        }
        return value

    initial_model_calls = []
    recovery_model_calls = []

    def initial_model(*_args, **_kwargs):
        initial_model_calls.append("called")
        return "x = 1", "model plan", None, {"total_tokens": 7}

    def model_fallback(*_args, **_kwargs):
        recovery_model_calls.append("called")
        return (
            "regenerate", "```python\ncomplete()\n```", "fallback", None, [],
            {"total_tokens": 11},
        )

    monkeypatch.setattr(trial_module, "_query_initial_code", initial_model)
    monkeypatch.setattr(trial_module, "_handle_multi_turn_step", model_fallback)

    hit_env = _FakeTrialEnv()
    hit = trial_module._run_single_trial(
        hit_env,
        1,
        _trial_args(),
        config(tmp_path / "hit", {"PREPARE_FAILED": ["recover"]}),
        "{executed_code} {console_stdout} {console_stderr}",
    )
    assert hit.task_completed is True
    assert hit_env.executed == ["x = 1", "recover()", "complete()"]
    assert initial_model_calls == ["called"]
    assert recovery_model_calls == []

    miss_env = _FakeTrialEnv()
    miss = trial_module._run_single_trial(
        miss_env,
        2,
        _trial_args(),
        config(tmp_path / "miss", {}),
        "{executed_code} {console_stdout} {console_stderr}",
    )
    assert miss.task_completed is True
    assert miss_env.executed == ["x = 1", "complete()"]
    assert recovery_model_calls == ["called"]

    failed_env = _FakeTrialEnv()
    failed = trial_module._run_single_trial(
        failed_env,
        3,
        _trial_args(),
        config(tmp_path / "failed", {"PREPARE_FAILED": ["bad_recover"]}),
        "{executed_code} {console_stdout} {console_stderr}",
    )
    assert failed.task_completed is True
    assert failed_env.executed == ["x = 1", "x = 2", "complete()"]
    assert initial_model_calls == ["called", "called", "called"]
    assert recovery_model_calls == ["called", "called"]

    retry_env = _FakeTrialEnv()
    retry = trial_module._run_single_trial(
        retry_env,
        4,
        _trial_args(),
        config(
            tmp_path / "retry",
            {"PREPARE_FAILED": ["bad_recover", "recover", "finish"]},
        ),
        "{executed_code} {console_stdout} {console_stderr}",
    )
    assert retry.task_completed is True
    assert retry_env.executed == ["x = 1", "x = 2", "recover()", "complete()"]
    assert recovery_model_calls == ["called", "called"]

    responses = json.loads(
        (Path(hit.code_path).parent / "all_responses.json").read_text()
    )
    assert responses[0]["plan_source"] == "model"
    assert responses[0]["model_request_count"] == 1
    assert any(item["decision"] == "policy_recovery" for item in responses)
    assert sum(item.get("model_request_count", 0) for item in responses) == 1

    miss_responses = json.loads(
        (Path(miss.code_path).parent / "all_responses.json").read_text()
    )
    assert any(
        item["decision"] == "policy_miss" and item["fallback_to_model"]
        for item in miss_responses
    )

    failed_responses = json.loads(
        (Path(failed.code_path).parent / "all_responses.json").read_text()
    )
    assert any(
        item["decision"] == "policy_failed" and item["fallback_to_model"]
        for item in failed_responses
    )

    retry_responses = json.loads(
        (Path(retry.code_path).parent / "all_responses.json").read_text()
    )
    assert any(item["decision"] == "policy_retry" for item in retry_responses)
    assert sum(item.get("model_request_count", 0) for item in retry_responses) == 1


def test_model_led_stage_uses_task_graph_postcondition_before_policy_recovery(
    tmp_path, monkeypatch
):
    config = _trial_config(tmp_path)
    config["use_oracle_code"] = False
    config["text_world_state_feedback"] = {
        "enabled": True,
        "action_aligned_window": True,
        "wait_for_stability": False,
    }
    config["hybrid_recovery"] = {
        "enabled": True,
        "policy_first": False,
        "model_led_policy_recovery": True,
        "task_graph": {
            "initial": ["stage_001"],
            "actions": [
                {
                    "id": "stage_001",
                    "code": "x = 1",
                    "success_conditions": {"prepared": True},
                    "failure_routes": {
                        "POSTCONDITION_FAILED": "PREPARE_FAILED"
                    },
                },
                {
                    "id": "recover",
                    "code": "recover()",
                    "success_conditions": {"prepared": True},
                    "next": "finish",
                },
                {
                    "id": "finish",
                    "code": "complete()",
                    "success_conditions": {"task_completed": True},
                },
            ],
        },
        "recovery_library": {"PREPARE_FAILED": ["recover"]},
        "stage_actions": ["stage_001"],
    }
    monkeypatch.setattr(
        trial_module,
        "_query_initial_code",
        lambda *_a, **_k: ("x = 1", "model plan", None, {"total_tokens": 7}),
    )

    def unexpected_model_fallback(*_args, **_kwargs):
        pytest.fail("known failure must use policy recovery before another model call")

    monkeypatch.setattr(
        trial_module, "_handle_multi_turn_step", unexpected_model_fallback
    )
    env = _FakeTrialEnv()
    summary = trial_module._run_single_trial(
        env,
        1,
        _trial_args(),
        config,
        "{executed_code} {console_stdout} {console_stderr}",
    )

    assert summary.task_completed is True
    assert env.executed == ["x = 1", "recover()", "complete()"]
    responses = json.loads(
        (Path(summary.code_path).parent / "all_responses.json").read_text()
    )
    assert any(item["decision"] == "policy_recovery" for item in responses)
    assert sum(item.get("model_request_count", 0) for item in responses) == 1


def test_model_led_replan_past_last_stage_keeps_terminal_monitor():
    metadata = assign_stage_indices(3, start=2, generation=1, regenerated=True)

    trial_module._attach_hybrid_stage_monitors(
        metadata, ("stage_001", "stage_002")
    )

    assert [item["hybrid_monitor_action_id"] for item in metadata] == [
        "stage_002",
        "stage_002",
        "stage_002",
    ]


def test_policy_first_completes_without_any_model_request(tmp_path, monkeypatch):
    env = _FakeTrialEnv()
    config = _trial_config(tmp_path)
    config["use_oracle_code"] = False
    config["text_world_state_feedback"] = {
        "enabled": True,
        "action_aligned_window": True,
        "wait_for_stability": False,
    }
    config["execution_guard"]["stage_expectations"] = [
        {"name": "prepared", "required": {"prepared": True}},
        {"name": "finished", "required": {"task_completed": True}},
    ]
    config["hybrid_recovery"] = {
        "enabled": True,
        "policy_first": True,
        "policy_preconditions": {"object_stable": True},
        "task_graph": {
            "initial": ["stage_001"],
            "actions": [
                {
                    "id": "stage_001",
                    "code": "recover()",
                    "success_conditions": {"prepared": True},
                    "next": "stage_002",
                },
                {
                    "id": "stage_002",
                    "code": "complete()",
                    "preconditions": {"prepared": True},
                    "success_conditions": {"task_completed": True},
                },
            ],
        },
        "recovery_library": {},
        "stage_actions": ["stage_001", "stage_002"],
    }

    monkeypatch.setattr(
        trial_module,
        "_query_initial_code",
        lambda *_a, **_k: pytest.fail("initial model must not be queried"),
    )
    monkeypatch.setattr(
        trial_module,
        "_handle_multi_turn_step",
        lambda *_a, **_k: pytest.fail("recovery model must not be queried"),
    )

    summary = trial_module._run_single_trial(
        env, 1, _trial_args(), config,
        "{executed_code} {console_stdout} {console_stderr}",
    )

    assert summary.task_completed is True
    assert env.executed == ["recover()", "complete()"]
    responses = json.loads((Path(summary.code_path).parent / "all_responses.json").read_text())
    assert responses[0]["plan_source"] == "offline_policy"
    assert sum(item.get("model_request_count", 0) for item in responses) == 0


def test_three_action_window_continues_fourth_link_after_fresh_state_checks(
    tmp_path, monkeypatch
):
    env = _FakeTrialEnv()
    config = _trial_config(tmp_path)
    config["use_oracle_code"] = False
    config["text_world_state_feedback"] = {
        "enabled": True,
        "action_aligned_window": True,
        "wait_for_stability": False,
    }
    config["execution_guard"]["stage_expectations"] = [
        {"name": "nominal", "required": {"prepared": True}},
    ]
    config["hybrid_recovery"] = {
        "enabled": True,
        "policy_first": True,
        "task_graph": {
            "initial": ["stage_001"],
            "actions": [
                {
                    "id": "stage_001",
                    "code": "x = 1",
                    "success_conditions": {"prepared": True},
                    "failure_routes": {"POSTCONDITION_FAILED": "EDGE_MISS"},
                },
                {
                    "id": "observe",
                    "code": "x = 2",
                    "success_conditions": {"object_stable": True},
                    "next": "recenter",
                    "metadata": {"interrupt_on_failure": True},
                },
                {
                    "id": "recenter",
                    "code": "x = 3",
                    "success_conditions": {"object_stable": True},
                    "next": "regrasp",
                    "metadata": {"interrupt_on_failure": True},
                },
                {
                    "id": "regrasp",
                    "code": "recover()",
                    "success_conditions": {"prepared": True},
                    "next": "replace",
                    "metadata": {"interrupt_on_failure": True},
                },
                {
                    "id": "replace",
                    "code": "complete()",
                    "success_conditions": {"task_completed": True},
                },
            ],
        },
        "recovery_library": {"EDGE_MISS": ["observe"]},
        "stage_actions": ["stage_001"],
    }

    monkeypatch.setattr(
        trial_module,
        "_handle_multi_turn_step",
        lambda *_a, **_k: pytest.fail("model must not be queried"),
    )

    summary = trial_module._run_single_trial(
        env,
        1,
        _trial_args(),
        config,
        "{executed_code} {console_stdout} {console_stderr}",
    )

    assert summary.task_completed is True
    assert env.executed == ["x = 1", "x = 2", "x = 3", "recover()", "complete()"]
    responses = json.loads(
        (Path(summary.code_path).parent / "all_responses.json").read_text()
    )
    assert any(item["decision"] == "policy_continue" for item in responses)
    assert sum(item.get("model_request_count", 0) for item in responses) == 0


def test_staged_model_recovery_rechecks_state_then_auto_continues(
    tmp_path, monkeypatch
):
    env = _FakeTrialEnv()
    config = _trial_config(tmp_path)
    config["use_oracle_code"] = False
    config["text_world_state_feedback"] = {
        "enabled": True,
        "action_aligned_window": True,
        "wait_for_stability": False,
    }
    config["execution_guard"]["stage_expectations"] = [
        {"name": "regrasped", "required": {"prepared": True}},
        {"name": "replaced", "required": {"task_completed": True}},
    ]
    config["hybrid_recovery"] = {
        "enabled": True,
        "policy_first": True,
        "model_recovery_stage_indices": [0, 1],
        "task_graph": {
            "initial": ["stage_001"],
            "actions": [{
                "id": "stage_001",
                "code": "x = 1",
                "failure_routes": {"POSTCONDITION_FAILED": "UNKNOWN_FAILURE"},
            }],
        },
        "recovery_library": {},
        "stage_actions": ["stage_001"],
    }
    model_calls = []

    def staged_fallback(*_args, **_kwargs):
        model_calls.append("called")
        return (
            "regenerate",
            "```python\nrecover()\nbreakpoint_code_block()\ncomplete()\n```",
            "two-stage recovery",
            None,
            [],
            {"total_tokens": 1},
        )

    monkeypatch.setattr(trial_module, "_handle_multi_turn_step", staged_fallback)

    summary = trial_module._run_single_trial(
        env,
        1,
        _trial_args(),
        config,
        "{executed_code} {console_stdout} {console_stderr}",
    )

    assert summary.task_completed is True
    assert env.executed == ["x = 1", "recover()", "complete()"]
    assert model_calls == ["called"]
    responses = json.loads(
        (Path(summary.code_path).parent / "all_responses.json").read_text()
    )
    assert any(item["decision"] == "model_recovery_continue" for item in responses)


def test_model_recovery_template_compiles_bounded_selector_to_two_stages():
    blocks, selection = trial_module._compile_bin_place_recovery_template(
        '{"template": "bin_place_regrasp_replace_v1", '
        '"grasp_mode": "sample_grasp_pose", "approach_clearance": 0.1, '
        '"lift_clearance": 0.15, "entry_clearance": 0.08, '
        '"pregrasp_home": True}'
    )

    assert selection["grasp_mode"] == "sample_grasp_pose"
    assert len(blocks) == 2
    assert 'sample_grasp_pose("primary")' in blocks[0]
    assert "abs(float(current_pos[1])) >= 0.27" in blocks[0]
    assert "np.copysign(0.01, current_pos[1])" in blocks[0]
    assert "home_pose()" in blocks[0]
    assert "get_target_region" not in blocks[0]
    assert blocks[1].rstrip().endswith("open_gripper()")
    assert "lift_pos = grasp_pos +" in blocks[0]
    assert "transit = 0.5 * (lift_pos + entry)" in blocks[1]
    assert "target_object_center[2]" not in blocks[1]
    assert "goto_pose" not in repr(selection)

    role_blocks, _ = trial_module._compile_bin_place_recovery_template(
        '{"template": "bin_place_regrasp_replace_v1", '
        '"grasp_mode": "sample_grasp_pose", "approach_clearance": 0.1, '
        '"lift_clearance": 0.15, "entry_clearance": 0.08, '
        '"pregrasp_home": True}',
        object_role="secondary",
        target_region="target_secondary",
    )
    assert 'get_object_pose("secondary"' in role_blocks[0]
    assert 'sample_grasp_pose("secondary")' in role_blocks[0]
    assert 'get_target_region("target_secondary")' in role_blocks[1]

    boundary_blocks, boundary_selection = (
        trial_module._compile_bin_place_recovery_template(
            '{"template": "bin_place_regrasp_replace_v1", '
            '"grasp_mode": "boundary_yaw_topdown", '
            '"approach_clearance": 0.1, "lift_clearance": 0.15, '
            '"entry_clearance": 0.08, "pregrasp_home": true}'
        )
    )
    assert boundary_selection["grasp_mode"] == "boundary_yaw_topdown"
    assert "yaw += 0.5 * np.pi" in boundary_blocks[0]
    assert "np.copysign(0.01, current_pos[1])" in boundary_blocks[0]
    assert 'sample_grasp_pose("primary")' in boundary_blocks[0]

    with pytest.raises(ValueError, match="fields must be exactly"):
        trial_module._compile_bin_place_recovery_template(
            '{"template": "bin_place_regrasp_replace_v1", '
            '"grasp_mode": "sample_grasp_pose", "approach_clearance": 0.1, '
            '"lift_clearance": 0.15, "entry_clearance": 0.08, '
            '"pregrasp_home": True, "code": "home_pose()"}'
        )

    with pytest.raises(ValueError, match="must not use markdown"):
        trial_module._compile_bin_place_recovery_template(
            "```python\n{'template': 'bin_place_regrasp_replace_v1'}\n```"
        )
    with pytest.raises(ValueError, match="not valid JSON"):
        trial_module._compile_bin_place_recovery_template(
            "goto_pose(target, quat)\nopen_gripper()"
        )

    with pytest.raises(ValueError, match="not valid JSON"):
        trial_module._compile_bin_place_recovery_template(
            'Use this selection: {"template":"bin_place_regrasp_replace_v1"}'
        )
    with pytest.raises(ValueError, match="fields must be exactly"):
        trial_module._compile_bin_place_recovery_template(
            '{"template":"bin_place_regrasp_replace_v1"}'
        )
    with pytest.raises(ValueError, match="approach_clearance must be one of"):
        trial_module._compile_bin_place_recovery_template(
            '{"template":"bin_place_regrasp_replace_v1",'
            '"grasp_mode":"sample_grasp_pose","approach_clearance":0.5,'
            '"lift_clearance":0.15,"entry_clearance":0.08,'
            '"pregrasp_home":true}'
        )


def test_l4_recovery_graph_compiles_whitelisted_skills_to_two_stages():
    graph = {
        "version": "recovery_graph_v1",
        "steps": [
            {"skill": "safe_home"},
            {"skill": "reobserve_object"},
            {"skill": "sample_grasp", "mode": "orthogonal_cuboid"},
            {
                "skill": "regrasp",
                "approach_clearance": 0.1,
                "lift_clearance": 0.15,
            },
            {"skill": "reobserve_target"},
            {"skill": "replace", "entry_clearance": 0.08},
            {"skill": "verify"},
        ],
    }

    blocks, normalized = trial_module._compile_model_recovery_selection(
        "recovery_graph_v1", graph, 1,
    )

    assert len(blocks) == 2
    assert normalized["version"] == "recovery_graph_v1"
    assert normalized["compiled_template"]["pregrasp_home"] is True
    assert 'sample_cuboid_grasp_pose("primary", candidate_index=1)' in blocks[0]
    assert "home_pose()" in blocks[0]
    assert 'get_target_region("target")' in blocks[1]


def test_l4_recovery_graph_rejects_code_and_reordered_skills():
    graph = {
        "version": "recovery_graph_v1",
        "steps": [
            {"skill": "reobserve_object", "code": "home_pose()"},
            {"skill": "sample_grasp", "mode": "sample_grasp_pose"},
            {
                "skill": "regrasp",
                "approach_clearance": 0.1,
                "lift_clearance": 0.15,
            },
            {"skill": "replace", "entry_clearance": 0.08},
            {"skill": "reobserve_target"},
            {"skill": "verify"},
        ],
    }

    with pytest.raises(ValueError, match="skills must be"):
        trial_module._compile_recovery_graph_v1(graph)


def test_l4_config_uses_model_plan_policy_recovery_and_graph_fallback(monkeypatch):
    monkeypatch.setenv("REGION_PLACE_OUTPUT_DIR", "outputs/test-region-l4")
    _, config, _ = _load_config(LaunchArgs(
        config_path="env_configs/region_place/l4_model_led_recovery_graph.yaml"
    ))
    hybrid = config["hybrid_recovery"]

    assert hybrid["policy_first"] is False
    assert hybrid["model_led_policy_recovery"] is True
    assert hybrid["model_initial_plan_template"] == "recovery_graph_v1"
    assert hybrid["model_recovery_template"] == "recovery_graph_v1"
    assert hybrid["model_recovery_candidate_limit"] == 2
    assert hybrid["max_model_fallbacks_total"] == 1


def test_l4_initial_graph_is_compiled_before_robot_execution(tmp_path, monkeypatch):
    class L4Env(_FakeTrialEnv):
        def step(self, code):
            self.executed.append(code)
            if "close_gripper()" in code:
                self.prepared = True
            if "get_target_region" in code:
                self.completed = True
            return (
                {"full_prompt": [{"role": "user", "content": []}]},
                1.0 if self.completed else 0.0,
                self.completed,
                False,
                {
                    "sandbox_rc": 0,
                    "stdout": "",
                    "stderr": "",
                    "task_completed": self.completed,
                },
            )

    graph = {
        "version": "recovery_graph_v1",
        "steps": [
            {"skill": "reobserve_object"},
            {"skill": "sample_grasp", "mode": "sample_grasp_pose"},
            {
                "skill": "regrasp",
                "approach_clearance": 0.1,
                "lift_clearance": 0.15,
            },
            {"skill": "reobserve_target"},
            {"skill": "replace", "entry_clearance": 0.1},
            {"skill": "verify"},
        ],
    }
    config = _trial_config(tmp_path)
    config["use_oracle_code"] = False
    config["text_world_state_feedback"] = {
        "enabled": True,
        "action_aligned_window": True,
        "wait_for_stability": False,
    }
    config["execution_guard"]["stage_expectations"] = [
        {"name": "regrasped", "required": {"prepared": True}},
        {"name": "replaced", "required": {"task_completed": True}},
    ]
    config["hybrid_recovery"] = {
        "enabled": True,
        "policy_first": False,
        "model_led_policy_recovery": True,
        "model_initial_plan_template": "recovery_graph_v1",
        "model_recovery_stage_indices": [0, 1],
        "model_recovery_template": "recovery_graph_v1",
        "task_graph": {
            "initial": ["stage_001"],
            "actions": [
                {"id": "stage_001", "code": "x = 1", "success_conditions": {"prepared": True}},
                {"id": "stage_002", "code": "x = 2", "success_conditions": {"task_completed": True}},
            ],
        },
        "recovery_library": {},
        "stage_actions": ["stage_001", "stage_002"],
    }
    monkeypatch.setattr(
        trial_module,
        "_query_initial_code",
        lambda *_a, **_k: (json.dumps(graph), "structured plan", None, {"total_tokens": 10}),
    )

    env = L4Env()
    summary = trial_module._run_single_trial(
        env, 1, _trial_args(), config,
        "{executed_code} {console_stdout} {console_stderr}",
    )

    assert summary.task_completed is True
    assert len(env.executed) == 2
    assert all("recovery_graph_v1" not in code for code in env.executed)
    responses = json.loads(
        (Path(summary.code_path).parent / "all_responses.json").read_text()
    )
    assert responses[0]["model_initial_plan"]["version"] == "recovery_graph_v1"
    assert responses[0]["model_request_count"] == 1


def test_model_recovery_role_stays_bound_to_failed_policy_action():
    state = {"active_object_role": "primary"}
    assert trial_module._recovery_role_for_failed_action(
        "candidate_replace_secondary", state
    ) == "secondary"
    assert trial_module._recovery_role_for_failed_action(None, state) == "primary"


def test_model_recovery_binding_keeps_role_specific_target():
    state = {
        "active_object_role": "primary",
        "target_regions": {
            "secondary": {"target_name": "target_secondary"},
        },
    }

    assert trial_module._recovery_binding("secondary", state) == (
        "secondary",
        "target_secondary",
    )


def test_model_recovery_uses_role_specific_stage_monitors():
    graph = SimpleNamespace(
        actions={
            "candidate_grasp_secondary": object(),
            "candidate_replace_secondary": object(),
        }
    )

    assert trial_module._recovery_monitor_actions(
        graph, "secondary", ("stage_001",)
    ) == ("candidate_grasp_secondary", "candidate_replace_secondary")
    assert trial_module._recovery_monitor_actions(
        graph, "quaternary", ("stage_001",)
    ) == ("stage_001",)


def test_model_recovery_compiles_one_selection_to_two_bounded_candidates():
    blocks, selection = trial_module._compile_bin_place_recovery_candidates(
        '{"template": "bin_place_regrasp_replace_v1", '
        '"grasp_mode": "yaw_topdown", "approach_clearance": 0.08, '
        '"lift_clearance": 0.15, "entry_clearance": 0.08, '
        '"pregrasp_home": False}',
        2,
    )

    assert len(blocks) == 4
    assert selection["candidate_count"] == 2
    assert selection["derived_second_candidate"]["grasp_mode"] == "sample_grasp_pose"
    assert selection["derived_second_candidate"]["pregrasp_home"] is True
    assert 'sample_grasp_pose("primary")' in blocks[2]


def test_model_recovery_second_candidate_is_orthogonal_cuboid_grasp():
    blocks, selection = trial_module._compile_bin_place_recovery_candidates(
        '{"template": "bin_place_regrasp_replace_v1", '
        '"grasp_mode": "sample_grasp_pose", "approach_clearance": 0.08, '
        '"lift_clearance": 0.15, "entry_clearance": 0.08, '
        '"pregrasp_home": false}',
        2,
        object_role="secondary",
        target_region="target_secondary",
    )

    assert selection["derived_second_candidate"]["grasp_mode"] == "orthogonal_cuboid"
    assert 'sample_cuboid_grasp_pose("secondary", candidate_index=1)' in blocks[2]


def test_model_recovery_grasp_mode_adapts_to_live_workspace_position():
    selector = lambda mode: (
        '{"template": "bin_place_regrasp_replace_v1", '
        f'"grasp_mode": "{mode}", "approach_clearance": 0.1, '
        '"lift_clearance": 0.15, "entry_clearance": 0.08, '
        '"pregrasp_home": True}'
    )

    def run(mode, y):
        block, _ = trial_module._compile_bin_place_recovery_template(
            selector(mode)
        )
        sample_calls = []
        poses = []

        def sample_grasp_pose(_name):
            sample_calls.append(_name)
            return (
                trial_module.np.array([0.5, y, -0.07]),
                trial_module.np.array([0.0, 0.0, 1.0, 0.0]),
            )

        exec(block[0], {
            "get_object_pose": lambda *_a, **_k: (
                trial_module.np.array([0.5, y, -0.08]),
                trial_module.np.array([1.0, 0.0, 0.0, 0.0]),
                trial_module.np.array([0.04, 0.04, 0.04]),
            ),
            "sample_grasp_pose": sample_grasp_pose,
            "goto_pose": lambda position, quaternion, **_k: poses.append(position.copy()),
            "home_pose": lambda: None,
            "open_gripper": lambda: None,
            "close_gripper": lambda: None,
        })
        return sample_calls, poses

    boundary_samples, boundary_poses = run("sample_grasp_pose", -0.304)
    inside_samples, _ = run("boundary_yaw_topdown", -0.132)

    assert boundary_samples == []
    assert boundary_poses[0][1] == pytest.approx(-0.294)
    assert inside_samples == ["primary"]


def test_staged_model_recovery_executes_compiled_template_not_model_code(
    tmp_path, monkeypatch
):
    class TemplateEnv(_FakeTrialEnv):
        def step(self, code):
            self.executed.append(code)
            if "close_gripper()" in code:
                self.prepared = True
            if "get_target_region" in code:
                self.completed = True
            return (
                {"full_prompt": [{"role": "user", "content": []}]},
                1.0 if self.completed else 0.0,
                self.completed,
                False,
                {
                    "sandbox_rc": 0,
                    "stdout": "",
                    "stderr": "",
                    "task_completed": self.completed,
                },
            )

    env = TemplateEnv()
    config = _trial_config(tmp_path)
    config["text_world_state_feedback"] = {
        "enabled": True,
        "action_aligned_window": True,
        "wait_for_stability": False,
    }
    config["execution_guard"]["stage_expectations"] = [
        {"name": "regrasped", "required": {"prepared": True}},
        {"name": "replaced", "required": {"task_completed": True}},
    ]
    config["hybrid_recovery"] = {
        "enabled": True,
        "model_recovery_stage_indices": [0, 1],
        "model_recovery_template": "bin_place_regrasp_replace_v1",
        "task_graph": {
            "initial": ["stage_001"],
            "actions": [{
                "id": "stage_001",
                "code": "x = 1",
                "failure_routes": {"POSTCONDITION_FAILED": "UNKNOWN_FAILURE"},
            }],
        },
        "recovery_library": {},
        "stage_actions": ["stage_001"],
    }
    selector = (
        '{"template": "bin_place_regrasp_replace_v1", '
        '"grasp_mode": "sample_grasp_pose", "approach_clearance": 0.1, '
        '"lift_clearance": 0.15, "entry_clearance": 0.08, '
        '"pregrasp_home": True}'
    )
    attached_monitors = []
    original_attach = trial_module._attach_hybrid_stage_monitors

    def record_attached_monitors(metadata, stage_actions):
        original_attach(metadata, stage_actions)
        attached_monitors.append([
            item.get("hybrid_monitor_action_id") for item in metadata
        ])

    monkeypatch.setattr(
        trial_module, "_attach_hybrid_stage_monitors", record_attached_monitors
    )
    monkeypatch.setattr(
        trial_module,
        "_handle_multi_turn_step",
        lambda *_a, **_k: (
            "regenerate", selector, "select trusted template", None, [],
            {"total_tokens": 1},
        ),
    )

    summary = trial_module._run_single_trial(
        env,
        1,
        _trial_args(),
        config,
        "{executed_code} {console_stdout} {console_stderr}",
    )

    assert summary.task_completed is True
    assert len(env.executed) == 3
    assert "bin_place_regrasp_replace_v1" not in "\n".join(env.executed)
    assert "close_gripper()" in env.executed[1]
    assert env.executed[2].rstrip().endswith("open_gripper()")
    assert attached_monitors == []
    responses = json.loads(
        (Path(summary.code_path).parent / "all_responses.json").read_text()
    )
    regenerated = next(item for item in responses if item["decision"] == "regenerate")
    assert regenerated["model_recovery_raw_selection"] == selector
    assert regenerated["model_recovery_template"]["grasp_mode"] == "sample_grasp_pose"


def test_staged_model_recovery_rechecks_k3_before_bounded_second_candidate(
    tmp_path, monkeypatch
):
    class TwoCandidateEnv(_FakeTrialEnv):
        def __init__(self):
            super().__init__()
            self.replace_calls = 0

        def reset(self, **kwargs):
            self.replace_calls = 0
            return super().reset(**kwargs)

        def step(self, code):
            self.executed.append(code)
            if "close_gripper()" in code:
                self.prepared = True
            if "get_target_region" in code:
                self.replace_calls += 1
                self.completed = self.replace_calls == 2
            return (
                {"full_prompt": [{"role": "user", "content": []}]},
                1.0 if self.completed else 0.0,
                self.completed,
                False,
                {
                    "sandbox_rc": 0,
                    "stdout": "",
                    "stderr": "",
                    "task_completed": self.completed,
                },
            )

    config = _trial_config(tmp_path)
    config["text_world_state_feedback"] = {
        "enabled": True,
        "action_aligned_window": True,
        "wait_for_stability": False,
    }
    config["execution_guard"]["stage_expectations"] = [
        {"name": "regrasped", "required": {"prepared": True}},
        {"name": "replaced", "required": {"task_completed": True}},
    ]
    config["hybrid_recovery"] = {
        "enabled": True,
        "max_model_fallbacks_per_exception": 1,
        "max_model_fallbacks_total": 1,
        "model_recovery_stage_indices": [0, 1],
        "model_recovery_template": "bin_place_regrasp_replace_v1",
        "model_recovery_candidate_limit": 2,
        "task_graph": {
            "initial": ["stage_001"],
            "actions": [
                {
                    "id": "stage_001",
                    "code": "x = 1",
                    "success_conditions": {"task_completed": True},
                    "failure_routes": {"POSTCONDITION_FAILED": "UNKNOWN_FAILURE"},
                },
                {"id": "safe_stop", "code": "open_gripper()"},
            ],
        },
        "recovery_library": {"RECOVERY_BUDGET_LOW": ["safe_stop"]},
        "stage_actions": ["stage_001"],
    }
    selector = (
        '{"template": "bin_place_regrasp_replace_v1", '
        '"grasp_mode": "yaw_topdown", "approach_clearance": 0.08, '
        '"lift_clearance": 0.15, "entry_clearance": 0.08, '
        '"pregrasp_home": False}'
    )
    model_calls = []

    def fallback(*_args, **_kwargs):
        model_calls.append("called")
        return "regenerate", selector, "select", None, [], {"total_tokens": 1}

    monkeypatch.setattr(trial_module, "_handle_multi_turn_step", fallback)
    env = TwoCandidateEnv()
    summary = trial_module._run_single_trial(
        env,
        1,
        _trial_args(),
        config,
        "{executed_code} {console_stdout} {console_stderr}",
    )

    assert summary.task_completed is True
    assert env.replace_calls == 2
    assert model_calls == ["called"]
    responses = json.loads(
        (Path(summary.code_path).parent / "all_responses.json").read_text()
    )
    assert any(
        item["decision"] == "model_recovery_candidate_retry"
        for item in responses
    )
    assert sum(item.get("model_request_count", 0) for item in responses) == 1


def test_split_policy_interrupts_before_replace_then_compiles_model_template(
    tmp_path, monkeypatch
):
    class TemplateEnv(_FakeTrialEnv):
        def step(self, code):
            self.executed.append(code)
            if "close_gripper()" in code:
                self.prepared = True
            if "get_target_region" in code:
                self.completed = True
            return (
                {"full_prompt": [{"role": "user", "content": []}]},
                1.0 if self.completed else 0.0,
                self.completed,
                False,
                {
                    "sandbox_rc": 0,
                    "stdout": "",
                    "stderr": "",
                    "task_completed": self.completed,
                },
            )

    config = _trial_config(tmp_path)
    config["text_world_state_feedback"] = {
        "enabled": True,
        "action_aligned_window": True,
        "wait_for_stability": False,
    }
    config["execution_guard"]["stage_expectations"] = [
        {"name": "regrasped", "required": {"prepared": True}},
        {"name": "replaced", "required": {"task_completed": True}},
    ]
    config["hybrid_recovery"] = {
        "enabled": True,
        "model_recovery_stage_indices": [0, 1],
        "model_recovery_template": "bin_place_regrasp_replace_v1",
        "task_graph": {
            "initial": ["stage_001"],
            "actions": [
                {
                    "id": "stage_001",
                    "code": "x = 1",
                    "failure_routes": {"POSTCONDITION_FAILED": "PREPARE_FAILED"},
                },
                {
                    "id": "regrasp",
                    "code": "x = 2",
                    "success_conditions": {"prepared": True},
                    "next": "replace",
                    "metadata": {"interrupt_on_failure": True},
                },
                {
                    "id": "replace",
                    "code": "complete()",
                    "preconditions": {"prepared": True},
                    "success_conditions": {"task_completed": True},
                },
            ],
        },
        "recovery_library": {"PREPARE_FAILED": ["regrasp"]},
        "stage_actions": ["stage_001"],
    }
    selector = (
        '{"template": "bin_place_regrasp_replace_v1", '
        '"grasp_mode": "sample_grasp_pose", "approach_clearance": 0.1, '
        '"lift_clearance": 0.15, "entry_clearance": 0.08, '
        '"pregrasp_home": True}'
    )
    monkeypatch.setattr(
        trial_module,
        "_handle_multi_turn_step",
        lambda *_a, **_k: (
            "regenerate", selector, "select trusted template", None, [],
            {"total_tokens": 1},
        ),
    )

    env = TemplateEnv()
    summary = trial_module._run_single_trial(
        env,
        1,
        _trial_args(),
        config,
        "{executed_code} {console_stdout} {console_stderr}",
    )

    assert summary.task_completed is True
    assert env.executed[0:2] == ["x = 1", "x = 2"]
    assert "complete()" not in env.executed
    assert "close_gripper()" in env.executed[2]
    assert env.executed[3].rstrip().endswith("open_gripper()")
    responses = json.loads(
        (Path(summary.code_path).parent / "all_responses.json").read_text()
    )
    assert any(item["decision"] == "policy_failed" for item in responses)
    regenerated = next(item for item in responses if item["decision"] == "regenerate")
    assert regenerated["model_recovery_template"] is not None


@pytest.mark.parametrize("fallback_mode", ["l2", "l3"])
def test_structural_candidate_failure_falls_back_once_to_configured_model_mode(
    tmp_path, monkeypatch, fallback_mode
):
    class FallbackEnv(_FakeTrialEnv):
        def step(self, code):
            self.executed.append(code)
            if "close_gripper()" in code:
                self.prepared = True
            if "complete()" in code or "get_target_region" in code:
                self.completed = True
            return (
                {"full_prompt": [{"role": "user", "content": []}]},
                1.0 if self.completed else 0.0,
                self.completed,
                False,
                {
                    "sandbox_rc": 0,
                    "stdout": "",
                    "stderr": "",
                    "task_completed": self.completed,
                },
            )

    config = _trial_config(tmp_path)
    config["text_world_state_feedback"] = {
        "enabled": True,
        "action_aligned_window": True,
        "wait_for_stability": False,
        "loop_guard_enabled": False,
    }
    config["hybrid_recovery"] = {
        "enabled": True,
        "policy_first": True,
        "max_model_fallbacks_per_exception": 1,
        "max_model_fallbacks_total": 1,
        "task_graph": {
            "initial": ["stage_001"],
            "actions": [
                {
                    "id": "stage_001",
                    "code": "x = 1",
                    "success_conditions": {"prepared": True},
                    "failure_routes": {
                        "POSTCONDITION_FAILED": "STRUCTURAL_RECOVERY_REQUIRED"
                    },
                },
                {
                    "id": "candidate_structurally_distinct_regrasp_replace",
                    "code": "candidate()",
                    "success_conditions": {"prepared": True},
                    "metadata": {"interrupt_on_failure": True},
                },
                {"id": "safe_stop", "code": "safe_stop()"},
            ],
        },
        "recovery_library": {
            "STRUCTURAL_RECOVERY_REQUIRED": [
                "candidate_structurally_distinct_regrasp_replace"
            ],
            "RECOVERY_BUDGET_LOW": ["safe_stop"],
        },
        "stage_actions": ["stage_001"],
    }
    if fallback_mode == "l2":
        config["hybrid_recovery"].update({
            "model_recovery_stage_indices": [0, 1],
            "model_recovery_template": "bin_place_regrasp_replace_v1",
        })
        model_code = (
            '{"template": "bin_place_regrasp_replace_v1", '
            '"grasp_mode": "sample_grasp_pose", "approach_clearance": 0.1, '
            '"lift_clearance": 0.15, "entry_clearance": 0.08, '
            '"pregrasp_home": True}'
        )
    else:
        model_code = "```python\ncomplete()\n```"

    model_calls = []

    def fallback(*_args, **_kwargs):
        model_calls.append(fallback_mode)
        return "regenerate", model_code, "recover", None, [], {"total_tokens": 1}

    monkeypatch.setattr(trial_module, "_handle_multi_turn_step", fallback)
    summary = trial_module._run_single_trial(
        FallbackEnv(), 1, _trial_args(), config,
        "{executed_code} {console_stdout} {console_stderr}",
    )

    assert summary.task_completed is True
    assert model_calls == [fallback_mode]
    responses = json.loads(
        (Path(summary.code_path).parent / "all_responses.json").read_text()
    )
    assert any(
        item["decision"] == "policy_failed"
        and item["offline_action_id"]
        == "candidate_structurally_distinct_regrasp_replace"
        for item in responses
    )
    assert sum(item.get("model_request_count", 0) for item in responses) == 1


def test_staged_model_recovery_preflights_both_blocks_before_execution(
    tmp_path, monkeypatch
):
    env = _FakeTrialEnv()
    config = _trial_config(tmp_path)
    config["text_world_state_feedback"] = {
        "enabled": True,
        "action_aligned_window": True,
        "wait_for_stability": False,
    }
    config["execution_guard"]["stage_expectations"] = [
        {"name": "regrasped", "required": {"prepared": True}},
        {"name": "replaced", "required": {"task_completed": True}},
    ]
    config["hybrid_recovery"] = {
        "enabled": True,
        "max_model_fallbacks_per_exception": 2,
        "model_recovery_stage_indices": [0, 1],
        "task_graph": {
            "initial": ["stage_001"],
            "actions": [{
                "id": "stage_001",
                "code": "x = 1",
                "failure_routes": {"POSTCONDITION_FAILED": "UNKNOWN_FAILURE"},
            }, {
                "id": "safe_stop",
                "code": "safe_stop()",
            }],
        },
        "recovery_library": {"RECOVERY_BUDGET_LOW": ["safe_stop"]},
        "stage_actions": ["stage_001"],
    }
    decisions = iter([
        (
            "regenerate",
            "```python\nrecover()\nbreakpoint_code_block()\nif :\n```",
            "invalid second block",
            None,
            [],
            {"total_tokens": 1},
        ),
        (
            "regenerate",
            "```python\nrecover()\nbreakpoint_code_block()\ncomplete()\n```",
            "valid pair",
            None,
            [],
            {"total_tokens": 1},
        ),
    ])
    monkeypatch.setattr(
        trial_module, "_handle_multi_turn_step", lambda *_a, **_k: next(decisions)
    )

    summary = trial_module._run_single_trial(
        env,
        1,
        _trial_args(),
        config,
        "{executed_code} {console_stdout} {console_stderr}",
    )

    assert summary.task_completed is True
    assert env.executed == ["x = 1", "recover()", "complete()"]
    responses = json.loads(
        (Path(summary.code_path).parent / "all_responses.json").read_text()
    )
    assert sum(item.get("model_request_count", 0) for item in responses) == 2
    assert any(
        item["decision"] == "model_recovery_preflight_rejected"
        for item in responses
    )


def test_staged_model_recovery_safe_stops_before_low_horizon_regrasp(
    tmp_path, monkeypatch
):
    env = _FakeTrialEnv()
    env.max_steps = 1000
    env._sim_step_count = 800
    config = _trial_config(tmp_path)
    config["text_world_state_feedback"] = {
        "enabled": True,
        "action_aligned_window": True,
        "wait_for_stability": False,
    }
    config["execution_guard"]["stage_expectations"] = [
        {"name": "regrasped", "required": {"prepared": True}},
        {"name": "replaced", "required": {"task_completed": True}},
    ]
    config["hybrid_recovery"] = {
        "enabled": True,
        "model_recovery_stage_indices": [0, 1],
        "min_remaining_sim_steps_before_regrasp": 320,
        "task_graph": {
            "initial": ["stage_001"],
            "actions": [{
                "id": "stage_001",
                "code": "x = 1",
                "failure_routes": {"POSTCONDITION_FAILED": "UNKNOWN_FAILURE"},
            }],
        },
        "recovery_library": {},
        "stage_actions": ["stage_001"],
    }
    monkeypatch.setattr(
        trial_module,
        "_handle_multi_turn_step",
        lambda *_a, **_k: (
            "regenerate",
            "```python\nrecover()\nbreakpoint_code_block()\ncomplete()\n```",
            "two-stage recovery",
            None,
            [],
            {"total_tokens": 1},
        ),
    )

    summary = trial_module._run_single_trial(
        env,
        1,
        _trial_args(),
        config,
        "{executed_code} {console_stdout} {console_stderr}",
    )

    assert summary.task_completed is False
    assert summary.truncated is True
    assert env.executed == ["x = 1"]
    responses = json.loads(
        (Path(summary.code_path).parent / "all_responses.json").read_text()
    )
    assert any(
        item["decision"] == "regrasp_horizon_safe_stop"
        and item["remaining_sim_steps"] == 200
        for item in responses
    )


def test_hybrid_budget_counts_decision_reprompts(tmp_path, monkeypatch):
    env = _FakeTrialEnv()
    config = _trial_config(tmp_path)
    config["text_world_state_feedback"] = {
        "enabled": True,
        "action_aligned_window": True,
        "wait_for_stability": False,
    }
    config["execution_guard"]["stage_expectations"] = [
        {"name": "prepared", "required": {"prepared": True}},
    ]
    config["hybrid_recovery"] = {
        "enabled": True,
        "max_model_fallbacks_per_exception": 2,
        "task_graph": {
            "initial": ["stage_001"],
            "actions": [{
                "id": "stage_001",
                "code": "x = 1",
                "failure_routes": {"POSTCONDITION_FAILED": "UNKNOWN_FAILURE"},
            }, {
                "id": "safe_stop",
                "code": "safe_stop()",
            }],
        },
        "recovery_library": {"RECOVERY_BUDGET_LOW": ["safe_stop"]},
        "stage_actions": ["stage_001"],
    }
    model_calls = []

    def premature_finish(*_args, **_kwargs):
        model_calls.append("called")
        return "finish", None, "not complete", None, [], {"total_tokens": 1}

    monkeypatch.setattr(trial_module, "_handle_multi_turn_step", premature_finish)

    summary = trial_module._run_single_trial(
        env,
        1,
        _trial_args(),
        config,
        "{executed_code} {console_stdout} {console_stderr}",
    )

    assert model_calls == ["called", "called"]
    assert summary.task_completed is False
    assert summary.truncated is True
    responses = json.loads(
        (Path(summary.code_path).parent / "all_responses.json").read_text()
    )
    assert sum(item.get("model_request_count", 0) for item in responses) == 2
    assert any(
        item["decision"] == "model_fallback_budget_exhausted"
        and item["fallback_attempts"] == 2
        for item in responses
    )


@pytest.mark.parametrize(
    ("fallback_limit", "total_limit", "expected_calls"),
    [(1, 0, 1), (2, 0, 2), (2, 1, 1)],
)
def test_hybrid_model_fallback_budget_executes_safe_stop(
    tmp_path, monkeypatch, fallback_limit, total_limit, expected_calls
):
    env = _FakeTrialEnv()
    output_dir = tmp_path / f"{fallback_limit}-{total_limit}"
    output_dir.mkdir()
    config = _trial_config(output_dir)
    config["use_oracle_code"] = False
    config["text_world_state_feedback"] = {
        "enabled": True,
        "action_aligned_window": True,
        "wait_for_stability": False,
        "loop_guard_enabled": False,
    }
    config["execution_guard"]["stage_expectations"] = [
        {"name": "prepared", "required": {"prepared": True}},
    ]
    config["hybrid_recovery"] = {
        "enabled": True,
        "max_model_fallbacks_per_exception": fallback_limit,
        "max_model_fallbacks_total": total_limit,
        "task_graph": {
            "initial": ["stage_001"],
            "actions": [
                {
                    "id": "stage_001",
                    "code": "x = 1",
                    "failure_routes": {"POSTCONDITION_FAILED": "UNKNOWN_FAILURE"},
                },
                {"id": "safe_stop", "code": "safe_stop()"},
            ],
        },
        "recovery_library": {"RECOVERY_BUDGET_LOW": ["safe_stop"]},
        "stage_actions": ["stage_001"],
    }
    model_calls = []

    monkeypatch.setattr(
        trial_module,
        "_query_initial_code",
        lambda *_a, **_k: ("x = 1", "plan", None, {"total_tokens": 1}),
    )

    def model_fallback(*_args, **_kwargs):
        model_calls.append(len(model_calls) + 1)
        return (
            "regenerate",
            f"```python\nx = {len(model_calls) + 1}\n```",
            "retry",
            None,
            [],
            {"total_tokens": 1},
        )

    monkeypatch.setattr(trial_module, "_handle_multi_turn_step", model_fallback)

    summary = trial_module._run_single_trial(
        env,
        1,
        _trial_args(),
        config,
        "{executed_code} {console_stdout} {console_stderr}",
    )

    assert model_calls == list(range(1, expected_calls + 1))
    assert env.executed[-1] == "safe_stop()"
    assert summary.task_completed is False
    assert summary.truncated is True
    responses = json.loads(
        (Path(summary.code_path).parent / "all_responses.json").read_text()
    )
    assert any(
        item["decision"] == "model_fallback_budget_exhausted"
        and item["fallback_attempts"] == expected_calls
        for item in responses
    )
    assert responses[-1]["decision"] == "model_fallback_budget_safe_stop"
    assert responses[-1]["model_request_count"] == 0


def test_policy_safe_stop_route_calls_model_once_when_budget_is_available(
    tmp_path, monkeypatch
):
    env = _FakeTrialEnv()
    config = _trial_config(tmp_path)
    config["use_oracle_code"] = False
    config["text_world_state_feedback"] = {
        "enabled": True,
        "action_aligned_window": True,
        "wait_for_stability": False,
        "loop_guard_enabled": False,
    }
    config["execution_guard"]["stage_expectations"] = [
        {"name": "prepared", "required": {"prepared": True}},
    ]
    config["hybrid_recovery"] = {
        "enabled": True,
        "policy_first": True,
        "max_model_fallbacks_per_exception": 1,
        "max_model_fallbacks_total": 1,
        "task_graph": {
            "initial": ["stage_001"],
            "actions": [
                {
                    "id": "stage_001",
                    "code": "x = 1",
                    "failure_routes": {"POSTCONDITION_FAILED": "KNOWN_FAILURE"},
                },
                {
                    "id": "recover",
                    "code": "x = 2",
                    "metadata": {
                        "interrupt_on_failure": True,
                        "failure_next_action": "recover_budget_safe_stop",
                    },
                },
                {"id": "recover_budget_safe_stop", "code": "safe_stop()"},
            ],
        },
        "recovery_library": {
            "KNOWN_FAILURE": ["recover"],
            "RECOVERY_BUDGET_LOW": ["recover_budget_safe_stop"],
        },
        "stage_actions": ["stage_001"],
    }
    model_calls = []

    def model_fallback(*_args, **_kwargs):
        model_calls.append("called")
        return "finish", None, "stop", None, [], {"total_tokens": 1}

    monkeypatch.setattr(trial_module, "_handle_multi_turn_step", model_fallback)

    summary = trial_module._run_single_trial(
        env, 1, _trial_args(), config,
        "{executed_code} {console_stdout} {console_stderr}",
    )

    responses = json.loads(
        (Path(summary.code_path).parent / "all_responses.json").read_text()
    )
    assert model_calls == ["called"]
    assert any(
        item.get("offline_action_id") == "recover"
        and item.get("fallback_to_model") is True
        for item in responses
    )


def test_observation_refresh_bypasses_stale_global_observation_gate(
    tmp_path, monkeypatch
):
    class ObservationRefreshEnv(_FakeTrialEnv):
        def __init__(self):
            super().__init__(oracle_code="fail()")
            self.observed = False

        def step(self, code):
            self.executed.append(code)
            if "reobserve()" in code:
                self.observed = True
            if "complete()" in code:
                self.completed = True
            return (
                {"full_prompt": []},
                1.0 if self.completed else 0.0,
                self.completed,
                False,
                {
                    "sandbox_rc": 0,
                    "stdout": "",
                    "stderr": "",
                    "task_completed": self.completed,
                },
            )

        def get_text_world_state(self):
            return {
                "observed": self.observed,
                "prepared": False,
                "object_stable": True,
                "task_completed": self.completed,
            }

    env = ObservationRefreshEnv()
    config = _trial_config(tmp_path)
    config["text_world_state_feedback"] = {
        "enabled": True,
        "action_aligned_window": True,
        "wait_for_stability": False,
        "loop_guard_enabled": False,
    }
    config["execution_guard"]["stage_expectations"] = [
        {"name": "prepared", "required": {"prepared": True}},
    ]
    config["hybrid_recovery"] = {
        "enabled": True,
        "policy_preconditions": {"observed": True},
        "task_graph": {
            "initial": ["stage_001"],
            "actions": [
                {
                    "id": "stage_001",
                    "code": "fail()",
                    "failure_routes": {"POSTCONDITION_FAILED": "NEEDS_OBSERVATION"},
                },
                {
                    "id": "refresh",
                    "code": "reobserve()",
                    "success_conditions": {"observed": True},
                    "next": "complete",
                    "metadata": {
                        "observation_refresh": True,
                        "interrupt_on_failure": True,
                    },
                },
                {
                    "id": "complete",
                    "code": "complete()",
                    "success_conditions": {"task_completed": True},
                },
            ],
        },
        "recovery_library": {"NEEDS_OBSERVATION": ["refresh"]},
        "stage_actions": ["stage_001"],
    }
    monkeypatch.setattr(
        trial_module,
        "_query_initial_code",
        lambda *_a, **_k: ("fail()", "plan", None, {"total_tokens": 1}),
    )
    config["use_oracle_code"] = False

    summary = trial_module._run_single_trial(
        env,
        1,
        _trial_args(),
        config,
        "{executed_code} {console_stdout} {console_stderr}",
    )

    assert env.executed == ["fail()", "reobserve()", "complete()"]
    assert summary.task_completed is True


def test_policy_chain_beyond_model_turn_limit_reaches_terminal_safe_stop(tmp_path):
    env = _FakeTrialEnv()
    config = _trial_config(tmp_path)
    config["text_world_state_feedback"] = {
        "enabled": True,
        "action_aligned_window": True,
        "wait_for_stability": False,
        "loop_guard_enabled": False,
    }
    actions = [
        {"id": f"step_{index}", "code": f"step_{index} = True"}
        for index in range(11)
    ]
    actions.extend([
        {
            "id": "terminal_gate",
            "code": "x = 1",
            "preconditions": {"prepared": True},
            "success_conditions": {"task_completed": True},
            "metadata": {
                "interrupt_on_failure": True,
                "precondition_failure_next_action": "safe_stop",
            },
        },
        {
            "id": "safe_stop",
            "code": "safe_stop()",
            "metadata": {"terminal": "safe_stop"},
        },
    ])
    config["hybrid_recovery"] = {
        "enabled": True,
        "policy_first": True,
        "task_graph": {
            "initial": [f"step_{index}" for index in range(11)] + ["terminal_gate"],
            "actions": actions,
        },
        "recovery_library": {"RECOVERY_BUDGET_LOW": ["safe_stop"]},
        "stage_actions": [f"step_{index}" for index in range(11)] + ["terminal_gate"],
    }

    summary = trial_module._run_single_trial(
        env,
        1,
        _trial_args(),
        config,
        "{executed_code} {console_stdout} {console_stderr}",
    )

    assert env.executed[-1] == "safe_stop()"
    assert summary.truncated is True
    responses = json.loads(
        (Path(summary.code_path).parent / "all_responses.json").read_text()
    )
    assert responses[-1]["decision"] == "model_fallback_budget_safe_stop"


def test_hybrid_config_reaches_trial_loader():
    _, config, _ = _load_config(LaunchArgs(
        config_path="env_configs/cube_restack/hybrid_policy_first.yaml"
    ))

    assert config["hybrid_recovery"]["enabled"] is True
    assert config["hybrid_recovery"]["stage_actions"] == [
        "stage_001", "stage_002", "stage_003"
    ]
