import json

from capx.envs.launch import LaunchArgs
from capx.recovery.known_policy_ab import (
    GEOMETRY_HOLDOUT_SCENARIOS,
    GROUPS,
    SCENARIOS,
    collect_benchmark,
)
from capx.utils.launch_utils import _load_config


def test_known_policy_configs_only_enable_library_in_hybrid():
    _, hybrid, _ = _load_config(LaunchArgs(config_path=GROUPS["hybrid"]))
    _, native, _ = _load_config(LaunchArgs(config_path=GROUPS["native"]))
    assert hybrid["hybrid_recovery"]["policy_first"] is True
    assert native["hybrid_recovery"]["enabled"] is False
    assert hybrid["text_world_state_feedback"] == native["text_world_state_feedback"]
    assert hybrid["execution_guard"] == native["execution_guard"]
    assert hybrid["total_trials"] == native["total_trials"] == 10
    assert set(GEOMETRY_HOLDOUT_SCENARIOS) == {"random_geometry_outside_container"}


def test_collect_known_policy_ab_reports_pairing_and_exact_effect(tmp_path):
    model_dir = "test_model"
    for scenario in SCENARIOS:
        for group in GROUPS:
            group_dir = tmp_path / scenario / model_dir / group
            group_dir.mkdir(parents=True)
            for trial in (1, 2):
                completed = group == "hybrid" or trial == 1
                trial_dir = group_dir / (
                    f"trial_{trial:02d}_sandboxrc_0_reward_{float(completed):.3f}_"
                    f"taskcompleted_{int(completed)}"
                )
                trial_dir.mkdir()
                (trial_dir / "summary.txt").write_text(
                    f"Environment response:\n  Task Completed: {completed}\n",
                    encoding="utf-8",
                )
                (trial_dir / "all_responses.json").write_text(json.dumps([{
                    "decision": "initial",
                    "model_request_count": int(group == "native"),
                    "model_response_seconds": 2.0 if group == "native" else 0.0,
                    "token_usage": ({"total_tokens": 10} if group == "native" else None),
                }]), encoding="utf-8")
                (trial_dir / "trial_metrics.json").write_text(json.dumps({
                    "elapsed_seconds": 4.0 if group == "native" else 1.0,
                    "model_request_count": int(group == "native"),
                    "model_response_seconds": 2.0 if group == "native" else 0.0,
                }), encoding="utf-8")
                (group_dir / f"initial_state_fingerprint_trial_{trial:03d}.json").write_text(
                    json.dumps({"fingerprint": f"{scenario}-{trial}"}), encoding="utf-8"
                )

    report = collect_benchmark(tmp_path, "test/model")
    assert report["all_initial_states_paired"] is True
    assert report["pooled_paired_effect"]["hybrid_successes"] == 6
    assert report["pooled_paired_effect"]["native_successes"] == 3
    assert report["pooled_paired_effect"]["total_token_delta"] == -60
    assert report["pooled_paired_effect"]["mean_model_response_delta_seconds"] == -2.0
