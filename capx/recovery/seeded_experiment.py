"""Run and summarize the three-group fixed-seed Cube Stack experiment."""

from __future__ import annotations

import argparse
import json
import re
import socket
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import urlparse


GROUP_CONFIGS = {
    "baseline": "env_configs/cube_stack/recovery_seeded_baseline.yaml",
    "single_frame": "env_configs/cube_stack/recovery_seeded_single_frame.yaml",
    "k3_recovery": "env_configs/cube_stack/recovery_seeded_k3.yaml",
}
GUARDED_GROUP_CONFIGS = {
    "baseline": "env_configs/cube_stack/recovery_seeded_baseline.yaml",
    "k3_unbounded": "env_configs/cube_stack/recovery_seeded_k3.yaml",
    "k3_guarded": "env_configs/cube_stack/recovery_seeded_k3_guarded.yaml",
}
IMPROVED_K3_GROUP_CONFIGS = {
    "k3_action_aligned": "env_configs/cube_stack/recovery_seeded_k3_guarded.yaml",
}

_TRIAL_DIR = re.compile(
    r"trial_(?P<trial>\d+)_sandboxrc_(?P<sandbox_rc>-?\d+)_"
    r"reward_(?P<reward>-?[\d.]+)_taskcompleted_(?P<completed>[01])"
)


def model_endpoint_reachable(server_url: str, timeout: float = 2.0) -> bool:
    parsed = urlparse(server_url)
    if not parsed.hostname:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


def run_groups(
    *, model: str, server_url: str, trials: int, temperature: float,
    group_configs: dict[str, str] | None = None,
    output_root: Path | None = None,
    record_video: bool | None = None,
    start_trial: int = 1,
) -> None:
    from capx.envs.launch import LaunchArgs, main as launch_main

    selected = group_configs or GROUP_CONFIGS
    for group, config_path in selected.items():
        print(f"\n=== Running {group}: {config_path} ===")
        output_dir = None
        if output_root is not None:
            # CaP-X inserts the model slug immediately before the final path
            # component, so pass <root>/<group> here.
            output_dir = str(output_root / group)
        launch_main(LaunchArgs(
            config_path=config_path,
            model=model,
            server_url=server_url,
            temperature=temperature,
            total_trials=start_trial + trials - 1,
            resume_idx=start_trial if start_trial > 1 else None,
            num_workers=1,
            record_video=record_video,
            output_dir=output_dir,
        ))


def collect_group(group_dir: Path) -> dict[str, Any]:
    # Recovery writes intermediate result directories because reward is part of
    # the directory name. Prefer snapshots with a complete environment response;
    # an incomplete snapshot can have a newer directory mtime than the terminal
    # result. Legacy/synthetic results without summary.txt retain mtime fallback.
    candidates_by_trial: dict[int, list[Path]] = {}
    duplicate_result_dirs = 0
    for trial_dir in sorted(path for path in group_dir.glob("trial_*") if path.is_dir()):
        match = _TRIAL_DIR.fullmatch(trial_dir.name)
        if match is None:
            continue
        trial_id = int(match.group("trial"))
        trial_candidates = candidates_by_trial.setdefault(trial_id, [])
        if trial_candidates:
            duplicate_result_dirs += 1
        trial_candidates.append(trial_dir)

    candidates: dict[int, Path] = {}
    for trial_id, trial_candidates in candidates_by_trial.items():
        terminal_candidates = []
        for trial_dir in trial_candidates:
            summary_path = trial_dir / "summary.txt"
            if not summary_path.exists():
                continue
            summary = summary_path.read_text(encoding="utf-8", errors="replace")
            if "Environment response:" in summary and "Task Completed:" in summary:
                terminal_candidates.append(trial_dir)
        selection_pool = terminal_candidates or trial_candidates
        candidates[trial_id] = max(
            selection_pool, key=lambda path: path.stat().st_mtime_ns
        )

    trials: list[dict[str, Any]] = []
    for trial_id, trial_dir in sorted(candidates.items()):
        match = _TRIAL_DIR.fullmatch(trial_dir.name)
        assert match is not None
        responses_path = trial_dir / "all_responses.json"
        responses = json.loads(responses_path.read_text(encoding="utf-8")) \
            if responses_path.exists() else []
        observed_model_request_count = sum(
            item.get("decision") in {
                "initial", "regenerate", "continue", "finish",
                "continue_rejected", "finish_rejected",
            }
            or (item.get("decision") == "guard_abort" and item.get("token_usage") is not None)
            for item in responses
        )
        usage_items = [
            item["token_usage"] for item in responses
            if isinstance(item.get("token_usage"), dict)
        ]
        token_usage = None
        if usage_items:
            token_usage = {
                key: sum(int(item.get(key, 0)) for item in usage_items)
                for key in (
                    "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"
                )
            }
        regenerations = sum(item.get("decision") == "regenerate" for item in responses)
        metrics_path = trial_dir / "trial_metrics.json"
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.exists()
            else None
        )
        summary_text = (
            (trial_dir / "summary.txt").read_text(encoding="utf-8", errors="replace")
            if (trial_dir / "summary.txt").exists() else ""
        )
        timed_out = "timed out after" in summary_text
        model_request_count = int(
            metrics.get("model_request_count", observed_model_request_count)
            if metrics else observed_model_request_count
        )
        token_accounted_requests = int(
            metrics.get("token_accounted_requests", len(usage_items))
            if metrics else len(usage_items)
        )
        feedback_path = group_dir / f"text_world_state_feedback_trial_{trial_id:03d}.json"
        feedback = json.loads(feedback_path.read_text(encoding="utf-8")) \
            if feedback_path.exists() else None
        confidences: list[float] = []
        unstable_count = 0
        snapshot_count = 0
        wrong_object_grasped = False
        switch_strategy_requests = 0
        max_repeated_failures = 0
        if feedback:
            for item in feedback.get("history", []):
                summary = item.get("summary", {})
                if "confidence" in summary:
                    confidences.append(float(summary["confidence"]))
                unstable_count += int(not summary.get("stable", False))
                snapshot_count += 1
                state = summary.get("state", {})
                wrong_object_grasped |= bool(state.get("wrong_object_grasped", False))
                directive = item.get("recovery_directive", {})
                switch_strategy_requests += int(
                    directive.get("action") == "switch_strategy"
                )
                max_repeated_failures = max(
                    max_repeated_failures,
                    int(directive.get("repeated_failures", 0)),
                )
        completed = bool(int(match.group("completed")))
        fingerprint_path = group_dir / f"initial_state_fingerprint_trial_{trial_id:03d}.json"
        fingerprint_payload = (
            json.loads(fingerprint_path.read_text(encoding="utf-8"))
            if fingerprint_path.exists() else None
        )
        trials.append({
            "trial": trial_id,
            "task_completed": completed,
            "reward": float(match.group("reward")),
            "sandbox_rc": int(match.group("sandbox_rc")),
            "regenerations": regenerations,
            "recovered_after_regeneration": completed and regenerations > 0,
            "feedback_available": feedback is not None,
            "mean_state_confidence": mean(confidences) if confidences else None,
            "unstable_snapshots": unstable_count,
            "snapshot_count": snapshot_count,
            "wrong_object_grasped": wrong_object_grasped,
            "guard_aborted": any(
                item.get("decision") == "guard_abort" for item in responses
            ),
            "code_preflight_rejections": int(
                metrics.get("code_preflight_rejections", 0) if metrics else sum(
                    item.get("decision") == "code_preflight_rejected"
                    for item in responses
                )
            ),
            "finish_rejections": int(
                metrics.get("finish_rejections", 0) if metrics else sum(
                    item.get("decision") == "finish_rejected" for item in responses
                )
            ),
            "continue_rejections": int(
                metrics.get("continue_rejections", 0) if metrics else sum(
                    item.get("decision") == "continue_rejected" for item in responses
                )
            ),
            "strategy_switch_requests": switch_strategy_requests,
            "max_repeated_failures": max_repeated_failures,
            "model_request_count": model_request_count,
            "model_response_seconds": float(
                metrics.get("model_response_seconds", 0.0)
                if metrics else sum(
                    float(item.get("model_response_seconds", 0.0))
                    for item in responses
                )
            ),
            "token_accounted_requests": token_accounted_requests,
            "metrics_complete": bool(
                metrics.get("metrics_complete", True) if metrics else True
            ) and not timed_out,
            "timed_out": timed_out,
            "token_usage": token_usage,
            "elapsed_seconds": (
                float(metrics["elapsed_seconds"])
                if metrics and metrics.get("elapsed_seconds") is not None
                else None
            ),
            "initial_state_fingerprint": (
                fingerprint_payload.get("fingerprint") if fingerprint_payload else None
            ),
        })

    count = len(trials)
    group_usage_items = [item["token_usage"] for item in trials if item["token_usage"]]
    group_token_usage = None
    if group_usage_items:
        group_token_usage = {
            key: sum(int(item.get(key, 0)) for item in group_usage_items)
            for key in (
                "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"
            )
        }
    model_request_count = sum(item["model_request_count"] for item in trials)
    token_accounted_requests = sum(item["token_accounted_requests"] for item in trials)
    elapsed_items = [
        item["elapsed_seconds"]
        for item in trials
        if item["elapsed_seconds"] is not None
    ]
    model_elapsed_items = [item["model_response_seconds"] for item in trials]
    return {
        "group_dir": str(group_dir),
        "trial_count": count,
        "task_success_rate": (
            sum(item["task_completed"] for item in trials) / count if count else None
        ),
        "mean_reward": mean(item["reward"] for item in trials) if trials else None,
        "mean_regenerations": mean(item["regenerations"] for item in trials) if trials else None,
        "recovered_after_regeneration": sum(
            item["recovered_after_regeneration"] for item in trials
        ),
        "wrong_object_grasp_trials": sum(item["wrong_object_grasped"] for item in trials),
        "guard_abort_trials": sum(item["guard_aborted"] for item in trials),
        "code_preflight_rejections": sum(
            item["code_preflight_rejections"] for item in trials
        ),
        "finish_rejections": sum(item["finish_rejections"] for item in trials),
        "continue_rejections": sum(item["continue_rejections"] for item in trials),
        "strategy_switch_requests": sum(
            item["strategy_switch_requests"] for item in trials
        ),
        "model_request_count": model_request_count,
        "total_model_response_seconds": sum(model_elapsed_items),
        "mean_model_response_seconds": (
            mean(model_elapsed_items) if model_elapsed_items else None
        ),
        "token_accounted_requests": token_accounted_requests,
        "token_usage_coverage": (
            token_accounted_requests / model_request_count if model_request_count else None
        ),
        "token_usage": group_token_usage,
        "incomplete_metric_trials": sum(
            not item["metrics_complete"] for item in trials
        ),
        "elapsed_time_coverage": len(elapsed_items) / count if count else None,
        "total_trial_elapsed_seconds": sum(elapsed_items) if elapsed_items else None,
        "mean_trial_elapsed_seconds": mean(elapsed_items) if elapsed_items else None,
        "feedback_coverage": (
            sum(item["feedback_available"] for item in trials) / count if count else None
        ),
        "duplicate_result_dirs_ignored": duplicate_result_dirs,
        "trials": trials,
    }


def collect_experiment(
    output_root: Path,
    model: str,
    group_configs: dict[str, str] | None = None,
) -> dict[str, Any]:
    selected = group_configs or GROUP_CONFIGS
    model_dir = model.replace("/", "_")
    groups = {
        group: collect_group(output_root / model_dir / group)
        for group in selected
    }
    seed_ids = {
        group: [item["trial"] for item in result["trials"]]
        for group, result in groups.items()
    }
    reference_group = next(iter(selected))
    paired = (
        len(selected) > 1
        and bool(seed_ids[reference_group])
        and len({tuple(ids) for ids in seed_ids.values()}) == 1
    )
    physical_pairing = paired and all(
        len({
            groups[group]["trials"][index]["initial_state_fingerprint"]
            for group in selected
        }) == 1
        and groups[reference_group]["trials"][index]["initial_state_fingerprint"] is not None
        for index in range(len(groups[reference_group]["trials"]))
    )
    validity_notes = [
        "Duplicate result directories prefer complete environment-terminal snapshots, "
        "then select the newest result per trial ID."
    ]
    if physical_pairing:
        validity_notes.insert(0, "Physical initial-state fingerprints match across all groups.")
    else:
        validity_notes.insert(0, "Physical initial-state pairing is missing or does not match.")
    return {
        "experiment": "cube_stack_text_world_state_fixed_seed",
        "model": model,
        "paired_trial_ids": paired,
        "physical_initial_state_pairing_verified": physical_pairing,
        "validity_notes": validity_notes,
        "groups": groups,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Cube Stack 文本世界状态固定种子实验",
        "",
        f"模型：`{report['model']}`",
        f"三组试次编号是否配对：{'是' if report['paired_trial_ids'] else '否或尚无结果'}",
        "物理初始状态是否已验证配对："
        + ("是" if report["physical_initial_state_pairing_verified"] else "否"),
        "",
        "| 组别 | 试验数 | 成功率 | 输入 token | 输出 token | 推理 token | 总 token | 记录覆盖率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, result in report["groups"].items():
        rate = result["task_success_rate"]
        reward = result["mean_reward"]
        regens = result["mean_regenerations"]
        usage = result["token_usage"]
        lines.append(
            f"| {group} | {result['trial_count']} | {_pct(rate)} | "
            f"{_token(usage, 'prompt_tokens')} | {_token(usage, 'completion_tokens')} | "
            f"{_token(usage, 'reasoning_tokens')} | {_token(usage, 'total_tokens')} | "
            f"{_pct(result['token_usage_coverage'])} |"
        )
    lines.extend([
        "",
        "有效性说明：" + (
            "所有试次的初态指纹跨组一致，可按严格配对实验解释。"
            if report["physical_initial_state_pairing_verified"]
            else "初态指纹缺失或跨组不一致，结果不得作为严格配对实验解释。"
        ),
        "结果目录处理：同一试次存在多个快照时，优先统计含完整环境终态的快照，再按修改时间选择；原始目录保留。",
        "",
        "说明：当前主流程以代码块而非动作原语为边界，因此这里的“重规划”指多轮代码 "
        "REGENERATE；完整逐原语分数需等待动作 API 边界接入。",
    ])
    return "\n".join(lines) + "\n"


def save_report(report: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _token(usage: dict[str, int] | None, key: str) -> str:
    return "—" if usage is None else f"{usage.get(key, 0):,}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemini-3.8-flash")
    parser.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--profile", choices=("initial", "guarded", "k3-improved"), default="initial"
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    group_configs = {
        "initial": GROUP_CONFIGS,
        "guarded": GUARDED_GROUP_CONFIGS,
        "k3-improved": IMPROVED_K3_GROUP_CONFIGS,
    }[args.profile]
    output_root = args.output_root or Path(
        {
            "initial": "outputs/text_world_state_seeded",
            "guarded": "outputs/text_world_state_guarded_paired",
            "k3-improved": "outputs/text_world_state_k3_action_aligned",
        }[args.profile]
    )
    report_dir = args.report_dir or Path(
        {
            "initial": "reports/text_world_state_seeded_simulator",
            "guarded": "reports/text_world_state_guarded_paired",
            "k3-improved": "reports/text_world_state_k3_action_aligned",
        }[args.profile]
    )
    if args.run:
        if args.trials < 1:
            parser.error("--trials must be >= 1")
        if not model_endpoint_reachable(args.server_url):
            parser.error(
                f"model endpoint is unreachable: {args.server_url}; "
                "start the model service before using --run"
            )
        run_groups(
            model=args.model,
            server_url=args.server_url,
            trials=args.trials,
            temperature=args.temperature,
            group_configs=group_configs,
            output_root=output_root,
        )
    report = collect_experiment(output_root, args.model, group_configs)
    save_report(report, report_dir)
    print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
