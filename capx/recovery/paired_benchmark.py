"""Paired original-CapX versus action-aligned K=3 benchmark on two tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from capx.recovery.seeded_experiment import (
    collect_group,
    model_endpoint_reachable,
    run_groups,
)


TASKS = {
    "simple_cube_lift": {
        "label": "简单任务：Cube Lift",
        "groups": {
            "baseline": "env_configs/cube_lifting/paired_baseline.yaml",
            "k3_action_aligned": "env_configs/cube_lifting/paired_k3_action_aligned.yaml",
        },
    },
    "complex_cube_restack": {
        "label": "复杂任务：Cube Restack",
        "groups": {
            "baseline": "env_configs/cube_restack/paired_baseline.yaml",
            "k3_action_aligned": "env_configs/cube_restack/paired_k3_action_aligned.yaml",
        },
    },
}


def run_benchmark(
    *, model: str, server_url: str, trials: int, temperature: float,
    output_root: Path,
) -> None:
    for task_name, task in TASKS.items():
        print(f"\n######## {task['label']} ########")
        run_groups(
            model=model,
            server_url=server_url,
            trials=trials,
            temperature=temperature,
            group_configs=task["groups"],
            output_root=output_root / task_name,
        )


def collect_benchmark(output_root: Path, model: str) -> dict[str, Any]:
    model_dir = model.replace("/", "_")
    task_results: dict[str, Any] = {}
    for task_name, task in TASKS.items():
        groups = {
            group: collect_group(output_root / task_name / model_dir / group)
            for group in task["groups"]
        }
        trial_maps = {
            group: {item["trial"]: item for item in result["trials"]}
            for group, result in groups.items()
        }
        common_ids = sorted(set.intersection(*(
            set(items) for items in trial_maps.values()
        ))) if trial_maps else []
        paired_trial_ids = bool(common_ids) and all(
            sorted(items) == common_ids for items in trial_maps.values()
        )
        paired_trials = []
        for trial_id in common_ids:
            baseline = trial_maps["baseline"][trial_id]
            k3 = trial_maps["k3_action_aligned"][trial_id]
            fingerprint_match = (
                baseline["initial_state_fingerprint"] is not None
                and baseline["initial_state_fingerprint"]
                == k3["initial_state_fingerprint"]
            )
            paired_trials.append({
                "trial": trial_id,
                "fingerprint_match": fingerprint_match,
                "initial_state_fingerprint": baseline["initial_state_fingerprint"],
                "baseline": _paired_trial_projection(baseline),
                "k3_action_aligned": _paired_trial_projection(k3),
            })
        physical_pairing = paired_trial_ids and all(
            item["fingerprint_match"] for item in paired_trials
        )
        baseline_rate = groups["baseline"]["task_success_rate"]
        k3_rate = groups["k3_action_aligned"]["task_success_rate"]
        task_results[task_name] = {
            "label": task["label"],
            "paired_trial_ids": paired_trial_ids,
            "physical_initial_state_pairing_verified": physical_pairing,
            "success_rate_delta_percentage_points": (
                100.0 * (k3_rate - baseline_rate)
                if baseline_rate is not None and k3_rate is not None
                else None
            ),
            "groups": groups,
            "paired_trials": paired_trials,
        }
    return {
        "experiment": "paired_capx_vs_action_aligned_k3_two_tasks",
        "model": model,
        "task_count": len(task_results),
        "all_tasks_physically_paired": bool(task_results) and all(
            task["physical_initial_state_pairing_verified"]
            for task in task_results.values()
        ),
        "tasks": task_results,
    }


def _paired_trial_projection(trial: dict[str, Any]) -> dict[str, Any]:
    usage = trial.get("token_usage") or {}
    return {
        "task_completed": trial["task_completed"],
        "reward": trial["reward"],
        "elapsed_seconds": trial.get("elapsed_seconds"),
        "total_tokens": usage.get("total_tokens"),
        "metrics_complete": trial.get("metrics_complete", True),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "model_request_count": trial["model_request_count"],
        "regenerations": trial["regenerations"],
        "guard_aborted": trial["guard_aborted"],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 原生 CapX 与动作对齐 K=3 双任务配对实验",
        "",
        f"模型：`{report['model']}`",
        f"全部任务物理初态严格配对：{'是' if report['all_tasks_physically_paired'] else '否'}",
    ]
    for task in report["tasks"].values():
        lines.extend([
            "",
            f"## {task['label']}",
            "",
            f"试次编号配对：{'是' if task['paired_trial_ids'] else '否'}；"
            f"初态指纹配对：{'是' if task['physical_initial_state_pairing_verified'] else '否'}",
            "",
            "| 方法 | 试次 | 成功率 | 平均耗时/次 | 总 token | token 覆盖率 | 模型请求 | 平均重规划 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for group_name, group in task["groups"].items():
            usage = group.get("token_usage") or {}
            lines.append(
                f"| {group_name} | {group['trial_count']} | {_pct(group['task_success_rate'])} | "
                f"{_seconds(group.get('mean_trial_elapsed_seconds'))} | "
                f"{_token_total(group)} | "
                f"{_pct(group.get('token_usage_coverage'))} | "
                f"{group['model_request_count']} | {_float(group['mean_regenerations'])} |"
            )
        lines.extend([
            "",
            "| 配对试次 | 初态匹配 | Baseline成功 | K3成功 | Baseline耗时 | K3耗时 | Baseline token | K3 token |",
            "| ---: | :---: | :---: | :---: | ---: | ---: | ---: | ---: |",
        ])
        for pair in task["paired_trials"]:
            baseline = pair["baseline"]
            k3 = pair["k3_action_aligned"]
            lines.append(
                f"| {pair['trial']} | {'是' if pair['fingerprint_match'] else '否'} | "
                f"{'是' if baseline['task_completed'] else '否'} | "
                f"{'是' if k3['task_completed'] else '否'} | "
                f"{_seconds(baseline['elapsed_seconds'])} | {_seconds(k3['elapsed_seconds'])} | "
                f"{_trial_tokens(baseline)} | {_trial_tokens(k3)} |"
            )
        delta = task["success_rate_delta_percentage_points"]
        lines.extend([
            "",
            "K=3 相对原生 CapX 的成功率差："
            + ("—" if delta is None else f"{delta:+.1f} 个百分点"),
        ])
    lines.extend([
        "",
        "判定口径：成功仅使用环境 `task_completed=True`；总 token 使用模型服务返回的 "
        "`total_tokens`；逐试次耗时覆盖该试次的全部重试。",
        "标记 `≥` 的 token 数是费用下界：该试次因基础设施异常中断，部分已完成请求的 "
        "usage 未能落盘。",
    ])
    return "\n".join(lines) + "\n"


def save_report(report: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def _seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}s"


def _integer(value: int | None) -> str:
    return "—" if value is None else f"{value:,}"


def _token_total(group: dict[str, Any]) -> str:
    total = (group.get("token_usage") or {}).get("total_tokens")
    prefix = "≥" if group.get("incomplete_metric_trials", 0) else ""
    return prefix + _integer(total)


def _trial_tokens(trial: dict[str, Any]) -> str:
    prefix = "" if trial.get("metrics_complete", True) else "≥"
    return prefix + _integer(trial.get("total_tokens"))


def _float(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemini-3.8-flash")
    parser.add_argument(
        "--server-url", default="http://127.0.0.1:8110/chat/completions"
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/paired_capx_k3_two_tasks_20260903"),
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=Path("reports/paired_capx_k3_two_tasks_20260903"),
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("--trials must be >= 2 for a paired comparison")
    if args.run:
        if not model_endpoint_reachable(args.server_url):
            parser.error(f"model endpoint is unreachable: {args.server_url}")
        run_benchmark(
            model=args.model,
            server_url=args.server_url,
            trials=args.trials,
            temperature=args.temperature,
            output_root=args.output_root,
        )
    report = collect_benchmark(args.output_root, args.model)
    save_report(report, args.report_dir)
    print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
