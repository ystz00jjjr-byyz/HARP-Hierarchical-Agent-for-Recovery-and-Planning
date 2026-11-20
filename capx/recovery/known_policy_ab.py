"""Paired current-hybrid versus model-only CAPX benchmark for known bin faults."""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from typing import Any, Iterator

from capx.recovery.seeded_experiment import (
    collect_group,
    model_endpoint_reachable,
    run_groups,
)


GROUPS = {
    "hybrid": "env_configs/bin_place/paired_known_hybrid.yaml",
    "native": "env_configs/bin_place/paired_known_native.yaml",
}
SCENARIOS = {
    "outside_container": {
        "BIN_PLACE_RELEASE_FAILURES": "1",
        "BIN_PLACE_RELEASE_OFFSET": "0.10",
    },
    "unstable_release": {
        "BIN_PLACE_RELEASE_FAILURES": "1",
        "BIN_PLACE_RELEASE_UNSTABLE": "true",
        "BIN_PLACE_RELEASE_OFFSET": "0.06",
    },
    "workspace_edge_grasp_miss": {
        "BIN_PLACE_WORKSPACE_EDGE_SWEEP": "true",
        "BIN_PLACE_GRASP_MISS": "true",
    },
}
GEOMETRY_HOLDOUT_SCENARIOS = {
    "random_geometry_outside_container": {
        "BIN_PLACE_SCENARIO_SWEEP": "true",
        "BIN_PLACE_RELEASE_FAILURES": "1",
        "BIN_PLACE_RELEASE_OFFSET": "0.10",
    },
}
_ENV_KEYS = {
    "BIN_PLACE_SCENARIO_SWEEP",
    "BIN_PLACE_WORKSPACE_EDGE_SWEEP",
    "BIN_PLACE_GRASP_MISS",
    "BIN_PLACE_RELEASE_FAILURES",
    "BIN_PLACE_RELEASE_UNSTABLE",
    "BIN_PLACE_RELEASE_OFFSET",
}


@contextmanager
def _scenario_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in _ENV_KEYS}
    try:
        for key in _ENV_KEYS:
            os.environ.pop(key, None)
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_benchmark(
    *, model: str, server_url: str, trials: int, temperature: float,
    output_root: Path, scenarios: dict[str, dict[str, str]] = SCENARIOS,
    start_trial: int = 1,
) -> None:
    for scenario, environment in scenarios.items():
        print(f"\n######## {scenario} ########")
        with _scenario_environment(environment):
            run_groups(
                model=model,
                server_url=server_url,
                trials=trials,
                temperature=temperature,
                group_configs=GROUPS,
                output_root=output_root / scenario,
                record_video=True,
                start_trial=start_trial,
            )


def collect_benchmark(
    output_root: Path, model: str,
    scenarios_to_collect: dict[str, dict[str, str]] = SCENARIOS,
    seed_start: int = 1,
) -> dict[str, Any]:
    model_dir = model.replace("/", "_")
    scenarios: dict[str, Any] = {}
    pooled_pairs: list[dict[str, Any]] = []
    for scenario in scenarios_to_collect:
        groups = {
            group: collect_group(output_root / scenario / model_dir / group)
            for group in GROUPS
        }
        maps = {
            group: {trial["trial"]: trial for trial in result["trials"]}
            for group, result in groups.items()
        }
        common = sorted(set(maps["hybrid"]) & set(maps["native"]))
        paired_ids = bool(common) and all(sorted(values) == common for values in maps.values())
        pairs = []
        for trial_id in common:
            hybrid = maps["hybrid"][trial_id]
            native = maps["native"][trial_id]
            pair = {
                "scenario": scenario,
                "trial": trial_id,
                "fingerprint_match": (
                    hybrid.get("initial_state_fingerprint") is not None
                    and hybrid.get("initial_state_fingerprint")
                    == native.get("initial_state_fingerprint")
                ),
                "hybrid": _trial_projection(hybrid),
                "native": _trial_projection(native),
            }
            pairs.append(pair)
            pooled_pairs.append(pair)
        scenarios[scenario] = {
            "paired_trial_ids": paired_ids,
            "physical_initial_state_pairing_verified": paired_ids and all(
                pair["fingerprint_match"] for pair in pairs
            ),
            "groups": groups,
            "paired_effect": _paired_effect(pairs),
            "paired_trials": pairs,
        }
    return {
        "experiment": "known_bin_policy_hybrid_vs_native_capx",
        "model": model,
        "temperature": 0.0,
        "scenario_count": len(scenarios),
        "seed_start": seed_start,
        "all_initial_states_paired": bool(scenarios) and all(
            value["physical_initial_state_pairing_verified"]
            for value in scenarios.values()
        ),
        "scenarios": scenarios,
        "pooled_paired_effect": _paired_effect(pooled_pairs),
        "paper_claim_ready": _paper_claim_ready(scenarios),
    }


def _trial_projection(trial: dict[str, Any]) -> dict[str, Any]:
    usage = trial.get("token_usage") or {}
    return {
        "task_completed": bool(trial["task_completed"]),
        "reward": trial["reward"],
        "elapsed_seconds": trial.get("elapsed_seconds"),
        "model_response_seconds": trial.get("model_response_seconds", 0.0),
        "model_request_count": trial["model_request_count"],
        "total_tokens": int(usage.get("total_tokens", 0)),
        "metrics_complete": trial.get("metrics_complete", True),
        "code_preflight_rejections": trial.get("code_preflight_rejections", 0),
        "finish_rejections": trial.get("finish_rejections", 0),
    }


def _paired_effect(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    hybrid_success = sum(pair["hybrid"]["task_completed"] for pair in pairs)
    native_success = sum(pair["native"]["task_completed"] for pair in pairs)
    hybrid_only = sum(
        pair["hybrid"]["task_completed"] and not pair["native"]["task_completed"]
        for pair in pairs
    )
    native_only = sum(
        pair["native"]["task_completed"] and not pair["hybrid"]["task_completed"]
        for pair in pairs
    )
    return {
        "pair_count": len(pairs),
        "hybrid_successes": hybrid_success,
        "native_successes": native_success,
        "success_rate_delta_percentage_points": (
            100.0 * (hybrid_success - native_success) / len(pairs) if pairs else None
        ),
        "hybrid_success_wilson_95": _wilson(hybrid_success, len(pairs)),
        "native_success_wilson_95": _wilson(native_success, len(pairs)),
        "hybrid_only_successes": hybrid_only,
        "native_only_successes": native_only,
        "mcnemar_exact_two_sided_p": _mcnemar_exact(hybrid_only, native_only),
        "mean_elapsed_delta_seconds": _mean_delta(pairs, "elapsed_seconds"),
        "mean_model_response_delta_seconds": _mean_delta(
            pairs, "model_response_seconds"
        ),
        "total_token_delta": sum(
            pair["hybrid"]["total_tokens"] - pair["native"]["total_tokens"]
            for pair in pairs
        ),
        "total_model_request_delta": sum(
            pair["hybrid"]["model_request_count"]
            - pair["native"]["model_request_count"]
            for pair in pairs
        ),
    }


def _mean_delta(pairs: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(pair["hybrid"][key]) - float(pair["native"][key])
        for pair in pairs
        if pair["hybrid"].get(key) is not None and pair["native"].get(key) is not None
    ]
    return mean(values) if values else None


def _wilson(successes: int, count: int) -> list[float] | None:
    if not count:
        return None
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [center - margin, center + margin]


def _mcnemar_exact(hybrid_only: int, native_only: int) -> float | None:
    discordant = hybrid_only + native_only
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(hybrid_only, native_only) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def _paper_claim_ready(scenarios: dict[str, Any]) -> bool:
    return bool(scenarios) and all(
        value["physical_initial_state_pairing_verified"]
        and value["groups"]["hybrid"]["trial_count"] >= 10
        and value["groups"]["native"]["trial_count"] >= 10
        and value["groups"]["hybrid"]["incomplete_metric_trials"] == 0
        and value["groups"]["native"]["incomplete_metric_trials"] == 0
        for value in scenarios.values()
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 当前混合调度与原生 CAPX：已知异常配对 A/B 实验",
        "",
        f"模型：`{report['model']}`；温度：0；seed 从 {report.get('seed_start', 1)} 开始。",
        "成功仅按环境 `task_completed=true`；A/B 使用相同初态指纹、故障注入、K=3 观测、执行门禁和时域。",
        "原生组禁用离线任务图与策略库，由模型生成初始方案及恢复；混合组策略优先，失败才调用同一模型。",
        "",
        f"全部初态严格配对：{'是' if report['all_initial_states_paired'] else '否'}；"
        f"论文主结论就绪：{'是' if report['paper_claim_ready'] else '否'}。",
    ]
    for name, scenario in report["scenarios"].items():
        lines.extend(["", f"## {name}", "", _summary_table(scenario), "", _effect_line(scenario["paired_effect"])])
    if report.get("geometry_manifest"):
        lines.extend(["", "## 随机几何清单", "", _geometry_table(report["geometry_manifest"])])
    lines.extend(["", "## 合并配对效应", "", _effect_line(report["pooled_paired_effect"]), ""])
    return "\n".join(lines)


def _summary_table(scenario: dict[str, Any]) -> str:
    lines = [
        "| 方法 | n | 成功 | 模型请求 | 总 token | 平均模型响应 | 平均总耗时 | 无效代码拒绝 | 错误完成拒绝 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("hybrid", "native"):
        group = scenario["groups"][name]
        usage = group.get("token_usage") or {}
        lines.append(
            f"| {name} | {group['trial_count']} | {sum(t['task_completed'] for t in group['trials'])} | "
            f"{group['model_request_count']} | {int(usage.get('total_tokens', 0)):,} | "
            f"{_seconds(group.get('mean_model_response_seconds'))} | "
            f"{_seconds(group.get('mean_trial_elapsed_seconds'))} | "
            f"{group['code_preflight_rejections']} | {group['finish_rejections']} |"
        )
    return "\n".join(lines)


def _effect_line(effect: dict[str, Any]) -> str:
    delta = effect["success_rate_delta_percentage_points"]
    p = effect["mcnemar_exact_two_sided_p"]
    return (
        f"混合版成功率差：{delta:+.1f} 个百分点；配对不一致："
        f"{effect['hybrid_only_successes']} vs {effect['native_only_successes']}；"
        f"McNemar 精确双侧 p={p:.4f}；平均总耗时差 "
        f"{effect['mean_elapsed_delta_seconds']:+.2f}s；Token 差 "
        f"{effect['total_token_delta']:+,}。"
    ) if delta is not None and p is not None else "尚无完整配对结果。"


def _seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}s"


def _geometry_table(manifest: list[dict[str, Any]]) -> str:
    lines = [
        "| seed | 物体半尺寸 (m) | 物体位置 (m) | 物体 yaw | 容器位置 (m) | 容器 yaw | 内部尺寸 (m) | 边沿高 (m) |",
        "| ---: | --- | --- | ---: | --- | ---: | --- | ---: |",
    ]
    for item in manifest:
        lines.append(
            f"| {item['seed']} | {item['object_half_sizes']} | {item['object_xy']} | "
            f"{item['object_yaw_degrees']}° | {item['bin_offset_xy']} | "
            f"{item['bin_yaw_degrees']}° | {item['target_inner_extent']} | "
            f"{item['rim_height']} |"
        )
    return "\n".join(lines)


def save_report(report: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemini-3.8-flash")
    parser.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--geometry-holdout", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/bin_place_known_ab_20260921"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/bin_place_known_ab_20260921"))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    selected_scenarios = GEOMETRY_HOLDOUT_SCENARIOS if args.geometry_holdout else SCENARIOS
    if args.run:
        if args.trials < 2:
            parser.error("--trials must be >= 2")
        if not model_endpoint_reachable(args.server_url):
            parser.error(f"model endpoint is unreachable: {args.server_url}")
        run_benchmark(
            model=args.model,
            server_url=args.server_url,
            trials=args.trials,
            temperature=args.temperature,
            output_root=args.output_root,
            scenarios=selected_scenarios,
            start_trial=args.seed_start,
        )
    report = collect_benchmark(
        args.output_root, args.model, selected_scenarios, args.seed_start,
    )
    if args.geometry_holdout:
        from capx.envs.simulators.robosuite_bin_place import bin_place_scenario

        report["geometry_manifest"] = [
            {"seed": seed, **bin_place_scenario("cube", seed)}
            for seed in range(args.seed_start, args.seed_start + args.trials)
        ]
    save_report(report, args.report_dir)
    print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
