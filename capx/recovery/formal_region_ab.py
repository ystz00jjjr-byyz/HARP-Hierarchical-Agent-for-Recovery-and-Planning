"""Formal same-seed Native/L1/L2/L3 region-place benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from capx.recovery.known_policy_ab import _paired_effect, _trial_projection
from capx.recovery.seeded_experiment import (
    collect_group,
    model_endpoint_reachable,
    run_groups,
)


GROUPS = {
    "native": "env_configs/region_place/formal_ab_native.yaml",
    "l1": "env_configs/region_place/formal_ab_l1.yaml",
    "l2": "env_configs/region_place/formal_ab_l2.yaml",
    "l3": "env_configs/region_place/formal_ab_l3.yaml",
}


def _artifact_audit(group_dir: Path, trial_ids: list[int]) -> dict[str, Any]:
    def covered(pattern: str) -> list[int]:
        return [trial for trial in trial_ids if any(group_dir.glob(pattern.format(trial=trial)))]

    return {
        "expected_trials": len(trial_ids),
        "combined_video_trials": covered("trial_{trial}_*/video_combined.mp4"),
        "response_trials": covered("trial_{trial}_*/all_responses.json"),
        "metrics_trials": covered("trial_{trial}_*/trial_metrics.json"),
        "k3_feedback_trials": covered("text_world_state_feedback_trial_{trial:03d}.json"),
        "world_state_trials": covered("world_state/trial_{trial:03d}/**/*"),
    }


def collect_benchmark(output_root: Path, model: str) -> dict[str, Any]:
    model_dir = output_root / model.replace("/", "_")
    groups = {name: collect_group(model_dir / name) for name in GROUPS}
    maps = {
        name: {trial["trial"]: trial for trial in group["trials"]}
        for name, group in groups.items()
    }
    common = sorted(set.intersection(*(set(mapping) for mapping in maps.values()))) \
        if all(maps.values()) else []
    fingerprints_match = bool(common) and all(
        maps["native"][trial]["initial_state_fingerprint"] is not None
        and len({maps[name][trial]["initial_state_fingerprint"] for name in GROUPS}) == 1
        for trial in common
    )
    pairwise = {}
    for name in ("l1", "l2", "l3"):
        pairs = [{
            "hybrid": _trial_projection(maps[name][trial]),
            "native": _trial_projection(maps["native"][trial]),
        } for trial in common]
        pairwise[f"{name}_vs_native"] = _paired_effect(pairs)

    expected_trials = list(range(21, 51))
    artifacts = {
        name: _artifact_audit(model_dir / name, expected_trials)
        for name in maps
    }
    return {
        "experiment": "formal_nonprivileged_region_place_native_l1_l2_l3_ab",
        "model": model,
        "seed_range": "21-50",
        "fault": "double_release",
        "world_state_source": "camera_rgbd_k3",
        "paired_trial_ids": common,
        "all_initial_state_fingerprints_match": fingerprints_match,
        "groups": groups,
        "pairwise_vs_native": pairwise,
        "artifacts": artifacts,
        "combined_video_count": sum(
            len(item["combined_video_trials"]) for item in artifacts.values()
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 非特权 region_place：原生 CAPX / L1 / L2 / L3 正式配对 A/B",
        "",
        "30 个相同 seeds（21–50），统一使用 RGB-D 非特权 K=3 世界状态、随机位置/姿态/尺寸/目标几何和双重释放故障。",
        "L1 仅用离线策略；L2 仅在策略失败后选择白名单模板参数；L3 仅在策略失败后生成一组受门禁的两段恢复代码；原生 CAPX 由模型生成初始方案及恢复。",
        "",
        f"初态指纹全部匹配：{'是' if report['all_initial_state_fingerprints_match'] else '否'}；"
        f"完整试次视频：{report['combined_video_count']}/120。",
        "",
        "| 方法 | n | 成功 | 模型请求 | 总 Token | 平均模型响应 | 平均总耗时 | 无效代码拒绝 | 错误完成拒绝 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in GROUPS:
        group = report["groups"][name]
        usage = group.get("token_usage") or {}
        lines.append(
            f"| {name} | {group['trial_count']} | "
            f"{sum(item['task_completed'] for item in group['trials'])} | "
            f"{group['model_request_count']} | {int(usage.get('total_tokens', 0)):,} | "
            f"{_seconds(group.get('mean_model_response_seconds'))} | "
            f"{_seconds(group.get('mean_trial_elapsed_seconds'))} | "
            f"{group['code_preflight_rejections']} | {group['finish_rejections']} |"
        )
    for label, effect in report["pairwise_vs_native"].items():
        delta = effect["success_rate_delta_percentage_points"]
        p = effect["mcnemar_exact_two_sided_p"]
        lines.extend([
            "",
            f"- `{label}`：成功率差 {_signed(delta, ' 个百分点')}；"
            f"McNemar 精确双侧 p={p:.6f}；Token 差 "
            f"{effect['total_token_delta']:+,}；模型请求差 "
            f"{effect['total_model_request_delta']:+,}。"
            if delta is not None and p is not None else
            f"- `{label}`：尚无完整配对结果。",
        ])
    lines.extend(["", "## 产物完整性", ""])
    for name, audit in report["artifacts"].items():
        expected = audit["expected_trials"]
        lines.append(
            f"- {name}：视频 {len(audit['combined_video_trials'])}/{expected}，"
            f"响应 {len(audit['response_trials'])}/{expected}，"
            f"Token/时延指标 {len(audit['metrics_trials'])}/{expected}，"
            f"K=3 {len(audit['k3_feedback_trials'])}/{expected}，"
            f"世界状态 {len(audit['world_state_trials'])}/{expected}。"
        )
    return "\n".join(lines) + "\n"


def _seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}s"


def _signed(value: float | None, suffix: str) -> str:
    return "—" if value is None else f"{value:+.1f}{suffix}"


def save_report(report: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemini-3.8-flash")
    parser.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/formal_region_ab_20260929"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/formal_region_ab_20260929"))
    parser.add_argument("--seed-start", type=int, default=21)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    selected_names = [item.strip() for item in args.groups.split(",") if item.strip()]
    unknown = sorted(set(selected_names) - set(GROUPS))
    if unknown:
        parser.error(f"unknown groups: {', '.join(unknown)}")
    if args.run:
        if not model_endpoint_reachable(args.server_url):
            parser.error(f"model endpoint is unreachable: {args.server_url}")
        run_groups(
            model=args.model,
            server_url=args.server_url,
            trials=args.trials,
            temperature=0.0,
            group_configs={name: GROUPS[name] for name in selected_names},
            output_root=args.output_root,
            record_video=True,
            start_trial=args.seed_start,
        )
    report = collect_benchmark(args.output_root, args.model)
    save_report(report, args.report_dir)
    print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
