"""Run and audit the structural-unknown same-checkpoint paired experiment."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from capx.recovery.known_policy_ab import _paired_effect
from capx.recovery.seeded_experiment import (
    collect_group,
    model_endpoint_reachable,
    run_groups,
)
from capx.recovery.structural_unknown_ab import (
    ANOMALY_CLASSES,
    DEFAULT_SEED_START,
    DEFAULT_SEEDS_PER_CLASS,
    anomaly_for_seed,
    generate_checkpoints,
)


GROUP_CONFIGS = {
    "l4": "env_configs/region_place/structural_unknown_l4.yaml",
    "l3": "env_configs/region_place/structural_unknown_l3.yaml",
    "native": "env_configs/region_place/structural_unknown_native.yaml",
}
_TRIAL_DIR = re.compile(
    r"trial_(?P<trial>\d+)_sandboxrc_(?P<sandbox_rc>-?\d+)_"
    r"reward_(?P<reward>-?[\d.]+)_taskcompleted_(?P<completed>[01])"
)
_INFRA_PATTERNS = (
    "read timed out",
    "connecttimeout",
    "connectionerror",
    "connection refused",
    "httpconnectionpool",
    "trial timed out",
    "timed out after",
    "openrouter api error",
)


def _terminal_trial_dirs(group_dir: Path) -> dict[int, Path]:
    candidates: dict[int, list[Path]] = {}
    for path in group_dir.glob("trial_*"):
        if not path.is_dir():
            continue
        match = _TRIAL_DIR.fullmatch(path.name)
        if match is not None:
            candidates.setdefault(int(match.group("trial")), []).append(path)
    selected = {}
    for trial, paths in candidates.items():
        complete = [
            path for path in paths
            if (path / "summary.txt").exists()
            and "Task Completed:" in (path / "summary.txt").read_text(
                encoding="utf-8", errors="replace"
            )
        ]
        selected[trial] = max(complete or paths, key=lambda path: path.stat().st_mtime_ns)
    return selected


def _world_truth(group_dir: Path, trial: int) -> dict[str, Any]:
    trajectory = (
        group_dir / "world_state" / f"trial_{trial:03d}" / "trajectory.jsonl"
    )
    result: dict[str, Any] = {
        "available": False,
        "physical_success": False,
        "checkpoint_sha256": None,
        "anomaly_class": None,
    }
    if not trajectory.exists():
        return result
    for line in trajectory.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("event") == "reset":
            checkpoint = (entry.get("metadata") or {}).get("recovery_checkpoint") or {}
            result["checkpoint_sha256"] = checkpoint.get("checkpoint_sha256")
            result["anomaly_class"] = checkpoint.get("anomaly_class")
            result["feasibility_precheck"] = checkpoint.get(
                "physical_feasibility_precheck"
            )
        if entry.get("event") == "trial_end":
            task_state = ((entry.get("world_state") or {}).get("task_state") or {})
            result.update({
                "available": True,
                "physical_success": bool(task_state.get("task_completed", False)),
                "object_inside_target_region": bool(
                    task_state.get("object_inside_target_region", False)
                ),
                "object_stable": bool(task_state.get("object_stable", False)),
                "gripper_open": bool(task_state.get("gripper_open", False)),
                "target_grasped": bool(task_state.get("target_grasped", False)),
            })
    return result


def _infrastructure_failure(trial_dir: Path | None) -> tuple[bool, str | None]:
    if trial_dir is None:
        return True, "missing_trial_artifacts"
    summary = (
        (trial_dir / "summary.txt").read_text(encoding="utf-8", errors="replace")
        if (trial_dir / "summary.txt").exists() else ""
    )
    lowered = summary.lower()
    matched = next((pattern for pattern in _INFRA_PATTERNS if pattern in lowered), None)
    return (matched is not None, matched)


def _group_trials(group_dir: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    aggregate = collect_group(group_dir)
    trial_dirs = _terminal_trial_dirs(group_dir)
    by_trial = {int(item["trial"]): dict(item) for item in aggregate["trials"]}
    for trial, item in by_trial.items():
        truth = _world_truth(group_dir, trial)
        infra, infra_reason = _infrastructure_failure(trial_dirs.get(trial))
        item.update({
            "physical_truth": truth,
            "physical_success": bool(truth["physical_success"]),
            "infrastructure_failure": infra,
            "infrastructure_failure_reason": infra_reason,
            "anomaly_class": truth.get("anomaly_class") or anomaly_for_seed(trial),
        })
    return aggregate, by_trial


def _effect_for_trials(
    l3: dict[int, dict[str, Any]],
    native: dict[int, dict[str, Any]],
    trials: list[int],
) -> dict[str, Any]:
    pairs = []
    for trial in trials:
        hybrid = dict(l3[trial])
        baseline = dict(native[trial])
        hybrid["task_completed"] = hybrid["physical_success"]
        baseline["task_completed"] = baseline["physical_success"]
        hybrid["total_tokens"] = int(
            (hybrid.get("token_usage") or {}).get("total_tokens", 0)
        )
        baseline["total_tokens"] = int(
            (baseline.get("token_usage") or {}).get("total_tokens", 0)
        )
        pairs.append({"hybrid": hybrid, "native": baseline})
    return _paired_effect(pairs)


def collect_benchmark(
    output_root: str | Path,
    *,
    model: str,
    seed_start: int = DEFAULT_SEED_START,
    seeds_per_class: int = DEFAULT_SEEDS_PER_CLASS,
    trial_count: int | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    model_dir = output_root / model.replace("/", "_")
    l3_aggregate, l3 = _group_trials(model_dir / "l3")
    native_aggregate, native = _group_trials(model_dir / "native")
    l4_dir = model_dir / "l4"
    l4_aggregate, l4 = _group_trials(l4_dir) if l4_dir.exists() else ({}, {})
    expected_count = (
        len(ANOMALY_CLASSES) * seeds_per_class
        if trial_count is None else trial_count
    )
    if expected_count < 1:
        raise ValueError("trial_count must be positive")
    expected = list(range(seed_start, seed_start + expected_count))
    common = [trial for trial in expected if trial in l3 and trial in native]
    paired_hashes = {
        trial: bool(l3[trial]["physical_truth"].get("checkpoint_sha256"))
        and l3[trial]["physical_truth"].get("checkpoint_sha256")
        == native[trial]["physical_truth"].get("checkpoint_sha256")
        for trial in common
    }
    infra_excluded = [
        trial for trial in common
        if not l3[trial]["infrastructure_failure"]
        and not native[trial]["infrastructure_failure"]
    ]
    per_class = {}
    for anomaly_class in ANOMALY_CLASSES:
        class_trials = [
            trial for trial in common
            if anomaly_for_seed(
                trial, seed_start=seed_start, seeds_per_class=seeds_per_class
            ) == anomaly_class
        ]
        usable = [trial for trial in class_trials if trial in infra_excluded]
        per_class[anomaly_class] = {
            "trial_ids": class_trials,
            "first_attempt": {
                "l3_successes": sum(l3[t]["physical_success"] for t in class_trials),
                "native_successes": sum(native[t]["physical_success"] for t in class_trials),
                "paired_trials": len(class_trials),
            },
            "infrastructure_excluded": {
                "l3_successes": sum(l3[t]["physical_success"] for t in usable),
                "native_successes": sum(native[t]["physical_success"] for t in usable),
                "paired_trials": len(usable),
            },
        }
    report = {
        "experiment": "structural_unknown_region_place_same_checkpoint_ab",
        "model": model,
        "seed_start": seed_start,
        "seeds_per_class": seeds_per_class,
        "expected_trial_count_per_group": len(expected),
        "paired_trial_ids": common,
        "checkpoint_hash_match_count": sum(paired_hashes.values()),
        "all_checkpoint_hashes_match": bool(common) and all(paired_hashes.values()),
        "missing_trial_ids": {
            "l3": [trial for trial in expected if trial not in l3],
            "native": [trial for trial in expected if trial not in native],
        },
        "first_attempt": {
            "paired_trials": len(common),
            "l3_physical_successes": sum(l3[t]["physical_success"] for t in common),
            "native_physical_successes": sum(
                native[t]["physical_success"] for t in common
            ),
            "paired_effect": _effect_for_trials(l3, native, common) if common else None,
        },
        "infrastructure_excluded": {
            "paired_trials": len(infra_excluded),
            "excluded_trial_ids": [t for t in common if t not in infra_excluded],
            "l3_physical_successes": sum(
                l3[t]["physical_success"] for t in infra_excluded
            ),
            "native_physical_successes": sum(
                native[t]["physical_success"] for t in infra_excluded
            ),
            "paired_effect": (
                _effect_for_trials(l3, native, infra_excluded)
                if infra_excluded else None
            ),
        },
        "per_anomaly_class": per_class,
        "groups": {"l3": l3_aggregate, "native": native_aggregate},
        "trial_results": {
            "l3": [l3[t] for t in sorted(l3)],
            "native": [native[t] for t in sorted(native)],
        },
    }
    if l4:
        l4_common = [trial for trial in common if trial in l4]
        l4_hashes = {
            trial: bool(l4[trial]["physical_truth"].get("checkpoint_sha256"))
            and l4[trial]["physical_truth"].get("checkpoint_sha256")
            == l3[trial]["physical_truth"].get("checkpoint_sha256")
            == native[trial]["physical_truth"].get("checkpoint_sha256")
            for trial in l4_common
        }
        report["l4_comparison"] = {
            "paired_trial_ids": l4_common,
            "checkpoint_hash_match_count": sum(l4_hashes.values()),
            "all_checkpoint_hashes_match": bool(l4_common) and all(l4_hashes.values()),
            "l4_physical_successes": sum(l4[t]["physical_success"] for t in l4_common),
            "l4_vs_l3": _effect_for_trials(l4, l3, l4_common),
            "l4_vs_native": _effect_for_trials(l4, native, l4_common),
        }
        report["groups"]["l4"] = l4_aggregate
        report["trial_results"]["l4"] = [l4[t] for t in sorted(l4)]
    return report


def render_markdown(report: dict[str, Any]) -> str:
    first = report["first_attempt"]
    clean = report["infrastructure_excluded"]
    has_l4 = "l4" in report["groups"]
    resource_rows = []
    if has_l4:
        resource_rows.append(_resource_row("L4", report["groups"]["l4"]))
    resource_rows.extend([
        _resource_row("L3", report["groups"]["l3"]),
        _resource_row("原生 CAPX", report["groups"]["native"]),
    ])
    lines = [
        "# 结构性未知异常恢复：L4、冻结 L3 与原生 CAPX 同检查点实验"
        if has_l4 else
        "# 结构性未知异常恢复：冻结 L3 与原生 CAPX 同检查点配对实验",
        "",
        "每个 seed 从同一个异常后 MuJoCo 检查点开始；两组使用同一模型、RGB-D K=3、图像、仿真时域和最多 5 次模型请求。L3 只返回结构化模板与有界参数，原生 CAPX 可自由生成受执行门禁约束的恢复代码。最终成功以 `trial_end` 的 MuJoCo 物理真值为准。",
        "",
        f"检查点哈希匹配：{report['checkpoint_hash_match_count']}/{len(report['paired_trial_ids'])}。",
        "",
        "| 统计口径 | 有效配对 | L3 成功 | 原生 CAPX 成功 |",
        "| --- | ---: | ---: | ---: |",
        f"| 首次尝试（基础设施故障计失败） | {first['paired_trials']} | {first['l3_physical_successes']} | {first['native_physical_successes']} |",
        f"| 排除基础设施故障 | {clean['paired_trials']} | {clean['l3_physical_successes']} | {clean['native_physical_successes']} |",
        "",
        "## 资源消耗",
        "",
        "| 方法 | 试次 | 模型请求 | 总 Token | 平均模型响应 | 平均总耗时 | Token 记录覆盖率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *resource_rows,
        "",
        "## 分异常结果",
        "",
        "| 异常 | 首次 L3 | 首次原生 | 排故后 L3 | 排故后原生 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for anomaly, values in report["per_anomaly_class"].items():
        raw = values["first_attempt"]
        filtered = values["infrastructure_excluded"]
        lines.append(
            f"| `{anomaly}` | {raw['l3_successes']}/{raw['paired_trials']} | "
            f"{raw['native_successes']}/{raw['paired_trials']} | "
            f"{filtered['l3_successes']}/{filtered['paired_trials']} | "
            f"{filtered['native_successes']}/{filtered['paired_trials']} |"
        )
    if has_l4:
        comparison = report["l4_comparison"]
        paired = comparison["paired_trial_ids"]
        l3_successes = sum(
            item["physical_success"] for item in report["trial_results"]["l3"]
            if item["trial"] in paired
        )
        native_successes = sum(
            item["physical_success"] for item in report["trial_results"]["native"]
            if item["trial"] in paired
        )
        lines.extend([
            "",
            "## L4 五种子冒烟对比",
            "",
            f"三组同检查点哈希匹配：{comparison['checkpoint_hash_match_count']}/{len(paired)}。",
            "",
            "| 方法 | 物理成功 | 配对试次 |",
            "| --- | ---: | ---: |",
            f"| L4 | {comparison['l4_physical_successes']} | {len(paired)} |",
            f"| 冻结 L3 | {l3_successes} | {len(paired)} |",
            f"| 原生 CAPX | {native_successes} | {len(paired)} |",
        ])
    lines.extend([
        "",
        "## 基础设施故障",
        "",
        "排除的配对 seed："
        + (", ".join(map(str, clean["excluded_trial_ids"])) or "无")
        + "。",
    ])
    return "\n".join(lines) + "\n"


def _resource_row(label: str, group: dict[str, Any]) -> str:
    usage = group.get("token_usage") or {}
    model_seconds = group.get("mean_model_response_seconds")
    elapsed_seconds = group.get("mean_trial_elapsed_seconds")
    coverage = group.get("token_usage_coverage")
    return (
        f"| {label} | {group['trial_count']} | {group['model_request_count']} | "
        f"{int(usage.get('total_tokens', 0)):,} | "
        f"{'—' if model_seconds is None else f'{model_seconds:.2f}s'} | "
        f"{'—' if elapsed_seconds is None else f'{elapsed_seconds:.2f}s'} | "
        f"{'—' if coverage is None else f'{100 * coverage:.1f}%'} |"
    )


def save_report(report: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
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
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("outputs/structural_unknown_ab/checkpoints"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/structural_unknown_ab/runs"),
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=Path("reports/structural_unknown_ab"),
    )
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seeds-per-class", type=int, default=DEFAULT_SEEDS_PER_CLASS)
    parser.add_argument("--run-start", type=int, default=None)
    parser.add_argument("--run-trials", type=int, default=None)
    parser.add_argument(
        "--report-trials", type=int, default=None,
        help="limit the report to this many consecutive seeds without changing class mapping",
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--groups", default="l3,native")
    args = parser.parse_args()

    if args.prepare:
        generate_checkpoints(
            args.checkpoint_dir,
            seed_start=args.seed_start,
            seeds_per_class=args.seeds_per_class,
            object_shape="cuboid",
        )
    if args.run:
        if not (args.checkpoint_dir / "manifest.json").exists():
            parser.error("checkpoint manifest is missing; run with --prepare first")
        if not model_endpoint_reachable(args.server_url):
            parser.error(f"model endpoint is unreachable: {args.server_url}")
        selected = [item.strip() for item in args.groups.split(",") if item.strip()]
        unknown = sorted(set(selected) - set(GROUP_CONFIGS))
        if unknown:
            parser.error(f"unknown groups: {', '.join(unknown)}")
        old_checkpoint_dir = os.environ.get("STRUCTURAL_UNKNOWN_CHECKPOINT_DIR")
        os.environ["STRUCTURAL_UNKNOWN_CHECKPOINT_DIR"] = str(
            args.checkpoint_dir.resolve()
        )
        try:
            run_start = args.seed_start if args.run_start is None else args.run_start
            run_trials = (
                len(ANOMALY_CLASSES) * args.seeds_per_class
                if args.run_trials is None else args.run_trials
            )
            run_groups(
                model=args.model,
                server_url=args.server_url,
                trials=run_trials,
                temperature=0.0,
                group_configs={name: GROUP_CONFIGS[name] for name in selected},
                output_root=args.output_root,
                record_video=True,
                start_trial=run_start,
            )
        finally:
            if old_checkpoint_dir is None:
                os.environ.pop("STRUCTURAL_UNKNOWN_CHECKPOINT_DIR", None)
            else:
                os.environ["STRUCTURAL_UNKNOWN_CHECKPOINT_DIR"] = old_checkpoint_dir
    report = collect_benchmark(
        args.output_root,
        model=args.model,
        seed_start=args.seed_start,
        seeds_per_class=args.seeds_per_class,
        trial_count=args.report_trials,
    )
    save_report(report, args.report_dir)
    print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
