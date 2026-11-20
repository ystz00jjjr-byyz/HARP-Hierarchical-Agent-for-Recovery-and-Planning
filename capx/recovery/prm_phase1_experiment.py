"""Collect the three-group fixed-seed Restack candidate-reranking experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from capx.recovery.seeded_experiment import collect_group


MODEL = "google/gemini-3.8-flash"
BASELINE_DIR = Path(
    "outputs/paired_capx_k3_two_tasks_gemini38_20260903/complex_cube_restack/"
    "google_gemini-3.8-flash/k3_action_aligned"
)
EXPERIMENT_ROOT = Path("outputs/restack_prm_phase1_gemini38_20260904")


def collect_phase1(
    *, baseline_dir: Path = BASELINE_DIR, experiment_root: Path = EXPERIMENT_ROOT
) -> dict[str, Any]:
    candidate_root = experiment_root / MODEL.replace("/", "_")
    group_dirs = {
        "k3_single_candidate": baseline_dir,
        "k3_random_4_candidates": candidate_root / "random_4_candidates",
        "k3_prm_proxy_4_candidates": candidate_root / "prm_proxy_4_candidates",
    }
    groups = {name: collect_group(path) for name, path in group_dirs.items()}
    for name, group in groups.items():
        group["candidate_selection"] = _collect_selection_stats(group_dirs[name], group)

    seed_lists = {
        name: [trial["trial"] for trial in group["trials"]]
        for name, group in groups.items()
    }
    paired_ids = bool(seed_lists["k3_single_candidate"]) and len(
        {tuple(seed_list) for seed_list in seed_lists.values()}
    ) == 1
    paired_trials = []
    if paired_ids:
        for index, trial_id in enumerate(seed_lists["k3_single_candidate"]):
            values = {
                name: group["trials"][index]
                for name, group in groups.items()
            }
            fingerprints = {
                value.get("initial_state_fingerprint") for value in values.values()
            }
            paired_trials.append({
                "trial": trial_id,
                "fingerprint_match": len(fingerprints) == 1 and None not in fingerprints,
                "groups": values,
            })
    return {
        "experiment": "restack_phase1_rule_guided_prm_reranking",
        "model": MODEL,
        "candidate_count": 4,
        "selection_seed": 20260904,
        "prm_type": "deterministic rule-guided proxy (not a trained neural PRM)",
        "paired_trial_ids": paired_ids,
        "physical_initial_state_pairing_verified": bool(paired_trials) and all(
            item["fingerprint_match"] for item in paired_trials
        ),
        "groups": groups,
        "paired_trials": paired_trials,
    }


def _collect_selection_stats(group_dir: Path, group: dict[str, Any]) -> dict[str, Any] | None:
    audits: list[dict[str, Any]] = []
    for trial in group.get("trials", []):
        pattern = (
            f"trial_{trial['trial']:02d}_sandboxrc_{trial['sandbox_rc']}_"
            f"reward_{trial['reward']:.3f}_taskcompleted_{int(trial['task_completed'])}"
        )
        matches = [path for path in group_dir.glob(pattern) if path.is_dir()]
        if not matches:
            continue
        response_path = max(matches, key=lambda path: path.stat().st_mtime_ns) / "all_responses.json"
        if not response_path.exists():
            continue
        for event in json.loads(response_path.read_text(encoding="utf-8")):
            audit = event.get("candidate_reranking")
            if isinstance(audit, dict):
                audits.append(audit)
    if not audits:
        return None
    selected_scores = [float(item["selected_score"]) for item in audits]
    candidates = [candidate for item in audits for candidate in item.get("candidates", [])]
    unique_per_event = [
        len({candidate.get("code", "") for candidate in item.get("candidates", [])})
        for item in audits
    ]
    return {
        "selection_events": len(audits),
        "generated_candidates": len(candidates),
        "hard_rejected_candidates": sum(
            bool((candidate.get("score") or {}).get("hard_reject"))
            for candidate in candidates
        ),
        "mean_selected_proxy_score": mean(selected_scores),
        "mean_unique_candidates_per_event": mean(unique_per_event),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Restack：K=3 候选重排序三组固定种子对照",
        "",
        f"模型：`{report['model']}`；物理种子：1–5；候选数：{report['candidate_count']}；"
        f"选择种子：{report['selection_seed']}。",
        "",
        "第三组是规则引导的 PRM proxy，不是已训练神经网络 PRM。成功只按环境 "
        "`task_completed=True` 判定。",
        "",
        f"试次编号完整配对：{'是' if report['paired_trial_ids'] else '否'}；"
        f"物理初态指纹完整匹配：{'是' if report['physical_initial_state_pairing_verified'] else '否'}。",
        "",
        "| 组别 | n | 成功率 | 平均耗时/次 | 总 token | 模型请求 | 平均重规划 | 候选事件 | 平均入选分 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, group in report["groups"].items():
        usage = group.get("token_usage") or {}
        selection = group.get("candidate_selection") or {}
        lines.append(
            f"| {name} | {group['trial_count']} | {_pct(group['task_success_rate'])} | "
            f"{_num(group.get('mean_trial_elapsed_seconds'))} s | "
            f"{_group_tokens(group)} | {group['model_request_count']} | "
            f"{_num(group.get('mean_regenerations'))} | "
            f"{_integer(selection.get('selection_events'))} | "
            f"{_num(selection.get('mean_selected_proxy_score'))} |"
        )
    lines.extend([
        "",
        "| 种子 | 初态匹配 | 单候选成功 | 随机4候选成功 | PRM-proxy成功 | 单候选 token | 随机 token | PRM token |",
        "| ---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: |",
    ])
    for pair in report["paired_trials"]:
        values = pair["groups"]
        lines.append(
            f"| {pair['trial']} | {'是' if pair['fingerprint_match'] else '否'} | "
            f"{_yes(values['k3_single_candidate']['task_completed'])} | "
            f"{_yes(values['k3_random_4_candidates']['task_completed'])} | "
            f"{_yes(values['k3_prm_proxy_4_candidates']['task_completed'])} | "
            f"{_trial_tokens(values['k3_single_candidate'])} | "
            f"{_trial_tokens(values['k3_random_4_candidates'])} | "
            f"{_trial_tokens(values['k3_prm_proxy_4_candidates'])} |"
        )
    lines.extend([
        "",
        "Token 说明：带 `≥` 的组或试次包含超时；数值只覆盖服务成功返回 usage 的请求，"
        "因此是费用下界。随机组种子 4 在协议修正前还有一次被中止的预检尝试，"
        "其已落盘额外消耗至少 193,828 token，不计入主表。",
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
    return "—" if value is None else f"{value:.2f}"


def _integer(value: int | None) -> str:
    return "—" if value is None else f"{value:,}"


def _yes(value: bool) -> str:
    return "是" if value else "否"


def _trial_tokens(trial: dict[str, Any]) -> str:
    usage = trial.get("token_usage") or {}
    prefix = "" if trial.get("metrics_complete", True) else "≥"
    return prefix + _integer(usage.get("total_tokens"))


def _group_tokens(group: dict[str, Any]) -> str:
    usage = group.get("token_usage") or {}
    prefix = "≥" if group.get("incomplete_metric_trials", 0) else ""
    return prefix + _integer(usage.get("total_tokens"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument(
        "--report-dir", type=Path,
        default=Path("reports/restack_prm_phase1_gemini38_20260904"),
    )
    args = parser.parse_args()
    report = collect_phase1(
        baseline_dir=args.baseline_dir, experiment_root=args.experiment_root
    )
    save_report(report, args.report_dir)
    print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
