"""Deterministic MVP probe for text-only world-state recovery."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from capx.recovery import ExecutionOutcome, PlannedAction, ThreeStepRecoveryHarness
from capx.recovery.validators import default_validator


@dataclass(frozen=True)
class ProbeScenario:
    name: str
    category: str
    actions: tuple[PlannedAction, ...]
    outcomes: dict[str, tuple[ExecutionOutcome, ...]]
    initial_state: dict[str, Any]
    expected_harness_status: str


class ReplayExecutor:
    def __init__(self, outcomes: dict[str, tuple[ExecutionOutcome, ...]]) -> None:
        self.outcomes = outcomes
        self.offsets: dict[str, int] = defaultdict(int)
        self.calls: list[str] = []

    def __call__(self, action: PlannedAction) -> ExecutionOutcome:
        sequence = self.outcomes[action.action_id]
        offset = self.offsets[action.action_id]
        self.offsets[action.action_id] += 1
        self.calls.append(action.action_id)
        return sequence[min(offset, len(sequence) - 1)]


def _outcome(state: dict[str, Any], *, window: tuple[dict[str, Any], ...] = ()):
    return ExecutionOutcome(state=state, state_window=window)


def build_scenarios() -> tuple[ProbeScenario, ...]:
    align = PlannedAction(
        "align", "move_to", expected={"alignment_error": {"lte": 0.02}}
    )
    grasp = PlannedAction("grasp", "grasp", expected={"target_grasped": True})
    place = PlannedAction(
        "place", "place",
        expected={"object_at_target": True, "object_stable": True},
    )
    return (
        ProbeScenario(
            "nominal_pick_place", "nominal", (align, grasp, place),
            {
                "align": (_outcome({"alignment_error": 0.01}),),
                "grasp": (_outcome({"target_grasped": True}),),
                "place": (_outcome({"object_at_target": True, "object_stable": True}),),
            },
            {"alignment_error": 0.08}, "success",
        ),
        ProbeScenario(
            "alignment_retry", "recoverable", (align, grasp),
            {
                "align": (
                    _outcome({"alignment_error": 0.06, "recoverable_environment": True}),
                    _outcome({"alignment_error": 0.01, "recoverable_environment": True}),
                ),
                "grasp": (_outcome({"target_grasped": True}),),
            },
            {"alignment_error": 0.10}, "success",
        ),
        ProbeScenario(
            "grasp_retry", "recoverable", (align, grasp, place),
            {
                "align": (_outcome({"alignment_error": 0.01}),),
                "grasp": (
                    _outcome({"target_grasped": False, "recoverable_environment": True}),
                    _outcome({"target_grasped": True, "recoverable_environment": True}),
                ),
                "place": (_outcome({"object_at_target": True, "object_stable": True}),),
            },
            {"alignment_error": 0.08}, "success",
        ),
        ProbeScenario(
            "unstable_place_retry", "recoverable", (grasp, place),
            {
                "grasp": (_outcome({"target_grasped": True}),),
                "place": (
                    _outcome({
                        "object_at_target": True,
                        "object_stable": False,
                        "recoverable_environment": True,
                    }),
                    _outcome({
                        "object_at_target": True,
                        "object_stable": True,
                        "recoverable_environment": True,
                    }),
                ),
            },
            {"target_grasped": False}, "success",
        ),
        ProbeScenario(
            "single_frame_grasp_false_negative", "sensor_noise", (grasp,),
            {
                "grasp": (_outcome(
                    {"target_grasped": False},
                    window=(
                        {"target_grasped": True},
                        {"target_grasped": True},
                        {"target_grasped": False},
                    ),
                ),),
            },
            {"target_grasped": False}, "success",
        ),
        ProbeScenario(
            "collision_hard_stop", "safety", (align,),
            {
                "align": (_outcome({"alignment_error": 0.01, "collision": True}),),
            },
            {"alignment_error": 0.08}, "aborted",
        ),
    )


def run_baseline(scenario: ProbeScenario) -> dict[str, Any]:
    executor = ReplayExecutor(scenario.outcomes)
    unsafe_pass = False
    status = "success"
    failed_action = None
    for action in scenario.actions:
        outcome = executor(action)
        validation = default_validator.validate(action, outcome.state)
        unsafe = any(bool(outcome.state.get(key)) for key in (
            "collision", "safety_violation", "out_of_bounds", "wrong_object_grasped"
        ))
        if validation.ok and unsafe:
            unsafe_pass = True
        if not validation.ok:
            status = "failed"
            failed_action = action.action_id
            break
    return {
        "status": status,
        "failed_action": failed_action,
        "unsafe_pass": unsafe_pass,
        "attempts": len(executor.calls),
        "call_trace": executor.calls,
    }


def run_harness(scenario: ProbeScenario) -> dict[str, Any]:
    executor = ReplayExecutor(scenario.outcomes)
    result = ThreeStepRecoveryHarness(executor, max_retries=1).run(
        list(scenario.actions), initial_state=scenario.initial_state
    )
    return {
        "status": result.status,
        "failed_action": result.failed_action,
        "unsafe_pass": any(
            score.safety_penalty > 0 and score.decision not in {"abort", "replan"}
            for score in result.scores
        ),
        "attempts": len(executor.calls),
        "call_trace": executor.calls,
        "score_trace": [asdict(score) for score in result.scores],
        "decision_trace": [score.decision for score in result.scores],
    }


def run_probe() -> dict[str, Any]:
    rows = []
    for scenario in build_scenarios():
        baseline = run_baseline(scenario)
        harness = run_harness(scenario)
        rows.append({
            "scenario": scenario.name,
            "category": scenario.category,
            "expected_harness_status": scenario.expected_harness_status,
            "expectation_met": harness["status"] == scenario.expected_harness_status,
            "baseline": baseline,
            "harness": harness,
        })

    task_rows = [row for row in rows if row["category"] != "safety"]
    recoverable_rows = [row for row in rows if row["category"] == "recoverable"]
    return {
        "experiment": "text_world_state_recovery_initial_probe",
        "scenario_count": len(rows),
        "summary": {
            "expected_outcomes_met": sum(row["expectation_met"] for row in rows),
            "baseline_task_successes": sum(
                row["baseline"]["status"] == "success" for row in task_rows
            ),
            "harness_task_successes": sum(
                row["harness"]["status"] == "success" for row in task_rows
            ),
            "recoverable_faults_recovered": sum(
                row["harness"]["status"] == "success" for row in recoverable_rows
            ),
            "recoverable_fault_count": len(recoverable_rows),
            "baseline_unsafe_passes": sum(row["baseline"]["unsafe_pass"] for row in rows),
            "harness_unsafe_passes": sum(row["harness"]["unsafe_pass"] for row in rows),
        },
        "scenarios": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# 文本世界状态异常恢复：初始验证",
        "",
        "这是确定性状态回放验证，不代表真实仿真成功率。",
        "",
        "| 场景 | 类别 | 无恢复基线 | Recovery Harness | 决策轨迹 | 符合预期 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["scenarios"]:
        lines.append(
            f"| {row['scenario']} | {row['category']} | {row['baseline']['status']} "
            f"| {row['harness']['status']} | "
            f"{', '.join(row['harness']['decision_trace'])} | "
            f"{'是' if row['expectation_met'] else '否'} |"
        )
    lines.extend([
        "",
        "## 结果汇总",
        "",
        f"- 符合预期：{summary['expected_outcomes_met']}/{report['scenario_count']}",
        f"- 非安全类任务成功：基线 {summary['baseline_task_successes']}/5；"
        f"Harness {summary['harness_task_successes']}/5",
        f"- 可恢复故障恢复：{summary['recoverable_faults_recovered']}/"
        f"{summary['recoverable_fault_count']}",
        f"- 不安全误通过：基线 {summary['baseline_unsafe_passes']}；"
        f"Harness {summary['harness_unsafe_passes']}",
        "",
        "## 结论边界",
        "",
        "本实验验证确定性恢复路由、K 帧聚合、重试和安全硬门控。"
        "下一阶段应使用实时任务状态信号开展小规模固定种子仿真实验。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    report = run_probe()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.output_dir / "report.md").write_text(
            render_markdown(report), encoding="utf-8"
        )
    print(render_markdown(report), end="")


if __name__ == "__main__":
    main()
