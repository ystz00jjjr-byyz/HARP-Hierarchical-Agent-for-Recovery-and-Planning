"""Record videos by deterministically replaying saved A/B trial code blocks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from capx.envs.configs.instantiate import instantiate
from capx.envs.launch import LaunchArgs
from capx.envs.runner import _start_api_servers, _stop_api_servers
from capx.recovery.known_policy_ab import GROUPS, SCENARIOS, _scenario_environment
from capx.utils.launch_utils import _load_config
from capx.utils.video_utils import _write_video


_TRIAL_DIR = re.compile(
    r"trial_(?P<trial>\d+)_sandboxrc_-?\d+_reward_-?[\d.]+_taskcompleted_[01]"
)
_BLOCK = re.compile(r"^# Code block \d+\s*$", re.MULTILINE)


def split_code_blocks(text: str) -> list[str]:
    return [part.strip() for part in _BLOCK.split(text) if part.strip()]


def terminal_trial_dirs(group_dir: Path) -> dict[int, Path]:
    """Select the terminal artifacts using the benchmark collector's rule."""
    grouped: dict[int, list[Path]] = {}
    for path in group_dir.glob("trial_*"):
        match = _TRIAL_DIR.fullmatch(path.name) if path.is_dir() else None
        if match:
            grouped.setdefault(int(match.group("trial")), []).append(path)

    selected: dict[int, Path] = {}
    for trial, paths in grouped.items():
        terminal = []
        for path in paths:
            summary = path / "summary.txt"
            if summary.exists():
                text = summary.read_text(encoding="utf-8", errors="replace")
                if "Environment response:" in text and "Task Completed:" in text:
                    terminal.append(path)
        selected[trial] = max(terminal or paths, key=lambda path: path.stat().st_mtime_ns)
    return selected


def replay_group(
    *, scenario: str, group: str, config_path: str, source_dir: Path,
    video_root: Path, trials: int,
) -> list[dict[str, Any]]:
    args = LaunchArgs(config_path=config_path, record_video=True, total_trials=trials)
    env_factory, _, api_servers = _load_config(args)
    server_procs = _start_api_servers(api_servers)
    env = instantiate(env_factory)
    selected = terminal_trial_dirs(source_dir)
    records: list[dict[str, Any]] = []
    try:
        for trial in range(1, trials + 1):
            source = selected.get(trial)
            if source is None:
                raise RuntimeError(f"missing saved trial {trial}: {source_dir}")
            code_path = source / "code.py"
            blocks = split_code_blocks(code_path.read_text(encoding="utf-8"))
            env.reset(seed=trial, options={"trial": trial})
            low_level = env.low_level_env
            low_level.enable_video_capture(True, clear=True)
            outcomes = []
            for index, block in enumerate(blocks):
                try:
                    _, reward, terminated, truncated, info = env.step(block)
                    outcomes.append({
                        "block": index,
                        "sandbox_rc": info.get("sandbox_rc"),
                        "reward": float(reward),
                        "task_completed": info.get("task_completed"),
                        "terminated": terminated,
                        "truncated": truncated,
                    })
                except BaseException as exc:
                    outcomes.append({"block": index, "runner_error": repr(exc)})
            frames = low_level.get_video_frames(clear=True)
            destination = video_root / scenario / group / f"trial_{trial:02d}"
            if not frames:
                raise RuntimeError(f"no captured frames for {scenario}/{group}/{trial}")
            _write_video(frames, str(destination), suffix="combined")
            record = {
                "scenario": scenario,
                "group": group,
                "trial": trial,
                "source_code": str(code_path),
                "video": str(destination / "video_combined.mp4"),
                "frame_count": len(frames),
                "block_count": len(blocks),
                "outcomes": outcomes,
            }
            (destination / "replay.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            records.append(record)
            print(f"Replayed {scenario}/{group}/trial_{trial:02d}: {len(frames)} frames")
    finally:
        if hasattr(env, "close"):
            env.close()
        _stop_api_servers(server_procs)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--model", default="google/gemini-3.8-flash")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS))
    parser.add_argument("--group", choices=tuple(GROUPS))
    args = parser.parse_args()
    model_dir = args.model.replace("/", "_")
    records: list[dict[str, Any]] = []
    for scenario, scenario_env in SCENARIOS.items():
        if args.scenario and scenario != args.scenario:
            continue
        with _scenario_environment(scenario_env):
            for group, config_path in GROUPS.items():
                if args.group and group != args.group:
                    continue
                records.extend(replay_group(
                    scenario=scenario,
                    group=group,
                    config_path=config_path,
                    source_dir=args.source_root / scenario / model_dir / group,
                    video_root=args.video_root,
                    trials=args.trials,
                ))
    args.video_root.mkdir(parents=True, exist_ok=True)
    (args.video_root / "manifest.json").write_text(
        json.dumps({"replay_mode": "saved_code_no_model", "trials": records},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(records)} replay videos to {args.video_root}")


if __name__ == "__main__":
    main()
