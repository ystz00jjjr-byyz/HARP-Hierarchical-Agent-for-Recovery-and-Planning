"""Recompute published HARP results from bundled records, without models or simulation."""
from pathlib import Path
import gzip
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    known = ROOT / 'results/known_ab_20260921'
    data = json.loads((known / 'results.json').read_text())
    published = json.loads((known / 'fault_coverage.json').read_text())
    totals = {'hybrid': 0, 'native': 0}
    both_count = 0
    pairs = 0
    for scenario, result in sorted(data['scenarios'].items()):
        rows = result['paired_trials']
        paired_ids = {f"trial_{row['trial']:03d}" for row in rows}
        assert len(paired_ids) == 10
        pairs += len(rows)
        injected = {}
        for group in totals:
            paths = sorted((known / 'trajectories' / scenario / group).glob('*.jsonl.gz'))
            assert len(paths) == 10
            injected[group] = set()
            for path in paths:
                with gzip.open(path, 'rt') as stream:
                    events = [json.loads(line) for line in stream if line.strip()]
                if any(event.get('event') == 'fault_injected' for event in events):
                    injected[group].add(path.name.removesuffix('.jsonl.gz'))
            assert sorted(injected[group]) == published[scenario][group]['injected_trial_ids']
            for row in rows:
                assert row['fingerprint_match']
                totals[group] += bool(row[group]['task_completed'])
        both = paired_ids & injected['hybrid'] & injected['native']
        both_count += len(both)
        print(f"{scenario}: both recorded injection {len(both)}/10")
    assert (pairs, totals['hybrid'], totals['native'], both_count) == (30, 28, 12, 13)
    print(f"Known A/B: hybrid {totals['hybrid']}/{pairs}; native {totals['native']}/{pairs}")

    smoke = ROOT / 'results/l4_smoke_seed301'
    metric = json.loads((smoke / 'trial/trial_metrics.json').read_text())
    assert metric['task_completed'] and metric['token_usage']['total_tokens'] == 11052
    with (smoke / 'trajectory.jsonl').open() as stream:
        ends = [row for line in stream if line.strip() and (row := json.loads(line)).get('event') == 'trial_end']
    assert len(ends) == 1
    terminal = ends[0]['world_state']['task_state']
    assert all(terminal[k] for k in ('object_inside_target_region', 'object_stable', 'gripper_open', 'task_completed'))
    assert not terminal['target_grasped']
    calls = [json.loads(p.read_text()) for p in sorted((smoke / 'model_calls').glob('*.json'))]
    total = sum(c['usage']['input_tokens'] + c['usage']['output_tokens'] for c in calls)
    assert len(calls) == 2 and total == 22034
    print('L4 seed 301: physical success; valid trial 11,052 tokens; both calls 22,034 tokens')
    print('PASS: archived summaries, injection coverage and L4 terminal evidence agree')


if __name__ == '__main__':
    main()
