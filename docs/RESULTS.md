# Results and evidence

## Known-anomaly experiment · 2026-09-21

Model: `google/gemini-3.8-flash`, temperature 0. Ten seeds per configured scenario, 30 paired tasks / 60 final records.

| Scenario | Hybrid completion | Native completion | Hybrid recorded injection | Native recorded injection | Both recorded injection |
|---|---:|---:|---:|---:|---:|
| `outside_container` | 9/10 | 6/10 | 10/10 | 6/10 | 6 pairs |
| `unstable_release` | 10/10 | 0/10 | 10/10 | 7/10 | 7 pairs |
| `workspace_edge_grasp_miss` | 9/10 | 6/10 | 1/10 | 0/10 | 0 pairs |
| Total | **28/30** | **12/30** | **21/30** | **13/30** | **13 pairs** |

Completion is the archived environment `task_completed` outcome. Matching object-initial-state fingerprints and configured faults establish the conditions for these end-to-end runs. Actual fault exposure differs and the methods do not branch from a single post-fault checkpoint. A trajectory with no recorded injection means no injection event was read from its parsable records.

The hybrid path recorded 24 model requests and 97,985 tokens, versus 201 requests and 1,280,203 tokens for the native control. Trial wall-clock means are 12.47 s and 198.47 s. These aggregates belong to this historical mixed scheduling comparison. Nineteen hybrid trials requested a model at least once.

### Evidence

- [Complete paired result table](../results/known_ab_20260921/results.json)
- [Recomputed fault coverage](../results/known_ab_20260921/fault_coverage.json)
- [All 60 trajectories, losslessly gzip-compressed](../results/known_ab_20260921/trajectories)
- [Final trial metrics, executed code and responses](../results/known_ab_20260921/trials)
- [Initial-state fingerprints](../results/known_ab_20260921/fingerprints)
- [Historical report](../results/known_ab_20260921/historical_report.zh-CN.md)

The historical report is preserved as an original artifact. Its same-fault wording and claims of strict recovery comparability are superseded by the exposure audit above. The experiment source was recorded as `458dd94` with a dirty working tree. `SOURCE_SNAPSHOT.json` inventories the current export independently, and does not reconstruct every uncommitted historical file.

## L4 Astra smoke · seed 301

Task: `region_place`. Exception checkpoint: `TARGET_POSITION_CHANGED`. Model: `gpt-6-astra`.

| Record | Requests | Input + output tokens | Model response | Trial wall time | Outcome |
|---|---:|---:|---:|---:|---|
| Valid seed 301 | 1 | **11,052** | 12.60 s | 37.31 s | Physically successful |
| Earlier schema-rejected preflight | 1 | **10,982** | 17.90 s | — | Aborted before a complete valid trial |
| Total calls | **2** | **22,034** | — | — | Kept as separate records |

The valid `trial_end` contains `object_inside_target_region=true`, `object_stable=true`, `gripper_open=true`, and `target_grasped=false`. A zero process exit alone does not satisfy this physical check. The model emitted a graph and the compiler produced two controlled grasp/place blocks.

A completed same-model Astra baseline has not been archived for this comparison. The single successful trial establishes an executed prototype and its recorded cost.

- [Trial metrics](../results/l4_smoke_seed301/trial/trial_metrics.json)
- [Physical trajectory](../results/l4_smoke_seed301/trajectory.jsonl)
- [Both model calls and raw graph outputs](../results/l4_smoke_seed301/model_calls)
- [Executed code](../results/l4_smoke_seed301/trial/code.py)
- [K=3 feedback](../results/l4_smoke_seed301/text_world_state_feedback_trial_301.json)
- [Original Chinese summary](../results/l4_smoke_seed301/historical_summary.zh-CN.md)

## Model response packaging

Model response text and prompt text are preserved in the compressed response files. Repeated embedded input-image bodies are replaced by `harp-input-image-sha256:` references, with media types and byte sizes in [the image hash index](../results/input_image_hashes.json). Original complete response files remain in the source experiment archives.

## Video interpretation

The known-policy and baseline videos are deterministic saved-code replays. The L4 video is the recorded valid trial. They are accelerated simulation captures. The animated L4 preview additionally slows the capture 2×. Video hashes and original paths are recorded in [video provenance](../assets/videos/provenance.json).
