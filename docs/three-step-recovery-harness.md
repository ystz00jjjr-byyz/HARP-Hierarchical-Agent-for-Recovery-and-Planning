# Three-step recovery harness

The initial harness is intentionally text-first and simulator-agnostic. It
wraps an existing CaP-X executor with:

1. a rolling planning window of exactly three `PlannedAction` objects;
2. ordered execution of the first action in that window;
3. a rule-based post-condition check from a text world-state mapping;
4. a transparent 0-100 score from pre/post world state;
5. an event stream suitable for the world-state recorder;
6. an optional retry/replan/abort policy.

## Minimal usage

```python
from capx.recovery import PlannedAction, ThreeStepRecoveryHarness

plan = [
    PlannedAction("align", "move_to", expected={"eef_aligned": True}),
    PlannedAction("grasp", "grasp", expected={"target_grasped": True}),
    PlannedAction("place", "place", expected={"object_at_target": True}),
]

def execute(action):
    # Replace with a call to the existing CaP-X env/API executor.
    return {"state": read_text_world_state_after(action)}

harness = ThreeStepRecoveryHarness(execute)
result = harness.run(plan, initial_state={"alignment_error": 0.08})
```

The current built-in primitive rules cover `move_to`, `goto_pose`, `align`,
`grasp`, `close_gripper`, `open_gripper`, `place`, `release`, and `rotate`.
Each rule accepts either a semantic boolean signal (for example
`target_grasped`) or a numeric error signal (for example `alignment_error`).

## Per-step scoring

`PrimitiveStepScorer` computes five components after every primitive:

- goal completion: 50%;
- progress from the pre-action state: 20%;
- execution quality: 15%;
- world-state confidence: 15%;
- safety penalty: a hard override.

The rule validator remains authoritative. A failed post-condition caps the
score below 70, and a collision, out-of-bounds state, wrong-object grasp, or
explicit safety violation forces the score to zero. Default routing is:

| Score / condition | Decision |
| --- | --- |
| safety violation | `abort` |
| execution error | `replan` |
| hard pass and score >= 85 | `pass` |
| hard pass and score >= 70 | `verify` |
| failed rule with positive progress or recoverable state | `retry` |
| other failed rule | `replan` |

Every `HarnessResult` contains a `scores` tuple, and every attempt emits a
`scoring.result` event with the total and component values. The recovery prompt
builder accepts the optional `score=` argument so the existing CaP-X multi-turn
model can see both the failed rule and its supporting numeric evidence.

## K-frame stability

`ExecutionOutcome.state_window` is now an active input rather than metadata.
The latest three frames are reduced before validation and scoring:

- booleans and categorical values use majority consensus;
- numeric values use the median;
- numeric agreement uses an absolute and relative tolerance;
- missing or disagreeing signals lower temporal confidence;
- unstable signal paths are attached to validation and scoring events.

The same aggregated state is used by the hard rule validator, the score, and
the next step's progress baseline. This prevents a single-frame grasp or
alignment glitch from independently changing the recovery decision.

```python
return ExecutionOutcome(
    state=latest_state,
    state_window=(frame_t_minus_2, frame_t_minus_1, latest_state),
)
```

## Multi-turn handoff

After a failed validation, call `build_recovery_prompt(...)` and append the
returned text to the existing multi-turn decision prompt. The LLM should only
choose a recovery intent. The executor and validator remain authoritative for
motion feasibility and post-conditions.

This first version does not add visual detectors, reset-skill learning, or
automatic code regeneration. Those can be connected after the text-signal
validation is stable on a small set of pick-place and stack tasks.

## Opt-in CaP-X trial feedback

The existing CaP-X trial loop executes generated code blocks, not typed action
primitives. It therefore does not pretend that a code block is a `grasp` or
`place` action. Instead, an opt-in bridge captures the environment's live
world-state provider after each code block, aggregates recent snapshots, and
adds the compact JSON state to the existing multi-turn prompt.

```yaml
text_world_state_feedback:
  enabled: true
  window_size: 3
  min_stable_fraction: 0.6667
  numeric_abs_tolerance: 0.01
  numeric_rel_tolerance: 0.05
  include_entities: true
  fail_open: true
  loop_guard_enabled: true
  max_same_failure_retries: 2
  min_progress_delta: 0.01
```

`fail_open: true` preserves the original CaP-X path when an environment has no
provider. Each enabled trial writes
`text_world_state_feedback_trial_XXX.json` under `output_dir`, containing raw
text snapshots and aggregated summaries. Full primitive-level scoring becomes
available when generated programs expose explicit action boundaries and return
`PlannedAction`/`ExecutionOutcome` records to the harness.

### Repeated-failure guard

The optional loop guard is enforced by the executor rather than left entirely
to the model. It uses semantic task progress and the ordered robot-API call
sequence:

1. the first two observations of the same non-progressing failure may retry;
2. the third requires a structurally different action sequence;
3. numeric-offset-only rewrites are rejected before execution;
4. if the switched strategy also makes no semantic progress, the trial stops.

Every feedback history entry stores `recovery_directive`. Trial responses use
`guard_abort` when the execution layer blocks a repeated strategy.

## Initial text-state validation

Run the deterministic fault-replay matrix before spending simulator time:

```bash
python -m scripts.text_world_state_recovery_probe \
  --output-dir reports/text_world_state_initial_validation
```

It covers nominal pick-place, alignment retry, grasp retry, unstable placement,
a one-frame false negative, and collision hard-stop. The generated report is an
MVP logic check only; simulator success-rate claims require seeded live runs.

## Fixed-seed Cube Stack comparison

Three paired five-seed configurations are available:

- `recovery_seeded_baseline.yaml`: no agent-visible text world state;
- `recovery_seeded_single_frame.yaml`: one current text-state snapshot;
- `recovery_seeded_k3.yaml`: a three-snapshot consensus window.

Run all groups and produce one comparison report with:

```bash
python -m scripts.run_text_world_state_seeded_experiment \
  --run --trials 5 --temperature 0
```

For the next strict paired comparison—baseline, unbounded K=3, and guarded
K=3—use:

```bash
python -m scripts.run_text_world_state_seeded_experiment \
  --profile guarded --run --trials 5 --temperature 0
```

This profile writes to a fresh output root and records a SHA-256 fingerprint of
the reset cube poses for every trial. The report marks physical pairing as
verified only when each trial fingerprint matches across all three groups.

## Token accounting

Each model response stores a normalized `token_usage` object in the trial's
`all_responses.json`:

- `prompt_tokens`;
- `completion_tokens`;
- `reasoning_tokens` (also included within completion tokens by providers);
- `total_tokens`.

The experiment report sums these fields independently for every group and
reports coverage as accounted model requests divided by all model requests.
Older responses without provider usage metadata remain unknown (`—`) instead
of being counted as zero.

Without `--run`, the command only rebuilds the report from existing outputs.
Trial IDs are reused across groups, so environment seeds are paired. The report
tracks task success, reward, code regeneration, feedback coverage, confidence,
unstable snapshots, and wrong-object grasp signals.
# Closed-loop execution guards

The production trial loop can enable deterministic gates with the top-level
`execution_guard` configuration. The gates run independently of candidate
reranking, so a low-cost single-candidate K=3 experiment can reject malformed
or unsafe decisions without generating four candidates per turn.

The guarded order is:

1. Parse each generated block with Python's AST before any robot API executes.
2. Reject empty, overlong, or expensive-loop blocks without changing the world.
3. Execute one short block and capture an action-aligned K=3 world-state window.
4. Check the configured post-condition for that nominal stage.
5. Reject `CONTINUE` when the stage failed and reject `FINISH` unless the
   environment reports `task_completed=true`.
6. Re-prompt in the same world state up to `max_decision_reprompts`, then abort
   safely rather than execute an unverified action.

For Restack, use
`env_configs/cube_restack/k3_closed_loop_guarded.yaml`. Its three hard
post-conditions are: green cleared and released; red grasped and lifted; red
stacked, stable, and released. Trial metrics include
`code_preflight_rejections`, `continue_rejections`, `finish_rejections`, and
`execution_guard_aborts`.
