# Offline task graph recovery

CAPX can run a task without sending post-failure world state to an LLM or
accepting dynamically generated recovery code. Enable the mode in a trial
configuration with an `offline_recovery` mapping:

```yaml
offline_recovery:
  enabled: true
  k_action: 3
  k_state: 3
  state_sampling:
    include_entities: true
    action_aligned_window: true
    wait_for_stability: true
  task_graph:
    initial: [clear_green]
    actions:
      - id: clear_green
        primitive: clear_precondition
        code: |
          # Static, reviewed primitive code. No generated recovery code is accepted.
          clear_green()
        success_conditions:
          restack_precondition_cleared: true
          gripper_open: true
        failure_routes:
          POSTCONDITION_FAILED: CLEAR_GREEN_FAILED
        next: grasp_red
      - id: grasp_red
        primitive: grasp_object
        code: |
          grasp_red()
        preconditions:
          restack_precondition_cleared: true
        success_conditions:
          target_grasped: true
          object_lifted: true
        next: place_red
      - id: place_red
        primitive: place_object
        code: |
          place_red()
        preconditions:
          target_grasped: true
        success_conditions:
          task_completed: true
          gripper_open: true
      - id: recover_clear_green
        primitive: recover_clear_precondition
        code: |
          recover_clear_green()
        success_conditions:
          restack_precondition_cleared: true
        next: grasp_red
  recovery_library:
    CLEAR_GREEN_FAILED: [recover_clear_green]
```

The `code` fields are the only executable code in this mode and must be
reviewed offline. The controller executes only the first item of its current
`[current, next_1, next_2]` window. After a successful action it retains valid
tail actions and follows the declared `next` edge to refill the window. After
a failure it looks up a predeclared route. Unknown failures produce a safe
stop and are never sent to a model.

`k_action` and `k_state` are intentionally fixed at `3`; changing either is
rejected by the controller so the experiment cannot silently mix the two
meanings of K.

## Compile traces

The minimal compiler uses successful trials for the nominal graph. A failed
trial becomes an executable recovery route only when its directory contains an
explicit `failure.json`, for example:

```json
{
  "exception_id": "GRASP_MISS",
  "failed_block_index": 1,
  "recovery_code_blocks": ["open_gripper()", "reobserve_red()", "alternate_grasp()"]
}
```

Compile the artifacts with:

```bash
python -m capx.recovery.compiler outputs/ \
  --graph task_graph.yaml \
  --library recovery_library.yaml
```

Unlabelled failures are deliberately not promoted to recovery routes; inspect
them and add `failure.json` only after the exception and recovery sequence have
been verified.

Set `"promote": false` in `failure.json` to preserve a labelled strategy under
`candidates` without making it executable. This is required for seed-specific
code, single-observation recoveries, and any strategy still awaiting replay.

## L4 structured skill graphs

L4 keeps model task understanding but removes model-authored robot code from
both initial planning and unknown-failure recovery. The model returns a
`recovery_graph_v1` object; the local compiler validates the exact skill order,
discrete parameters, active object role, and target binding, then emits the
same trusted two-stage `regrasp -> K=3 -> replace -> K=3` execution used by the
existing recovery runtime.

```json
{
  "version": "recovery_graph_v1",
  "steps": [
    {"skill": "safe_home"},
    {"skill": "reobserve_object"},
    {"skill": "sample_grasp", "mode": "orthogonal_cuboid"},
    {
      "skill": "regrasp",
      "approach_clearance": 0.1,
      "lift_clearance": 0.15
    },
    {"skill": "reobserve_target"},
    {"skill": "replace", "entry_clearance": 0.1},
    {"skill": "verify"}
  ]
}
```

The graph accepts no code, API calls, extra fields, arbitrary ordering, or
continuous free-form parameters. A valid first candidate may receive one
deterministic orthogonal fallback without another model request. Known
exceptions still use the formal recovery library first; infeasible geometry,
an exhausted model budget, or a failed bounded candidate ends at
`recover_budget_safe_stop`.

The first candidate configuration is
`env_configs/region_place/l4_model_led_recovery_graph.yaml`. It is intentionally
marked `candidate`; promote it only after real same-checkpoint and unseen-seed
validation.
