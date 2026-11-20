# HARP code map

| Component | Entry point |
|---|---|
| Task graphs, parameter binding, policy library, window controller | [`capx/recovery/offline.py`](../capx/recovery/offline.py) |
| Execution gates and stage postconditions | [`execution_guard.py`](../capx/recovery/execution_guard.py) |
| K=3 aggregation | [`stability.py`](../capx/recovery/stability.py) |
| World-state feedback and exception diagnosis | [`trial_feedback.py`](../capx/recovery/trial_feedback.py) |
| Loop and repeated-failure budget | [`loop_guard.py`](../capx/recovery/loop_guard.py) |
| Structured template / skill-graph validation and trusted compilation | [`capx/envs/trial.py`](../capx/envs/trial.py) |
| Compile successful traces and labelled recovery candidates | [`compiler.py`](../capx/recovery/compiler.py) |
| Bin-placement simulator and fault hooks | [`robosuite_bin_place.py`](../capx/envs/simulators/robosuite_bin_place.py) |
| Region placement environment | [`franka_region_place.py`](../capx/envs/tasks/franka/franka_region_place.py) |
| RGB-D control interface | [`control_sort_rgbd.py`](../capx/integrations/franka/control_sort_rgbd.py) |
| Recording schema, registry, events and compressed arrays | [`capx/recording/`](../capx/recording) |
| Known-policy A/B runner | [`run_known_policy_ab.py`](../scripts/run_known_policy_ab.py) |
| Structural-unknown checkpoint runner | [`run_structural_unknown_ab.py`](../scripts/run_structural_unknown_ab.py) |
| Read-only publication audit | [`audit_published_results.py`](../scripts/audit_published_results.py) |

## Configurations

- Policy-first known-anomaly experiment: [`paired_known_hybrid.yaml`](../env_configs/bin_place/paired_known_hybrid.yaml).
- Model-driven comparison: [`paired_known_native.yaml`](../env_configs/bin_place/paired_known_native.yaml).
- Controlled template variant: [`paired_known_l2.yaml`](../env_configs/bin_place/paired_known_l2.yaml).
- L3 model-led policy recovery variant: [`l3_model_led_policy_recovery.yaml`](../env_configs/region_place/l3_model_led_policy_recovery.yaml).
- L4 initial and recovery skill graphs: [`l4_model_led_recovery_graph.yaml`](../env_configs/region_place/l4_model_led_recovery_graph.yaml).
- Same-checkpoint L4 configuration: [`structural_unknown_l4.yaml`](../env_configs/region_place/structural_unknown_l4.yaml).
- Shared policy graph and recovery definitions: [`bin_place/`](../env_configs/bin_place), [`region_place/`](../env_configs/region_place).

Each YAML defines model authority, policy priority, state sampling and budget. Historical L1–L4 scheduling labels are distinct from task-difficulty labels in other benchmarks.

## Tests

The exported tests cover the offline controller, recovery compiler, K=3 aggregation, execution guards, known-policy summaries, structural checkpoints and task geometry. See [`tests/`](../tests). Simulator-independent core tests can be run with NumPy, PyYAML and pytest; integration tests additionally require the CaP-X runtime.
