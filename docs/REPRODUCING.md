# Installation and reproduction

## 1. Audit existing results

Python 3.10+ is sufficient. From the repository root:

```bash
python3 scripts/audit_published_results.py
```

This checks archived outcomes and fault exposure. It does not spend model credits or launch simulation.

## 2. Simulator-independent core tests

```bash
uv run --no-project --python 3.10 --with numpy --with pyyaml --with pytest \
  python -m pytest -q tests/test_recovery_compiler.py tests/test_three_step_recovery_harness.py
```

## 3. Full CaP-X runtime

Use Linux, Python 3.10 and the same Robosuite dependency family as the upstream project. L4 RGB-D execution additionally needs a supported GPU, SAM3 weights and the IK service. Current dependency versions are captured in `uv.lock`.

```bash
git submodule update --init --recursive
uv sync --extra robosuite --extra dev
git -C capx/third_party/robosuite apply --check ../../../docs/vendor/robosuite-local.patch
git -C capx/third_party/robosuite apply ../../../docs/vendor/robosuite-local.patch
```

The patch records the local Robosuite changes present during this source export. Upstream source and submodules retain their original package names and licenses. The exported runtime targets the HARP bin-placement and region-placement tasks; optional upstream real-robot asset bundles are obtained from the original project when needed.

SAM3 weights are supplied separately:

```bash
export SAM3_CHECKPOINT=/absolute/path/to/sam3.pt
export MUJOCO_GL=egl
```

Public YAML files resolve the checkpoint through `SAM3_CHECKPOINT`. Configure the model endpoint and credentials locally. Original key files, model weights and account state are not part of this repository.

## 4. Run an explicitly selected experiment

Known-policy runner (starts new model/simulator work only with `--run`):

```bash
uv run --no-sync python scripts/run_known_policy_ab.py --help
uv run --no-sync python scripts/run_known_policy_ab.py \
  --run --model google/gemini-3.8-flash --trials 10 --seed-start 1 \
  --output-root /absolute/path/to/new/run \
  --report-dir /absolute/path/to/new/report
```

The model must be available through your configured endpoint. Model names in the historical records identify the model used for those archived runs.

For structural-unknown experiments, inspect the checkpoint generator and runner options first:

```bash
uv run --no-sync python scripts/run_structural_unknown_ab.py --help
```

The L4 configuration requires a compatible failed checkpoint directory via `STRUCTURAL_UNKNOWN_CHECKPOINT_DIR`. The archived seed 301 checkpoint is supplied in `results/l4_smoke_seed301/checkpoints/`; checkpoint compatibility is validated by the runtime.

A new run uses the checked-out code and newly resolved service/model versions. The historical code was dirty; exact historical source reconstruction is not implied by checking out the exported runtime.

## 5. Storage

For the original workstation, new CAP-X reports, metrics, model responses, logs, trajectories and videos belong under `/media/byyz/PHILIPS/CAP-X/结果/<experiment>/` after checking mount point `/media/byyz/PHILIPS`, UUID `B60D-CA43` and write access. Use `/mnt/experiment-data/cap-x-main` as pending-sync staging when the drive is unavailable. Unix-socket/locking/symlink-dependent runs stay on local ext4 and are archived to the verified external drive afterward. Other machines should choose their own durable experiment location.
