# RTX 4090 formal CAM16 training runbook — workers8 v1

## 1. Authorization boundary

This runbook authorizes the current CAM16 train/validation entry on an RTX 4090; it
does not itself start training. The JSON passed through `--authorization` is a lightweight
authorization record, not a Git, code, tag, commit, path, or config identity.

- Run ID: `phase1-cam16-baseline-b32-v2`.
- Test access remains false. Do not enumerate, hash, load, or evaluate the test
  effective split. Patient-level isolation remains `not_evaluated`.

Use a Linux cloud environment with one RTX 4090. Dataset download or split
modification is outside this runbook.

## 2. Set paths without changing the contract

From the repository root:

```bash
set -euo pipefail
export PYTHONPATH="$PWD/src"
export DATA_ROOT="/absolute/path/to/approved/cam16_patch"
export RELEASE_ID="phase1-training-b32-workers8-v1"
export RUN_ID="phase1-cam16-baseline-b32-v2"
export PREFLIGHT_REPORT="artifacts/preflight/${RELEASE_ID}/preflight.json"
export FORMAL_OUTPUT="artifacts/formal_runs/${RUN_ID}"
export SEED="3407"
test -d "$DATA_ROOT"
case "$SEED" in 1729|3407) ;; *) exit 1 ;; esac
```

`DATA_ROOT` must already contain the approved package. Do not create aliases for a
different manifest or data package.

## 3. Verify authorization

Confirm the authorization JSON is readable and retains
`formal_training_authorized = true`, `test_access_authorized = false`, and no
external blockers. Repository cleanliness, tags, commits, parents, and changed
paths are not training gates.

## 4. Verify environment and static contract

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 1
name = torch.cuda.get_device_name(0)
total_memory = torch.cuda.get_device_properties(0).total_memory
assert name == "NVIDIA GeForce RTX 4090", name
assert total_memory >= 23 * 1024**3, total_memory
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": name,
    "total_memory_bytes": total_memory,
})
PY

python -m pytest tests -q
python -m compileall -q src tests tools
python -m ruff check .
git diff --check
```

The tests, compile check, and Ruff must all pass. These checks do not access CAM16
test images or start training.

## 5. Confirm the selected seed destination is new

The Run ID and output root are shared by both approved seeds. The completed
`seed-1729` artifacts remain in place; only the selected seed directory must be absent:

```bash
test ! -e "$FORMAL_OUTPUT/seed-$SEED"
```

Do not delete, rename, merge, or overwrite an earlier formal artifact to make these
checks pass. Stop and preserve it.

## 6. Ensure the standalone preflight exists

```bash
if [ ! -e "$PREFLIGHT_REPORT" ]; then
  python -m cg_pipeline formal-preflight \
    --config configs/phase1_baseline.toml \
    --data-root "$DATA_ROOT" \
    --authorization configs/formal_training_authorization.json \
    --output "$PREFLIGHT_REPORT"
fi
```

The first invocation writes the report exclusively; later seed invocations reuse
that report and must not overwrite it. The preflight checks current config parsing,
train/validation manifest and patch existence, nonempty splits, isolation, CUDA,
the fixed frontend, Morlet numerical/spectral correctness, optimizer ownership,
determinism, authorization, and disabled test access.

Do not compute a test effective hash. Confirm the command exits zero and the report
shows `status = PASS`, `blocking_gates = []`, `training_started = false`, and
`test_split_accessed = false`.

## 7. Start formal training by consuming that report

```bash
python -m cg_pipeline formal-train \
  --config configs/phase1_baseline.toml \
  --data-root "$DATA_ROOT" \
  --authorization configs/formal_training_authorization.json \
  --preflight-report "$PREFLIGHT_REPORT" \
  --seed "$SEED"
```

Training does not rerun the full standalone preflight. Before its first batch it
checks that the report is readable, passed, unblocked, did not start training, and
did not access test. It revalidates current authorization, CUDA availability,
train/validation files, and split isolation. Report path, extra fields, Git state,
code identity, and config identity are not gates. This invocation runs only
`$SEED`; other approved seed directories are neither required nor modified.

## 8. Interruption and resume

Do not use `--resume` for a new run. After a genuine interruption, resume against
the existing output:

```bash
python -m cg_pipeline formal-train \
  --config configs/phase1_baseline.toml \
  --data-root "$DATA_ROOT" \
  --authorization configs/formal_training_authorization.json \
  --preflight-report "$PREFLIGHT_REPORT" \
  --seed "$SEED" \
  --resume
```

Resume requires a continuous checkpoint/report history and matching
source/effective manifest, fixed frontend, model, optimizer, seed, and epoch state.
It never repairs or overwrites artifacts.

## 9. Stop conditions and claims

Stop immediately on any nonzero command, invalid authorization, existing selected
seed destination,
non-4090 device, changed manifest/split, unsafe report result, or
resume discontinuity. Preserve the terminal output and artifacts for audit; do not
retry by changing the contract.

Training and validation may report only the frozen slide-level metrics. The sole
isolation statement remains `group_id/slide_id split isolation verified`. Do not
claim patient-level leakage prevention, do not access test data, and do not advance
the project phase from this runbook alone.
