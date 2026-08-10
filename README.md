# Fixed H/E Morlet pathology classifier — Phase 1 entry

This simulation-only checkout implements the frozen fixed H/E first-order Morlet-
modulus classifier, electronic interaction/backend, strict existing-patch adapter,
training/evaluation contracts, and Decision 30 numerical-equivalence gate. It does
not download data, read or generate WSI candidates, or contain physical/clinical
deployment code.

## Public interfaces

- `compare_object(...)`: independent CPU binary64 comparator for one formal
  object.
- `run_calibration_gate(...)`: real non-CPU calibration across the two forward
  objects, three separate backward `dZ` fixtures, quarter-margin gate, negative
  controls, and audit identities.
- `python -m cg_pipeline exploratory-train`: real CAM16 train/validation for
  profiling, performance tuning, and other explicitly non-formal experiments.
- `python -m cg_pipeline formal-preflight`: formal configuration, data, model, precision,
  isolation, and release gates.
- `python -m cg_pipeline formal-train`: the sole formal training entry; it consumes the
  release-bound standalone preflight report, revalidates every live identity, and
  starts no batch when any gate fails.

## Verification

From `E:\cg` in PowerShell:

```powershell
$env:PYTHONPATH = 'E:\cg\src'
python -m pytest tests -q
python -m cg_acceptance `
  --device cuda:0 `
  --mode local_smoke `
  --run-id decision30-local-smoke-autograd-cuda0-20260731 `
  --output artifacts\decision30_local_smoke_autograd_cuda0_20260731.json
python -m compileall -q src tests
python -m ruff check .
git diff --check
```

The CLI exits with status 1 unless the real non-CPU environment, every formal
object, every quarter-margin check, zero-variance evidence, input identity, and
every negative control pass.

`local_smoke` is evidence for the local implementation only and cannot close
formal acceptance. `formal_acceptance` is available through
`run_calibration_gate(...)` and requires an explicitly pre-registered fixture,
formal input shape and hashes, plus the expected environment identity. Reports
are created exclusively; an existing JSON file is never overwritten.

## Training modes

Exploratory training accepts one seed and controlled engineering overrides without
requiring a release, tag, clean tree, or formal preflight. Its outputs are confined
to `artifacts/exploratory_runs/<run_id>/` and every report/checkpoint records
`formal_experiment=false` and `experiment_mode=exploratory_train`. Exploratory
results cannot be relabelled or automatically promoted to formal results.

Formal training remains release-bound, hash-bound, preflight-gated, multi-seed,
immutable, and non-overwriting. Both modes construct only train and validation
datasets; neither CLI exposes test access.

```powershell
$env:PYTHONPATH = 'E:\cg\src'
python -m cg_pipeline exploratory-train `
  --config configs\exploratory_train.toml `
  --data-root cam16_patch `
  --device cuda:0 `
  --seed 1729 `
  --output artifacts/exploratory_runs/profile-1729 `
  --run-id profile-1729 `
  --batch-size 32 `
  --num-workers 4 `
  --max-epochs 1 `
  --max-steps 100
```

## Formal verification and training

```powershell
$env:PYTHONPATH = 'E:\cg\src'
python -m pytest tests -q
python -m compileall -q src tests
python -m ruff check .
git diff --check

python -m cg_pipeline formal-preflight `
  --config configs\phase1_baseline.toml `
  --data-root cam16_patch `
  --release configs\phase1_training_release_b32_v3.json `
  --output artifacts\preflight\phase1-training-b32-v3\preflight.json

python -m cg_pipeline formal-train `
  --config configs\phase1_baseline.toml `
  --data-root cam16_patch `
  --release configs\phase1_training_release_b32_v3.json `
  --preflight-report artifacts\preflight\phase1-training-b32-v3\preflight.json
```

The exploratory configuration is non-formal. The formal configuration is a
preregistered starting baseline, not an empirically established
optimum. Phase 0 is closed and preflight authorizes the frozen CAM16 Phase 1
train/validation entry when every applicable gate passes. The only isolation
statement is `group_id/slide_id split isolation verified`; machine-readable state
remains `patient_level_isolation = not_evaluated` and
`patient_level_claim_allowed = false`. Patient mapping and approval files are not
preflight inputs.

## Formal training entry

Phase 0 is closed. Formal entry consumes one independently generated preflight
report and rechecks its current code, release, config, and train/validation data
identities before the first batch:

The v3 commands below become active only when the exact v3 release JSON exists in
the three-file release commit, its decision is recorded, and the annotated v3 tag
resolves to that commit. The validator enforces those conditions; before publication
the prior release remains the repository's latest published evidence.

The active formal identity is `phase1-cam16-baseline-b32-v2`, with batch size 32,
2,487 train updates per complete epoch, and at most 49,740 updates over 20 epochs.
Run the controlled checklist in `docs/PHASE1_TRAINING_RUNBOOK.md`; do not reuse the
historical unbound `configs/phase0_release.json` for this revised contract.

```powershell
$env:PYTHONPATH = 'E:\cg\src'
python -m cg_pipeline formal-train `
  --config configs\phase1_baseline.toml `
  --data-root cam16_patch `
  --release configs\phase1_training_release_b32_v3.json `
  --preflight-report artifacts\preflight\phase1-training-b32-v3\preflight.json
```

Training never loads the test split. Test evaluation requires a later, separate
final-once authorization that names the frozen config, data, checkpoint, and
validation-threshold identities.
