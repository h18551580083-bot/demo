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
  isolation, CUDA, and lightweight authorization gates.
- `python -m cg_pipeline formal-train`: the sole formal training entry; it consumes a
  passing standalone preflight report and revalidates current data, split, CUDA,
  test-access, and authorization safety before preparation.

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

Formal training remains authorization- and preflight-gated, multi-seed, immutable,
and non-overwriting. Git state, tags, code identity, config identity, and report
checksums are not startup gates. Both modes construct only train and validation
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
  --authorization configs\formal_training_authorization.json `
  --output artifacts\preflight\phase1-training-b32-workers8-v1\preflight.json

python -m cg_pipeline formal-train `
  --config configs\phase1_baseline.toml `
  --data-root cam16_patch `
  --authorization configs\formal_training_authorization.json `
  --preflight-report artifacts\preflight\phase1-training-b32-workers8-v1\preflight.json
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

Phase 0 is closed. Formal entry consumes one independently generated passing
preflight report and rechecks current authorization, CUDA availability,
train/validation files, and split isolation before the first batch. The JSON passed
through `--authorization` is a lightweight authorization record.

The active formal run is `phase1-cam16-baseline-b32-v2`, with batch size 32,
2,487 train updates per complete epoch, and at most 49,740 updates over 20 epochs.
Run the controlled checklist in `docs/PHASE1_TRAINING_RUNBOOK.md` with the current
lightweight authorization record.

```powershell
$env:PYTHONPATH = 'E:\cg\src'
python -m cg_pipeline formal-train `
  --config configs\phase1_baseline.toml `
  --data-root cam16_patch `
  --authorization configs\formal_training_authorization.json `
  --preflight-report artifacts\preflight\phase1-training-b32-workers8-v1\preflight.json
```

Training never loads the test split. Test evaluation requires a later, separate
final-once authorization that names the data, checkpoint, and validation-threshold
identities.
