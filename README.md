# Fixed H/E Morlet pathology classifier — Phase 1 entry

This simulation-only checkout implements the frozen fixed H/E first-order Morlet-
modulus classifier, electronic interaction/backend, strict existing-patch adapter,
training/evaluation contracts, and Decision 30 numerical-equivalence gate. It does
not download data, read or generate WSI candidates, or contain physical/clinical
deployment code.

## Documentation navigation

[AGENTS](AGENTS.md) is the only default project context. Load specialized documents
by task: [highest-level specification](docs/DEVELOPMENT_SPEC.md),
[fixed frontend](docs/specs/FIXED_OPTICAL_FRONTEND_SPEC.md),
[electronic backend](docs/specs/ELECTRONIC_BACKEND_SPEC.md),
[training](docs/TRAINING_PROTOCOL.md), [evaluation](docs/EVALUATION_PROTOCOL.md),
and relevant [terminology](CONTEXT.md) sections or ADRs.
Search [decisions](docs/DECISIONS.md) by topic/date; do not load the full history.
[Historical archive](docs/archive/README.md) and [refactor audit](docs/SPEC_REFACTOR_MAP.md)
are never default context. Reports do not authorize a phase transition or seed change.

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
- `python -m cg_pipeline formal-train --seed <SEED>`: the sole formal training entry;
  each invocation trains exactly one approved seed, consumes a
  passing standalone preflight report and revalidates current data, split, CUDA,
  test-access, and authorization safety before preparation.

## Verification

From `E:\cg` in PowerShell:

```powershell
$env:PYTHONPATH = 'E:\cg\src'
python -m pytest tests -q
python -m compileall -q src tests
python -m ruff check .
git diff --check
```

The current project smoke is `python -m pytest tests/test_pipeline_entrypoints.py -q`.
It checks CLI/config routing and pipeline control flow with synthetic files and
mocked training. Device numerical acceptance remains a separate check under
`cg_acceptance`, as required by [electronic backend section 3.14](docs/specs/ELECTRONIC_BACKEND_SPEC.md#314-required-cross-device-numerical-equivalence-reference-interface).

See [the repository audit](docs/archive/REPOSITORY_AUDIT_2026-09-07.md) for retained tests
and historical paths, [the contract reconciliation](docs/archive/CONTRACT_RECONCILIATION_2026-09-07.md)
for the current fact matrix and remaining human decision, and
[minimal maintenance governance](docs/MAINTENANCE.md) for Skills evaluation and
report-only automation designs. No Skills or scheduled jobs are installed.

Phase 0 total-acceptance tooling and closure documents are archived at local tag
`archive/pre-slim-2026-09-07`. Read historical files with
`git show archive/pre-slim-2026-09-07:<path>`. References in DECISIONS and the TBD
register to removed files refer to that snapshot. The tag has not been pushed.

## Training modes

Exploratory training accepts one seed and controlled engineering overrides without
requiring a release, tag, clean tree, or formal preflight. Its outputs are confined
to `artifacts/exploratory_runs/<run_id>/` and every report/checkpoint records
`formal_experiment=false` and `experiment_mode=exploratory_train`. Exploratory
results cannot be relabelled or automatically promoted to formal results.

Formal training remains authorization- and preflight-gated, immutable, and
non-overwriting. Seeds `1729` and `3407` are repeats under the same formal
baseline, while each invocation runs only the seed passed through `--seed`. Git
state, tags, code identity, config identity, and report
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
  --preflight-report artifacts\preflight\phase1-training-b32-workers8-v1\preflight.json `
  --seed 3407
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
Its approved seeds are `1729` and `3407`; they share this Run ID and
`artifacts/formal_runs/phase1-cam16-baseline-b32-v2/` output root.
Follow [TRAINING_PROTOCOL](docs/TRAINING_PROTOCOL.md) with the current lightweight
authorization record; the [runbook](docs/PHASE1_TRAINING_RUNBOOK.md) is a supporting checklist.

```powershell
$env:PYTHONPATH = 'E:\cg\src'
python -m cg_pipeline formal-train `
  --config configs\phase1_baseline.toml `
  --data-root cam16_patch `
  --authorization configs\formal_training_authorization.json `
  --preflight-report artifacts\preflight\phase1-training-b32-workers8-v1\preflight.json `
  --seed 3407
```

Training never loads the test split. Test evaluation requires a later, separate
final-once authorization that names the data, checkpoint, and validation-threshold
identities.
