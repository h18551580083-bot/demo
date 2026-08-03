# Fixed H/E Morlet pathology classifier — Phase 0

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
- `python -m cg_pipeline dry-run`: complete non-formal synthetic pipeline.
- `python -m cg_pipeline preflight`: formal configuration, data, model, precision,
  isolation, and release gates.
- `python -m cg_pipeline train`: the sole formal training entry; it calls preflight
  first and starts no batch when any gate fails.

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

## Phase 0 verification

```powershell
$env:PYTHONPATH = 'E:\cg\src'
python -m pytest tests -q
python -m compileall -q src tests
python -m ruff check .
git diff --check

python -m cg_pipeline dry-run `
  --config configs\phase0_dry_run.toml `
  --workspace-root .

python -m cg_pipeline preflight `
  --config configs\phase1_baseline.toml `
  --data-root cam16_patch `
  --release configs\phase0_release.json `
  --output artifacts\phase0_preflight.json
```

The dry run is synthetic and cannot support a performance claim. The formal
configuration is a preregistered starting baseline, not an empirically established
optimum. At the current checkout, preflight must fail before training because the
package has no validated reliable patient-to-slide mapping or attributable
provenance-reliability approval, and the Phase 0 release record is closed. See
`docs/PHASE0_ACCEPTANCE_MATRIX.md` for the exact external
input and revalidation command.

## Formal training entry

Do not execute while Phase 0 is open:

```powershell
$env:PYTHONPATH = 'E:\cg\src'
python -m cg_pipeline train `
  --config configs\phase1_baseline.toml `
  --data-root cam16_patch `
  --patient-mapping <approved-local-patient-mapping.csv> `
  --patient-mapping-approval <approved-provenance-artifact.json> `
  --release configs\phase0_release.json
```

Training never loads the test split. Test evaluation requires a later, separate
final-once authorization that names the frozen config, data, checkpoint, and
validation-threshold identities.
