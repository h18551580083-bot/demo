# Decision 30 numerical-equivalence gate

This checkout implements the frozen cross-device acceptance contract in
`docs/DEVELOPMENT_SPEC.md` Section 3.14. It is simulation-only and does not
download or read pathology data.

## Public interfaces

- `compare_object(...)`: independent CPU binary64 comparator for one formal
  object.
- `run_calibration_gate(...)`: real non-CPU calibration across the two forward
  objects, three separate backward `dZ` fixtures, quarter-margin gate, negative
  controls, and audit identities.

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
