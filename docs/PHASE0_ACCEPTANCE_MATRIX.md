# Phase 0 acceptance evidence matrix

Audit date: 2026-08-03. Scope is the eleven deliverables in
`docs/DEVELOPMENT_SPEC.md` Section 4. `PASS` requires code, configuration, tests,
fail-closed acceptance, and an evidence artifact. Decision 30 formal RTX 4090
acceptance remains separate from repository-wide Phase 0 closure.

## Deliverables

| ID | Deliverable | Code | Configuration | Test / command | Acceptance rule | Evidence | Status |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | Public experiment pipeline | `src/cg_pipeline/pipeline.py`, `__main__.py` | both Phase 0/1 TOML files | `pytest tests/test_pipeline_entrypoints.py -q` | dry/preflight/train are one public seam; train always preflights | `artifacts/phase0_dry_run_v1/report.json` | **COMPLETE** |
| 2 | Typed end-to-end contracts | `data.py`, `frontend.py`, `interaction.py`, `pooling.py`, `model.py`, `evaluation.py` | model/data/evaluation tables | module contract tests | exact axes, shapes, dtypes, masks, identities, logits, and predictions | full pytest report | **COMPLETE** |
| 3 | Explicit strict configuration | `config.py`, `identity.py` | `configs/phase1_baseline.toml` | `pytest tests/test_pipeline_config.py -q` | all formal scientific values exact; missing/unknown/illegal/TBD/float/mutation/test access fail; normalized SHA-256 | preflight config hash | **COMPLETE** |
| 4 | Existing-patch CAM16 adapter | `data.py` | data table | `pytest tests/test_data_contract.py -q` | exact CSV/PNG/path/label/disk inventory; no invalid-row exclusion | preflight manifest hashes/counts | **COMPLETE** |
| 5 | Split-leakage check | `validate_manifest`, `validate_patient_mapping` | `identity_level = slide_id` | data tests and preflight | `slide_id` crossing fails; patient claim requires complete mapping plus independent provenance approval bound to mapping/source hashes | slide-ID report passes; patient mapping status `not_evaluated` | **EXTERNAL BLOCKER** |
| 6 | Fixed-frontend non-trainability | `frontend.py`, `training.py` | fixed model/precision tables | model and training tests | zero optical Parameters; optimizer contains only 9473 backend scalars; bytes/hash unchanged after step | dry and preflight identity audits | **COMPLETE** |
| 7 | Deterministic Morlet generator | `morlet.py` | locked model table | frontend and spectral tests | 32 canonical kernels, separate hashes, tolerance and all spectral gates, shared H/E tensor | fixed hash vectors and spectral report | **COMPLETE** |
| 8 | Unit and negative tests | `tests/` | pytest/Ruff settings | `pytest tests -q` | all tests pass; no required skip converts failure to pass | consolidated acceptance report | **COMPLETE** |
| 9 | Synthetic public-pipeline dry run | `run_dry_run` | `configs/phase0_dry_run.toml` | `python -m cg_pipeline dry-run ...` | complete real backward/update/checkpoint/restore/evaluation twice exactly; duplicate-ID negative detected; no test | ignored strict JSON and checkpoint | **COMPLETE** |
| 10 | Commands and evidence documentation | README, protocol docs, gap register, this matrix | command blocks | acceptance CLI | commands, evidence, failures, skips, assumptions, locked impact recorded | consolidated acceptance report | **COMPLETE** |
| 11 | Decisions for all five groups | `docs/DECISIONS.md`, ADR 0010 | unique formal TOML | config/protocol tests | classifier, loss/optimizer, evaluation, training budget/CI/test gate, transfer disposition frozen | decision append and config hash | **COMPLETE** |

Summary: **10 complete, 1 externally blocked**. All repository-internal Phase 0
work is closed. Phase 0 itself remains open because the approved patient-level gate
cannot be satisfied from `slide_id` alone.

## Mandatory gate commands

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

python -m cg_pipeline.acceptance `
  --repository . `
  --config configs\phase1_baseline.toml `
  --data-root cam16_patch `
  --release configs\phase0_release.json `
  --dry-report artifacts\phase0_dry_run_v1\report.json `
  --decision30-report artifacts\decision30_formal_acceptance_rtx4090_20260802.json `
  --output artifacts\phase0_total_acceptance.json
```

The current preflight and total-acceptance commands must return failure with only
`patient_level_isolation` and `phase0_release` blocked. This is correct fail-closed
behavior, not a skipped check.

## Minimum external input and verification

Supply (1) a local, untracked UTF-8 RFC 4180 CSV with exactly
`slide_id,patient_id,provenance`, one row for every in-scope slide, no extra or
duplicate slide, and no patient identity crossing train/validation/test; and
(2) an attributable approval artifact that establishes the mapping provenance is
reliable. The mapping raw SHA-256 must replace `data.patient_mapping_evidence` in
a newly approved formal configuration. After review, `configs/phase0_release.json`
must bind the mapping hash, source-manifest hash, approval-artifact hash, approver,
timestamp, and `provenance_reliability_approved = true`, with no external blocker.
The separate approval artifact is strict JSON with schema
`patient-mapping-provenance-approval-v1` and the same mapping/source hashes,
approver, timestamp, and approval Boolean. Then run:

```powershell
$env:PYTHONPATH = 'E:\cg\src'
python -m cg_pipeline preflight `
  --config <approved-config-with-mapping-sha256> `
  --data-root cam16_patch `
  --patient-mapping <local-untracked-patient-mapping.csv> `
  --patient-mapping-approval <approved-provenance-artifact.json> `
  --release configs\phase0_release.json `
  --output artifacts\phase0_preflight_with_patient_mapping.json
```

Until that command passes, formal training and all test-split access are prohibited.
