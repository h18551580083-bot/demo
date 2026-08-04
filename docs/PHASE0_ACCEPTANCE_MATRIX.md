# Phase 0 acceptance evidence matrix

Audit date: 2026-08-03. Scope is the eleven deliverables in
`docs/DEVELOPMENT_SPEC.md` Section 4. `PASS` requires code, configuration, tests,
fail-closed acceptance, and an evidence artifact. Decision 30 formal RTX 4090
acceptance remains separate from repository-wide Phase 0 closure.

The 2026-08-04 Phase 1 batch-32 contract revision does not rewrite this historical
Phase 0 acceptance. Active training preflight now consumes the hash-bound
`configs/phase1_training_release_b32_v3.json`; its commands are in
`docs/PHASE1_TRAINING_RUNBOOK.md`. Commands below remain the historical
`phase0-closed-v1` evidence and are not current-checkout Phase 1 commands.

## Deliverables

| ID | Deliverable | Code | Configuration | Test / command | Acceptance rule | Evidence | Status |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | Public experiment pipeline | `src/cg_pipeline/pipeline.py`, `__main__.py` | both Phase 0/1 TOML files | `pytest tests/test_pipeline_entrypoints.py -q` | dry/preflight/train are one public seam; train always preflights | `artifacts/phase0_dry_run_v1/report.json` | **COMPLETE** |
| 2 | Typed end-to-end contracts | `data.py`, `frontend.py`, `interaction.py`, `pooling.py`, `model.py`, `evaluation.py` | model/data/evaluation tables | module contract tests | exact axes, shapes, dtypes, masks, identities, logits, and predictions | full pytest report | **COMPLETE** |
| 3 | Explicit strict configuration | `config.py`, `identity.py` | `configs/phase1_baseline.toml` | `pytest tests/test_pipeline_config.py -q` | all formal scientific values exact; missing/unknown/illegal/TBD/float/mutation/test access fail; normalized SHA-256 | preflight config hash | **COMPLETE** |
| 4 | Existing-patch CAM16 adapter | `data.py` | data table | `pytest tests/test_data_contract.py -q` | exact CSV/PNG/path/label/disk inventory; no invalid-row exclusion | preflight manifest hashes/counts | **COMPLETE** |
| 5 | Split-leakage check | `validate_manifest`, preflight release validation | `identity_level = slide_id`; release v2 patient state | data, config, and preflight negative tests | supplied `group_id`/`slide_id` crossing fails; exact permitted claim is `group_id/slide_id split isolation verified`; any patient-level safety claim fails | patient status `not_evaluated`; claim flag false; patient gate `NOT APPLICABLE` | **COMPLETE** |
| 6 | Fixed-frontend non-trainability | `frontend.py`, `training.py` | fixed model/precision tables | model and training tests | zero optical Parameters; optimizer contains only 9473 backend scalars; bytes/hash unchanged after step | dry and preflight identity audits | **COMPLETE** |
| 7 | Deterministic Morlet generator | `morlet.py` | locked model table | frontend and spectral tests | 32 canonical kernels, separate hashes, tolerance and all spectral gates, shared H/E tensor | fixed hash vectors and spectral report | **COMPLETE** |
| 8 | Unit and negative tests | `tests/` | pytest/Ruff settings | `pytest tests -q` | all tests pass; no required skip converts failure to pass | consolidated acceptance report | **COMPLETE** |
| 9 | Synthetic public-pipeline dry run | `run_dry_run` | `configs/phase0_dry_run.toml` | `python -m cg_pipeline dry-run ...` | complete real backward/update/checkpoint/restore/evaluation twice exactly; duplicate-ID negative detected; no test | ignored strict JSON and checkpoint | **COMPLETE** |
| 10 | Commands and evidence documentation | README, protocol docs, gap register, this matrix | command blocks | acceptance CLI | commands, evidence, failures, skips, assumptions, locked impact recorded | consolidated acceptance report | **COMPLETE** |
| 11 | Decisions for all five groups | `docs/DECISIONS.md`, ADR 0010 | unique formal TOML | config/protocol tests | classifier, loss/optimizer, evaluation, training budget/CI/test gate, transfer disposition frozen | decision append and config hash | **COMPLETE** |

Summary: **11 complete, 0 blocked**. Patient-level isolation is outside the current
CAM16 claim scope and is therefore `NOT APPLICABLE`, not a failed deliverable.
Phase 0 is closed; formal CAM16 train/validation is release-authorized and final
test access remains false.

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

The current preflight and total-acceptance commands must return `PASS`. Their
machine-readable isolation state is:

- `isolation_claim = "group_id/slide_id split isolation verified"`;
- `patient_level_isolation = "not_evaluated"`;
- `patient_level_claim_allowed = false`;
- `patient_level_isolation` appears in `not_applicable_gates`, never in
  `blocking_gates`.

A patient mapping or mapping-approval file is not a CAM16 Phase 1 release input.
Supplying, inferring, or parsing a patient identity cannot upgrade the current
claim. Final test access remains separately prohibited until its final-once gate.
