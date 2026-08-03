# Phase 0 acceptance evidence matrix

This matrix is the authoritative working inventory for the Phase 0
total-acceptance closure. Its scope is limited to the eleven deliverables in
`docs/DEVELOPMENT_SPEC.md` Section 4 and the five blocking decision groups in
Section 6.

No row may be marked **complete** unless it names all of the following:

1. code location;
2. configuration location;
3. test location and command;
4. acceptance metric or fail-closed rule;
5. produced evidence artifact.

`MISSING` is an explicit current-state finding, not a planned implementation
location. Decision 30 calibration evidence may support a row, but it cannot by
itself close the repository-wide Phase 0 gate.

## Current baseline

- Audit date: `2026-08-03`.
- Git HEAD at audit start: `417d910` (`main`, one commit ahead of `origin/main`).
- Current tracked implementation: `src/cg_acceptance/` only.
- Current unit verification: `python -m pytest tests -q` passes all 19 collected
  cases; `python -m compileall -q src tests`, `python -m ruff check .`, and
  `git diff --check` pass.
- Accepted local formal artifact:
  `artifacts/decision30_formal_acceptance_rtx4090_20260802.json`, SHA-256
  `fe0c0a3d704a2ae458e26e894ef82be0fddcdf13ad430ac5f483bf72b1836117`.
  It is ignored and closes only Decision 30.
- Overall deliverable inventory: **0 complete, 7 partial, 4 missing**.

## Deliverable inventory

| ID | Deliverable | Code location | Configuration location | Test location / command | Acceptance metric | Evidence artifact | Status |
|---:|---|---|---|---|---|---|---|
| 1 | Documented public experiment-pipeline interface | `MISSING`; `src/cg_acceptance/__init__.py` exposes only Decision 30 interfaces | `MISSING`; no experiment configuration exists | `MISSING`; no public-pipeline contract or smoke test exists | One public seam accepts synthetic RGB patches, slide identifiers, and explicit configuration and returns typed patch and slide predictions | `MISSING`; no interface report or public-pipeline smoke report | **Partial**: the specification describes the seam, but there is no signature, implementation, configuration, or test |
| 2 | Typed end-to-end contracts | Partial local types in `src/cg_acceptance/fixture.py`, `device_operator.py`, `comparator.py`, and `calibration.py`; RGB through slide-prediction contracts are `MISSING` | `MISSING` | Decision 30 tensor-object tests in `tests/test_comparator.py`, `test_calibration.py`, and `test_calibration_autograd.py`; full-chain contract tests are `MISSING` | Contracts cover RGB, H/E, `F_H`, `F_E`, interaction, pooled, patch-prediction, and slide-prediction values with exact axes, shapes, dtypes, masks, and identities | Decision 30 JSON reports exist; full-chain contract report is `MISSING` | **Partial**: only the pooled-statistic calibration boundary is typed and tested |
| 3 | Explicit configuration schema | `MISSING` | `MISSING`; `pyproject.toml` is package/tool configuration, not experiment configuration | `MISSING` | Every scientific and experimental field is explicit; every unresolved field is literal `TBD`; validation fails if code, a fixture, or a test silently supplies a `TBD` value | `MISSING`; no validated configuration or schema-validation report | **Missing** |
| 4 | CAM16 dataset-adapter contract | `MISSING` | `MISSING` | `MISSING` | Stable existing group or slide identity; explicit `identity_level` and `identity_column`; external immutable split manifest; reliable mapping evidence before any patient-level claim; full traceability without exposing patient metadata | `MISSING`; no adapter contract, manifest schema, manifest hash report, or mapping-validation report | **Partial**: normative requirements exist in the specification, but no executable or typed contract exists |
| 5 | Split-leakage check | `MISSING` | `MISSING` | `MISSING` | Fail when one declared identity crosses splits; report the verified identity level exactly; patient-level pass requires a validated reliable patient-to-slide mapping; group-ID evidence cannot be promoted | `MISSING`; no isolation report containing identity level, manifest identity, conflict count, and fail-closed verdict | **Missing** |
| 6 | Fixed-frontend non-trainability check | `MISSING`; no fixed H/E/Morlet frontend implementation exists | `MISSING` | `MISSING` | Optical trainable-parameter count is exactly zero; no optical value belongs to the optimizer; all optical values and identities remain unchanged across a backend optimizer step; precision guards pass | `MISSING`; no parameter-ownership or before/after identity report | **Missing** |
| 7 | Deterministic Morlet-generator contract | `MISSING`; no generator or immutable kernel-bundle implementation exists | Locked formulas and values exist in `docs/DEVELOPMENT_SPEC.md` and ADRs `0002` through `0006`; executable configuration is `MISSING` | `MISSING` | Separate parameter and tensor hashes; explicit channel metadata; shared H/E tensor identity; repeated generation identity; canonical `[32,105,105,2]`; complex128 zero-DC/L2 error at most `1e-12`; runtime complex64 error at most `1e-6`; `abs(beta_disc-beta_inf) <= 1e-2`; all spectral, boundary, and FFT gates pass | `MISSING`; no parameter specification, immutable bundle, stored preimage/digest vectors, or generation report | **Partial**: the contract is detailed, but implementation and executable evidence are absent |
| 8 | Unit tests for all implemented Phase 0 modules | `src/cg_acceptance/` is the only implemented package | Pytest configuration in `pyproject.toml` | `tests/test_comparator.py`, `tests/test_calibration.py`, and `tests/test_calibration_autograd.py`; `python -m pytest tests -q` currently passes 19 cases | Every implemented module has passing unit and negative-control coverage; required evidence is not converted to pass by a skip; the full Section 5 gate matrix is covered | Current ignored Decision 30 reports; full Phase 0 test report is `MISSING` | **Partial**: current modules pass, but most Phase 0 modules and tests do not exist |
| 9 | Synthetic public-pipeline smoke test | Decision 30 fixture and calibration code exists but is not the public experiment pipeline | Decision 30 CLI arguments only; public-pipeline configuration is `MISSING` | Decision 30 local/formal tests exist; public-pipeline smoke test is `MISSING` | The public seam runs synthetic RGB patches and slide identifiers under explicit configuration and produces patch and slide outputs without accessing CAM16 or treating smoke as scientific evidence | Decision 30 smoke JSON exists; Phase 0 public-pipeline smoke report is `MISSING` | **Missing** |
| 10 | Test-command, expected-evidence, failure, and skip documentation | Not applicable as executable code | Tool configuration in `pyproject.toml` | Decision 30 commands are documented in `README.md`; project-wide commands and skip audit are `MISSING` | Documentation lists exact commands, expected evidence, observed result, failures, every skipped check, unresolved issues, assumptions, and locked-spec impact for all eleven rows | This matrix and the Decision 30 README are partial evidence; consolidated Phase 0 acceptance report is `MISSING` | **Partial** |
| 11 | Human decisions for all blocking `TBD` groups | Approved `linear-logit-v1` implementation is not yet present | `MISSING`; no experiment configuration exists | Classifier and configuration-rejection tests are `MISSING` | All five groups have explicit human approval in `docs/DECISIONS.md`; effective configuration matches those decisions; no hidden default resolves a field; phase transition still requires a separate approval | Conditional closure and `linear-logit-v1` decisions exist; `cam16-eval-v1` has principle approval but its calculation contract is incomplete; remaining decision records and configuration audit are `MISSING` | **Partial**: 1 of 5 groups is frozen; group 2 remains conditional |

## Isolation evidence boundary

The ignored local CAM16 package currently exposes slide identity but no verified
patient identity or supplied `group_id`. Its current split evidence may therefore
be described only at the supplied slide-identifier level. Any artifact field that
labels this evidence as patient-level is invalid under the approved claim rule and
cannot close the patient-level Phase 0 gate.

Historical code deleted by commit `5ea1c1e` treated `slide_id` as a patient key.
It is prior art only and must not be restored with that identity conflation.

## Update rule

Update a row immediately when its evidence changes. Replacing `MISSING` requires a
real path in the current worktree and a reproducible command; a planned path or an
unexecuted test is not evidence. Phase 0 remains active until every row is complete,
every Section 5 gate passes, and a human separately approves the phase transition.
