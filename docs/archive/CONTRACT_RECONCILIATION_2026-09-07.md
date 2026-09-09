> **Historical report / audit / proposal; never default-load.**
> This body records its original scope, results and paths; it is not current
> scientific authorization. [Archive index](README.md) |
> [Current specification](../DEVELOPMENT_SPEC.md) | [Decisions](../DECISIONS.md).

# Research contract reconciliation — 2026-09-07

## Authority and scope

Authority is applied by subject matter, not file date: locked scientific requirements in
`DEVELOPMENT_SPEC.md`; approved, still-effective changes in `DECISIONS.md`; training and
evaluation protocols; then implementation/tests as drift evidence. README and AGENTS are
maintenance summaries and must follow those sources.

This audit did not run training, read a dataset, access the test split, or load checkpoint
contents. Static inspection covered the requested documents, authorization/config, formal
pipeline modules, safety tests, Git tracking state, and artifact metadata/JSON only.

## Fact matrix

| Requirement | AGENTS | DEVELOPMENT_SPEC | DECISIONS | README | Code | Tests | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Formal startup uses lightweight human authorization; Git status, release, tag, commit, code/config/report identity are not gates | Synchronized to protocol | §§2, 3: lightweight record; no Git/code/config/release/path binding | 2026-08-17 explicitly supersedes identity/governance gates | States lightweight authorization and no Git/tag/identity startup gates | `preflight.py` validates authorization fields and has no repository-identity gate | Pipeline safety tests exercise authorization/preflight without a release identity | STALE-DOC |
| A passing standalone preflight remains required | Required by reference | §§2, 3 require standalone preflight | 2026-08-17 retains preflight correctness gates | Sole formal entry consumes a passing report | `consume_preflight_report` fails closed on status, blockers, training/test flags and required gates | Safety tests cover report production/consumption rejection paths | CONSISTENT |
| Formal runs use only approved fixed seeds and one seed per invocation | Requires frozen seeds by reference | §2 retains fixed-seed gate | 2026-08-31 fixes seeds 1729 and 3407, one per invocation | Names both seeds and one-seed invocation | Config exact values and `run_formal_training` reject unapproved seeds | Config/pipeline tests cover seed contract | CONSISTENT |
| Training is train/validation only; final test needs separate final-once authorization | Prohibits test use | §§2, 3, 8 prohibit training-time test access | 2026-08-17 retains test prohibition | Both CLIs expose no test access | Preflight requires `test_access_authorized=false`; reports record no test access | Safety/config tests retain test rejection | CONSISTENT |
| Optical frontend remains fixed and outside optimizer ownership | Fixed, non-trainable | §§3.2, 3.11 and prohibited actions lock it | 2026-08-17 preserves fixed optical correctness | Describes fixed H/E frontend | Preflight checks empty frontend parameters and optimizer ownership | Training tests verify frontend identity survives a step/checkpoint restore | CONSISTENT |
| Formal epochs/checkpoints/provenance are complete and outputs are never overwritten | Required by reference | §§2, 3 retain immutable output and checkpoint/resume integrity | 2026-08-17 retains complete checkpoint/resume restoration | States immutable, non-overwriting formal output | Existing seed/output raises `FileExistsError`; resume validates paired history | Training tests cover exclusive writes, tamper and resume failures | CONSISTENT |
| Patient isolation is not evaluated; no patient-level claim is allowed | States exact fields | §§2, 8 state exact fields and claim boundary | 2026-08-17 does not expand the claim | States exact fields | Authorization/preflight validate exact fields | Safety/claims tests enforce the boundary | CONSISTENT |
| No checkpoint is tracked for commit | Prohibits checkpoint commits | §7 prohibits commits; §5 records no tracked checkpoint as a closure condition | 2026-09-07 preserves formal artifacts during slimming | Does not authorize new checkpoints | `.gitignore` now blocks checkpoint suffixes; 22 historical `.pt` files remain tracked | Ignore/tracking checks distinguish new files from legacy tracked evidence | LOCKED-SPEC-CONFLICT |

The sole `STALE-DOC` row was resolved by replacing the old AGENTS
release/tag/hash/preflight sentence with a reference to the approved lightweight contract.
No implementation or scientific gate changed.

## 1. Resolved conflicts

- Formal startup governance now has one operational source: `TRAINING_PROTOCOL.md`, backed
  by the explicit 2026-08-17 Decision and locked summary in `DEVELOPMENT_SPEC.md`.
- AGENTS now requires lightweight authorization, standalone preflight, fixed seeds,
  complete epoch/checkpoint/provenance handling and non-overwrite, while explicitly stating
  that Git/release/tag/hash identity is not a startup gate.
- README already matched the active contract and now links here instead of duplicating the
  detailed matrix.

## 2. Human decisions required

`HUMAN-DECISION-REQUIRED`: the locked specification says checkpoints must not be tracked,
but 22 formal baseline checkpoints are already tracked and are active evidence. This task
records a narrow legacy exception only in `MAINTENANCE.md`; it does not amend the locked
specification. A future human decision must choose either (a) amend the locked wording to
distinguish legacy evidence from prohibited new commits, or (b) approve verified external
archival and a later current-branch removal. No history rewrite is proposed.

No other unresolved formal-gate conflict was found in the inspected scope.

## 3. Legacy tracked artifacts

| Path | Git tracked | Referenced by paper/project | Formal evidence | Classification | Recommendation |
| --- | ---: | ---: | ---: | --- | --- |
| `artifacts/formal_runs/phase1-cam16-baseline-b32-v2/seed-1729/epoch-0000.pt` … `epoch-0014.pt` (15 files) | yes | yes | yes: completed formal seed and validation summary | ACTIVE-FORMAL-EVIDENCE | Retain under legacy exception; prohibit new checkpoint commits |
| `artifacts/formal_runs/phase1-cam16-baseline-b32-v2/seed-3407/epoch-0000.pt` … `epoch-0006.pt` (7 files) | yes | yes | yes: completed formal seed and validation summary | ACTIVE-FORMAL-EVIDENCE | Retain under legacy exception; prohibit new checkpoint commits |

No tracked `.ckpt`, `.pth`, `.onnx`, `.safetensors`, `.bin`, or `.pkl` model artifact was
found. Checkpoint contents and hashes were not read; existing completion and validation JSON
report 15 + 7 completed epochs, 22/22 checkpoint presence, both seeds complete, and
`test_split_accessed=false`.

## 4. Repository freeze state

- Pre-slim archive tag: local annotated tag `archive/pre-slim-2026-09-07`, peeled target
  `7e8d8cffaa14d99de596c9e11bdf3db611d1e062`; verified unchanged. It has not been pushed.
- Current HEAD: `7e8d8cffaa14d99de596c9e11bdf3db611d1e062`.
- Working tree: intentionally dirty and uncommitted; staged slimming deletions coexist with
  reviewed maintenance/code/test/document changes. Suggested commits below use explicit
  paths and do not include local `picture/` content.
- `.pytest-formal-run-id-*`: all formerly tracked copies are staged for deletion;
  `.pytest-*/` prevents recurrence.
- Formal artifacts: no tracked path under `artifacts/` is modified, staged, or deleted.
- Repository hygiene: no new checkpoint or recognized credential/patient-metadata path is
  staged; `cam16_patch/`, checkpoint suffixes, secrets, and local `picture/` are ignored.
- Core validation: `python -m pytest tests -q` passed all 182 collected tests;
  `python -m compileall -q src tests`, `python -m ruff check src tests`, unstaged and
  staged `git diff --check`, and the contract text checks all passed.

## 5. No-scope-change declaration

This reconciliation and freeze work:

- did not train;
- did not access the test split;
- did not modify data splits;
- did not modify the fixed optical frontend;
- did not advance the research phase;
- did not rewrite Git history.

No locked specification was modified. The remaining checkpoint wording conflict is reported
as `HUMAN-DECISION-REQUIRED` rather than silently resolved.
