# Specification refactor map — 2026-09-09

Acceptance evidence for a documentation-only refactor, not a new scientific authority.
Original source: `48cc2935754741966e04dbe7d18aaf67d3b73064:docs/DEVELOPMENT_SPEC.md`.
All original lines are partitioned exactly once below, including headings and blank lines.
SHA-256 is over each original UTF-8 span with LF newlines; destination text must match
exactly after newline normalization. New navigation text is outside these spans.
No original clause is removed, no scientific value is changed, and no unresolved value is filled.

| Original DEVELOPMENT_SPEC section / lines | New location / lines | Status | Span SHA-256 |
| --- | --- | --- | --- |
| Development specification; 1-2 | [DEVELOPMENT_SPEC.md](DEVELOPMENT_SPEC.md); 1-2 | retained | `169d92e01ab06447008dd6fa704703fdba7869961384f4c23b385e3c5e195eac` |
| 1. Authority and change control; 3-18 | [DEVELOPMENT_SPEC.md](DEVELOPMENT_SPEC.md); 3-18 | retained | `6169c0afec6d9ecf1cb041c20a3c227571b4c75c40fc53e68f85b301abc9ec50` |
| 2. Current phase; 19-85 | [DEVELOPMENT_SPEC.md](DEVELOPMENT_SPEC.md); 41-107 | retained | `68bb89429be785e9745714cbd0d09fbaa1a1339fc2fcdffdd563da73e922dbee` |
| 3. Locked project scope; 86-112 | [DEVELOPMENT_SPEC.md](DEVELOPMENT_SPEC.md); 108-134 | retained | `958b367d3e0f037df1e8a9feeb698bf1e31cb6f35807dae9000162bbb79a9ffe` |
| 3. Locked project scope; 113-309 | [specs/FIXED_OPTICAL_FRONTEND_SPEC.md](specs/FIXED_OPTICAL_FRONTEND_SPEC.md); 13-209 | moved verbatim | `dd6c36aa425974bfada558ae24857fd7c02a342a1f1eaa7a16eeb5b26436e2eb` |
| 3. Locked project scope; 310-313 | [DEVELOPMENT_SPEC.md](DEVELOPMENT_SPEC.md); 138-141 | retained | `30453818ea8c5fa4cad27c73c2ec1f2386c2cd6d17277d101281a1c8f098d48f` |
| 3.2 Required H/E interaction-boundary interface; 314-335 | [specs/ELECTRONIC_BACKEND_SPEC.md](specs/ELECTRONIC_BACKEND_SPEC.md); 11-32 | moved verbatim | `40b18b46788cb78e5164b59ef670025d9f1b4ba3f98915f7dd03b508ca08a37c` |
| 3.3 Required cross-stain gating interface; 336-361 | [specs/ELECTRONIC_BACKEND_SPEC.md](specs/ELECTRONIC_BACKEND_SPEC.md); 33-58 | moved verbatim | `e16e818353f1ec9ae35ce3a8c231e6430ebbb63e81296387f3c3f859bfce3067` |
| 3.4 Required same-location co-occurrence interface; 362-386 | [specs/ELECTRONIC_BACKEND_SPEC.md](specs/ELECTRONIC_BACKEND_SPEC.md); 59-83 | moved verbatim | `3fcfcdf1db518edbdf4fac3367830d0fb8a3cccf0915e11e30352726ecf88446` |
| 3.5 Required neighborhood-interaction interface; 387-423 | [specs/ELECTRONIC_BACKEND_SPEC.md](specs/ELECTRONIC_BACKEND_SPEC.md); 84-120 | moved verbatim | `e75eee56aa63c663e06212f9f3628972539f4ca18e7b5b52ad226f1c0cf9f3fb` |
| 3.6 Required difference-feature interface; 424-453 | [specs/ELECTRONIC_BACKEND_SPEC.md](specs/ELECTRONIC_BACKEND_SPEC.md); 121-150 | moved verbatim | `14e0748e68e05520b38e49233ebe669c6aab68158d48aa6263914152aaeb16d9` |
| 3.7 Required combined interaction-feature interface; 454-493 | [specs/ELECTRONIC_BACKEND_SPEC.md](specs/ELECTRONIC_BACKEND_SPEC.md); 151-190 | moved verbatim | `ace9ef2cfc2138441a440d92dc868976c7f5a3c792169754fe1125b762cd245e` |
| 3.8 Required spatial-pooling support interface; 494-533 | [specs/ELECTRONIC_BACKEND_SPEC.md](specs/ELECTRONIC_BACKEND_SPEC.md); 191-230 | moved verbatim | `e3b746e4fa33bc7a100a1d613494f77155eee0df20ef24d17cd5781edcca09a3` |
| 3.9 Required spatial-pyramid region interface; 534-588 | [specs/ELECTRONIC_BACKEND_SPEC.md](specs/ELECTRONIC_BACKEND_SPEC.md); 231-285 | moved verbatim | `a1f38a69b5f39bcfe53068f03a5a139a90976f2cc2164d766ac59bfd9e8c0c93` |
| 3.10 Required spatial-pyramid statistic interface; 589-636 | [specs/ELECTRONIC_BACKEND_SPEC.md](specs/ELECTRONIC_BACKEND_SPEC.md); 286-333 | moved verbatim | `02c95bc50f53be966ab33bdd280443198f58a74b81a65286c771ce4255f0b37e` |
| 3.11 Required pooled-feature handoff interface; 637-681 | [specs/ELECTRONIC_BACKEND_SPEC.md](specs/ELECTRONIC_BACKEND_SPEC.md); 334-378 | moved verbatim | `00989901332343f9163c40ae9799141935d4cc5ad2aba67a5c664917fbb63ef9` |
| Primary `linear-logit-v1` classifier; 682-711 | [specs/ELECTRONIC_BACKEND_SPEC.md](specs/ELECTRONIC_BACKEND_SPEC.md); 379-408 | moved verbatim | `7db41f9e9e586470cf886963d0a008849f1372049eaf802034b32022328e4e3e` |
| 3.12 Required electronic-backend precision interface; 712-774 | [specs/ELECTRONIC_BACKEND_SPEC.md](specs/ELECTRONIC_BACKEND_SPEC.md); 409-471 | moved verbatim | `a9b3f0c8b4a148b1193c1210736337429a0fc6ebae872acfc88b52821ec85841` |
| 3.13 Required statistical summation-order interface; 775-798 | [specs/ELECTRONIC_BACKEND_SPEC.md](specs/ELECTRONIC_BACKEND_SPEC.md); 472-495 | moved verbatim | `91778420dbc47bf78c9512cbd1f781331fc1348dd94c8433f0c993cf91a29921` |
| 3.14 Required cross-device numerical-equivalence reference interface; 799-1038 | [specs/ELECTRONIC_BACKEND_SPEC.md](specs/ELECTRONIC_BACKEND_SPEC.md); 496-735 | moved verbatim | `86e9ea747e837f167e8e8a05f9e949a11aa3208289ec22a68f65943724afcd92` |
| 3.15 Frozen `cam16-eval-v1` calculation contract; 1039-1077 | [DEVELOPMENT_SPEC.md](DEVELOPMENT_SPEC.md); 145-183 | retained | `89bc8c0f5ace0a8e6ddb3d003cf729ba7a80e3a71a9b1bc4aeaaaf633721e825` |
| 3.1 Required Morlet generation interface; 1078-1097 | [specs/FIXED_OPTICAL_FRONTEND_SPEC.md](specs/FIXED_OPTICAL_FRONTEND_SPEC.md); 210-229 | moved verbatim | `daa2cd8be275118d345c83f5eb54b598b5d3ba16aefa140f861587d07596fade` |
| 4. Phase 0 deliverables; 1098-1143 | [DEVELOPMENT_SPEC.md](DEVELOPMENT_SPEC.md); 184-229 | retained | `6ccc4909d7cafa32037ad8bdf2d231b9a2b0dace5d199e96877e884f239797fa` |
| 5. Phase 0 acceptance gate; 1144-1160 | [DEVELOPMENT_SPEC.md](DEVELOPMENT_SPEC.md); 230-246 | retained | `a5c890f2d39fec7c9d04f8ea8f739e707cb1f9e50bdb5656245b6e70b021e96d` |
| 5. Phase 0 acceptance gate; 1161-1179 | [specs/FIXED_OPTICAL_FRONTEND_SPEC.md](specs/FIXED_OPTICAL_FRONTEND_SPEC.md); 232-250 | moved verbatim | `f75716267fb1893e004dd5db6eee92da737452f1f8df5728f6565c6fd47e8a12` |
| 5. Phase 0 acceptance gate; 1180-1264 | [specs/ELECTRONIC_BACKEND_SPEC.md](specs/ELECTRONIC_BACKEND_SPEC.md); 738-822 | moved verbatim | `94f4eee76d7ae05aae3a5e428fce2cda455aad40e14a6372c8ae95f44388e059` |
| 5. Phase 0 acceptance gate; 1265-1316 | [specs/FIXED_OPTICAL_FRONTEND_SPEC.md](specs/FIXED_OPTICAL_FRONTEND_SPEC.md); 253-304 | moved verbatim | `a22895714b292b81cc2689c52268cf420ea0d1e96213dc3e2249329b0e54f078` |
| 5. Phase 0 acceptance gate; 1317-1325 | [DEVELOPMENT_SPEC.md](DEVELOPMENT_SPEC.md); 252-260 | retained | `78296ae294ceab030b12230fcc589128072e18792312ae1ae87f6b49eb48eef6` |
| 6. Blocking decision groups and current disposition; 1326-1355 | [DEVELOPMENT_SPEC.md](DEVELOPMENT_SPEC.md); 261-290 | retained | `60d964397f08a912840637a30c42159a5831df28856197d61d47cd43f37f78ff` |
| 7. Prohibited actions; 1356-1389 | [DEVELOPMENT_SPEC.md](DEVELOPMENT_SPEC.md); 291-324 | retained | `d2cd2dd80eec5728e5bbbc0c0bf8c39cf0245e7185ebdf520db17ac76140573f` |
| 8. Data and evaluation invariants; 1390-1412 | [DEVELOPMENT_SPEC.md](DEVELOPMENT_SPEC.md); 325-347 | retained | `15a15817eca7901043981be21f0a0e3def8b84b17b6a6b7dd1db8e794c9e7a7b` |
| 9. Required completion report; 1413-1421 | [DEVELOPMENT_SPEC.md](DEVELOPMENT_SPEC.md); 348-356 | retained | `fa1b107537f80f795ea2f23d76ebc5fe6a1a05f7e90e64c2d609cbca09e943f6` |

## Dependency and authority audit

Pre-move inspection covered AGENTS, CONTEXT, README, pyproject, all Markdown under
docs (including ADR/agent instructions), and document references across configs,
src, tests and scripts. No data or checkpoint contents were read.

- `src/cg_pipeline/claims.py:143-152` and `tests/test_claims.py:11-21` require the
  original runbook path: BLOCKED_BY_ACTIVE_DEPENDENCY. It remains in place.
- No runtime/config/test path dependency was found for the other five archive candidates.
- pyproject contains no document-path dependency and remains unchanged.
- ADR 0001-0009 remain byte-identical. ADR 0010/0011 receive status headers only.
- The 2026-08-17 decision explicitly supersedes release/Git/tag/hash/clean-tree startup
  governance; the 2026-08-31 two-seed decision supersedes ADR 0010's three-seed count.
  Historical bodies, result claims and all decision bodies remain unchanged.
- README active archive links are repaired. Historical decision/report path text is
  preserved and resolved by the archive index. The existing missing historical
  PHASE0_ACCEPTANCE_MATRIX.md link is documented there, not recreated.
- Sections 3.15 and 6 retain their original wording: they include scope, claim and
  historical-disposition clauses beyond the execution protocols. Shared execution
  details continue to cite those protocols; no divergent rule is silently merged.
- CONTEXT definitions, canonical terms and Avoid boundaries are unchanged; only
  topic headings were added/replaced. Maintenance and issue rules load only by task.

## Unresolved authority findings (not authorizations)

- HUMAN_DECISION_REQUIRED: PHASE_STATUS. Section 2 still names Phase 1 entry.
  The 2026-09-05 decisions authorize bounded Phase2-A implementation/validity work,
  not an explicit phase transition; the local 2026-09-09 decision is exploratory-only.
  Existing Phase2-A work cannot itself approve transition. The status is retained.
- HUMAN_DECISION_REQUIRED: FIVE_SEED_CONTRACT. Four current Phase2-A formal TOMLs
  and config.py use [1729, 3407, 5113, 7717, 9109]; the retained current stability
  report describes five observed seeds. The authority chain inspected here still
  records the two-seed baseline and no approved five-seed amendment synchronized
  across DEVELOPMENT_SPEC, DECISIONS and TRAINING_PROTOCOL. No configuration,
  result, formal eligibility claim, or approval history is changed by this refactor.
- NORMATIVE_CONFLICT: LEGACY_TRACKED_CHECKPOINTS. Sections 5 and 7 prohibit tracked
  checkpoints while MAINTENANCE describes a legacy exception and Git still tracks
  22 Phase1 .pt paths. The historical reconciliation already records this conflict;
  no new highest-level approval resolving it was found. Keep the evidence unchanged.
- BLOCKED_BY_ACTIVE_DEPENDENCY: PHASE1_TRAINING_RUNBOOK.md, as detailed above.

## Verification boundary

The root specification is structurally changed, with its locked scientific clauses
preserved verbatim by the map. No scientific decision, new TBD, training, data
access, test access, model/config change, artifact mutation, phase advancement,
history rewrite, or push is part of this work.
Unrelated pre-existing edits to DECISIONS, config.py, test_pipeline_config.py and
the stability config/report are excluded from the governance commit.
The map is evidence for this refactor only; it is never default context.

## Reproduce clause-preservation verification

Run this Python snippet from the repository root. It reads only Git's original
specification and the mapped Markdown files; it performs no data access or writes.

```python
from pathlib import Path
import hashlib
import re
import subprocess

source = subprocess.check_output([
    "git", "show",
    "48cc2935754741966e04dbe7d18aaf67d3b73064:docs/DEVELOPMENT_SPEC.md",
]).decode("utf-8").replace("\r\n", "\n").splitlines(keepends=True)
mapping = Path("docs/SPEC_REFACTOR_MAP.md").read_text(encoding="utf-8")
pattern = r"^\| .*; (\d+)-(\d+) \| \[.*?\]\((.*?)\); (\d+)-(\d+) \| (?:retained|moved verbatim) \| `([0-9a-f]{64})` \|$"
covered = []
for a, b, path, c, d, digest in re.findall(pattern, mapping, re.M):
    a, b, c, d = map(int, (a, b, c, d))
    original = "".join(source[a - 1:b])
    target = (Path("docs") / path).read_text(encoding="utf-8").splitlines(keepends=True)
    assert original == "".join(target[c - 1:d]), (path, a, b)
    assert hashlib.sha256(original.encode("utf-8")).hexdigest() == digest
    covered.extend(range(a, b + 1))
assert covered == list(range(1, len(source) + 1)), "Incomplete or repeated mapping"
print(f"PASS: all {len(source)} original lines preserved exactly once")
```

## Verification results

- DEVELOPMENT_SPEC: 1421 lines / 88998 bytes before; 356 lines / 20995 bytes after.
  Frontend spec: 304 lines / 19246 bytes; backend spec: 822 lines / 52212 bytes.
- The runnable span check passes for all 1421 original lines across 32 disjoint
  spans. Exact text equality also preserves every original formula, number, dtype,
  shape, seed, threshold and claim; unmapped clauses: zero.
- `PYTHONPATH=src python -m pytest tests/test_claims.py tests/test_pipeline_entrypoints.py -q`:
  18 passed, including the non-training CLI/config/control-flow smoke. Training
  runners are mocked in the control-flow tests; no real dataset is used.
- `audit_documentation(Path('.'))`: PASS; nine required documents present, no
  active blocking TBD and no forbidden isolation claim.
- Active Markdown link targets: PASS. All moved documents have archive-index
  navigation. The sole missing historical target is documented in the index.
- Historical bodies (except the recorded live navigation repair), glossary
  definitions, decision bodies, ADR 0001-0009, both protocols, maintenance,
  runbook and the pre-existing stability report were checked unchanged.
- All snapshotted non-document files match their pre-task bytes. No new artifact
  changes appear in Git status; checkpoint contents were not read.
- `git diff --check`: PASS. Initial PowerShell wildcard searches and one final
  metrics print hit path/encoding errors; directory filters and explicit UTF-8
  reads resolved them. No test failed or was skipped in the selected test run.
- Full-suite/CUDA/training/data checks were not run, per the document-only scope.
