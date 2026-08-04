# ADR 0010: Phase 1 preregistered baseline and fail-closed release

Status: accepted on 2026-08-03 under the explicit autonomous Phase 0 closure
authorization recorded in `docs/DECISIONS.md`.

The Phase 1 starting point is the single machine-validated contract
`configs/phase1_baseline.toml`: uniform once-per-epoch rows, no augmentation,
unweighted float32 BCE-with-logits, float32-state AdamW, no scheduler, three fixed
seeds, validation slide-AUROC checkpoint selection, and manifest-bounded maximum-
logit slide aggregation. These are conservative preregistered starting values,
not empirically established optima. Test access remains separately release-gated.

Formal training requires one separately generated immutable preflight report. The
`train --preflight-report` entry does not repeat the model/optimizer/spectral
preflight; it verifies the report identity and recomputes the current Git release,
configuration, source-manifest, and train/validation effective-split identities
before the first batch.
The Phase 0 release is closed successfully and authorizes the frozen CAM16 Phase 1
train/validation baseline. Patient-level isolation is `not_evaluated`, the patient-
level claim flag is false, and no patient mapping or approval artifact is a release
input. No WSI input, candidate generation, physical deployment, stain
normalization, adaptive optical parameter, or transfer dataset is added.

Amendment, 2026-08-04: the human-approved batch-32 revision is a new formal
training-contract identity, `phase1-cam16-baseline-b32-v2`, bound by
`configs/phase1_training_release_b32_v2.json`. The batch-4 decision and
`phase0-closed-v1` tag remain historical evidence. Learning rate `0.001`, the
20-epoch cap, optimizer parameters, evaluation rules, seeds, fixed frontend, and
test prohibition are unchanged. Their suitability under the lower optimizer-step
budget is not claimed without formal training evidence.

Amendment, 2026-08-04: release governance v3 uses a two-commit identity. The
annotated `phase1-training-b32-v3` tag names a single-parent release commit whose
only parent is the release-bound formal code commit. The release commit may change
only `configs/phase1_training_release_b32_v3.json`, `docs/DECISIONS.md`, and
`docs/PHASE1_TRAINING_RUNBOOK.md`. The release freezes the raw source-manifest
SHA-256 and the domain-separated train/validation effective identities; no test
effective identity is frozen or computed by formal preflight. The unchanged v2 Run
ID remains the training-configuration identity, while v3 is explicitly the release-
governance identity.
