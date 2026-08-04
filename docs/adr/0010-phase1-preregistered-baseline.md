# ADR 0010: Phase 1 preregistered baseline and fail-closed release

Status: accepted on 2026-08-03 under the explicit autonomous Phase 0 closure
authorization recorded in `docs/DECISIONS.md`.

The Phase 1 starting point is the single machine-validated contract
`configs/phase1_baseline.toml`: uniform once-per-epoch rows, no augmentation,
unweighted float32 BCE-with-logits, float32-state AdamW, no scheduler, three fixed
seeds, validation slide-AUROC checkpoint selection, and manifest-bounded maximum-
logit slide aggregation. These are conservative preregistered starting values,
not empirically established optima. Test access remains separately release-gated.

Formal training always calls the same preflight used by the standalone command.
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
