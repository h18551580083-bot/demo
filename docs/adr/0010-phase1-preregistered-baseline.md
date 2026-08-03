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
