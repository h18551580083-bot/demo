# `cam16-eval-v1` frozen calculation contract

## Scope and access

Evaluation consumes only an immutable existing-patch manifest and an immutable
prediction ledger. It never reads a WSI, generates a candidate patch, claims full-
slide tissue coverage, or screens a row using labels, annotations, tumor location,
quality scores, or model output. The sole split-isolation statement for the current
CAM16 study is `group_id/slide_id split isolation verified`. Patient-level
isolation is `not_evaluated` and `patient_level_claim_allowed = false`; no patient
identity is inferred or consumed by this protocol.

Binary target `1` is tumor and `0` is normal. Model output is one finite float32 raw
logit per patch. Sigmoid values are called uncalibrated evaluation scores and are
not clinical or population probabilities. Raw float32 logits are the ranking and
threshold keys.

## Manifest and ledger

The source UTF-8 RFC 4180 CSV requires nonempty unique `patch_id`, unique normalized
relative PNG `patch_path`, `split` in `train|val|test`, nonempty `slide_id`, integer
`label`, exact `label_name`, exact `patch_label`, and consistent `slide_label`.
The immutable class map is `normal -> 0`, `tumor -> 1`. Extra columns remain audit
input but never affect row inclusion. Missing, duplicate, conflicting, escaping,
nonexistent, extra-on-disk, or cross-split identities fail the manifest.

Each split identity is a domain-separated SHA-256 over the canonical UTF-8
`patch_id` order and complete effective rows. A prediction ledger contains exactly
one finite float32 logit for every authorized effective row and no extras. Missing,
duplicate, or failed predictions invalidate the split; rows cannot be removed after
prediction.

## Aggregation and endpoint

For each manifest-bounded slide, the slide logit is the maximum patch logit. An
exact tie records the smallest UTF-8 `patch_id` only as provenance. Every slide
must be nonempty and have one consistent target. This is not WSI inference.

The sole primary endpoint is slide AUROC. Patch AUROC is secondary. AUROC uses the
exact Mann-Whitney pair count

`AUROC = (2*W + T) / (2*P*N)`,

where `W` is the exact number of positive-greater-than-negative pairs and `T` the
exact number of tied pairs. The implementation uses an equivalent sorted tie-group
count, not a quadratic pair loop. No positive, no negative, empty input, missing
logit, or non-finite logit makes AUROC undefined and non-reportable.

## Thresholds and secondary metrics

Patch and slide thresholds are selected independently on validation only. The
candidate set is every distinct finite validation raw logit at that level;
prediction is positive for `z >= t`. The candidate maximizing the exact rational
Youden statistic is selected; an exact tie chooses the numerically largest logit.
No midpoint, sentinel, quantile, score rounding, test row, or library default is a
candidate.

Using the frozen validation thresholds, report exact confusion counts,
sensitivity, specificity, accuracy, balanced accuracy, PPV, NPV, F1, and Youden J.
A zero denominator yields JSON `null` plus an explicit reason, never zero. The
score diagnostics are mean score, prevalence, calibration bias, Brier score, and
ten equal-width-bin ECE. No calibration transform is fitted.

## Uncertainty, seeds, and identity

Primary slide-AUROC uncertainty is a 95% stratified slide bootstrap percentile
interval with 2000 replicates, separately resampling positive and negative slides
with replacement. The run seed drives the frozen Python MT19937 sequence. Sorted
replicate indices 49 and 1949 are the lower and upper bounds. This is slide-level,
not patient-cluster uncertainty.

Every evaluation first proves its prediction ledger is exactly equal to the
authorized effective rows; a complete subset is still invalid. Every result records
config, code, source/effective manifest, fixed frontend, checkpoint, seed,
aggregation, threshold, count, and result identities. Floating
identity values use their exact IEEE-754 bits. Non-finite JSON, duplicate keys,
negative-zero JSON numbers, and overwriting an existing report are prohibited.

## Test boundary and final-once gate

Training and checkpoint/threshold selection access train and validation only. Test
cannot choose hyperparameters, threshold, epoch, checkpoint, structure, retry, or
failure handling. A future final-once authorization must name the already frozen
config, code, data, checkpoint, and validation-threshold identities and explicitly
set test access true. It is loaded from a strict immutable JSON authorization plus
a separate approval-evidence artifact; both hashes are rechecked when evaluation
starts. Frozen threshold numbers are recomputed into their validation identity,
and the bootstrap seed must equal the recorded run seed. The current
`configs/phase0_release.json` keeps test access false, so no test ledger may be
created now.
