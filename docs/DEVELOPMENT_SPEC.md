# Development specification

## 1. Authority and change control

This document is the highest-level development specification for this repository.
Implementation, configuration, tests, experiment scripts, and lower-level
documentation must conform to it.

- Only decisions explicitly approved by a human may replace a `TBD`.
- Approved decisions must also be recorded in `docs/DECISIONS.md`.
- A phase transition requires an explicit human approval recorded in
  `docs/DECISIONS.md`; satisfying technical checks alone does not advance the
  project.
- If code, data, an issue, or another document conflicts with this specification,
  work must stop at the conflict and the conflict must be reported.
- Changes to dataset splits, evaluation gates, or locked research boundaries require
  explicit human approval before the change is made.

### Normative document map

Authority: DEVELOPMENT_SPEC -> approved DECISIONS -> task-specific normative
specifications/protocols -> ADR -> current reports -> historical reports/audits.
An explicit approved amendment must be checked for its scope; an apparent conflict
without that evidence must be reported, not resolved by choosing the newer file.
Reports and this refactor's mapping are evidence, never scientific authorization.

| Task / original section | Normative location |
| --- | --- |
| Fixed stain/Morlet details; 3.1; frontend acceptance clauses from 5 | [Fixed optical frontend spec](specs/FIXED_OPTICAL_FRONTEND_SPEC.md) |
| Interaction, pooling, classifier, precision; 3.2-3.14; backend acceptance clauses from 5 | [Electronic backend spec](specs/ELECTRONIC_BACKEND_SPEC.md) |
| Training execution and optimization | [TRAINING_PROTOCOL](TRAINING_PROTOCOL.md) |
| Evaluation calculation, uncertainty, final-once gate | [EVALUATION_PROTOCOL](EVALUATION_PROTOCOL.md), together with section 3.15 below |
| Approved changes; search by topic/date | [DECISIONS](DECISIONS.md) |
| Original clause locations and governance audit (non-normative) | [SPEC_REFACTOR_MAP](SPEC_REFACTOR_MAP.md) |

The linked scientific clauses are incorporated by reference with unchanged lock
and approval requirements. Section references 3.1-3.14, including historical
references to DEVELOPMENT_SPEC, resolve through this map. Moving a clause does
not revoke it, weaken its gate, or confer new experimental authorization.

## 2. Current phase

**Current phase: Phase 1 entry — preregistered CAM16 baseline training**

**Status: Phase 0 closed on 2026-08-03 after the total-acceptance gate passed.
Formal CAM16 train/validation execution is explicitly authorized; final test access
and transfer evaluation remain separately closed.**

The purpose of this phase is to turn the approved research architecture into
reviewable and executable contracts. Under the explicit 2026-08-03 autonomous
closure authorization recorded in `docs/DECISIONS.md`, work in this phase may:

- document typed interfaces and tensor/data contracts;
- implement and validate the uniquely frozen configuration schema;
- define module boundaries for dataset adapters, the fixed optical frontend, the
  trainable electronic backend, evaluation, and the public experiment pipeline;
- add synthetic fixtures and contract tests that do not encode an unapproved
  scientific choice;
- inspect local metadata and validate an explicitly supplied split manifest without
  moving samples or changing split membership.

Phase 0 did **not** itself authorize formal model training, test-set evaluation, or
transfer evaluation. Development may use `exploratory_train` with real CAM16
train/validation data for profiling, engineering performance work, bounded-step
checks, and explicitly non-formal experiments. Exploratory execution requires no
formal authorization or preflight; every artifact must state
`formal_experiment=false` and `experiment_mode=exploratory_train`, and no result may
be promoted or renamed as formal evidence. Formal training reads current parameters
from `configs/phase1_baseline.toml` or the controlled frontend-replacement profile
`configs/phase1_matched_control.toml`, and a lightweight authorization record from
`configs/formal_training_authorization.json`. The standalone preflight
checks data availability and isolation, CUDA availability, the fixed frontend,
frontend-specific correctness, optimizer ownership, determinism, and
disabled test access. Its report is not bound to Git, code, config, release, or path
identity. Historical release and tag records remain historical evidence only.

The default and primary frontend remains `fixed-he-morlet-linear-v1`. The approved
`fixed-he-matched-control-linear-v1` comparison replaces only that fixed frontend
with the frozen envelope-matched random-phase control; all other baseline training
and evaluation conditions remain identical. Morlet retains its numerical and
spectral checks. Matched-control preflight verifies the frozen control bank's DC
and unit energy through the `matched_control_numerical` gate. The matched-control
frontend variant is managed by configuration (`frontend_variant`, `contract_id`)
and Git commit, without a runtime frontend identity hash gate. Observed hashes
remain provenance metadata; they are not compared with a fixed SHA-256 allowlist.
Optimizer-step immutability and checkpoint/resume integrity checks remain enforced.
Morlet's existing behavior is unchanged. Morlet spectral coverage is
not applicable and must not be reported as passed for this control. Preflight
consumption requires the gate for the configured frontend. This control does not
claim complete spectral isolation. See the 2026-09-03 decisions.

Phase2-A `fixed-he-morlet-phase2a-linear-v1` uses the separate
`phase2a_morlet_validity` gate. It recomputes fixed shape, dtype, finite, DC and
unit-energy checks, checks both kernel precisions against the configured sampled
zero-DC theoretical spectral peak, and validates the backend interface with a
synthetic forward. Overlap, ring uniformity and continuous beta deviation are
diagnostics, not Phase1 threshold requirements. Legacy Morlet continues to use
its unchanged identity and spectral coverage gates. See the Phase2-A validity
decision in `docs/DECISIONS.md` for numerical tolerances and evidence boundaries.

The only real CAM16 training modes are `exploratory_train` and `formal_train`.
Both are restricted to train/validation and expose no test-access parameter.
They share manifest readability, legal split, and supplied `group_id`/`slide_id`
cross-split isolation checks. `formal_train` additionally retains lightweight human
authorization, CUDA, frontend-specific correctness, optimizer, determinism, fixed-seed,
complete-epoch, validation-checkpoint, immutable-output, and provenance gates.

## 3. Locked project scope

The project studies simulation-only digital pathology tumor classification.

- CAM16 is the primary development dataset.
- Other pathology datasets are reserved for transfer evaluation.
- The sole valid data input is an existing, already split patch data package.
  Complete WSI files and any route that derives patches from them are outside the
  project scope.
- Public outcomes are patch-level and slide-level binary tumor classification.
- The optical frontend is fixed and must not be trainable.
- The intended processing path is:
  1. accept an RGB pathology patch;
  2. separate it into hematoxylin (H) and eosin (E) channels using fixed
     optical-density color deconvolution without baseline stain normalization or
     data-adaptive stain-vector estimation;
  3. process H and E using the same parameter-sharing, first-order fixed
     wavelet-modulus frontend;
  4. combine `F_H` and `F_E` in a structured `HEInteractionBlock`;
  5. represent cross-stain gating, same-location co-occurrence, neighborhood
     interaction, and difference features;
  6. summarize interaction features using spatial-pyramid statistical pooling;
  7. classify them using a lightweight trainable digital binary-classification
     head.
- The primary end-to-end test seam is the public experiment-pipeline entry point,
  exercised with synthetic RGB patches, slide identifiers, and explicit
  configuration.
Detailed fixed-frontend requirements are incorporated from the
[fixed optical frontend spec](specs/FIXED_OPTICAL_FRONTEND_SPEC.md).

The stain-separation component, fixed wavelet backbone, interaction block,
electronic backend, evaluation, and dataset adapters must remain separate modules.
Configuration must be explicit rather than hidden in source code.

Interaction, pooling, classifier and precision requirements (3.2-3.14)
are incorporated from the [electronic backend spec](specs/ELECTRONIC_BACKEND_SPEC.md).

### 3.15 Frozen `cam16-eval-v1` calculation contract

- The patch-level evaluation object is defined only by an immutable patch
  manifest.
- Labels, outcomes, tumor annotations, patch labels, and derived tumor-location
  fields must not be used to screen or filter the existing patches before their
  predictions are fixed for metric calculation.
- Existing validated patches may be aggregated only by identifiers already present
  in the current data package. A slide-level result requires an identifier declared
  and validated as `slide_id`; a generic `group_id` result must remain group-level.
  Such aggregation does not represent a complete WSI, coverage of all tissue
  regions, a complete patch set, or patches produced by a uniform WSI candidate
  algorithm.
- `sigmoid(z_patch)` and any sigmoid-transformed aggregate output are called
  **uncalibrated evaluation scores**. They must not be described as calibrated,
  clinical, or natural-population probabilities.
- The sole primary endpoint is slide-level AUROC.
- Patch and slide decision thresholds are distinct and use validation data only.
  The candidate set is every distinct finite raw float32 validation logit at that
  level; prediction is positive for `z >= t`; maximum exact rational Youden J wins;
  and the numerically largest raw-logit threshold wins an exact tie.
- Primary slide and secondary patch AUROC use exact Mann-Whitney win and tie counts
  on raw float32 logits. Missing classes, empty populations, and non-finite or
  missing logits are undefined and fail reportability rather than receiving a
  substituted value.
- Manifest-bounded slide aggregation selects the maximum patch logit. A tie records
  the smallest UTF-8 `patch_id` only as provenance. This is not complete WSI
  inference or all-tissue coverage.
- Threshold metrics, uncalibrated-score diagnostics, operation order, exceptional
  cases, result identity, and the 2000-replicate stratified slide bootstrap interval
  are frozen in `docs/EVALUATION_PROTOCOL.md` and the unique effective
  configuration.
- Test access requires a separate final-once authorization naming the data,
  checkpoint, and validation-threshold identities. The current authorization record
  keeps test access false.
- CAM16 split evidence is limited to `group_id/slide_id split isolation verified`.
  Patient-level isolation is `not_evaluated`, patient-level claims are forbidden,
  and this non-applicable property is not a Phase 0 or formal-training blocker.

## 4. Phase 0 deliverables

Phase 0 is complete only when all of the following artifacts exist and agree:

1. A documented public experiment-pipeline interface.
2. Typed contracts, where practical, for RGB input, H/E channels, `F_H`, `F_E`,
   interaction features, pooled features, patch predictions, and slide
   predictions.
3. An explicit configuration schema covering every scientific and experimental
   choice, rejecting unknown, missing, illegal, floating-TOML, and unresolved
   values rather than supplying code defaults.
4. A CAM16 dataset-adapter contract that requires a stable existing group or
   slide identifier, an explicit `identity_level` and `identity_column`, and an
   externally supplied, immutable split manifest. A patient-level identity
   additionally requires a reliable patient-to-slide mapping with recorded
   provenance, in-scope mapping coverage, and assignment-consistency validation.
5. A leakage check that fails when one identity at the explicitly declared level
   occurs in more than one split and reports
   `group_id/slide_id split isolation verified` without upgrading the claim.
   Patient-level isolation is recorded as `not_evaluated` and the patient-level
   check is `NOT APPLICABLE` to Phase 0 acceptance.
6. A fixed-frontend check proving that no optical-frontend value is registered as a
   trainable parameter or changed by an optimizer step.
7. A deterministic Morlet-generator contract with separate parameter-specification
   and kernel-tensor hashes, explicit channel metadata, and shared H/E tensor
   identity.
8. Unit tests for all implemented Phase 0 modules.
9. Historical Phase 0 acceptance evidence includes the then-public synthetic
   pipeline smoke. The retired dry-run entry need not remain active; the current
   project smoke validates CLI/config routing and pipeline control flow without
   starting a training run.
10. Documentation of test commands, expected evidence, and any skipped checks.
11. Explicit human decisions for every formerly blocking group, recorded in
    `docs/DECISIONS.md`.

The Phase 0 total-acceptance closure is limited to these eleven deliverables and
the five decision-group dispositions in Section 6. It must not add
a new research module, broaden the primary model, or expand the research scope.

Completion evidence for each deliverable must identify its code location,
configuration location, tests, acceptance metrics, and produced artifact. A bare
statement that a deliverable is "implemented" is not acceptance evidence.

Deliverables may be added in separate work items. Their existence does not by
itself pass the acceptance gate.

## 5. Phase 0 acceptance gate

The gate is **closed by default**. It passes only when:

- every Phase 0 deliverable is present;
- all unit tests for changed modules pass;
- the current non-training CLI/config/pipeline-control-flow smoke test passes;
- no test, fixture, or configuration supplies a hidden or unresolved value;
- the supplied `group_id`/`slide_id` split isolation check passes with no
  cross-split identifier conflict;
- the machine-readable patient fields are exactly
  `patient_level_isolation = not_evaluated` and
  `patient_level_claim_allowed = false`; the patient-level gate result is
  `NOT APPLICABLE`, not `FAIL`, and no patient identity is inferred from filenames
  or identifiers;
- the optical frontend is demonstrated to be non-trainable and unchanged during a
  backend optimization step;

The detailed acceptance clauses in the [frontend spec](specs/FIXED_OPTICAL_FRONTEND_SPEC.md)
and [backend spec](specs/ELECTRONIC_BACKEND_SPEC.md) are also mandatory parts
of this gate, with exactly the original scope and thresholds.

- no dataset split has changed relative to the approved manifest;
- no image, checkpoint, credential, or patient metadata is tracked for commit;
- a human reviews the evidence and explicitly approves entry into the next phase in
  `docs/DECISIONS.md`.

These conditions were satisfied and the human-approved Phase 0 closure was recorded
on 2026-08-03. Later phases and final-test access still require their own explicit
scope and authorization; Phase 0 closure does not authorize them implicitly.

## 6. Blocking decision groups and current disposition

The five Phase 0 groups now have explicit human disposition under the 2026-08-03
authorization. Executable values occur only in `configs/phase1_baseline.toml`:

1. `linear-logit-v1` and the exact 9473-scalar backend are frozen in Section 3.11.
2. Loss, loss precision, optimizer, optimizer-state precision, regularization, and
   schedule are frozen in `docs/TRAINING_PROTOCOL.md`: unweighted mean float32
   BCE-with-logits and float32-state AdamW with the exact configured parameters and
   no scheduler.
3. `cam16-eval-v1` manifest, sampling, aggregation, AUROC, Youden, secondary
   metrics, calibration diagnostics, uncertainty, and exceptional cases are frozen
   in Section 3.15 and `docs/EVALUATION_PROTOCOL.md`.
4. Seeds, batch/epoch budget, early stopping, checkpoint/resume, failed-run,
   multi-seed, 2000-replicate confidence interval, and final-once test rules are
   frozen in the two protocol documents and the unique configuration. Under the
   explicit 2026-08-04 revision, batch size is 32 with `drop_last = false`: 79,570
   train rows give 2,487 optimizer updates per complete epoch and 49,740 maximum
   updates at 20 epochs; validation reuses batch 32 and retains all 18,171 rows in
   568 batches. The AdamW learning rate remains `0.001` without linear scaling.
5. Transfer datasets, physical-scale adaptation, and transfer protocol are not part
   of the CAM16 Phase 1 starting baseline. They require a later separate
   preregistration and do not authorize transfer work now.

These are conservative preregistered starting values, not empirically demonstrated
optima. No repository-internal or external Phase 0 blocker remains. A patient-to-
slide mapping and mapping-approval artifact are not release inputs for the current
CAM16 Phase 1 baseline. Patient-level isolation remains `not_evaluated` and no
patient-level claim is allowed.

## 7. Prohibited actions

The following actions are prohibited in every phase unless this specification is
changed through explicit human approval:

- making the optical frontend trainable, including learned filters, trainable stain
  separation, trainable wavelets, or optimizer-owned optical parameters;
- adding physical deployment, fabrication, SLM control, hardware-control, or
  clinical-deployment code;
- representing the work as a clinical system or making clinical-performance claims;
- using a transfer-evaluation dataset for development, tuning, model selection, or
  threshold selection;
- allowing a supplied `group_id` or `slide_id` to cross training, validation, or
  test splits;
- asserting patient-level leakage protection or isolation without a separately
  approved, reliable patient-to-slide mapping;
- changing, regenerating, or rebalancing dataset splits silently;
- downloading any dataset automatically;
- committing pathology images, checkpoints, credentials, secrets, or patient
  metadata;
- embedding scientific or experiment configuration as hidden source-code defaults;
- applying baseline stain normalization or estimating stain vectors from the
  training set, a slide, or a patch;
- evaluating the final test set before the approved final-once test gate;
- resolving a `TBD` without explicit human approval;
- advancing to another phase automatically.

During Phase 0, the following are additionally prohibited:

- training a model or tuning a parameter;
- running comparative, ablation, transfer, or final-test experiments;
- implementing a scientific component whose behavior depends on a blocking `TBD`;
- treating synthetic smoke-test results as scientific evidence.

## 8. Data and evaluation invariants

- Dataset isolation identity is declared at the strongest externally validated
  level. Patient-level identity requires a reliable patient-to-slide mapping;
  otherwise the declared level remains the supplied group or slide identifier.
- Every sample used by an experiment must be traceable to its declared isolation
  identity, slide, split, and source manifest without exposing patient metadata.
  Patient traceability is additionally required before a patient-level claim.
- Every split-isolation result must state the exact verified identity level. It
  may be called patient-level only when supported by a reliable patient-to-slide
  mapping; otherwise it must be reported at the exact supplied identifier level,
  such as `group_id` or `slide_id`, without implying patient-level protection.
- A `slide_id` must not be treated as or renamed to `patient_id`; in the current
  CAM16 package it denotes only `group_id`. Patient identity must not be inferred
  from filenames or identifier syntax.
- Split validation must occur before feature extraction, training, or evaluation.
- Split membership must be supplied explicitly; dataset adapters must not invent a
  split.
- CAM16 development results and transfer-evaluation results must be reported
  separately.
- Test-set access and transfer evaluation remain unavailable until their respective
  protocols are approved.

## 9. Required completion report

Every work item must report:

1. changed files;
2. tests executed, including commands, results, failures, and skipped tests;
3. unresolved issues;
4. assumptions;
5. whether any locked specification was affected.
