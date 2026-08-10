# Fixed H/E Wavelet-Modulus Pathology Classification

This context defines the research language for a simulation-only digital pathology
classifier with a fixed optical frontend and a trainable digital backend.

## Language

**H/E stain separation**:
Decomposition of an RGB pathology patch into spatially aligned hematoxylin (H) and
eosin (E) contribution channels.
_Avoid_: Stain normalization, color normalization

**Stain normalization**:
A transformation that reduces stain-appearance variation relative to a reference
domain or target appearance; it is distinct from H/E stain separation.
_Avoid_: Stain separation, color deconvolution

**Fixed H/E basis**:
The ordered pair of hematoxylin and eosin RGB absorbance vectors that defines the
project's stain-separation coordinate system.
_Avoid_: Learned stain matrix, adaptive stain vectors, HED default

**Optical density**:
The nonnegative logarithmic absorbance representation of an RGB patch relative to
the white reference value.
_Avoid_: RGB intensity, stain concentration

**H/E concentration**:
The ordered nonnegative hematoxylin and eosin contributions obtained by unmixing
optical density against the fixed H/E basis.
_Avoid_: HED color, normalized stain intensity

**First-order fixed wavelet-modulus frontend**:
The shared, immutable transform that applies one complex wavelet convolution and a
modulus to each H or E concentration channel while preserving spatially indexed
scale-orientation responses.
_Avoid_: Full scattering network, multi-order scattering, learned frontend

**Complex Morlet channel**:
A direction- and scale-indexed response defined by a paired real and imaginary
Morlet kernel.
_Avoid_: LoG channel, real Gabor channel

**Morlet scale index**:
The integer `j` beginning at zero that orders Morlet channels from highest center
frequency and smallest spatial envelope toward lower frequencies and larger
envelopes.
_Avoid_: Pyramid level, scattering order

**Discrete zero-DC projection**:
The deterministic correction computed on the actual finite discrete support so a
generated complex Morlet kernel has zero discrete sum.
_Avoid_: Mean subtraction, empirical correction

**Carrier direction**:
The direction of the Morlet frequency vector along which complex phase varies;
constant-phase ridges are perpendicular to it.
_Avoid_: Ridge direction, edge direction

**Valid-support mask**:
A Boolean output map whose `True` positions have a complete convolution receptive
field drawn only from the original patch and whose `False` positions use at least
one reflected padding sample.
_Avoid_: Tissue mask, foreground mask, attention mask

**First-order modulus map**:
A nonnegative spatial feature map formed from the stable complex magnitude of one
scale-orientation Morlet response.
_Avoid_: Intensity map, squared modulus, second-order coefficient

**Fixed-frontend identity**:
The auditable identity set covering both the locked H/E stain-separation contract
and the shared first-order Morlet wavelet-modulus contract.
_Avoid_: Kernel hash alone, model checkpoint

**H/E interaction boundary**:
The transition where fixed H/E modulus maps enter the electronic
`HEInteractionBlock` with their axis semantics, valid-support mask, and complete
fixed-frontend identity.
_Avoid_: Optical interaction, stain fusion

**Symmetric cross-stain mutual gating**:
An electronic interaction in which the H response conditions multiplicative
modulation of the aligned E response and the E response conditions multiplicative
modulation of the aligned H response, with the two directions structurally
symmetric.
_Avoid_: Stain fusion, cross-stain replacement, spatial attention

**Same-location symmetric product co-occurrence**:
A nonnegative electronic interaction feature equal to the product of aligned H
and E modulus responses at one scale, orientation, and spatial location.
_Avoid_: Correlation, co-occurrence probability, normalized co-occurrence score

**Center-excluded eight-neighbor cross-stain interaction**:
A paired electronic interaction relating each stain's center response to the
other stain's mean response over the aligned eight-neighbor spatial ring.
_Avoid_: Same-location co-occurrence, spatial attention, masked-neighbor average

**Bidirectional nonnegative stain-excess response**:
A paired electronic interaction feature separating the amount by which H exceeds
aligned E from the amount by which E exceeds aligned H.
_Avoid_: Absolute difference alone, stain ratio, normalized difference

**Conservative pooling support mask**:
The common Boolean spatial support used to pool every interaction feature,
identical in value to the neighborhood-valid support so all channels use the same
valid positions.
_Avoid_: Tissue mask, foreground mask, attention mask

**Support-aligned spatial pyramid**:
A hierarchy of disjoint-within-level spatial regions anchored to the canonical
valid-support rectangle rather than to sample content or a sample-specific mask
bounding box.
_Avoid_: Tissue-adaptive pyramid, content-adaptive bins, overlapping adaptive pooling

**Mean-and-population-standard-deviation pooling**:
A two-statistic regional summary pairing response level with within-region
heterogeneity while preserving the source feature's physical units.
_Avoid_: Sample standard deviation, variance output, extrema pooling

**Identity pooled-feature handoff**:
The shape-only transition from structured spatial-pyramid statistics to the
classifier input, preserving every value, gradient, coordinate meaning, and unit.
_Avoid_: Feature normalization, learned preprocessing, dataset standardization

**Protected float32 electronic backend**:
The primary electronic execution policy using IEEE float32 semantics for model
state and ordinary operations, with explicit float64 spatial-statistic reductions.
_Avoid_: Mixed-precision backend, TF32 backend, automatic precision selection

**Canonical balanced adjacent-pair reduction**:
The fixed row-major float64 summation topology used by both passes of the
spatial-statistic computation.
_Avoid_: Native reduction, atomic accumulation, device-dependent reduction

**Normative CPU statistical reference**:
The sole auditable CPU execution path against which supported device statistics
are judged numerically equivalent under approved tolerances.
_Avoid_: Bitwise oracle, device-local reference, implementation default

**Reliable patient-to-slide mapping**:
An externally supplied mapping whose provenance, in-scope mapping coverage, and
patient assignment consistency have been verified before it is used to support
an isolation claim. Structural CSV validation alone is insufficient: an
attributable provenance-reliability approval must bind the mapping and source-
manifest identities.
_Avoid_: Filename-inferred patient identity, assumed patient mapping

**Patient-level split isolation**:
A split property in which no verified patient identity occurs in more than one
split, supported by a reliable patient-to-slide mapping.
_Avoid_: Group-ID-level isolation

**Group-ID-level split isolation**:
A split property in which no supplied `group_id` occurs in more than one split;
its claim is limited to that identifier unless it is verified as a patient ID.
_Avoid_: Patient-level isolation, assumed patient grouping

**Slide-ID-level split isolation**:
A split property in which no supplied `slide_id` occurs in more than one split;
it does not establish patient-level isolation without a reliable patient-to-slide
mapping.
_Avoid_: Patient-level isolation, group-ID-level isolation

**Current CAM16 group/slide isolation claim**:
For the current existing-patch package only, `slide_id` is defined as the supplied
`group_id` rather than as patient identity. The frozen report statement is exactly
`group_id/slide_id split isolation verified`; it means exact-identifier isolation
at this declared alias level and does not merge the two general glossary concepts.
Patient-level isolation remains `not_evaluated` and no patient-level claim is
allowed.
_Avoid_: Patient identity, patient-level isolation, inferred filename grouping

**Linear-logit primary classifier**:
The trainable digital head that maps the canonical 9408-coordinate pooled vector
through one zero-initialized affine operation to one raw binary logit per patch.
_Avoid_: MLP head, probability head, normalized classifier

**Uncalibrated evaluation score**:
The sigmoid transform of an approved raw logit used only for ranking and
thresholded research evaluation, without a claim of clinical or natural-population
probability.
_Avoid_: Calibrated probability, clinical risk, population probability

**Manifest-bounded slide score**:
The maximum raw patch logit among the immutable existing-patch manifest rows that
share one validated `slide_id`; it summarizes only those rows and makes no claim of
complete WSI or all-tissue coverage.
_Avoid_: WSI score, complete-slide inference, group score

**Preregistered starting baseline**:
The single machine-validated Phase 1 configuration frozen before formal training;
it is a reproducible starting point and not an empirically demonstrated optimum.
_Avoid_: Optimal parameters, tuned baseline, implementation defaults

**Exploratory training**:
Real CAM16 train/validation execution for profiling, performance optimization, and
non-formal engineering experiments. It may use controlled runtime overrides and a
dirty source tree, but its artifacts are permanently non-formal.
_Avoid_: Dry run, formal evidence, release candidate result

**Formal training**:
Release-bound CAM16 train/validation execution that consumes the standalone formal
preflight report and preserves the frozen seeds, epochs, checkpoint selection,
provenance, and immutable output contract.
_Avoid_: Exploratory training, ad hoc training, test evaluation

**Final-once test gate**:
A separate human authorization that binds the already frozen configuration, code,
data, checkpoint, and validation thresholds before any test prediction ledger is
created.
_Avoid_: Automatic test evaluation, validation gate, repeated test access
