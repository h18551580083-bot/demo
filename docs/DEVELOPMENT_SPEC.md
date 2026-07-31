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

## 2. Current phase

**Current phase: Phase 0 — specification baseline and interface contracts**

**Status: active; the Phase 0 acceptance gate has not passed.**

The purpose of this phase is to turn the approved research architecture into
reviewable contracts without choosing any unresolved scientific or experimental
value. Work in this phase may:

- document typed interfaces and tensor/data contracts;
- define configuration schemas with unresolved values left as `TBD`;
- define module boundaries for dataset adapters, the fixed optical frontend, the
  trainable electronic backend, evaluation, and the public experiment pipeline;
- add synthetic fixtures and contract tests that do not encode an unapproved
  scientific choice;
- inspect local metadata and validate an explicitly supplied split manifest without
  moving samples or changing split membership.

This phase does **not** authorize model training, comparative experiments, test-set
evaluation, transfer evaluation, or implementation choices that resolve a `TBD`.

## 3. Locked project scope

The project studies simulation-only digital pathology tumor classification.

- CAM16 is the primary development dataset.
- Other pathology datasets are reserved for transfer evaluation.
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
- The fixed stain basis has rows H then E and columns R, G, then B:
  - H: `[0.644211, 0.716556, 0.266844]`;
  - E: `[0.092789, 0.954111, 0.283111]`.
- H/E concentrations use the Moore–Penrose pseudoinverse of this two-vector basis.
  DAB is not a third stain, and a library-provided HED default must not silently
  define the basis.
- The stain-basis values, ordering, numeric type, and derived pseudoinverse are
  included in the fixed-frontend identity hash.
- Canonical stain-separation input is three-channel sRGB `uint8` in `[0, 255]`.
  Floating input is accepted only when explicitly declared as normalized to
  `[0, 1]`.
- For 8-bit value `I`, normalized intensity is `x = max(I, 1) / 255`, optical
  density is `OD = -ln(x)`, and ordered H/E concentration is
  `C = max(OD @ pinv(B), 0)`.
- The pseudoinverse is computed in `float64`; H/E runtime output is `float32` in
  optical-density units without per-patch, per-slide, or dataset-fitted
  normalization and without rescaling to `[0, 1]`.
- White background maps to approximately zero H/E concentration, and stain
  separation does not depend on a tissue mask.
- Invalid channel counts, out-of-range values, and non-finite inputs are errors.
- The primary fixed wavelet backbone uses only two-dimensional complex Morlet
  wavelets. Each scale-orientation entry consists of paired fixed real and
  imaginary kernels with explicit zero-DC correction.
- H and E share exactly the same Morlet kernel bank. LoG, real Gabor, and other
  wavelet families are not primary-frontend channels and require separate approval
  as ablation baselines.
- The primary frontend computes one complex Morlet convolution followed by a
  modulus and preserves the resulting spatial scale-orientation feature maps. It
  does not compute second-order scattering paths.
- The canonical term is **first-order fixed wavelet-modulus frontend**. Project
  documentation and claims must not describe the primary model as a full or
  multi-order scattering network.
- The primary frontend fixes `J = 4` and Morlet scale indices
  `j in {0, 1, 2, 3}`. Index `j = 0` is the highest center frequency and
  smallest spatial envelope. The frozen parameters are:
  - `sigma_j = 0.8 * 2^j`;
  - `xi_j = (3*pi/4) * 2^(-j)` radians/pixel;
  - `L = 8`;
  - `gamma = 4/L = 0.5`.
- The four approximate carrier wavelengths are `8/3`, `16/3`, `32/3`, and
  `64/3` pixels for `j = 0`, `1`, `2`, and `3`, respectively. A `j = 4` scale
  is not part of the primary frontend.
- For orientation `theta`, rotated coordinates and the Gaussian envelope are:
  - `u_parallel = u_x*cos(theta) + u_y*sin(theta)`;
  - `u_perp = -u_x*sin(theta) + u_y*cos(theta)`;
  - `g_(j,theta)(u) = exp(-(u_parallel^2 + gamma^2*u_perp^2) /
    (2*sigma_j^2))`.
- The envelope standard deviation is `sigma_j` along the carrier direction and
  `sigma_j/gamma` perpendicular to it.
- The continuous infinite-support reference correction is
  `beta_inf = exp(-sigma_j^2*xi_j^2/2)`. It is used only as a theoretical
  generation cross-check.
- For the actual discrete support `Omega`, generation must compute
  `beta_disc = sum_Omega(g*exp(i*xi_j*u_parallel))/sum_Omega(g)` and
  `psi = g*(exp(i*xi_j*u_parallel) - beta_disc)`. This discrete zero-DC
  projection, rather than the continuous reference value alone, defines the final
  finite kernel.
- Coordinates, parameters, Gaussian envelopes, and correction terms are computed
  in `float64`; complex kernels are generated in `complex128` before any later
  approved runtime conversion.
- Kernel generation is implemented explicitly by this project. Kymatio is neither
  a runtime dependency nor a source of hidden experiment defaults; it may be used
  only for numerical and spectral cross-checks.
- A parameter-specification hash and kernel-tensor hash are distinct identities.
  The specification hash covers formula version, `J`, `L`, scale indices, angle
  convention, coordinate convention, and generation precision. The tensor hash is
  generated only after direction order, discrete support, zero-DC handling, and
  normalization are frozen.
- H and E must reference one shared generated Morlet kernel tensor rather than
  generate separate or stain-specific tensors.
- `gamma = 4/L` is a fixed standard initialization. After `L` is approved, the
  bank must be checked for frequency coverage, neighboring-orientation overlap,
  and coverage holes. If `L != 8`, use of this formula alone does not permit a
  claim of full equivalence to Kymatio's standard two-dimensional configuration.
- Each stain path exposes 32 ordered complex Morlet channels and 32 first-order
  modulus feature maps. H and E therefore expose 64 raw feature maps at the
  interaction boundary while sharing the same 32-kernel tensor.
- Discrete kernel coordinates are centered with `u_x` increasing rightward by
  column and `u_y` increasing downward by row. Direction indices are
  `ell in {0, ..., 7}` with `theta_ell = ell*pi/8`.
- Positive `theta` appears clockwise in image coordinates. `theta` is the complex
  carrier/frequency-vector direction along which phase varies; constant-phase
  ridges are perpendicular to it.
- Channels are scale-major then orientation-major with `c = 8*j + ell`.
- The public operator is true convolution,
  `(x*psi)[p] = sum_u x[p-u]*psi[u]`. A backend implemented with a
  cross-correlation primitive must flip both spatial kernel axes.
- The parameter-specification hash covers the convolution convention. The final
  kernel-tensor hash covers the execution-ready kernel tensor and its channel
  ordering.
- Finite support should use an odd kernel size so `(0, 0)` is an exact center
  sample.
- All 32 Morlet kernels use the common support
  `Omega = {-52, ..., 52}^2`, an odd `105 x 105` grid. Its half-width is
  `R = ceil(4 * max_j(sigma_j/gamma)) = 52`.
- Envelopes, discrete zero-DC corrections, and complex kernels are computed
  directly on the full support rather than generated at per-scale sizes and then
  padded. Values outside the kernel support are zero.
- Before runtime conversion, the scale-major, orientation-major kernel tensor has
  shape `[32, 105, 105]` and dtype `complex128`.
- After H/E separation, concentration maps are padded by 52 pixels on each side
  using `reflect` semantics that do not duplicate edge samples. `symmetric`,
  `replicate`, circular, periodic, and wraparound padding are prohibited.
- Runtime reflection requires `H > 52` and `W > 52`; a nonempty region with no
  padding influence exists only when `H > 104` and `W > 104`.
- Reflection padding is followed by valid true convolution, yielding the original
  `H x W` output size.
- Every frontend output carries a Boolean `valid_support_mask`. `True` means the
  full `105 x 105` receptive field used only original patch samples; `False` means
  at least one reflected sample was used. For a `256 x 256` patch, `True` occupies
  exactly `[52:204, 52:204]`.
- The valid-support mask describes boundary influence only and must not be called a
  tissue, foreground, or attention mask. Its use by pooling remains subject to a
  separate explicit decision.
- FFT execution computes zero-padded linear convolution on the reflected input,
  never equal-size or circular convolution. With reflected shape
  `(H + 104, W + 104)` and kernel shape `(105, 105)`, the FFT grid is fixed
  exactly to `(H + 208, W + 208)`; `next_fast_len` substitution is prohibited.
- This minimal grid freezes spectral sampling, crop semantics, and execution
  identity. It does not lock or make claims about the FFT library's internal
  algorithm, plan, or scheduling.
- Reflected input and canonical kernel are placed at the top-left of zero-filled
  arrays. Input becomes zero-imaginary `complex64`; `fft2` and `ifft2` use
  `norm="backward"`; spectra are multiplied pointwise; no `fftshift` or
  `ifftshift` is used.
- The valid FFT response crop is `[104:104+H, 104:104+W]`.
- After discrete zero-DC projection, every complex kernel is normalized as one
  complex object to unit discrete L2 energy:
  `psi_hat = psi / sqrt(sum_Omega(abs(psi)^2))`.
- Energy accumulation and normalization use the `float64`/`complex128` generation
  representation. Real and imaginary components must not be normalized
  separately; L1, peak, and additional scale-dependent normalization are
  prohibited.
- Pre-normalization energy is generation audit metadata. Unit energy and discrete
  zero-DC residual are checked after normalization and checked again after
  conversion to the approved runtime dtype.
- H/E concentrations and first-order modulus feature maps use `float32`. Generated
  `complex128` kernels convert to `complex64` execution kernels, representable as
  paired `float32` real and imaginary components.
- The fixed frontend runs inside an explicit autocast-disabled precision guard and
  checks effective TF32 configuration. Formal execution must fail if TF32,
  `float16`, `bfloat16`, or automatic mixed precision can affect the frontend.
- Spatial execution computes real and imaginary convolutions in `float32`; FFT
  execution uses `complex64`. Electronic-backend precision is not fixed here.
- The canonical normalized `complex64` kernel array has axes `[channel, y, x]`,
  shape `[32, 105, 105]`, channel `c = 8*j + ell`, center index `[52, 52]`,
  `u_x = x - 52`, and `u_y = y - 52`. It stores generated `psi(u)` before a
  spatial cross-correlation flip.
- The canonical complex64 kernel has a distinct hash over a CPU, C-contiguous,
  little-endian IEEE-754 `float32` payload of shape `[32, 105, 105, 2]`, with the
  final axis ordered real then imaginary. The spatial execution-kernel hash covers
  the two-axis-flipped execution view. Neither identity uses device-native complex
  memory layout.
- The FFT cache key directly includes the canonical complex64 kernel hash, spatial
  execution-kernel hash, input dimensions, FFT grid, dtype, normalization,
  no-shift convention, crop convention, backend name/version, and device class.
  A frequency-tensor payload hash may detect cache corruption but is not a
  cross-platform scientific identity.
- All approved identities use
  `SHA256(T || 0x00 || uint64be(len(H)) || H || P)`, rendered as lowercase
  `sha256:` plus 64 hexadecimal characters. `T` is the approved UTF-8 domain tag,
  `H` is true RFC 8785 JCS output encoded as UTF-8, and `P` is the optional
  canonical raw payload.
- Domain tags are `cg/stain-separation-spec/v1`,
  `cg/morlet-param-spec/v1`,
  `cg/morlet-kernel-canonical/v1`,
  `cg/morlet-kernel-spatial-exec/v1`, `cg/fft-cache-key/v1`, and
  `cg/fft-cache-payload/v1`.
- Ordinary key-sorted JSON is not JCS. Duplicate keys, NaN, Infinity, and JSON
  negative zero are rejected before canonicalization.
- JSON integers are restricted to the interoperable safe-integer range; larger
  exact integers use schema-defined canonical decimal strings.
- Metadata is ASCII-first. Unicode receives no implicit normalization, and exact
  code points are identity-bearing.
- Tensor `-0.0` values are converted to `+0.0` before execution and hashing.
- `len(H)` is the unsigned 64-bit big-endian count of UTF-8 bytes in the JCS
  header. An empty payload has length zero and contributes no bytes. Header
  `payload_length` must match the actual payload byte count.
- Tensor headers include dtype, shape, axis semantics, C-contiguous layout,
  endianness, and payload length. Raw bytes are hashed directly, not as Base64.
- Scientific identities exclude paths, timestamps, hostnames, and device serial
  numbers. Every domain has a fixed test vector storing the complete envelope
  preimage and expected digest.
- Spatial and FFT paths are required to agree within approved numerical tolerances,
  not bitwise.
- For complex response components `r` and `q`, the first-order modulus uses the
  stable epsilon-free `float32` primitive:
  - `a = abs(r)`, `b = abs(q)`;
  - `h = max(a, b)`, `l = min(a, b)`;
  - output zero when `h == 0`, otherwise `h * sqrt(1 + (l/h)^2)`.
- Direct `sqrt(r*r + q*q)` is not the normative definition. Squared modulus,
  intensity, log magnitude, signed complex response, post-modulus clipping,
  normalization, standardization, and learned scaling are not fixed-frontend
  outputs.
- Non-finite response components or modulus values are errors. The
  `valid_support_mask` is propagated unchanged.

The stain-separation component, fixed wavelet backbone, interaction block,
electronic backend, evaluation, and dataset adapters must remain separate modules.
Configuration must be explicit rather than hidden in source code.

### 3.2 Required H/E interaction-boundary interface

- `F_H` and `F_E` are `float32` with semantic shape
  `[B, J, L, H, W] = [B, 4, 8, H, W]`.
- A combined representation uses `[B, stain, J, L, H, W]` with stain order
  `[H, E]`.
- The shared Boolean `valid_support_mask` is `[B, 1, H, W]`.
- Axis semantics must not be lost. Temporary reshape, flatten, permutation, or
  packing is allowed only with explicit, deterministically reversible metadata for
  stain, scale, direction, and spatial axes.
- A branch may add an explicit feature axis with canonical semantic shape
  `[B, J, L, C_branch, H, W]`.
- The entry identity includes a stain-separation specification hash plus the
  Morlet parameter, canonical-kernel, and spatial-execution-kernel hashes.
- Domain `cg/stain-separation-spec/v1` covers the ordered H/E basis,
  pseudoinverse convention, RGB-to-optical-density formula, clipping, dtype,
  background semantics, and no-normalization policy under the approved JCS/SHA-256
  envelope.
- `HEInteractionBlock` is electronic-backend computation. Any trainable parameter
  it contains is excluded from the optical frontend and included in backend
  capacity, optimizer, and checkpoint audits.

### 3.3 Required cross-stain gating interface

- For each `(j, ell)`, symmetric cross-stain mutual gating is defined by
  `g_H^(j,ell) = 2 * sigmoid(a_(j,ell) * F_E^(j,ell) + b_(j,ell))` and
  `g_E^(j,ell) = 2 * sigmoid(a_(j,ell) * F_H^(j,ell) + b_(j,ell))`.
- The gated outputs are
  `Y_H^(j,ell) = F_H^(j,ell) elementwise-multiplied by g_H^(j,ell)` and
  `Y_E^(j,ell) = F_E^(j,ell) elementwise-multiplied by g_E^(j,ell)`.
- `a_(j,ell)` and `b_(j,ell)` are trainable scalars specific to each
  scale-orientation pair and shared between the two stain directions. The branch
  therefore has exactly `2 * J * L = 64` trainable parameters.
- All gate parameters initialize to zero. Consequently every gate initializes to
  one and both gated streams initially equal their corresponding input streams
  exactly.
- Each gate lies in `(0, 2)`. The opposite stain computes a gate that suppresses
  or enhances the current target-stain response. The branch performs no additive
  or replacement mixing between stain responses.
- Gating is pointwise at the same `(j, ell, y, x)` coordinate and does not mix
  spatial neighborhoods, scales, or orientations.
- The canonical output shape is `[B, J, L, C_branch, H, W]`, with
  `C_branch = 2` ordered `[H_gated, E_gated]`.
- The Boolean `valid_support_mask` is propagated bitwise unchanged and does not
  participate in the gating computation.
- All gate parameters are electronic-backend parameters and are excluded from the
  fixed-frontend identity.

### 3.4 Required same-location co-occurrence interface

- For every `(j, ell)`, same-location symmetric product co-occurrence is defined
  by `C^(j,ell) = F_H^(j,ell) elementwise-multiplied by F_E^(j,ell)`.
- The branch reads the raw fixed-frontend responses `F_H` and `F_E` and runs in
  parallel with cross-stain gating. It must not consume `Y_H` or `Y_E`.
- The product is pointwise at the same `(j, ell, y, x)` coordinate and does not
  mix spatial neighborhoods, scales, or orientations.
- The output is nonnegative and invariant under exchange of H and E. It is zero
  when either input response is zero.
- The branch has no trainable parameters and adds zero parameters to the
  electronic-backend budget.
- Its canonical output shape is `[B, J, L, C_branch, H, W]`, with
  `C_branch = 1` ordered `[HE_product]`.
- No bias, epsilon, square root, logarithm, clipping, normalization, or trainable
  scaling is part of this branch.
- The Boolean `valid_support_mask` is propagated bitwise unchanged and does not
  participate in the product.
- The feature has squared modulus-response units and denotes joint response
  strength, not a probability, correlation coefficient, or normalized score.
- Electronic-backend execution precision remains a separate unresolved decision.
- The branch does not distinguish balanced from imbalanced H/E pairs having the
  same product; stain imbalance belongs to the separately required
  difference-feature family.

### 3.5 Required neighborhood-interaction interface

- For a feature map `X`, `A_8(X)(y,x)` is one eighth of the sum at offsets
  `{-1, 0, 1} x {-1, 0, 1}` excluding `(0, 0)`, with values outside the feature
  map fixed to zero.
- For every `(j, ell)`, the branch outputs
  `N_(H<-E)^(j,ell) = F_H^(j,ell) elementwise-multiplied by
  A_8(F_E^(j,ell))` and
  `N_(E<-H)^(j,ell) = F_E^(j,ell) elementwise-multiplied by
  A_8(F_H^(j,ell))`.
- The branch reads raw `F_H` and `F_E` and runs in parallel with cross-stain
  gating and same-location co-occurrence.
- The center offset is excluded. Spatial mixing occurs only within the same
  `(j, ell)` and does not mix scales or orientations.
- Exchanging H and E exchanges the two feature channels.
- The branch has no trainable parameters and adds zero parameters to the
  electronic-backend budget.
- Its canonical output is `[B, J, L, C_branch, H, W]`, with `C_branch = 2`
  ordered `[H_x_E_ring, E_x_H_ring]`.
- Feature-map boundaries use zero padding and the sum is always divided by eight;
  the divisor is not reduced at an edge.
- The original Boolean `valid_support_mask` remains available bitwise unchanged.
  The branch additionally emits a Boolean `neighborhood_valid_support_mask` with
  shape `[B, 1, H, W]`. It is `True` only when the center and all eight
  neighboring values of the original mask are `True`; mask values outside the
  map are `False`.
- Neither mask participates in the numeric neighborhood mean or product.
- For a `256 x 256` input, the original valid region `[52:204, 52:204]` produces
  the exact neighborhood-valid region `[53:203, 53:203]`, with size
  `150 x 150`.
- No bias, epsilon, nonlinearity, clipping, normalization, or trainable scaling is
  part of the branch. Its outputs have squared modulus-response units.
- Electronic-backend execution precision remains a separate unresolved decision.
- Larger, dilated, orientation-aligned, scale-dependent, and learned neighborhoods
  are excluded from the primary branch and require explicit approval as future
  ablations.

### 3.6 Required difference-feature interface

- For every `(j, ell)`, bidirectional nonnegative stain-excess responses are
  `D_H^(j,ell) = max(F_H^(j,ell) - F_E^(j,ell), 0)` and
  `D_E^(j,ell) = max(F_E^(j,ell) - F_H^(j,ell), 0)`.
- The branch reads raw `F_H` and `F_E` and runs in parallel with cross-stain
  gating, same-location co-occurrence, and neighborhood interaction.
- The operation is pointwise at the same `(j, ell, y, x)` coordinate and does not
  mix spatial neighborhoods, scales, or orientations.
- `D_H` is H excess over E and `D_E` is E excess over H. Both are zero when the
  inputs are equal.
- At every position, at most one output is nonzero,
  `D_H + D_E = abs(F_H - F_E)`, and `D_H - D_E = F_H - F_E`.
- Exchanging H and E exchanges the two feature channels.
- The branch has no trainable parameters. The four approved interaction branches
  contain exactly 64 trainable parameters in total, all belonging to cross-stain
  gating.
- Its canonical output is `[B, J, L, C_branch, H, W]`, with `C_branch = 2`
  ordered `[H_excess, E_excess]`.
- The Boolean `valid_support_mask` is propagated bitwise unchanged and does not
  participate in the calculation. This branch does not consume
  `neighborhood_valid_support_mask`.
- Outputs are nonnegative and retain modulus-response units. No bias, epsilon,
  ratio, logarithm, normalization, trainable scaling, or additional clipping is
  part of the branch.
- Non-finite inputs or results are errors and must not be replaced by zero.
- Electronic-backend execution precision remains a separate unresolved decision.
- Combining the four branch outputs is frozen in Section 3.7. Selecting a support
  mask for later spatial pooling remains a separate unresolved decision.

### 3.7 Required combined interaction-feature interface

- The four approved interaction branches remain parallel and are combined only
  by deterministic concatenation along the interaction-feature axis.
- The canonical combined output is
  `[B, J, L, C_interaction, H, W] = [B, 4, 8, 7, H, W]`.
- The `C_interaction` axis is ordered exactly as
  `[H_gated, E_gated, HE_product, H_x_E_ring, E_x_H_ring, H_excess,
  E_excess]`.
- Combination must not add summation, averaging, projection, renewed gating,
  nonlinearity, normalization, or implicit scaling.
- The combination step has no trainable parameters. The complete interaction
  module has exactly 64 trainable scalars, all belonging to cross-stain gating.
- `J`, `L`, and `C_interaction` must not be irreversibly or implicitly flattened.
  A temporary representation with `4 * 8 * 7 = 224` channels is allowed only
  when accompanied by a deterministic, reversible `(j, ell, feature_name)`
  mapping back to the canonical layout.
- Unit metadata is mandatory:
  - `H_gated`, `E_gated`, `H_excess`, and `E_excess` have
    modulus-response units;
  - `HE_product`, `H_x_E_ring`, and `E_x_H_ring` have squared
    modulus-response units.
- The combination step must not reconcile, normalize, or implicitly rescale the
  two unit families.
- The combined interface carries the unmodified Boolean `valid_support_mask` and
  the unmodified Boolean `neighborhood_valid_support_mask`, each with shape
  `[B, 1, H, W]`.
- Feature-to-mask bindings are fixed as:
  - `H_gated`, `E_gated`, `HE_product`, `H_excess`, and `E_excess` bind to
    `valid_support_mask`;
  - `H_x_E_ring` and `E_x_H_ring` bind to
    `neighborhood_valid_support_mask`.
- Under exchange of H and E, the zero-based `C_interaction` permutation is
  `[1, 0, 2, 4, 3, 6, 5]`. Both support masks remain bitwise unchanged.
- Electronic-backend execution precision, AMP policy, and output dtype are frozen
  in Section 3.12. The spatial-pooling support-mask policy is frozen in Section
  3.8.
- This interface does not alter any branch contract frozen in Sections 3.3
  through 3.6.

### 3.8 Required spatial-pooling support interface

- Define `pooling_support_mask` as
  `valid_support_mask AND neighborhood_valid_support_mask`. It must be bitwise
  identical to `neighborhood_valid_support_mask`.
- `neighborhood_valid_support_mask` must be a subset of
  `valid_support_mask`. The invariant is checked before pooling and any violation
  fails closed.
- `valid_support_mask`, `neighborhood_valid_support_mask`, and
  `pooling_support_mask` are separate named Boolean interface fields, each with
  shape `[B, 1, H, W]`. Each retains its distinct semantic identity and none may
  overwrite or mutate another.
- For every sample and every spatial-pyramid region, all
  `J * L * C_interaction = 4 * 8 * 7 = 224` channels use exactly the same set of
  positions selected by `pooling_support_mask`.
- Feature-specific native mask bindings remain available for audit but must not
  select different pooling positions for different feature channels.
- Pooling uses genuine mask-aware reductions. Invalid locations are excluded
  before evaluating a statistic and must not be zeroed and then included in an
  ordinary reduction.
- The valid-pixel count is computed exactly for every sample and every region and
  is the common sample count for all 224 channels in that region.
- An empty region is an error detected before reduction. It must not produce
  zero, NaN, a default statistic, or a silently omitted output.
- For a `256 x 256` input, `pooling_support_mask` is `True` exactly on
  `[53:203, 53:203]`, containing `150 * 150 = 22500` positions.
- Relative to the original `[52:204, 52:204]` valid square, the pooling support
  contracts by a one-pixel border on each of the square's four sides. This
  removes 604 of 23104 original-valid positions, approximately 2.61 percent.
- Nonempty global pooling support requires `H > 106` and `W > 106`. Every later
  spatial-pyramid configuration must independently prove that each configured
  region is nonempty.
- All three masks describe computational support only. They are not tissue,
  foreground, attention, lesion, label, or stain-intensity masks and are not
  intersected with such data-dependent masks.
- This policy has no trainable parameters.
- Spatial-pyramid levels and region-boundary allocation are frozen in Section
  3.9, statistics are frozen in Section 3.10, and electronic-backend precision
  is frozen in Section 3.12.

### 3.9 Required spatial-pyramid region interface

- Spatial-pyramid levels are ordered `[1, 2, 4]`, giving
  `P = 1^2 + 2^2 + 4^2 = 21` regions.
- For input dimensions `(H, W)`, the canonical support rectangle is fixed by
  geometry as `[53:H-53, 53:W-53]`, with `H_s = H - 106` and
  `W_s = W - 106`.
- At level `n`, boundaries are
  `y_(n,r) = 53 + floor(r * H_s / n)` and
  `x_(n,c) = 53 + floor(c * W_s / n)` for indices from zero through `n`.
  Region `(n, r, c)` is the half-open rectangle
  `[y_(n,r):y_(n,r+1), x_(n,c):x_(n,c+1)]`.
- Boundaries are anchored only to the canonical rectangle determined by `H` and
  `W`. They must not be derived from an individual sample's realized mask
  bounding box, tissue extent, labels, or feature values.
- Levels are traversed in `[1, 2, 4]` order. Within a level, rows proceed from top
  to bottom and columns within each row proceed from left to right.
- The zero-based region index is `p = o_n + r * n + c`, with offsets
  `o_1 = 0`, `o_2 = 1`, and `o_4 = 5`. Therefore `p = 0` is the `1 x 1`
  region, `p = 1..4` are the `2 x 2` regions, and `p = 5..20` are the
  `4 x 4` regions.
- Regions within a level are disjoint, leave no gaps, and exactly tile the
  canonical support rectangle. Overlap between different pyramid levels is
  intentional.
- Nondivisible dimensions produce region extents differing by at most one pixel.
  Padding, resampling, overlapping adaptive-pooling windows, and discarded
  remainder pixels are prohibited.
- Each region is intersected with `pooling_support_mask` and counted under the
  approved mask-aware pooling contract.
- For a `256 x 256` input, the support is `150 x 150`; the `2 x 2` boundaries
  are `[53, 128, 203]`, the `4 x 4` boundaries are
  `[53, 90, 128, 165, 203]`, and the finest-level extents along each axis are
  `[37, 38, 37, 38]`.
- The `4 x 4` level requires `H >= 110` and `W >= 110`. Smaller inputs fail
  closed before reduction; the fixed frontend retains its separate input
  contract.
- Static geometry metadata is batch-independent for a fixed `(H, W)` and contains
  21 ordered records
  `(p, level, row, column, y_start, y_end, x_start, x_end)`.
- Per-sample counts are stored separately as exact integer values in
  `valid_count` with shape `[B, 21]`. `valid_count[b, p]` must not be embedded in
  the static geometry records.
- All 224 scale-orientation-interaction channels share the same 21 regions and
  the same `valid_count[b, p]` within a sample-region pair.
- Each approved statistic produces `224 * 21 = 4704` scalar values before
  later normalization or classifier processing.
- Region construction has no trainable parameters and performs no weighting,
  averaging, or fusion between regions.
- Spatial-pyramid statistics and statistic-axis order are frozen in Section 3.10.
  The pooled-feature normalization policy is frozen in Section 3.11.
  Electronic-backend execution precision, AMP policy, and output dtype are frozen
  in Section 3.12. Classifier structure remains unresolved.
- Across different input dimensions, the pyramid is relative to the canonical
  support rectangle and makes no fixed-physical-scale claim.

### 3.10 Required spatial-pyramid statistic interface

- For sample `b` and region `p`, let `S_(b,p)` be the selected-position set and
  `N_(b,p) = valid_count[b,p]`.
- For each `(b, j, ell, c, p)`, regional mean is
  `mu = sum_(y,x in S_(b,p)) Z[b,j,ell,c,y,x] / N_(b,p)`.
- Regional population standard deviation is
  `sigma = sqrt(sum_(y,x in S_(b,p)) (Z[b,j,ell,c,y,x] - mu)^2 /
  N_(b,p))`.
- The statistic set and order are exactly `[mean, population_std]`. Population
  standard deviation uses denominator `N_(b,p)`, never `N_(b,p) - 1`.
- `N_(b,p) = 0` fails before reduction. At `N_(b,p) = 1`, mean is the sole
  selected value and population standard deviation is exactly zero.
- The normative algorithm is two-pass over the same selected positions: first
  compute mean, then compute the mean centered squared deviation.
  `E[Z^2] - E[Z]^2` is not a normative variance formula.
- At zero variance, forward `population_std` is exactly zero. Its backward
  convention is `d sigma / d variance = 0`; at positive variance it is
  `1 / (2 * sqrt(variance))`. Backward values must remain finite, and epsilon
  must not be added to the forward variance or square root.
- Non-finite checks cover every selected input value participating in reduction,
  first-pass sums and means, centered deviations, squared deviations, second-pass
  sums, variances, standard deviations, and every emitted statistic. Any
  non-finite value is an error.
- The canonical output is
  `[B, J, L, C_interaction, P, S_stat] = [B, 4, 8, 7, 21, 2]`, with
  `S_stat` ordered `[mean, population_std]`.
- Both statistics inherit source feature units. The four modulus-response
  features retain modulus-response units and the three squared-modulus-response
  features retain squared-modulus-response units.
- No statistic mixes scale, orientation, interaction feature, or pyramid region.
- H/E exchange applies `[1, 0, 2, 4, 3, 6, 5]` only to `C_interaction`; `P` and
  `S_stat` remain unchanged.
- Each sample emits exactly `4 * 8 * 7 * 21 * 2 = 9408` pooled scalars.
- Canonical flattening traverses
  `(j, ell, feature, p, statistic)`, with statistic varying fastest. The
  zero-based linear index is
  `q = ((((j * 8 + ell) * 7 + feature) * 21 + p) * 2 + statistic)`,
  covering exactly `q = 0..9407`.
- Flattening retains a complete reversible
  `(j, ell, feature_name, p, statistic_name)` mapping.
- These statistics add no trainable parameters. Maxima, minima, medians,
  quantiles, RMS, skewness, kurtosis, and higher moments are excluded from the
  primary model and require explicit approval as ablations.
- Pooled-feature normalization is frozen in Section 3.11, reduction execution and
  accumulation dtypes are frozen in Section 3.12, and statistical summation order
  is frozen in Section 3.13. Classifier structure remains unresolved.

### 3.11 Required pooled-feature handoff interface

- The structured pooled output
  `[B, J, L, C_interaction, P, S_stat] = [B, 4, 8, 7, 21, 2]` is reshaped
  directly to classifier input `[B, 9408]` using the approved canonical index.
- This handoff permits shape change only. Every flattened value must be
  numerically identical to its structured source.
- Centering, standardization, L1 or L2 normalization, RMS normalization,
  LayerNorm, BatchNorm, whitening, logarithms, clipping, and all other numeric
  transformations are prohibited between pooled output and classifier input.
- No dataset normalization mean, variance, quantile, scale, or other fitted
  statistic may be computed or stored from training, validation, test, or
  transfer data.
- The handoff adds no trainable parameters, running state, or persistent buffers.
  The interaction and pooling path has exactly 64 trainable scalars, all in
  cross-stain gating.
- Unit metadata and coordinate counts remain explicit: 5376 coordinates derive
  from the four modulus-response features and 4032 derive from the three
  squared-modulus-response features.
- The classifier has the capacity to adapt to coordinate-scale differences, but
  realized behavior depends on the later-approved optimizer, initialization, and
  regularization.
- All 9408 classifier-input coordinates are checked for finiteness before
  classifier evaluation. Any non-finite value fails closed.
- For batches `b,b'` and flattened coordinates `q,q'`, the handoff Jacobian is
  `d x_head[b,q] / d v[b',q'] = 1` exactly when `b = b'` and `q = q'`, and
  zero otherwise. The handoff performs no stop-gradient or gradient clipping.
- Define `pi = [1, 0, 2, 4, 3, 6, 5]` and
  `q(j,ell,c,p,s) = ((((j*8+ell)*7+c)*21+p)*2+s)`.
  The exact flattened H/E exchange permutation is
  `Pi(q(j,ell,c,p,s)) = q(j,ell,pi[c],p,s)`.
- Equivalently,
  `v_swap[b, q(j,ell,c,p,s)] =
  v[b, q(j,ell,pi[c],p,s)]`. `Pi` must be an involutive bijection over
  `0..9407`.
- Read-only distribution diagnostics may report summaries, histograms,
  non-finite counts, and scale disparities. Diagnostics must not change forward
  values, create model state, or automatically control normalization, clipping,
  loss weighting, sampling, optimizer settings, schedules, early stopping, or
  any other data-adaptive training rule.
- Diagnostics may inform a later explicit human decision but may not silently
  modify the primary pipeline.
- The canonical dtype conversion and AMP policy are frozen in Section 3.12, and
  statistical summation order is frozen in Section 3.13. Classifier architecture
  remains unresolved.
- Normalization inside a future classifier is not implicitly authorized and must
  be resolved with the classifier architecture.

### 3.12 Required electronic-backend precision interface

- Fixed-frontend outputs, interaction computations, combined interaction output,
  pooled output, classifier input, and every parameter, activation, and binary
  logit in the primary classifier use float32.
- Cross-stain gating parameters and their gradients are float32.
- The two-pass spatial mean and population-standard-deviation path promotes
  selected float32 values explicitly to float64. First-pass sums and means,
  centered deviations, squared deviations, second-pass sums, population
  variances, and square roots all execute in float64.
- The only canonical forward float64-to-float32 conversion occurs after both
  pooled statistics are complete. The structured pooled tensor, 9408-coordinate
  identity vector, and classifier input are float32.
- Before conversion, every float64 result must be finite and satisfy
  `abs(x) <= finfo(float32).max`. Every converted float32 value must also be
  finite. Either failure fails closed. Exact representability in float32 is not
  required.
- `valid_support_mask`, `neighborhood_valid_support_mask`, and
  `pooling_support_mask` are Boolean. Region indices, boundary coordinates, and
  `[B, 21] valid_count` are int64. A count converts explicitly to float64 only
  when used in statistical arithmetic.
- Exact-zero variance uses the approved zero-gradient convention through a
  validated operator or safe custom backward implementation. Float64 precision
  alone does not make naive square-root backward finite at zero.
- Constant-region and singleton tests must prove exact-zero forward standard
  deviation and finite zero gradients without forward epsilon.
- The complete primary training step prohibits AMP, float16, bfloat16, automatic
  dtype downgrade, and dynamic or static gradient scaling. This covers forward
  computation, loss evaluation, backward computation, and the optimizer step.
- The electronic backend executes inside an explicit autocast-disabled precision
  domain.
- TF32 is prohibited to preserve IEEE float32 semantics. Effective TF32 state is
  checked through APIs appropriate to the installed PyTorch version, including
  audit of relevant environment-variable and backend overrides. Formal execution
  fails closed if TF32 can affect the path.
- Runtime checks cover known reduced-precision modes, automatic downcasts,
  stochastic rounding, and other discoverable random-rounding settings. The
  policy does not claim cross-device bitwise identity or prohibit unspecified
  implementation differences that cannot be inspected.
- Interaction values, pooled values, classifier inputs, logits, and gating
  gradients receive fail-closed non-finite checks. Every float64 value crossing
  to float32 also receives the approved finite-range check.
- Backpropagation through float64 statistical intermediates returns across an
  explicit float64-to-float32 gradient boundary. Gating gradients must be finite
  float32 values.
- Explicit configuration, checkpoint metadata, and runtime audit include the
  precision-policy version, all approved dtypes, autocast state, AMP and
  gradient-scaling state, effective TF32 state, relevant environment overrides,
  and observed cast boundaries.
- This precision policy adds no trainable parameters or persistent normalization
  state. The interaction and pooling path retains exactly 64 trainable gating
  scalars.
- Float32 is mandatory for the primary classifier. Any lower, higher, or mixed
  classifier precision requires a separate explicit decision.
- Statistical summation order is frozen in Section 3.13. The cross-device
  reference strategy and forward comparison objects are frozen in Section 3.14.
  The backward comparison object and injection boundary are also frozen in
  Section 3.14, including the upstream-gradient fixtures and non-finite handling.
  The tolerance constants, comparator arithmetic precision, and normative-zero
  and near-zero comparison rules are frozen there as well. Loss definition,
  explicitly selected loss-internal precision, and optimizer-state precision
  remain unresolved.

### 3.13 Required statistical summation-order interface

- For each sample-region pair, selected spatial values form the reduction leaves
  in strict row-major `(y, x)` order.
- Every leaf is converted to float64 before it enters the reduction tree.
- The normative sum is a balanced adjacent-pair reduction. At each level,
  adjacent entries are paired from the start of the sequence and each pair
  performs exactly one float64 addition.
- If a level contains an odd number of entries, its final unpaired entry passes
  unchanged into the next level. The same rule repeats until one result remains.
- The first-pass sum used to compute the mean and the second-pass sum of centered
  squared deviations use exactly the same leaf ordering, pair relationships, and
  level topology.
- Centered squared deviations are computed in float64 before becoming leaves of
  the second-pass reduction tree.
- A framework-native sum whose reduction order is not explicitly guaranteed,
  parallel atomic accumulation, or a device-dependent reduction implementation
  must not supply the normative result.
- An optimized implementation may execute independent nodes within one level in
  parallel, but it must not change leaf order, pair relationships, or the
  topology between levels.
- Statistical execution and accumulation dtypes remain governed by Section 3.12.
  The cross-device reference strategy is frozen in Section 3.14.

### 3.14 Required cross-device numerical-equivalence reference interface

- The normative CPU statistical implementation is the sole reference path for
  cross-device numerical acceptance.
- The CPU reference and every device under test receive content-identical
  serialized float32 inputs, Boolean masks, and int64 valid counts.
- Before execution, input identity is confirmed through length, shape, dtype, and
  content-hash checks. An upstream input difference must not be classified as a
  device-computation difference.
- The CPU reference strictly executes the approved float64 two-pass statistics
  and canonical balanced adjacent-pair reduction, including the frozen spatial
  leaf order, odd-node carry rule, and precision-conversion boundaries.
- Device outputs are compared numerically against CPU reference outputs under
  the fixed tolerances and comparator semantics below.
- Cross-device acceptance does not require bitwise identity. Differing bytes
  alone must not cause failure.
- The CPU reference implementation version, code identity, and effective
  execution configuration are part of its auditable identity. A reference
  implementation update must not silently change the acceptance baseline.
- The first formal forward comparison object is the float64 structured statistic
  tensor containing `[mean, population_std]` for every valid region, with shape
  `[B, 4, 8, 7, 21, 2]` and the canonical pooled axis and region order.
- The second formal forward comparison object is the canonical float32 pooled
  tensor with shape `[B, 4, 8, 7, 21, 2]`, produced from the first comparison
  object by exactly one approved float64-to-float32 conversion.
- Both formal layers are compared elementwise against the CPU reference at the
  same logical indices. The first layer validates the two-pass statistics,
  reduction tree, and standard-deviation result; the second validates the
  canonical conversion boundary and the values that enter the classifier path.
- The `[B, 9408]` classifier input is not a separate cross-device numerical
  comparison object because it is an arithmetic-free reshape of the canonical
  float32 pooled tensor.
- Local tests still verify equal element counts before and after reshape, the
  approved flattening order, and elementwise identical stored values between the
  reshaped classifier input and its canonical float32 pooled source.
- Reduction-tree internal nodes, the first-pass total, centered squared
  deviations, and the second-pass total are failure diagnostics only. They are
  not independent pass conditions, do not replace either formal comparison
  layer, and need not be exported for an implementation to pass.
- The formal backward comparison injects a content-identical float32 upstream
  gradient `G` at the canonical post-conversion pooled tensor
  `[B, 4, 8, 7, 21, 2]`. `G` uses the fixed pooled axis order and auditable
  serialized content.
- Backward is the vector-Jacobian product equivalent to
  `backward(gradient=G)`. It must not construct an explicit scalar loss through
  `sum`, `mean`, or another additional reduction.
- The sole formal backward comparison object is the float32 pooling-input
  gradient `dZ` with shape `[B, 4, 8, 7, H, W]`.
- `dZ` covers the complete spatial input. Positions excluded by the applicable
  region or mask remain within the comparison object, with the normative result
  reflecting their zero contribution.
- CPU and device runs start with no historical gradients and perform exactly one
  backward pass. Gradient accumulation must not affect the result.
- The CPU reference backward reuses the frozen forward precision path: float32
  input, float64 internal statistics, two-pass computation, canonical balanced
  adjacent-pair reductions, and one float64-to-float32 statistics conversion.
- Boolean masks, region indices, and int64 valid counts are nondifferentiable
  inputs and have no compared gradients. Classifier, gating, and other
  electronic-backend parameter gradients are outside this acceptance scope.
- Intermediate-statistic, mean, and standard-deviation gradients are optional
  failure diagnostics, not independent pass conditions.
- The approved zero-variance mathematics remains unchanged:
  `d sigma / d variance = 0`, forward adds no epsilon, backward results are
  finite, and the standard-deviation branch contributes zero pooling-input
  gradient.
- A nonzero upstream gradient in the mean slot may make the complete `dZ`
  nonzero in a zero-variance region. When a fixture injects gradient only into
  the standard-deviation slot, its normative `dZ` is zero.
- The tolerance rules below do not alter, relax, or redefine the approved
  zero-variance mathematical semantics.
- Fixture indices are zero-based integers:
  `b in [0, B-1]`, `j in [0, 3]`, `ell in [0, 7]`, `c in [0, 6]`, and
  `p in [0, 20]`. The final structured slot axis is ordered `[mean, std]`.
- Define
  `r = (((j * 8 + ell) * 7 + c) * 21 + p)` and `u = b * 4704 + r`.
  `mod` is nonnegative integer remainder and `floor(u/4)` is integer floor.
- The mean-branch fixture is
  `G[..., mean] = 1.0` and `G[..., std] = 0.0`.
- The standard-deviation-branch fixture is
  `G[..., mean] = 0.0` and `G[..., std] = 1.0`.
- The joint signed fixture is
  `G[..., mean] = (-1)^u * (1 + (u mod 4)) / 4` and
  `G[..., std] = (-1)^floor(u/4) * (1 + ((3*u + 1) mod 4)) / 8`.
- Fixture signs and numerators are determined through integer operations before
  exact conversion to float32. Floating-point `pow`, floating-point remainder,
  and device-dependent random generation are prohibited.
- All fixture values are exactly representable binary fractions in float32. Each
  fixture is generated and serialized in the same canonical axis order, and the
  CPU and device paths receive content-identical float32 values.
- Each of the three fixtures runs as a separate backward acceptance test from a
  fresh gradient state. They must not be added together and tested through only
  one backward pass.
- Every formal comparison object undergoes a complete finiteness check before
  absolute or relative error is evaluated. The rule covers the float64 pre-cast
  statistic tensor, the float32 canonical pooled tensor, and the float32 `dZ`
  from each of the three separate upstream-gradient fixtures.
- If either side contains `NaN`, `+Inf`, or `-Inf`, tolerance comparison for that
  object must not proceed. A non-finite value is never accepted as equal to
  another non-finite value, including at the same index with the same type.
- A non-finite CPU reference marks the reference execution or fixture invalid,
  blocks device acceptance for that object and fixture, and is not recorded as a
  device failure. Device-to-device comparison must not replace the CPU baseline.
- When the CPU reference is wholly finite and a tested device contains a
  non-finite value, that device fails the corresponding object and fixture.
- Non-finite values must not be replaced, clipped, clamped, zeroed, skipped, or
  excluded from error statistics.
- Diagnostics distinguish `NaN`, `+Inf`, and `-Inf`. Each failure or block record
  includes comparison layer and dtype, complete logical index, CPU or tested
  device identity, forward fixture or backward `G` fixture, non-finite type, and
  verdict `reference_invalid` or `device_failed`.
- The first non-finite value determines that the object cannot pass. An
  implementation may continue scanning that object to collect additional
  diagnostic indices, but must not resume tolerance evaluation or alter the
  verdict.
- After both sides pass finiteness checks, every logical element uses the
  asymmetric CPU-reference rule
  `abs(x_device - x_CPU) <= atol + rtol * abs(x_CPU)`.
- The CPU value is always the reference. The operands must not be exchanged and
  symmetric relative error must not replace the approved rule.
- The inequality is closed: an error exactly equal to its tolerance bound passes.
- Every element is judged independently at its complete logical index. One
  failing element fails its entire formal comparison object.
- RMSE, MAE, maximum error, other aggregates, mean pass rate, quantiles, or an
  allowed failure fraction must not replace the elementwise gate. Aggregate
  metrics may be diagnostic only and cannot alter the verdict.
- The normative tolerance profile is:

  | Formal comparison object | Normative dtype | `atol` | `rtol` | Near-zero threshold |
  | --- | --- | --- | --- | --- |
  | Pre-cast structured `[mean, population_std]` tensor | float64 | `0x1p-48` | `0x1p-47` | `0x1p-40` |
  | Post-conversion canonical pooled tensor | float32 | `0x1p-22` | `0x1p-21` | `0x1p-16` |
  | Pooling-input gradient `dZ`, all three backward fixtures | float32 | `0x1p-21` | `0x1p-19` | `0x1p-16` |

- The hexadecimal values above denote exact powers of two. `atol`, `rtol`, and
  the near-zero threshold are stored and used as normative binary64 constants;
  decimal parsing must not define their values.
- Let binary64 unit roundoff be `u64 = 0x1p-53` and binary32 unit roundoff be
  `u32 = 0x1p-24`. The float64 profile allows `32*u64` absolute and `64*u64`
  relative error around a fixed two-pass tree, covering a small number of
  correctly rounded elementary-operation differences without accepting a
  changed reduction topology. The pooled float32 profile allows `4*u32`
  absolute and `8*u32` relative error after the single conversion. The `dZ`
  profile allows `8*u32` absolute and `32*u32` relative error for the additional
  derivative and cast path. These are fixed conservative budgets, not measured
  values or runtime-tuned thresholds.
- Calibration evidence must include deterministic synthetic fixtures spanning
  exact zeros, subnormal-adjacent values with FTZ/DAZ disabled, values on both
  sides of each near-zero threshold, signed cancellation, nonzero variance,
  zero variance, every structured axis, and every formal dtype boundary. For
  every declared supported device/configuration, the largest observed valid
  error must be no more than one quarter of the applicable normative bound.
  Exceeding that calibration margin blocks acceptance and requires root-cause
  analysis; it never widens a tolerance automatically.
- Negative calibration controls must demonstrate that axis permutation, mean/std
  slot exchange, sign inversion, one-position spatial displacement, wrong mask
  contribution, changed reduction order, and an extra or missing float32 cast
  fail at least one element. A tolerance profile that lets any applicable
  negative control pass is invalid even if ordinary fixtures pass.
- After formal values are returned to CPU in their specified original dtype,
  each float32 value is converted to binary64 by exact widening. A float64 value
  remains binary64 and must not be downcast, requantized, or otherwise rounded
  before comparison. A tested device must not produce the final pass/fail
  verdict.
- After the approved object-level finiteness checks, comparator intermediates
  are formed on CPU in exactly this order:
  `difference = round64(x_device - x_CPU)`;
  `error = abs(difference)`;
  `relative_term = round64(rtol * abs(x_CPU))`;
  `bound = round64(atol + relative_term)`.
  The final gate is `error <= bound`, so equality passes. On finite binary64
  inputs, `abs` changes only the sign bit and introduces no numerical rounding.
- The comparator uses IEEE 754 round-to-nearest, ties-to-even. Multiplication and
  addition must not be fused into FMA; arithmetic reassociation, extended-
  precision temporaries, lower-precision intermediates, FTZ, and DAZ are
  prohibited.
- The formal object non-finite rules above remain authoritative. If both formal
  inputs are finite but `difference`, `relative_term`, or `bound` becomes
  non-finite, the verdict is `comparator_invalid`. It blocks that acceptance
  run, is not a device failure, and must not fall back to another formula.
- If `x_CPU` is either signed zero, `x_device` must be either signed zero;
  `+0` and `-0` are equal for this rule. Thus every normative zero, including an
  excluded-position gradient and a zero-variance standard-deviation-only `dZ`,
  uses an exact numerical-zero gate with no `atol` allowance.
- If `0 < abs(x_CPU) <= near_zero_threshold`, the relative term is computed and
  recorded normally, but the tightened pass condition is `error <= atol`.
  Values above the threshold use the general `error <= bound` condition. The
  near-zero rule can only tighten the general formula.
- All three backward fixtures use the single `dZ` tolerance profile. In a
  zero-variance region, the mean-only and joint fixtures compare the complete
  normative `dZ` normally, including the exact-zero rule at every coordinate
  whose CPU reference is zero. For the standard-deviation-only fixture, the
  complete normative `dZ` of that region is zero and every corresponding device
  coordinate must be numerical zero.
- For every compared object, diagnostics record acceptance-run ID, comparator
  version and code hash, tolerance-profile version, CPU-reference and tested-
  device identities, effective floating-point configuration, object name and
  dtype, fixture name and content hashes, shape, element count, `atol`, `rtol`,
  near-zero threshold, verdict, failing-element count, and whether
  `reference_invalid`, `device_failed`, or `comparator_invalid` occurred.
- The complete logical index uses canonical `[b, j, ell, c, p, slot]` order for
  pooled statistics and `[b, j, ell, c, y, x]` order for `dZ`. Canonical
  row-major scan order determines the first failing index. Diagnostics at that
  index record CPU value, device value, `difference`, `error`,
  `relative_term`, `bound`, and the active general, near-zero, or exact-zero
  rule.
- The maximum finite absolute error is recorded for every object, including a
  passing object. Ties use the first canonical logical index. It is diagnostic
  only and cannot change the verdict. If tolerance comparison is prohibited by
  a formal non-finite value, no numerical maximum is reported; the first
  non-finite record is used. If a later comparator intermediate is invalid, the
  finite input values and first canonical `comparator_invalid` index are
  recorded.
- One formal object passes only if identities, shape, dtype, and element count
  match; both formal inputs pass finiteness checks; no comparator intermediate
  is invalid; and every element passes its active rule.
- One fixture passes on one device only if its serialized inputs, masks, valid
  counts, and fixture content match the CPU reference and every formal object
  required by that fixture passes. A backward fixture has exactly one formal
  object, `dZ`; a forward fixture requires both forward objects.
- One tested device passes only if every required forward fixture and each of
  the three separate backward fixtures pass, and all required audit-identity and
  floating-point-environment checks pass. Missing or skipped required evidence
  blocks rather than passes the device.
- Section 3.14 passes overall only if the acceptance manifest names at least one
  non-CPU tested device and every named device passes. Device pass rates, fixture
  pass rates, or object pass rates cannot substitute for this conjunction.
- Required tests cover exact parsing of every hexadecimal constant; exact
  float32-to-float64 widening; preservation of float64 inputs; operation order;
  boundary equality and the immediately larger representable error; signed
  zeros; both sides of every near-zero threshold; formal and comparator-created
  non-finites; all negative calibration controls; complete index reconstruction;
  deterministic first-failure and maximum-error tie handling; the three fresh
  backward fixtures including zero variance; reshape identity; hierarchy
  propagation from element through overall verdict; and audit-identity changes
  for comparator code or effective floating-point configuration.
- Comparator implementation version, comparator code identity, tolerance-profile
  version, and every effective setting capable of changing floating-point
  behavior are part of the acceptance audit identity. Any mismatch between the
  recorded identity and the executed comparator blocks acceptance.

### 3.1 Required Morlet generation interface

The Phase 0 interface contract must distinguish:

- a Morlet parameter specification containing a formula-version identifier, `J`,
  `L`, scale-index convention, angle convention, coordinate convention,
  generation precision, finite-support policy, zero-DC policy, normalization
  policy, and boundary policy;
- a generated immutable kernel bundle containing the ordered complex kernel tensor,
  scale-orientation metadata, parameter-specification hash, canonical complex64
  kernel hash, spatial execution-kernel hash, FFT cache key, and evidence from
  generation-time validation;
- a frontend response containing the ordered first-order feature maps and the
  Boolean `valid_support_mask` under the locked True/False semantics.

The generator must fail while a required field remains `TBD`. It must generate the
bundle once and provide the same immutable tensor reference to the H and E paths.
The interface must expose enough metadata to reproduce channel ordering without
relying on source-code iteration order.

## 4. Phase 0 deliverables

Phase 0 is complete only when all of the following artifacts exist and agree:

1. A documented public experiment-pipeline interface.
2. Typed contracts, where practical, for RGB input, H/E channels, `F_H`, `F_E`,
   interaction features, pooled features, patch predictions, and slide
   predictions.
3. An explicit configuration schema covering every scientific and experimental
   choice, with unresolved values represented as `TBD` rather than defaults.
4. A CAM16 dataset-adapter contract that requires stable patient and slide
   identifiers and an externally supplied, immutable split manifest.
5. A leakage check that fails when a patient occurs in more than one split.
6. A fixed-frontend check proving that no optical-frontend value is registered as a
   trainable parameter or changed by an optimizer step.
7. A deterministic Morlet-generator contract with separate parameter-specification
   and kernel-tensor hashes, explicit channel metadata, and shared H/E tensor
   identity.
8. Unit tests for all implemented Phase 0 modules.
9. A smoke test that runs the public pipeline seam on synthetic data only.
10. Documentation of test commands, expected evidence, and any skipped checks.
11. Explicit human decisions for every blocking `TBD`, recorded in
    `docs/DECISIONS.md`.

Deliverables may be added in separate work items. Their existence does not by
itself pass the acceptance gate.

## 5. Phase 0 acceptance gate

The gate is **closed by default**. It passes only when:

- every Phase 0 deliverable is present;
- all unit tests for changed modules pass;
- the project smoke test passes;
- no test, fixture, or configuration silently selects a value for a `TBD`;
- patient-level split isolation is demonstrated from the approved split manifest;
- the optical frontend is demonstrated to be non-trainable and unchanged during a
  backend optimization step;
- every discrete Morlet kernel passes the approved discrete zero-DC tolerance and
  its continuous reference correction is checked against the generated
  parameters;
- repeated generation from the same specification produces identical parameter
  and kernel hashes, channel metadata, and kernel values;
- each generated complex kernel has approved unit-L2 energy and zero-DC residual
  both before and after runtime dtype conversion, and real/imaginary components
  have not been independently rescaled;
- frontend tests prove autocast is explicitly disabled, prohibited reduced
  precisions do not enter the frontend, and effective TF32 state is checked;
- canonical kernel serialization is invariant to execution device and uses the
  approved CPU `[32, 105, 105, 2]` real/imaginary layout;
- canonical-kernel tests verify `[channel, y, x]` axes, center `[52, 52]`,
  coordinate mapping, distinction from the spatial flipped execution view, and
  direct inclusion of the canonical kernel hash in the FFT cache key;
- identity tests use true RFC 8785 JCS, reject duplicate keys and prohibited
  numeric values, enforce safe integers and no implicit Unicode normalization,
  canonicalize tensor negative zero, cross-check UTF-8 header and payload lengths,
  and reproduce every domain's stored preimage/digest vector;
- interaction-boundary tests preserve or reversibly restore every axis semantic,
  verify `[H,E]` stain order and valid-support mask shape, permit explicit
  `[B,J,L,C_branch,H,W]` branch features, and reject an identity that omits the
  stain-separation specification hash;
- cross-stain gating tests verify the exact 64-scalar parameter contract,
  parameter sharing between stain directions, zero initialization, exact
  identity output at initialization, gate range `(0, 2)`, pointwise-only
  dependence, `[H_gated,E_gated]` feature order on
  `[B,J,L,C_branch,H,W]`, and bitwise-unchanged `valid_support_mask`;
- same-location co-occurrence tests verify direct use of raw `F_H` and `F_E`,
  exact elementwise multiplication, H/E exchange invariance, zero output when
  either response is zero, absence of trainable parameters and implicit
  transforms, `[HE_product]` feature order on
  `[B,J,L,C_branch,H,W]`, and bitwise-unchanged `valid_support_mask`;
- neighborhood-interaction tests verify the exact eight center-excluded offsets,
  raw `F_H` and `F_E` inputs, constant division by eight, zero feature padding,
  H/E exchange equivariance, absence of trainable parameters and implicit
  transforms, `[H_x_E_ring,E_x_H_ring]` feature order on
  `[B,J,L,C_branch,H,W]`, preservation of the original mask, and exact derivation
  of the `[B,1,H,W]` `neighborhood_valid_support_mask`, including the
  `[53:203,53:203]` result for a `256 x 256` input;
- difference-feature tests verify exact positive-part subtraction, zero output at
  equal inputs, mutual exclusivity, the signed- and absolute-difference
  identities, H/E exchange equivariance, rejection of non-finite values, absence
  of trainable parameters and implicit transforms, `[H_excess,E_excess]` feature
  order on `[B,J,L,C_branch,H,W]`, and bitwise-unchanged
  `valid_support_mask`;
- combined-interaction tests verify exact deterministic concatenation to
  `[B,4,8,7,H,W]`, the frozen seven-feature order, absence of additional
  computation and trainable parameters, the exact total of 64 interaction
  scalars, reversible 224-channel metadata, unit-family metadata, preservation of
  both support masks and every feature-to-mask binding, and the H/E exchange
  permutation `[1,0,2,4,3,6,5]`;
- spatial-pooling-support tests verify all three Boolean `[B,1,H,W]` mask
  interfaces, bitwise equality of pooling and neighborhood masks, fail-closed
  subset validation, one common selected-position set and exact count across all
  224 channels per sample-region pair, true mask-aware exclusion, pre-reduction
  rejection of empty regions, and the exact `[53:203,53:203]` 22500-position
  support for a `256 x 256` input;
- spatial-pyramid-region tests verify `[1,2,4]` level order, all 21 region
  indices under `p = o_n + r*n + c`, top-to-bottom row and left-to-right column
  traversal, geometry anchoring to `(H,W)` rather than sample mask bounding
  boxes, disjoint gap-free within-level tiling, nonoverlapping floor boundaries,
  fail-closed `H >= 110` and `W >= 110` requirements, the exact `256 x 256`
  boundary vectors, separation of 21 static geometry records from integer
  `[B,21]` `valid_count`, and the 4704-values-per-statistic count;
- spatial-pyramid-statistic tests verify two-pass masked mean and population
  standard deviation, denominator `N` rather than `N-1`, exact singleton and
  zero-variance forward behavior, finite zero-gradient backward behavior at zero
  variance without forward epsilon, non-finite rejection for every participating
  input and intermediate statistic, `[B,4,8,7,21,2]` output and statistic order,
  source-unit preservation, exact 9408-scalar size, and complete bijection of
  `q = ((((j*8+ell)*7+feature)*21+p)*2+statistic)` over `0..9407`;
- pooled-feature-handoff tests verify shape-only flattening with exact value
  preservation, absence of normalization parameters, state, and buffers, the
  5376/4032 unit-family counts, fail-closed finiteness checking of all 9408
  coordinates, the identity Jacobian with zero cross-coordinate derivatives, and
  the exact involutive flattened H/E permutation
  `Pi(q(j,ell,c,p,s)) = q(j,ell,pi[c],p,s)`;
- diagnostic-isolation tests verify that read-only summaries cannot alter forward
  values, model state, loss, sampling, optimizer or schedule configuration, early
  stopping, or any other data-adaptive training rule;
- electronic-backend-precision tests verify float32 interaction and classifier
  tensors, float64 two-pass statistic intermediates, the sole canonical forward
  float64-to-float32 boundary, Boolean masks, int64 geometry and counts,
  pre-cast float32 finite-range checks, post-cast finiteness, float32 finite gating
  gradients, and absence of added trainable parameters or normalization state;
- zero-variance-backward tests use constant and singleton regions to verify the
  validated safe operator, exact-zero forward standard deviation, finite
  zero-gradient backward behavior, and absence of forward epsilon;
- precision-guard tests verify that the complete training step rejects AMP,
  float16, bfloat16, gradient scaling, TF32, automatic downcasts, and discoverable
  stochastic-rounding modes; audits version-appropriate PyTorch state and
  environment overrides; and does not claim cross-device bitwise identity;
- every kernel independently passes `complex128` zero-DC and unit-energy absolute
  error limits of `1e-12`;
- every runtime `complex64` kernel, accumulated for audit in `complex128`,
  independently passes zero-DC and unit-energy absolute error limits of `1e-6`;
- `abs(beta_disc - beta_inf) <= 1e-2` is checked per kernel as a finite-support
  construction check rather than a response-equivalence metric;
- modulus tests cover exact zero, unbalanced real/imaginary magnitudes, non-finite
  rejection, absence of epsilon and post-modulus transforms, and propagation of
  the unchanged valid-support mask;
- the H and E paths demonstrably reference the same generated kernel tensor;
- the approved frequency-coverage, adjacent-orientation overlap, and coverage-hole
  checks pass for `L = 8`;
- boundary tests distinguish `reflect` from `symmetric` and `replicate`, reject
  `H <= 52` or `W <= 52`, and verify whether a full-valid region exists under the
  separate `H > 104` and `W > 104` conditions;
- spatial and FFT implementations return the same `H x W` true-convolution result
  and modulus maps under the per-element rule
  `abs(a-b) <= 2e-5 + 2e-4*abs(b)`, with the FFT path using zero-padded linear
  rather than equal-size circular convolution;
- per-kernel complex response RMSE is
  `sqrt(mean((real(a)-real(b))^2 + (imag(a)-imag(b))^2))` over all compared batch
  and spatial elements for that kernel and is at most `2e-6`;
- a CPU `complex128` reference path independently regenerates kernels and performs
  reflection plus true spatial convolution without reusing runtime kernels, FFT
  spectra, or derived caches;
- fixtures include constants, deterministic seeded random inputs,
  scale-orientation sinusoids, and unit impulses at the center, four edge
  midpoints, and four corners;
- every formally supported backend and device class passes the same thresholds;
  no tolerance widens automatically, non-finite values fail before comparison, and
  the valid-support mask is bitwise identical;
- spectral validation uses modular negation
  `(-k_y mod 464, -k_x mod 464)` and symmetrized magnitude
  `B(k) = sqrt(abs(F(k))^2 + abs(F(-k))^2)`, with all metrics accumulated in
  `float64`;
- peak location uses a deterministic `float64` quadratic fit to the `3 x 3`
  modular `log(power)` neighborhood around the positive-carrier discrete maximum.
  The Hessian must be negative definite, sub-bin offsets remain in `[-1, 1]`,
  carrier-direction error is at most 1 degree, and radial error is
  `abs(rho_hat-xi_j) <= min(0.075, 0.10*xi_j)` per kernel;
- symmetrized adjacent-orientation cosine overlap is in `[0.50, 0.70]` per pair,
  including modular wraparound, and adjacent-scale overlap is in `[0.45, 0.60]`
  per same-orientation pair;
- at each `xi_j` and `sqrt(xi_j*xi_(j+1))`, 1440-angle periodic bilinear sampling
  of total symmetrized power independently passes `min/median >= 0.85`,
  `min/max >= 0.75`, and angular coefficient of variation `<= 0.10`;
- these ring metrics are described only as angular coverage uniformity gates at
  specified radii, not as standalone proof of complete two-dimensional no-hole
  coverage; full per-kernel/pair/radius metrics and worst locations are retained;
- FFT tests verify the exact `(H + 208, W + 208)` grid, top-left placement,
  `norm="backward"`, absence of shifts, `[104:104+H, 104:104+W]` crop, and that
  these freeze execution identity rather than a library-internal FFT algorithm;
- no dataset split has changed relative to the approved manifest;
- no image, checkpoint, credential, or patient metadata is tracked for commit;
- a human reviews the evidence and explicitly approves entry into the next phase in
  `docs/DECISIONS.md`.

Until every condition is satisfied, the repository remains in Phase 0. Agents must
report the unmet conditions and must not begin the next phase.

## 6. Blocking unresolved decisions

The following values remain `TBD` and must not be inferred, defaulted, or selected
by an agent:

- loss definition and explicitly selected loss-internal precision, plus
  optimizer-state precision;
- trainable backend architecture and parameter budget;
- patch sampling, slide-level aggregation, decision threshold, calibration, and
  evaluation metrics;
- training seeds, optimization budget, confidence-interval method, and final-once
  test gate;
- transfer datasets, physical-scale adaptation, and transfer protocol.

Any additional unresolved scientific, data, training, or evaluation choice
discovered during Phase 0 must be added to this list as `TBD` and presented for
human approval.

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
- allowing patient-level leakage across training, validation, or test splits;
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

- Dataset identity is determined at patient level first, then slide and patch level.
- Every sample used by an experiment must be traceable to a patient, slide, split,
  and source manifest without exposing patient metadata.
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
