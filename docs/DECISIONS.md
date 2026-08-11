# Approved decisions

This file records explicit human decisions. It does not replace
`docs/DEVELOPMENT_SPEC.md`, and unresolved values remain TBD until approved.

## 2026-07-28 — Fixed H/E wavelet-modulus classification architecture

Approved:

- The system targets digital pathology tumor classification in simulation only.
- CAM16 is the primary development dataset; other pathology datasets are reserved
  for transfer evaluation.
- An RGB pathology patch is separated into hematoxylin (H) and eosin (E) channels
  before fixed-feature extraction.
- H and E are processed by the same parameter-sharing, fixed multiscale and
  multidirectional wavelet backbone.
- The optical frontend is not trainable.
- The two feature streams, `F_H` and `F_E`, are combined by a structured
  `HEInteractionBlock`.
- The interaction block represents four interaction families: cross-stain gating,
  same-location co-occurrence, neighborhood interaction, and difference features.
- Interaction features are summarized by spatial-pyramid statistical pooling and a
  lightweight trainable digital binary-classification head.
- The intended public outcomes are patch-level and slide-level tumor
  classification.
- The primary end-to-end test seam is the public experiment-pipeline entry point,
  exercised with synthetic RGB patches, slide identifiers, and explicit
  configuration.

Not decided:

- The exact equations, parameterization, and capacity budget of each H/E interaction
  family.
- Spatial-pyramid levels, statistics, neighborhood definition, and tensor layout.
- The trainable backend architecture and parameter budget.
- Patch sampling, slide-level aggregation, decision threshold, calibration, and
  evaluation metrics.
- Training seeds, optimization budget, confidence-interval method, and final-once
  test gate.
- Transfer datasets, physical-scale adaptation, and transfer protocol.

## 2026-07-28 — Separate stains without baseline stain normalization

Approved:

- Baseline H/E stain separation uses fixed optical-density color deconvolution.
- Stain separation and stain normalization are distinct concepts.
- The baseline does not perform Macenko or another stain-normalization transform.
- The baseline does not estimate stain vectors from the training set, a slide, or a
  patch.
- The H/E stain matrix is explicit locked configuration rather than a hidden
  library default.

Not decided:

- None for the baseline stain-separation transform.

## 2026-07-28 — Lock the Ruifrok–Johnston H/E basis

Approved:

- The fixed H/E basis is stored as two ordered RGB absorbance vectors:
  - hematoxylin: `[0.644211, 0.716556, 0.266844]`;
  - eosin: `[0.092789, 0.954111, 0.283111]`.
- Matrix rows are ordered H then E; columns are ordered R, G, then B.
- H/E concentrations are obtained using the explicitly computed Moore–Penrose
  pseudoinverse of the two-vector basis.
- DAB is not treated as a third stain, and no library HED default is an implicit
  part of the contract.
- Matrix values, ordering, numeric type, and the derived pseudoinverse are included
  in the fixed-frontend identity hash.
- The basis is not re-estimated from CAM16 or any other dataset.

## 2026-07-28 — Lock RGB-to-optical-density and concentration semantics

Approved:

- Canonical input is three-channel sRGB `uint8` in `[0, 255]`. A floating input is
  accepted only when explicitly declared as normalized to `[0, 1]`.
- For 8-bit channel value `I`, normalized intensity is
  `x = max(I, 1) / 255` and optical density is `OD = -ln(x)`.
- H/E concentration is `C = OD @ pinv(B)`, ordered H then E, followed by
  nonnegative clipping `C = max(C, 0)`.
- The pseudoinverse is computed in `float64` from the locked basis and included in
  the fixed-frontend identity hash. Runtime H/E output is `float32`.
- H/E concentrations remain in optical-density units; there is no per-patch,
  per-slide, or dataset-fitted normalization and no rescaling to `[0, 1]`.
- White background maps to approximately zero H/E concentration. Stain separation
  does not use a tissue mask.
- Invalid channel counts, out-of-range values, and non-finite inputs fail
  validation rather than being silently clipped.

## 2026-07-28 — Use complex Morlet wavelets for the primary frontend

Approved:

- The primary fixed wavelet backbone uses two-dimensional complex Morlet wavelets
  as its only wavelet family.
- Each scale-orientation entry is represented by paired fixed real and imaginary
  kernels.
- Morlet kernels receive an explicit zero-DC correction.
- H and E share exactly the same Morlet kernel bank.
- LoG, real Gabor, and other kernel families are excluded from the primary
  frontend. They may be considered later only as explicitly approved ablation
  baselines.

Not decided:

- None for the primary first-order Morlet filter-bank definition.

## 2026-07-28 — Use a first-order fixed wavelet-modulus frontend

Approved:

- The primary fixed frontend computes only first-order paths: one complex Morlet
  convolution followed by a modulus.
- The fixed frontend does not apply a second wavelet transform to first-order
  modulus responses.
- The canonical project term is **first-order fixed wavelet-modulus frontend**.
- The primary model must not be described as a full or multi-order scattering
  network.
- Second-order scattering paths require a later explicit decision and may only
  enter as a separately identified future model or ablation.

## 2026-07-28 — Freeze the deterministic Morlet generation formula

Approved:

- Scale indices are `j in {0, 1, ..., J - 1}`. Index `j = 0` has the highest
  center frequency and smallest spatial envelope; each increment doubles the
  envelope scale and halves the center frequency.
- The frozen parameters are `sigma_j = 0.8 * 2^j`,
  `xi_j = (3*pi/4) * 2^(-j)` radians/pixel, and `gamma = 4/L`.
- For orientation `theta`, coordinates are
  `u_parallel = u_x*cos(theta) + u_y*sin(theta)` and
  `u_perp = -u_x*sin(theta) + u_y*cos(theta)`.
- The Gaussian envelope is
  `g = exp(-(u_parallel^2 + gamma^2*u_perp^2)/(2*sigma_j^2))`.
  Its standard deviation is `sigma_j` along the carrier direction and
  `sigma_j/gamma` perpendicular to it.
- The continuous infinite-support reference correction is
  `beta_inf = exp(-sigma_j^2*xi_j^2/2)`. It is a theoretical cross-check and is
  not the sole discrete zero-DC guarantee.
- On the actual finite discrete support `Omega`, the correction is
  `beta_disc = sum(g*exp(i*xi_j*u_parallel))/sum(g)`, and the generated kernel is
  `psi = g*(exp(i*xi_j*u_parallel) - beta_disc)`.
- The discrete correction is a deterministic zero-DC projection on the generated
  grid, not an empirical approximation.
- Coordinates, parameters, envelopes, and correction values are computed in
  `float64`; complex kernels are generated in `complex128`.
- The project generates kernels explicitly and does not invoke Kymatio filter
  generation at runtime. Kymatio is a numerical and spectral cross-check only.
- The project locks formulas, values, scale numbering, coordinate conventions, and
  generation algorithm rather than a Kymatio version.
- The parameter-specification hash is separate from the kernel-tensor hash. The
  former covers the formula version, `J`, `L`, scale indices, angle convention,
  coordinate convention, and generation precision. The latter is produced only
  after direction order, finite support, discrete zero-DC handling, and
  normalization are frozen.
- H and E reference the same generated kernel tensor; they must not independently
  generate kernels or use stain-specific wavelet parameters.
- `gamma = 4/L` is the fixed standard initialization. Once `L` is chosen, frequency
  coverage, neighboring-orientation overlap, and coverage holes must be checked.
  If `L != 8`, the filter bank must not be claimed to be fully equivalent to
  Kymatio's standard two-dimensional configuration solely because this formula is
  used.
- The already approved scattering order remains first-order.

Not decided:

- None for Morlet generation, execution, and numerical validation.

## 2026-07-29 — Use four dyadic Morlet scales

Approved:

- The primary frontend uses `J = 4` with scale indices `j = {0, 1, 2, 3}`.
- The scale parameters and approximate carrier wavelengths are:
  - `j = 0`: `sigma = 0.8`, `xi = 3*pi/4`, wavelength `8/3` pixels;
  - `j = 1`: `sigma = 1.6`, `xi = 3*pi/8`, wavelength `16/3` pixels;
  - `j = 2`: `sigma = 3.2`, `xi = 3*pi/16`, wavelength `32/3` pixels;
  - `j = 3`: `sigma = 6.4`, `xi = 3*pi/32`, wavelength `64/3` pixels.
- A fifth `j = 4` scale is not part of the primary frontend and requires explicit
  approval as a future ablation.

## 2026-07-29 — Use eight Morlet orientations

Approved:

- The primary frontend fixes `L = 8`, hence `gamma = 4/L = 0.5`.
- Each stain channel has `J * L = 32` ordered complex Morlet kernels and 32
  first-order modulus feature maps.
- H and E together expose 64 raw feature maps to the interaction boundary while
  referencing the same 32-kernel tensor.
- Frequency coverage, adjacent-orientation overlap, and coverage-hole checks remain
  mandatory; choosing `L = 8` does not by itself establish numerical equivalence
  with Kymatio.

## 2026-07-29 — Lock Morlet orientation and convolution conventions

Approved:

- Discrete image coordinates are centered on the kernel, with `u_x` increasing
  rightward by column and `u_y` increasing downward by row.
- Orientation indices are `ell in {0, ..., 7}` with
  `theta_ell = ell*pi/8`, ordered from 0 through 157.5 degrees.
- Positive `theta` appears clockwise in the image coordinate system.
- `theta` is the complex carrier/frequency-vector direction along which phase
  varies. Constant-phase ridges are perpendicular to `theta`.
- Rotated coordinates remain
  `u_parallel = u_x*cos(theta) + u_y*sin(theta)` and
  `u_perp = -u_x*sin(theta) + u_y*cos(theta)`.
- Channel order is scale-major then orientation-major, with `c = 8*j + ell`.
- The public response operator is true mathematical convolution:
  `(x*psi)[p] = sum_u x[p-u]*psi[u]`.
- An execution backend whose primitive computes cross-correlation must flip both
  spatial kernel axes. The parameter-specification hash covers the convolution
  convention, and the final kernel-tensor hash covers the execution-ready tensor
  ordering supplied to the primitive.
- Finite support should preferentially use an odd kernel size so the discrete
  center is the exact sample `(0, 0)`; the approved size is recorded below.

## 2026-07-29 — Use one 105-by-105 Morlet support

Approved:

- All 32 Morlet kernels use the common square support
  `Omega = {-52, ..., 52}^2`, giving an odd `105 x 105` grid centered exactly at
  `(0, 0)`.
- The half-width follows
  `R = ceil(4 * max_j(sigma_j/gamma)) = 52`; the kernel size is `2*R + 1`.
- Every scale and orientation computes its envelope, discrete zero-DC correction,
  and complex kernel directly on the full support. Smaller kernels are not
  generated and then zero-padded.
- Values outside the finite kernel support are zero. Input-image padding remains a
  separate unresolved decision.
- Before runtime-precision conversion, the ordered generated kernel tensor has
  shape `[32, 105, 105]` and dtype `complex128`.

## 2026-07-29 — Use reflection padding with an explicit valid-support mask

Approved:

- After H/E separation, each concentration channel is padded by 52 pixels on every
  side using `reflect` semantics that do not duplicate the edge sample.
- `symmetric`, `replicate`, circular, periodic, and wraparound padding are each
  explicitly prohibited.
- Reflect padding requires runtime input dimensions `H > 52` and `W > 52`.
- A nonempty region unaffected by padding exists only when `H > 104` and
  `W > 104`.
- Reflection padding is followed by valid true convolution, producing an output
  with the original `H x W` spatial size.
- The output carries a Boolean `valid_support_mask`. `True` means the complete
  `105 x 105` receptive field came from the original patch; `False` means at least
  one reflected sample was used.
- For a `256 x 256` patch, the mask is `True` exactly on
  `[52:204, 52:204]`, a `152 x 152` center.
- The mask records boundary influence only. It is not a tissue, foreground, or
  attention mask, and this decision does not determine how later pooling consumes
  it.
- An FFT backend must compute zero-padded linear convolution of the reflected input
  and kernel. It must not use an equal-size FFT or otherwise rely on circular
  convolution.
- For reflected input size `(H + 104, W + 104)` and kernel size `(105, 105)`, FFT
  sizes must be at least `(H + 208, W + 208)` before extracting the valid
  `H x W` result.
- Spatial and FFT backends must implement the same reflection, true-convolution,
  valid-crop, and mask contract.

## 2026-07-29 — Normalize each complex Morlet kernel to unit L2 energy

Approved:

- After discrete zero-DC projection, each complex kernel is independently
  normalized as a whole by
  `psi_hat = psi / sqrt(sum_Omega(abs(psi)^2))`.
- Energy accumulation and normalization occur in the `float64`/`complex128`
  generation stage.
- Real and imaginary components must not be normalized separately.
- No L1, peak, or additional scale-dependent normalization is applied.
- Pre-normalization energy is retained as generation audit metadata.
- Non-finite, non-positive, or insufficient energy is an error under the approved
  numerical tolerance.
- Discrete zero-DC residual and unit energy are audited after normalization.
- After conversion to the approved runtime dtype, unit energy and discrete zero-DC
  residual are audited again under separately approved runtime tolerances.
- The parameter-specification hash covers discrete complex L2 normalization; the
  final kernel-tensor hash covers normalized, convolution-adapted execution kernels
  in the approved runtime dtype.

## 2026-07-29 — Use protected float32/complex64 frontend execution

Approved:

- H/E concentrations enter the fixed frontend as `float32`.
- Normalized `complex128` generation kernels are converted to `complex64`
  execution kernels, equivalently represented by paired `float32` real and
  imaginary components.
- Spatial execution performs the real and imaginary convolutions in `float32`.
  FFT execution uses `complex64`. First-order modulus maps `F_H` and `F_E` are
  `float32`.
- The fixed frontend executes inside an explicit autocast-disabled precision guard.
  It checks effective TF32 state and rejects formal execution if TF32 can affect
  the frontend path.
- `float16`, `bfloat16`, automatic mixed precision, and TF32 are prohibited inside
  the fixed frontend. Electronic-backend precision remains a separate decision.
- The canonical execution-kernel byte payload is produced on CPU as a C-contiguous
  little-endian IEEE-754 `float32` array with shape `[32, 105, 105, 2]`; the last
  axis is ordered real then imaginary, and the preceding axes retain the approved
  scale-major/orientation-major execution-kernel order.
- Kernel identity uses this canonical CPU payload rather than device-native complex
  memory layout. The digest algorithm and metadata serialization envelope remain
  undecided.
- Runtime conversion is followed by fresh unit-energy and zero-DC audits and a
  comparison against the `complex128` reference response.
- The FFT path follows the separately frozen dimensions, normalization, shift,
  crop, and derived-cache identity recorded below.
- The complex-modulus primitive requires a separate frozen definition.
- Spatial and FFT responses must be equivalent within approved tolerances; bitwise
  identity is not required.

## 2026-07-29 — Use a stable epsilon-free complex modulus

Approved:

- For `float32` real part `r` and imaginary part `q`, define
  `a = abs(r)`, `b = abs(q)`, `h = max(a, b)`, and `l = min(a, b)`.
- The modulus is exactly zero when `h == 0`; otherwise it is
  `h * sqrt(1 + (l/h)^2)`.
- All primitive operations and outputs are `float32`.
- The normative definition is not direct `sqrt(r*r + q*q)`, and it adds no
  epsilon.
- The frontend does not output squared modulus, optical intensity, log magnitude,
  or signed complex response.
- No clipping, normalization, standardization, or learned scaling follows the
  modulus inside the fixed frontend.
- Non-finite components or modulus results are errors.
- The `valid_support_mask` is propagated unchanged with the modulus feature maps.
- Spatial and FFT paths use the same modulus contract and are compared within
  approved tolerances rather than bitwise.
- The parameter specification covers the modulus formula version and epsilon-free
  convention.

## 2026-07-29 — Freeze the FFT linear-convolution grid and cache identity

Approved:

- The canonical normalized `complex64` kernel array has axes
  `[channel, y, x]`, shape `[32, 105, 105]`, channel
  `c = 8*j + ell`, and center index `[52, 52]`.
- Array coordinates are `u_x = x - 52` and `u_y = y - 52`. The canonical array
  stores generated `psi(u)` before the two-axis flip required by a spatial
  cross-correlation primitive.
- The canonical complex64 kernel has its own hash over the canonical CPU
  `[32, 105, 105, 2]` little-endian float32 real/imaginary byte layout. It is
  distinct from the flipped spatial execution-kernel hash.
- For input `(H, W)`, reflection produces `(H + 104, W + 104)`. The frozen FFT
  grid is exactly the minimal full linear-convolution grid
  `(H + 208, W + 208)`; equal-size FFT and `next_fast_len` substitution are
  prohibited.
- The reflected input and canonical kernel are placed at the top-left of
  zero-filled arrays on that grid.
- Input is represented as zero-imaginary `complex64`. Both `fft2` and `ifft2` use
  `norm="backward"`, spectra are multiplied pointwise, and no `fftshift` or
  `ifftshift` is used.
- The valid complex response is cropped at
  `[104:104+H, 104:104+W]`, then passed to the approved modulus primitive.
- The frozen minimal grid defines transform sampling, crop semantics, and execution
  identity. It does not lock or make claims about the FFT library's internal
  algorithm, plan, or scheduling.
- The FFT cache key directly includes the canonical complex64 kernel hash, spatial
  execution-kernel hash, `H`, `W`, FFT grid, dtype, normalization, no-shift and
  crop conventions, backend name/version, and device class.
- An optional frequency-tensor payload hash may detect cache corruption but is not
  a cross-platform scientific identity.
- Spatial and FFT paths require tolerance equivalence, not bitwise equality.

## 2026-07-29 — Use domain-separated SHA-256 with RFC 8785 JCS envelopes

Approved:

- Every identity digest is
  `SHA256(T || 0x00 || uint64be(len(H)) || H || P)` where `T` is a UTF-8 domain
  tag, `H` is RFC 8785 JCS output encoded as UTF-8, and `P` is the optional
  canonical raw payload.
- Digest text is lowercase `sha256:` followed by exactly 64 hexadecimal
  characters.
- Domain tags are:
  - `cg/stain-separation-spec/v1`;
  - `cg/morlet-param-spec/v1`;
  - `cg/morlet-kernel-canonical/v1`;
  - `cg/morlet-kernel-spatial-exec/v1`;
  - `cg/fft-cache-key/v1`;
  - `cg/fft-cache-payload/v1`.
- JCS must truly conform to RFC 8785; ordinary key-sorted JSON is insufficient.
- Duplicate keys, NaN, Infinity, and JSON negative zero are rejected before
  canonicalization.
- JSON integers are limited to the interoperable safe-integer range. Larger exact
  integers are represented by a schema-defined canonical decimal string.
- Metadata is ASCII-first. Unicode is allowed only when required and receives no
  implicit Unicode normalization; exact code points are part of the identity.
- Tensor payload `-0.0` values are canonicalized to `+0.0` before execution and
  hashing.
- `len(H)` is the number of bytes in the RFC 8785 output after UTF-8 encoding and
  is stored as an unsigned 64-bit big-endian integer.
- An empty payload has length zero and contributes no payload bytes. Header
  `payload_length` must equal the actual payload byte count before hashing.
- Tensor headers include dtype, shape, axis semantics, C-contiguous layout,
  endianness, and payload length. Raw tensor bytes are hashed directly rather than
  through Base64.
- Formal parameter-specification identity is unavailable while any required field
  is `TBD`.
- Scientific identities exclude paths, timestamps, hostnames, and device serial
  numbers. The FFT cache key may include the approved backend version and device
  class.
- Each domain stores a complete envelope preimage and expected digest as a fixed
  test vector.
- These hashes provide identity and integrity audit only, not digital signatures or
  clinical trust.

## 2026-07-29 — Lock fixed-frontend numerical acceptance tolerances

Approved:

- All kernel-generation and response checks run per kernel. A global mean or an
  aggregate pass cannot hide a failing scale-orientation channel.
- For every normalized `complex128` kernel,
  `abs(sum(psi)) <= 1e-12` and
  `abs(sum(abs(psi)^2) - 1) <= 1e-12`.
- After conversion to `complex64`, residuals are accumulated in `complex128`, with
  `abs(sum(psi)) <= 1e-6` and
  `abs(sum(abs(psi)^2) - 1) <= 1e-6` for every kernel.
- `abs(beta_disc - beta_inf) <= 1e-2` is a finite-support construction check, not
  a runtime-response equivalence metric.
- Spatial and FFT complex components and modulus outputs use the elementwise rule
  `abs(a-b) <= 2e-5 + 2e-4*abs(b)`, with `b` the reference path.
- Per-kernel complex RMSE is
  `sqrt(mean((real(a)-real(b))^2 + (imag(a)-imag(b))^2))` over all compared batch
  and spatial elements for that kernel and must be at most `2e-6`.
- The `valid_support_mask` must be bitwise identical.
- A CPU `complex128` reference path independently regenerates kernels and performs
  the locked reflection and true spatial convolution without reusing runtime
  `complex64` kernels, FFT spectra, or derived caches.
- Fixtures include a constant image, deterministic seeded random image, selected
  scale-orientation sinusoids, and unit impulses at the center, four edge
  midpoints, and four corners.
- Non-finite values fail before tolerance comparison.
- Tolerances never widen automatically by device, backend, batch, or sample.
- Every formally supported backend and device class must pass the same fixed
  thresholds before its results are accepted.
- A change to kernels, FFT grid, or runtime precision requires a new explicit
  tolerance decision.

## 2026-07-29 — Lock Morlet spectral coverage validation

Approved:

- Coverage is evaluated on the `464 x 464` FFT grid associated with the primary
  `256 x 256` patch.
- For modular frequency index `k = (k_y, k_x)`, negative frequency is exactly
  `-k = ((-k_y) mod 464, (-k_x) mod 464)`.
- Symmetrized per-kernel magnitude is
  `B(k) = sqrt(abs(F(k))^2 + abs(F((-k) mod 464))^2)`.
- All powers, inner products, norms, interpolation, peak fitting, summary
  statistics, and threshold comparisons accumulate in `float64`.
- Peak estimation first chooses the discrete maximum in the expected positive
  carrier half-plane with deterministic index tie-breaking.
- Around that bin, a fixed `3 x 3` modular neighborhood of `log(power)` is fitted
  in `float64` to
  `a*dx^2 + b*dy^2 + c*dx*dy + d*dx + e*dy + f`.
  The fitted vertex is `-[ [2a,c],[c,2b] ]^(-1) * [d,e]`.
- The Hessian must be negative definite and each sub-bin offset must lie in
  `[-1, 1]`; failure is an error rather than a fallback to the grid maximum.
- Signed sub-grid frequencies use `2*pi*fftfreq(464)` plus the fitted offset times
  `2*pi/464`.
- Per kernel, carrier-direction error is at most 1 degree and radial peak error is
  `abs(rho_hat - xi_j) <= min(0.075, 0.10*xi_j)`.
- Adjacent orientation pairs, including modular wraparound, have symmetrized
  spectral cosine overlap in `[0.50, 0.70]`.
- Same-orientation adjacent-scale pairs have symmetrized spectral cosine overlap in
  `[0.45, 0.60]`.
- Total symmetrized power is checked on four radii `xi_j` and three radii
  `sqrt(xi_j*xi_(j+1))`, sampling 1440 equally spaced angles with periodic bilinear
  interpolation. Each radius separately requires `min/median >= 0.85`,
  `min/max >= 0.75`, and angular coefficient of variation `<= 0.10`.
- The ring checks are **angular coverage uniformity gates at specified radii**.
  They do not alone prove that the complete two-dimensional frequency plane has no
  holes.
- All peak and overlap gates run per kernel, pair, or radius rather than on global
  averages. Full metric arrays, worst locations, and generating identities are
  retained.
- Passing these project-specific gates is not a claim of numerical equivalence with
  Kymatio.

## 2026-07-29 — Place structured H/E interaction in the electronic backend

Approved:

- The fixed optical frontend ends after the stable first-order modulus.
- `F_H` and `F_E` are `float32` tensors with semantic shape
  `[B, J, L, H, W] = [B, 4, 8, H, W]`.
- A combined view uses `[B, stain, J, L, H, W]` with stain order `[H, E]`.
- The shared Boolean `valid_support_mask` has shape `[B, 1, H, W]`.
- Axis semantics must never be lost. Temporary reshaping, flattening, permutation,
  or packing is allowed only when it is explicitly described, deterministically
  reversible, and restores or carries the `stain`, `J`, `L`, and spatial metadata.
- A branch may add an explicit feature axis. Its canonical semantic output is
  `[B, J, L, C_branch, H, W]`; branch-specific `C_branch` meaning and size require
  later approval.
- `HEInteractionBlock` belongs entirely to the electronic backend. Any approved
  trainable interaction parameters count toward the electronic-backend budget,
  optimizer audit, and checkpoint.
- H/E separation, Morlet generation, convolution, and modulus remain fixed during
  interaction and classifier training.
- The interaction entry carries a stain-separation specification hash in addition
  to the Morlet parameter, canonical-kernel, and spatial-execution-kernel hashes.
- The new stain-separation hash uses domain
  `cg/stain-separation-spec/v1` and covers the ordered H/E basis, pseudoinverse
  convention, RGB-to-optical-density transform, clipping, dtype, background, and
  no-normalization policy.

## 2026-07-29 — Use symmetric cross-stain mutual gating

Approved:

- For every scale-orientation pair `(j, ell)`, the cross-stain gating branch uses
  `g_H^(j,ell) = 2 * sigmoid(a_(j,ell) * F_E^(j,ell) + b_(j,ell))` and
  `g_E^(j,ell) = 2 * sigmoid(a_(j,ell) * F_H^(j,ell) + b_(j,ell))`.
- Its outputs are
  `Y_H^(j,ell) = F_H^(j,ell) elementwise-multiplied by g_H^(j,ell)` and
  `Y_E^(j,ell) = F_E^(j,ell) elementwise-multiplied by g_E^(j,ell)`.
- `a_(j,ell)` and `b_(j,ell)` are trainable scalars specific to each
  scale-orientation pair and shared between the H-to-E and E-to-H directions.
  With `J = 4` and `L = 8`, the branch has exactly `2 * J * L = 64` trainable
  parameters.
- Every `a_(j,ell)` and `b_(j,ell)` is initialized to zero. Every gate therefore
  initializes to one, and both output streams initially equal their corresponding
  input streams exactly.
- Each gate lies in `(0, 2)`. A gate computed from the opposite stain suppresses
  or enhances the current target-stain response; the branch does not add, replace,
  or otherwise directly mix the two stain responses.
- The branch operates only at the same `(j, ell, y, x)` position. It does not mix
  spatial neighborhoods, scales, or orientations.
- The canonical output is `[B, J, L, C_branch, H, W]`, following the interaction
  boundary frozen in the development specification. Here `C_branch = 2`, ordered
  `[H_gated, E_gated]`.
- The `valid_support_mask` is propagated unchanged. It is neither cleared nor used
  in the gating computation.
- All 64 trainable parameters belong to the electronic-backend capacity,
  optimizer, checkpoint, and audit budget. The fixed optical frontend remains
  unchanged.

## 2026-07-29 — Use parameter-free same-location product co-occurrence

Approved:

- For every scale-orientation pair `(j, ell)`, the same-location co-occurrence
  branch is `C^(j,ell) = F_H^(j,ell) elementwise-multiplied by F_E^(j,ell)`.
- The branch reads the raw fixed-frontend modulus responses `F_H` and `F_E`
  directly and runs in parallel with cross-stain gating. It does not consume
  `Y_H` or `Y_E` from the gating branch.
- The product is computed only at the same `(j, ell, y, x)` coordinate and does
  not mix spatial neighborhoods, scales, or orientations.
- The feature is invariant under exchange of H and E. It is zero when either
  response is zero and increases when the two nonnegative responses jointly
  increase.
- The branch has no trainable parameters and adds zero parameters to the
  electronic-backend budget.
- The canonical output is `[B, J, L, C_branch, H, W]`, with `C_branch = 1` and
  the feature axis ordered `[HE_product]`.
- The output remains nonnegative. The branch adds no bias, epsilon, square root,
  logarithm, clipping, normalization, or trainable scaling.
- The `valid_support_mask` is propagated unchanged and does not participate in
  the product.
- The product has squared modulus-response units. It represents joint response
  strength and must not be described as a probability, correlation coefficient,
  or normalized co-occurrence score.
- Electronic-backend execution precision remains unresolved and is not frozen by
  this decision.
- Equal product values can arise from balanced or imbalanced H/E response pairs.
  This branch does not encode that distinction; the separately required
  difference-feature family remains responsible for expressing stain imbalance.

## 2026-07-29 — Use parameter-free center-excluded eight-neighbor interaction

Approved:

- For a feature map `X`, define `A_8(X)(y,x)` as one eighth of the sum at offsets
  `{-1, 0, 1} x {-1, 0, 1}` excluding `(0, 0)`. Values outside the feature map
  are fixed to zero.
- For every scale-orientation pair `(j, ell)`, the two outputs are
  `N_(H<-E)^(j,ell) = F_H^(j,ell) elementwise-multiplied by
  A_8(F_E^(j,ell))` and
  `N_(E<-H)^(j,ell) = F_E^(j,ell) elementwise-multiplied by
  A_8(F_H^(j,ell))`.
- The branch reads raw `F_H` and `F_E` directly and runs in parallel with
  cross-stain gating and same-location co-occurrence.
- Excluding `(0, 0)` prevents this branch from duplicating the approved
  same-location product. The branch mixes space only within the same `(j, ell)`
  and never mixes scales or orientations.
- Exchanging H and E exchanges the two feature channels. The paired branch is
  H/E-equivariant rather than collapsed to one invariant feature.
- The branch has no trainable parameters and adds zero parameters to the
  electronic-backend budget.
- The canonical output is `[B, J, L, C_branch, H, W]`, with `C_branch = 2`
  ordered `[H_x_E_ring, E_x_H_ring]`.
- Feature-map boundary values are zero padded and the neighborhood sum is always
  divided by eight. Available neighbors are not counted and the mean is not
  renormalized at an edge.
- The original `valid_support_mask` remains available bitwise unchanged. The
  branch additionally emits `neighborhood_valid_support_mask`, which is `True`
  only when the center and all eight neighboring positions are `True` in the
  original mask. Mask values outside the map are `False`.
- Neither mask participates in the numeric neighborhood mean or product.
- For a `256 x 256` input whose original valid region is
  `[52:204, 52:204]`, the derived neighborhood-valid region is exactly
  `[53:203, 53:203]`, with size `150 x 150`.
- The branch adds no bias, epsilon, nonlinearity, clipping, normalization, or
  trainable scaling. Its outputs have squared modulus-response units.
- Electronic-backend execution precision remains unresolved and is not frozen by
  this decision.
- Larger, dilated, orientation-aligned, scale-dependent, or learned neighborhoods
  are outside the primary branch and require separate approval as future
  ablations.

## 2026-07-29 — Use bidirectional nonnegative stain-excess responses

Approved:

- For every scale-orientation pair `(j, ell)`, the difference branch is
  `D_H^(j,ell) = max(F_H^(j,ell) - F_E^(j,ell), 0)` and
  `D_E^(j,ell) = max(F_E^(j,ell) - F_H^(j,ell), 0)`.
- The branch reads raw `F_H` and `F_E` directly and runs in parallel with
  cross-stain gating, same-location co-occurrence, and neighborhood interaction.
- The operation is pointwise at the same `(j, ell, y, x)` coordinate and does not
  mix spatial neighborhoods, scales, or orientations.
- `D_H` is the amount by which H exceeds E, and `D_E` is the amount by which E
  exceeds H. Both are exactly zero when the two inputs are equal.
- At every position, at most one output is nonzero, with
  `D_H + D_E = abs(F_H - F_E)` and `D_H - D_E = F_H - F_E`.
- Exchanging H and E exchanges the two feature channels. The paired branch is
  H/E-equivariant and retains both difference magnitude and dominant-stain
  direction.
- The branch has no trainable parameters. Across all four approved interaction
  branches, the only trainable interaction parameters remain the 64 cross-stain
  gating scalars.
- The canonical output is `[B, J, L, C_branch, H, W]`, with `C_branch = 2`
  ordered `[H_excess, E_excess]`.
- The Boolean `valid_support_mask` is propagated bitwise unchanged and does not
  participate in the difference calculation. The neighborhood branch's derived
  mask is not required by this pointwise branch.
- Outputs are nonnegative and retain modulus-response units. No bias, epsilon,
  ratio, logarithm, normalization, trainable scaling, or additional clipping is
  part of the branch.
- Non-finite inputs or difference results are errors and must not be replaced by
  zero.
- Electronic-backend execution precision remains unresolved and is not frozen by
  this decision.
- This decision does not determine how the four branch outputs are combined or
  which support mask later spatial pooling consumes.

## 2026-07-29 — Concatenate a canonical seven-channel interaction feature stack

Approved:

- The four approved interaction branches remain parallel. Their outputs are
  combined only by deterministic concatenation along the interaction-feature
  axis.
- The canonical combined output layout is
  `[B, J, L, C_interaction, H, W] = [B, 4, 8, 7, H, W]`.
- The `C_interaction` axis is ordered exactly as
  `[H_gated, E_gated, HE_product, H_x_E_ring, E_x_H_ring, H_excess,
  E_excess]`.
- Combining branches must not add summation, averaging, projection, renewed
  gating, nonlinearity, normalization, or implicit scaling.
- Combination has no trainable parameters. The complete interaction module
  retains exactly 64 trainable scalars, all from cross-stain gating.
- `J`, `L`, and `C_interaction` must not be irreversibly or implicitly flattened.
  A temporary 224-channel representation is allowed only with a deterministic,
  reversible `(j, ell, feature_name)` mapping back to the canonical layout.
- Unit metadata remains explicit:
  - `H_gated`, `E_gated`, `H_excess`, and `E_excess` have
    modulus-response units;
  - `HE_product`, `H_x_E_ring`, and `E_x_H_ring` have squared
    modulus-response units.
- The combination step does not reconcile, normalize, or otherwise alter the two
  unit families.
- The combined output carries both the unmodified `valid_support_mask` and the
  unmodified `neighborhood_valid_support_mask`.
- Feature-to-mask bindings are fixed as:
  - `H_gated`, `E_gated`, `HE_product`, `H_excess`, and `E_excess` bind to
    `valid_support_mask`;
  - `H_x_E_ring` and `E_x_H_ring` bind to
    `neighborhood_valid_support_mask`.
- Exchanging H and E applies the zero-based `C_interaction` permutation
  `[1, 0, 2, 4, 3, 6, 5]`. Both support masks remain unchanged.
- Electronic-backend execution precision, AMP policy, combined-output dtype,
  later spatial-pooling mask policy, and normalization policy remain unresolved.
- This decision does not modify the approved contracts for interaction Decisions
  20 through 23.

## 2026-07-29 — Use one conservative neighborhood-valid pooling support

Approved:

- `pooling_support_mask` is defined as
  `valid_support_mask AND neighborhood_valid_support_mask` and is bitwise
  identical to `neighborhood_valid_support_mask`.
- `neighborhood_valid_support_mask` must be a subset of
  `valid_support_mask`. This invariant is checked before pooling and any
  violation fails closed.
- `valid_support_mask`, `neighborhood_valid_support_mask`, and
  `pooling_support_mask` are separate named Boolean interface fields, each with
  shape `[B, 1, H, W]`. All three retain their distinct semantic identities and
  none may overwrite or mutate another.
- For each sample and spatial-pyramid region, all
  `4 * 8 * 7 = 224` scale-orientation-interaction channels use exactly the same
  set of positions selected by `pooling_support_mask`.
- The feature-specific native mask bindings approved for the combined interaction
  interface remain available for audit, but they do not select different pooling
  samples for different feature channels.
- Pooling uses genuine mask-aware reductions. Invalid locations are excluded
  before a statistic is evaluated; they must not be multiplied by zero and then
  included in an ordinary reduction.
- The valid-pixel count is computed exactly for every sample and every region and
  is the common sample count for all 224 channels in that region.
- An empty region is an error detected before reduction. It must not produce
  zero, NaN, a default statistic, or a silently omitted output.
- For a `256 x 256` input, `pooling_support_mask` is `True` exactly on
  `[53:203, 53:203]`, containing `150 * 150 = 22500` positions.
- Relative to the original `[52:204, 52:204]` valid square, the pooling support
  contracts by a one-pixel border on each of the square's four sides. It removes
  604 of 23104 original-valid positions, approximately 2.61 percent.
- Nonempty global pooling support requires `H > 106` and `W > 106`. Any later
  spatial-pyramid configuration must separately prove that every configured
  region is nonempty.
- The three masks express computational support only. None is a tissue,
  foreground, attention, lesion, label, or stain-intensity mask, and none is
  intersected with such data-dependent masks.
- This policy has no trainable parameters.
- Spatial-pyramid levels, region-boundary allocation, statistics, normalization,
  execution precision, AMP policy, and output dtype remain unresolved.

## 2026-07-29 — Use a support-aligned 1-by-1, 2-by-2, 4-by-4 spatial pyramid

Approved:

- The spatial-pyramid levels are ordered `[1, 2, 4]`, giving
  `P = 1^2 + 2^2 + 4^2 = 21` regions.
- For input shape `(H, W)`, the canonical pooling-support rectangle is fixed by
  geometry as `[53:H-53, 53:W-53]`, with height `H_s = H - 106` and width
  `W_s = W - 106`.
- Region boundaries at level `n` are
  `y_(n,r) = 53 + floor(r * H_s / n)` and
  `x_(n,c) = 53 + floor(c * W_s / n)` for indices from zero through `n`.
  Region `(n, r, c)` is the half-open rectangle
  `[y_(n,r):y_(n,r+1), x_(n,c):x_(n,c+1)]`.
- Boundaries are anchored only to the canonical rectangle determined by `H` and
  `W`. They must not be derived from any individual sample's realized mask
  bounding box, tissue extent, labels, or feature values.
- Levels are traversed in `[1, 2, 4]` order. Within a level, rows proceed from top
  to bottom and columns within each row proceed from left to right.
- The zero-based region index is `p = o_n + r * n + c`, with offsets
  `o_1 = 0`, `o_2 = 1`, and `o_4 = 5`. Thus `p = 0` is the `1 x 1` region,
  `p = 1..4` are the `2 x 2` regions, and `p = 5..20` are the `4 x 4`
  regions.
- Within each level, regions are disjoint, leave no gaps, and tile the entire
  canonical support rectangle. Hierarchical overlap between different levels is
  intentional.
- When a dimension is not divisible by `n`, region extents may differ by at most
  one pixel. Padding, resampling, overlapping adaptive-pooling windows, and
  discarded remainder pixels are prohibited.
- Every region is still intersected with `pooling_support_mask`, and its valid
  positions are counted under the approved mask-aware pooling contract.
- For a `256 x 256` input, the support is `150 x 150`; `2 x 2` boundaries are
  `[53, 128, 203]`, `4 x 4` boundaries are
  `[53, 90, 128, 165, 203]`, and the finest-level extents per axis are
  `[37, 38, 37, 38]`.
- Supporting the `4 x 4` level requires `H >= 110` and `W >= 110`; smaller
  inputs fail closed before reduction without changing the fixed frontend's
  separate input contract.
- Geometry metadata is batch-independent for a fixed `(H, W)` and contains 21
  ordered records
  `(p, level, row, column, y_start, y_end, x_start, x_end)`.
- Sample-dependent valid counts are stored separately as exact integer values in
  `valid_count` with shape `[B, 21]`; `valid_count[b, p]` must not be embedded in
  the static geometry record.
- All 224 scale-orientation-interaction channels share the same 21 regions and
  the same `valid_count[b, p]` within a sample-region pair.
- Each later-approved statistic produces `224 * 21 = 4704` scalar values before
  any later normalization or classifier processing.
- Region construction has no trainable parameters and performs no weighting,
  averaging, or fusion between regions.
- Spatial-pyramid statistics, statistic-axis order, normalization, execution
  precision, AMP policy, output dtype, and classifier structure remain unresolved.
- For differing input dimensions, the pyramid is relative to the canonical
  support rectangle and does not claim a fixed physical scale.

## 2026-07-29 — Pool regional mean and population standard deviation

Approved:

- For each sample `b`, region `p`, scale `j`, orientation `ell`, and interaction
  feature `c`, let `S_(b,p)` be the approved selected-position set and
  `N_(b,p) = valid_count[b,p]`.
- Regional mean is
  `mu = sum_(y,x in S_(b,p)) Z[b,j,ell,c,y,x] / N_(b,p)`.
- Regional population standard deviation is
  `sigma = sqrt(sum_(y,x in S_(b,p)) (Z[b,j,ell,c,y,x] - mu)^2 /
  N_(b,p))`.
- The statistic set and order are exactly `[mean, population_std]`. Population
  standard deviation uses denominator `N_(b,p)`, never `N_(b,p) - 1`.
- `N_(b,p) = 0` fails before reduction. When `N_(b,p) = 1`, mean is the sole
  selected value and population standard deviation is exactly zero.
- The normative computation is two-pass: first compute mean, then compute the
  mean centered squared deviation over the same selected positions. The
  cancellation-prone identity `E[Z^2] - E[Z]^2` is not normative.
- Forward population standard deviation is exactly zero when variance is zero.
  Its backward convention is `d sigma / d variance = 0` at zero variance and
  `1 / (2 * sqrt(variance))` at positive variance. This zero-gradient convention
  must keep backward values finite and must not be implemented by adding epsilon
  to the forward variance or square root.
- Non-finite checks cover every selected input value that actually participates
  in reduction, first-pass sums and means, centered deviations, squared
  deviations, second-pass sums, variances, standard deviations, and all emitted
  statistics. Any non-finite value is an error.
- The canonical output layout is
  `[B, J, L, C_interaction, P, S_stat] = [B, 4, 8, 7, 21, 2]`, with
  `S_stat` ordered `[mean, population_std]`.
- Both statistics inherit their source feature units. Statistics from
  `H_gated`, `E_gated`, `H_excess`, and `E_excess` have modulus-response units;
  statistics from `HE_product`, `H_x_E_ring`, and `E_x_H_ring` have squared
  modulus-response units.
- Statistics are not mixed across scale, orientation, interaction feature, or
  spatial-pyramid region.
- H/E exchange retains the approved `C_interaction` permutation
  `[1, 0, 2, 4, 3, 6, 5]`; region and statistic axes remain unchanged.
- Each sample emits exactly `4 * 8 * 7 * 21 * 2 = 9408` pooled scalars.
- Canonical flattening traverses axes as
  `(j, ell, feature, p, statistic)`, with statistic varying fastest. Its
  zero-based linear index is
  `q = ((((j * 8 + ell) * 7 + feature) * 21 + p) * 2 + statistic)`,
  covering exactly `q = 0..9407`.
- A flattened representation retains the complete reversible
  `(j, ell, feature_name, p, statistic_name)` mapping.
- The pooling statistics add no trainable parameters. Maxima, minima, medians,
  quantiles, RMS, skewness, kurtosis, and higher moments are excluded from the
  primary model and require explicit approval as ablations.
- Reduction execution dtype, accumulation dtype, summation order, AMP policy,
  subsequent normalization, and classifier structure remain unresolved.

## 2026-07-30 — Use an unnormalized identity pooled-feature handoff

Approved:

- The six-dimensional pooled output
  `[B, J, L, C_interaction, P, S_stat] = [B, 4, 8, 7, 21, 2]` is reshaped
  directly to classifier input `[B, 9408]` using the approved canonical index.
- The handoff permits shape change only. Every flattened value is numerically
  identical to its structured source value.
- No centering, standardization, L1 or L2 normalization, RMS normalization,
  LayerNorm, BatchNorm, whitening, logarithm, clipping, or other numeric
  transformation is permitted between pooled output and classifier input.
- No dataset normalization mean, variance, quantile, scale, or other fitted
  statistic is computed or stored from training, validation, test, or transfer
  data.
- The handoff adds no trainable parameter, running state, or persistent buffer.
  The interaction and pooling path retains exactly 64 trainable scalars, all in
  cross-stain gating.
- The two unit families and their coordinate identities remain explicit:
  5376 flattened coordinates derive from the four modulus-response features and
  4032 derive from the three squared-modulus-response features.
- The classifier has the capacity to adapt to coordinate-scale differences, but
  the realized result depends on later-approved optimizer, initialization, and
  regularization choices.
- All 9408 classifier-input coordinates are checked for finiteness before the
  classifier is evaluated. Any non-finite value fails closed.
- For batches `b,b'` and flattened coordinates `q,q'`, the handoff Jacobian is
  `d x_head[b,q] / d v[b',q'] = 1` when `b = b'` and `q = q'`, and zero
  otherwise. No stop-gradient or gradient clipping is part of the handoff.
- Let `pi = [1, 0, 2, 4, 3, 6, 5]` be the approved H/E exchange permutation on
  the interaction-feature axis and
  `q(j,ell,c,p,s) = ((((j*8+ell)*7+c)*21+p)*2+s)`.
  The flattened H/E permutation is frozen by
  `Pi(q(j,ell,c,p,s)) = q(j,ell,pi[c],p,s)`.
- Equivalently, after H/E exchange,
  `v_swap[b, q(j,ell,c,p,s)] =
  v[b, q(j,ell,pi[c],p,s)]`. `Pi` is an involutive bijection of
  `0..9407`.
- Read-only distribution diagnostics are allowed. They may report summaries,
  histograms, non-finite counts, or scale disparities, but must not change
  forward values, create model state, or automatically control normalization,
  clipping, loss weighting, sampling, optimizer settings, schedules, early
  stopping, or another data-adaptive training rule.
- A diagnostic may inform a later explicit human decision, but it cannot silently
  modify the approved primary pipeline.
- A dtype conversion is not authorized by this normalization decision. Execution
  dtype, accumulation dtype, AMP policy, and classifier architecture remain
  unresolved.
- Normalization internal to a future classifier is not implicitly authorized and
  must be addressed with the classifier architecture.

## 2026-07-30 — Use protected float32 electronic execution with float64 statistics

Approved:

- Fixed-frontend outputs, interaction computations, combined interaction output,
  pooled output, classifier input, and every parameter, activation, and binary
  logit in the primary classifier use float32.
- Cross-stain gating parameters and their gradients are float32.
- The two-pass spatial mean and population-standard-deviation path explicitly
  promotes selected float32 values to float64. First-pass sums and means,
  centered deviations, squared deviations, second-pass sums, population
  variances, and square roots all execute in float64.
- The only canonical forward float64-to-float32 boundary occurs after both pooled
  statistics have been completed. The structured pooled tensor, identity
  flattened vector, and classifier input are float32.
- Before that cast, every float64 result must be finite and satisfy
  `abs(x) <= finfo(float32).max`. After casting, every float32 value must be
  finite. Failure at either check fails closed; exact float32 representability is
  not required.
- `valid_support_mask`, `neighborhood_valid_support_mask`, and
  `pooling_support_mask` are Boolean. Region indices, boundary coordinates, and
  `[B, 21] valid_count` are int64. Counts convert explicitly to float64 only for
  statistical arithmetic.
- Exact-zero variance uses the approved zero-gradient convention through a
  validated operator or safe custom backward implementation. Float64 precision
  alone is not evidence that a naive square-root backward is finite at zero.
- Constant-region and singleton tests must prove exact-zero forward standard
  deviation and finite zero gradients without adding epsilon.
- The complete primary training step prohibits automatic mixed precision,
  float16, bfloat16, dynamic or static gradient scaling, and automatic dtype
  downgrades. The prohibition covers forward computation, loss evaluation,
  backward computation, and the optimizer step, not only the electronic module.
- The electronic backend runs inside an explicit autocast-disabled protection
  domain.
- TF32 is prohibited to preserve IEEE float32 semantics. Effective TF32 state is
  checked using APIs appropriate to the installed PyTorch version and must audit
  relevant environment-variable or backend overrides. Formal execution fails
  closed if TF32 can affect the path.
- Checks target known reduced-precision execution modes, automatic downcasts,
  stochastic rounding, and other discoverable random-rounding settings. This
  policy does not claim cross-device bitwise identity or prohibit unspecified
  device implementation differences that cannot be inspected.
- Interaction values, pooled values, classifier inputs, logits, and gating
  gradients receive fail-closed non-finite checks. Any float64 value crossing to
  float32 must also pass the finite-range check before conversion.
- Backpropagation through float64 statistical intermediates returns gradients
  across an explicit float64-to-float32 boundary. Gating gradients must be finite
  float32 values.
- Precision-policy version, all approved dtypes, autocast state, AMP and
  gradient-scaling state, effective TF32 state, relevant environment overrides,
  and observed cast boundaries belong to explicit configuration, checkpoint
  metadata, and runtime audit.
- This policy adds no trainable parameters or persistent normalization state.
  The interaction and pooling path retains exactly 64 trainable gating scalars.
- Float32 is mandatory for the primary classifier. Any lower, higher, or mixed
  classifier precision requires a separate explicit decision.
- Statistical summation order is frozen by the following decision. Cross-device
  numerical-equivalence tolerances, loss definition and any explicitly selected
  loss-internal precision, and optimizer-state precision remain unresolved.

## 2026-07-30 — Use a canonical balanced adjacent-pair reduction topology

Approved:

- Selected spatial values form the reduction leaf sequence in strict row-major
  `(y, x)` order.
- Every leaf converts to float64 before entering the reduction tree.
- At every level, adjacent entries are paired from the start of the sequence and
  each pair performs exactly one float64 addition.
- When a level has an odd number of entries, its final unpaired entry passes
  unchanged into the next level.
- The same adjacent-pair rule repeats level by level until only one result
  remains.
- The first-pass sum for the regional mean and the second-pass sum of centered
  squared deviations reuse exactly the same leaf ordering, pair relationships,
  and level topology.
- Centered squared deviations are computed in float64 before entering the
  second-pass reduction tree.
- A framework-native `sum` without an explicitly guaranteed reduction order,
  parallel atomic accumulation, or a device-dependent reduction implementation
  cannot provide the normative result.
- Optimized implementations may execute independent nodes within the same level
  in parallel, but cannot alter leaf order, pair relationships, or inter-level
  topology.
- Cross-device numerical-equivalence tolerances and comparator semantics are
  frozen by the subsections below.

### Normative CPU reference for cross-device equivalence

Approved:

- The normative CPU statistical implementation is the sole reference path for
  cross-device numerical acceptance.
- The CPU reference and every device under test receive content-identical
  serialized float32 inputs, Boolean masks, and int64 valid counts.
- Before execution, length, shape, dtype, and content hash are checked to confirm
  input identity. An upstream input difference cannot be reported as a
  device-computation difference.
- The CPU reference strictly executes the approved float64 two-pass statistics
  and canonical balanced adjacent-pair reduction, including the frozen spatial
  ordering, odd-node carry rule, and precision-conversion boundaries.
- Device outputs are compared numerically with CPU reference outputs under the
  fixed tolerances and comparator semantics below.
- Cross-device acceptance does not require bitwise identity, and differing bytes
  alone do not cause failure.
- The CPU reference implementation version, code identity, and effective
  execution configuration belong to its auditable identity so an implementation
  update cannot silently change the reference baseline.
- Forward and backward comparison objects, upstream-gradient fixtures,
  non-finite handling, absolute and relative tolerances, comparator arithmetic,
  and near-zero rules are frozen below.

### Two-layer forward comparison objects

Approved:

- The first formal forward comparison object is the float64 structured statistic
  tensor containing `[mean, population_std]` for every valid region. Its shape
  and canonical axis order are `[B, 4, 8, 7, 21, 2]`.
- This first layer directly validates the two-pass statistics, canonical
  reduction tree, and population-standard-deviation results.
- The second formal forward comparison object is the canonical float32 pooled
  tensor with shape `[B, 4, 8, 7, 21, 2]`.
- The second layer is produced from the first by exactly one approved
  float64-to-float32 conversion and validates the actual numerical boundary into
  the classifier path.
- Both layers are compared elementwise with the normative CPU reference at the
  same logical indices.
- Absolute tolerance, relative tolerance, and near-zero rules for both layers
  are frozen below. Non-finite handling remains frozen below.
- The `[B, 9408]` classifier input is not a separate cross-device numerical
  comparison object because it is an arithmetic-free reshape.
- Local tests still verify equal element counts, the approved flattening order,
  and elementwise identical stored values between the reshaped classifier input
  and the canonical float32 pooled tensor.
- Reduction-tree internal nodes, first-pass totals, centered squared deviations,
  and second-pass totals are failure diagnostics only. They are not independent
  pass conditions, do not replace either formal layer, and need not be exported
  for an implementation to pass.
- The backward comparison object and injection boundary are frozen below.

### Pooling-input gradient as the sole backward comparison object

Approved:

- A content-identical float32 upstream gradient `G` is injected at the canonical
  post-conversion pooled tensor with shape `[B, 4, 8, 7, 21, 2]`.
- `G` uses the fixed pooled axis order and auditable serialized content.
- Backward is the vector-Jacobian product equivalent to
  `backward(gradient=G)`. It does not construct an explicit scalar loss through
  `sum`, `mean`, or another additional reduction.
- The sole formal backward comparison object is the float32 pooling-input
  gradient `dZ` with shape `[B, 4, 8, 7, H, W]`.
- `dZ` covers the complete spatial input. Positions excluded by the applicable
  region or mask remain formal comparison coordinates, and the normative result
  reflects their zero contribution.
- The CPU reference backward reuses the frozen forward precision path: float32
  input, float64 internal statistics, two-pass computation, canonical balanced
  adjacent-pair reductions, and one float64-to-float32 statistics conversion.
- Every acceptance run begins without historical gradients and performs exactly
  one backward pass. Gradient accumulation cannot affect the result.
- Boolean masks, region indices, and int64 valid counts are nondifferentiable
  inputs and have no compared gradients.
- Classifier, gating, and other electronic-backend parameter gradients are
  outside this acceptance scope.
- Intermediate-statistic, mean, and standard-deviation gradients may be emitted
  for failure diagnosis but are not independent pass conditions.
- The approved zero-variance mathematics remains unchanged:
  `d sigma / d variance = 0`, forward adds no epsilon, backward results are
  finite, and the standard-deviation branch contributes zero pooling-input
  gradient.
- A nonzero upstream gradient in the mean slot may make the complete `dZ`
  nonzero in a zero-variance region. If a fixture injects gradient only into the
  standard-deviation slot, the corresponding normative `dZ` is zero.
- The tolerance rules below do not alter, relax, or redefine the approved
  zero-variance mathematical semantics.
- General `dZ` tolerances, zero-variance fixture rules, and near-zero comparison
  rules are frozen below. Non-finite handling remains frozen below.

### Three exact upstream-gradient fixtures

Approved:

- All indices are zero-based integers:
  `b in [0, B-1]`, `j in [0, 3]`, `ell in [0, 7]`, `c in [0, 6]`, and
  `p in [0, 20]`. The final structured slot axis is `[mean, std]`.
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
- Signs and numerators are determined through integer operations before exact
  conversion to float32. Floating-point `pow`, floating-point remainder, and
  device-dependent random generation are prohibited.
- Every fixture value is an exactly representable float32 binary fraction. All
  fixtures use the same canonical axis order and auditable serialization, and
  CPU and device paths receive content-identical float32 content.
- The mean and standard-deviation fixtures isolate their respective backward
  branches. The joint signed fixture covers signs, multiple magnitudes, batch
  boundaries, and all structured axes to expose axis, slot, batch, or sign
  propagation errors.
- Each fixture runs as a separate backward acceptance test from a fresh gradient
  state. The three fixtures cannot be summed and substituted by one backward
  pass.
- The single cross-device `dZ` tolerance profile shared by all three fixtures is
  frozen below.

### Fail closed on non-finite comparison values

Approved:

- Every formal comparison object receives a complete finiteness check before any
  absolute or relative tolerance calculation.
- This check covers the float64 pre-cast structured statistic tensor, the
  float32 canonical pooled tensor, and the float32 `dZ` from each of the three
  separate upstream-gradient fixtures.
- If either side contains `NaN`, `+Inf`, or `-Inf`, that object cannot proceed to
  tolerance comparison.
- Non-finite values are never accepted as equal, including when both sides have
  the same non-finite type at the same logical index.
- A non-finite CPU result makes the reference execution or fixture invalid,
  blocks device acceptance for that object and fixture, and is not recorded as a
  device failure. Device-to-device comparison cannot replace the CPU baseline.
- If the CPU reference is wholly finite and a tested device contains a
  non-finite value, that device fails the corresponding object and fixture.
- Non-finite values cannot be replaced, clipped, clamped, zeroed, skipped, or
  excluded from error statistics.
- Diagnostics distinguish `NaN`, `+Inf`, and `-Inf`. A failure or block record
  includes comparison layer and dtype, complete logical index, CPU or tested
  device, forward fixture or backward `G` fixture, non-finite type, and verdict
  `reference_invalid` or `device_failed`.
- The first non-finite value is sufficient to determine that the object cannot
  pass. Scanning may continue only to collect more diagnostic indices; it cannot
  resume tolerance evaluation or change the verdict.

### Asymmetric elementwise CPU-reference tolerance gate

Approved:

- After both sides pass finiteness checks, every logical element is judged by
  `abs(x_device - x_CPU) <= atol + rtol * abs(x_CPU)`.
- The CPU value is always the reference. Operands cannot be exchanged, and a
  symmetric relative-error formula cannot replace this rule.
- The inequality is closed, so an error exactly equal to its tolerance bound
  passes.
- Every element is judged independently at its complete logical index. One
  failing element fails the complete formal comparison object.
- RMSE, MAE, maximum error, other aggregate metrics, mean pass rate, quantiles,
  and allowed failure fractions cannot replace the elementwise gate.
- Aggregate error metrics may be recorded for diagnosis but cannot alter the
  elementwise verdict.
- The complete fixed tolerance profile and comparator contract are recorded in
  the following final decision.

### Fixed tolerance profile, comparator arithmetic, and acceptance closure

Approved:

- The float64 pre-cast structured statistic tensor uses
  `atol = 0x1p-48`, `rtol = 0x1p-47`, and near-zero threshold `0x1p-40`.
- The float32 post-conversion canonical pooled tensor uses
  `atol = 0x1p-22`, `rtol = 0x1p-21`, and near-zero threshold `0x1p-16`.
- The float32 pooling-input gradient `dZ` uses
  `atol = 0x1p-21`, `rtol = 0x1p-19`, and near-zero threshold `0x1p-16`.
  This one profile applies to all three separate backward fixtures.
- These hexadecimal powers of two are exact normative binary64 constants.
  Decimal parsing cannot define them.
- With unit roundoff `u64 = 0x1p-53` and `u32 = 0x1p-24`, the budgets are,
  respectively, `32*u64` absolute plus `64*u64` relative for float64
  statistics, `4*u32` plus `8*u32` for the converted float32 pool, and
  `8*u32` plus `32*u32` for `dZ`. This ordering reflects the fixed two-pass
  arithmetic, one conversion, and the longer derivative/cast path without
  permitting a changed algorithm.
- Calibration uses deterministic fixtures covering zeros, subnormal-adjacent
  values with FTZ/DAZ disabled, near-zero boundaries, signed cancellation,
  nonzero and zero variance, all structured axes, and all dtype boundaries.
  On every declared supported device/configuration, the maximum observed valid
  error must be at most one quarter of its applicable normative bound. Exceeding
  that margin blocks acceptance and cannot widen a tolerance automatically.
- Negative controls for axis permutation, mean/std slot exchange, sign
  inversion, one-position spatial displacement, wrong mask contribution,
  changed reduction order, and an extra or missing float32 cast must each fail
  at least one element when applicable. Passing any such negative control
  invalidates the tolerance calibration.
- Device results return to CPU in each formal object's original dtype before
  comparison. Float32 values widen exactly to binary64. Float64 values cannot be
  downcast, requantized, or otherwise rounded before comparison. A GPU,
  accelerator, or other tested device cannot generate the final verdict.
- After formal-object finiteness checks, CPU comparator intermediates are formed
  in this exact order:
  `difference = round64(x_device - x_CPU)`;
  `error = abs(difference)`;
  `relative_term = round64(rtol * abs(x_CPU))`;
  `bound = round64(atol + relative_term)`.
  The final condition is `error <= bound`; equality passes. For finite binary64
  inputs, `abs` changes only the sign bit and adds no rounding.
- Comparator execution uses IEEE 754 round-to-nearest, ties-to-even. FMA,
  arithmetic reassociation, extended-precision temporaries, lower-precision
  intermediates, FTZ, and DAZ are prohibited.
- Formal-object non-finites retain the previously approved verdicts. If both
  formal inputs are finite but `difference`, `relative_term`, or `bound` becomes
  non-finite, the verdict is `comparator_invalid`. It blocks that acceptance
  run, is not a device failure, and cannot fall back to another formula.
- If the CPU reference is either signed zero, the tested-device value must be
  either signed zero. Signed-zero bit differences are accepted, but no nonzero
  value receives an `atol` allowance at a normative-zero coordinate.
- When `0 < abs(x_CPU) <= near_zero_threshold`, the relative term is still
  computed and recorded, but the tightened gate is `error <= atol`. Larger CPU
  references use the general gate. This special rule cannot relax the formula.
- In a zero-variance region, mean-only and joint fixtures compare the complete
  normative `dZ` normally, including exact zero at every CPU-zero coordinate.
  In the std-only fixture, the complete normative `dZ` for that region is zero,
  so every corresponding device coordinate must be numerical zero.
- Every object record includes run, comparator, tolerance-profile, CPU,
  device, effective floating-point configuration, object, dtype, fixture and
  content-hash identities; shape and element count; all three tolerance
  constants; verdict; failing-element count; and any `reference_invalid`,
  `device_failed`, or `comparator_invalid` classification.
- Logical indices are `[b, j, ell, c, p, slot]` for pooled statistics and
  `[b, j, ell, c, y, x]` for `dZ`. The first failure is selected by canonical
  row-major scan. Its record contains both values, `difference`, `error`,
  `relative_term`, `bound`, and the active rule.
- Maximum finite absolute error and its first canonical tie index are recorded
  for every object, including passing objects, as diagnosis only. A formal
  non-finite object has no numerical maximum; a comparator-invalid object
  records the finite inputs and first invalid intermediate index.
- A formal object passes only with matching identities, shape, dtype, and count;
  finite formal inputs; valid comparator intermediates; and all elements
  passing. One fixture passes on one device only when its inputs and content
  identities match and all objects required by that fixture pass.
- One device passes only when every required forward fixture passes both forward
  objects, all three fresh backward fixtures pass `dZ`, and every audit and
  floating-point-environment check passes. Missing required evidence blocks.
- Cross-device equivalence passes overall only when the acceptance manifest
  names at least one non-CPU tested device and every named device passes. No
  object, fixture, or device pass rate can replace these conjunctions.
- Required tests cover exact constant parsing and widening; preservation of
  float64; comparator order and rounding; equality and next-representable
  failure boundaries; signed and near zeros; formal and comparator-created
  non-finites; all negative controls; canonical index and tie handling; all
  three fresh backward fixtures including zero variance; reshape identity;
  verdict propagation; and changes to comparator code or floating-point audit
  identity.
- Comparator implementation version, code identity, tolerance-profile version,
  and every effective configuration capable of changing floating-point behavior
  are part of the audit identity. An identity/execution mismatch blocks
  acceptance.

## 2026-08-02 — Accept the Decision 30 RTX 4090 formal calibration result

Approved:

- Decision 30 formal cross-device numerical-equivalence acceptance names one
  non-CPU tested device: `NVIDIA GeForce RTX 4090`, GPU UUID
  `GPU-302ac56f-e8de-c41b-76ed-3c66fc66ea95`.
- The accepted run ID is
  `decision30-formal-acceptance-rtx4090-20260802`, executed in
  `formal_acceptance` mode from Git commit
  `9b5d2ea61eabbb61e54570696f14783baba27ee7`.
- The pre-registered formal fixture is
  `decision30-formal-rtx4090-v1` with shape `[1, 4, 8, 7, 9, 11]` and input
  hashes:
  - `z`:
    `248bfed97cc17b71b5fa0553e622155ec6a309f64d32f4c4d285aed8c7a877b9`;
  - `mask`:
    `466fbff22dd950cb181e52dc62ebf5de2311733b59b74125e01a78bf34299767`;
  - `valid_counts`:
    `43350a995b402b25021884314a636d4305864ac4fe7de19fa6f4a8e3a6b29687`.
- Both forward comparison objects and all three fresh-autograd backward `dZ`
  objects passed their elementwise gates and quarter-margin requirements.
- All eight candidate-operator negative controls were detected. Input identity,
  the protected floating-point environment, and every required zero-variance
  forward/backward check passed.
- The audited report is
  `artifacts/decision30_formal_acceptance_rtx4090_20260802.json`, with SHA-256
  `fe0c0a3d704a2ae458e26e894ef82be0fddcdf13ad430ac5f483bf72b1836117`.
  The report remains an ignored local acceptance artifact and is not approved
  for Git tracking.
- The report's CPU reference, device operator, comparator, and calibration
  orchestrator code identities match the named commit. The recorded runtime
  used PyTorch `2.3.0a0+6ddf5cf85e.nv24.04`, CUDA `12.4`, cuDNN `90100`, and
  driver `580.126.09` on Linux x86-64.
- This approval closes only the Decision 30 formal RTX 4090 calibration gate.
  It does not pass the repository-wide Phase 0 acceptance gate, approve any
  remaining `TBD`, authorize training or evaluation, or approve entry into the
  next phase.

## 2026-08-03 — Conditionally approve Phase 0 total-acceptance closure

Approved:

- The sole current next work is the **Phase 0 total-acceptance closure**.
- Until all Phase 0 deliverables pass their acceptance criteria and a human
  explicitly approves the phase transition, the project must not run CAM16
  training, model selection, comparative, ablation, transfer, or final-test
  experiments.
- The closure scope is limited to the eleven Phase 0 deliverables in Section 4 of
  `docs/DEVELOPMENT_SPEC.md` and freezing the five existing blocking decision
  groups in Section 6. It must not add a new research module or broaden the
  primary model.
- Evidence for every Phase 0 deliverable must identify the code location,
  configuration location, tests, acceptance metrics, and produced artifact. A
  statement that a deliverable is merely "implemented" is insufficient.
- Patient-level isolation may be claimed only when a reliable patient-to-slide
  mapping exists and has been validated. If that mapping is absent or not
  validated, evidence must state the actually verified supplied identifier level,
  such as `group_id` or `slide_id`, and must not imply patient-level isolation.
- `group_id`-level or `slide_id`-level evidence does not satisfy the existing
  patient-level Phase 0 acceptance gate unless that identifier has itself been
  verified as a patient identity.

Not approved by this decision:

- None of the five blocking decision groups is resolved; every value in those
  groups remains `TBD` pending separate human review.
- The Phase 0 acceptance gate has not passed, and entry into the next phase is not
  approved.

## 2026-08-03 — Approve `linear-logit-v1` and the exact backend budget

Approved:

- The primary trainable classifier is `linear-logit-v1`.
- Its input is the approved float32 identity handoff with shape `[B, 9408]`.
- It contains exactly one biased affine map,
  `z[b] = b_head + sum_(q=0)^9407 w[q] * x[b,q]`, and emits one float32 raw
  binary logit per patch with shape `[B]`.
- `w` has shape `[1, 9408]`, `b_head` has shape `[1]`, and both are explicitly
  initialized to exact zero. Framework-default or random initialization is not
  part of the primary classifier.
- The classifier has no sigmoid, hidden layer, activation, normalization,
  dropout, feature selection, attention, residual connection, trainable
  temperature, additional parameter, running state, or persistent buffer.
- Probability conversion, thresholding, and calibration are outside the
  classifier and remain unresolved evaluation decisions.
- The classifier has exactly 9409 trainable scalars. Together with the approved
  64 cross-stain gating scalars, the complete electronic backend has exactly
  **9473** trainable scalars. The fixed optical frontend has exactly zero.
- This parameter budget is an equality rather than an upper bound. Every
  electronic-backend parameter must occur exactly once in optimizer-ownership
  audits, no optical parameter may occur, and any extra trainable scalar fails
  acceptance.
- No additional classifier weight tying or H/E-exchange invariance is imposed.

Still unresolved:

- Patch sampling, slide-level aggregation, probability conversion, decision
  threshold, calibration, and evaluation metrics.
- Loss definition, loss-internal precision, and optimizer-state precision.
- Optimizer algorithm, regularization, schedule, training seeds, optimization
  budget, confidence intervals, and the final-once test gate.
- Transfer datasets, physical-scale adaptation, and transfer protocol.

This decision freezes one of the five blocking decision groups. It does not
authorize implementation-dependent defaults, training, evaluation, or a phase
transition.

## 2026-08-03 — Approve the governing principles of `cam16-eval-v1`

Approved in principle:

- The primary patch-level evaluation object is defined by an immutable patch
  manifest.
- Labels, outcomes, tumor annotations, patch labels, and derived tumor-location
  fields must not be used to screen or filter existing patches before their
  predictions are fixed for metric calculation.
- Existing validated patches may be aggregated only by identifiers already present
  in the current data package. A slide-level result requires an identifier declared
  and validated as `slide_id`; a generic `group_id` result must remain group-level.
  Such aggregation does not represent a complete WSI, coverage of all tissue
  regions, a complete patch set, or patches produced by a uniform WSI candidate
  algorithm.
- Sigmoid-transformed patch and aggregate outputs are called **uncalibrated
  evaluation scores**. They are not clinical or natural-population probabilities.
- Patch and slide thresholds are separate, use validation data only, and are
  selected by a fully frozen Youden algorithm.
- The sole primary endpoint is slide-level AUROC.
- Every manifest identity calculation, metric, calibration estimate, threshold
  candidate, tie rule, and exceptional case must have one unique auditable
  computation definition before the protocol is frozen.
- The missing reliable patient-to-slide mapping keeps the patient-level Phase 0
  gate unmet. This decision does not authorize training or test evaluation.

Still unresolved within this decision group:

- Existing-patch manifest canonicalization and identity contracts.
- Training patch sampling, group or slide aggregation, and any permitted
  transformation policy.
- Exact score-widening, metric, calibration-estimation, and undefined-case
  arithmetic.
- The complete validation Youden candidate set, operation order, tie handling,
  and exceptional-case rules.

Accordingly, `cam16-eval-v1` is **not yet frozen**, and this decision group does
not yet count as the second completed blocking group.

Historical governance note: the complete WSI candidate-generation route was
withdrawn when the project data entry changed to an existing patch data package;
it is no longer within the current project scope.

## 2026-08-03 — Freeze all remaining repository-internal Phase 0 decisions

Authorization:

- The human attached a long-line autonomous closure instruction that explicitly
  authorizes conservative decisions for all repository-internal Phase 0 `TBD`
  groups without pausing for item-by-item approval.
- This approval does not supply or validate a patient-to-slide mapping and does not
  authorize an automatic phase transition, formal training, or test access.

Approved loss and optimizer contract:

- The formal loss is mean binary cross entropy with raw logits, evaluated in
  float32 with no class weight, sample weight, label smoothing, focal term, or
  probability pre-conversion.
- The optimizer is AdamW with float32 parameter, gradient, and state tensors;
  learning rate `0.001`, betas `(0.9, 0.999)`, epsilon `0.00000001`, and weight
  decay `0.0001`. There is no learning-rate scheduler.
- Every one of the 9473 electronic parameters appears exactly once in the
  optimizer and no optical value appears. AMP, gradient scaling, TF32, float16,
  bfloat16, and gradient clipping are absent.

Approved training baseline:

- Each training-manifest row is used exactly once per epoch in the frozen
  SHA-256 keyed order. The baseline uses no resampling, weighting, augmentation,
  stochastic image transform, or dropped partial batch.
- Batch size is 4, the maximum is 20 epochs, and early stopping monitors validation
  slide AUROC with patience 5 and minimum improvement 0.
- An immutable checkpoint is saved after every complete epoch. The checkpoint with
  greatest validation slide AUROC is selected; an exact tie selects the earliest
  epoch. Resume is explicit and accepts only the latest complete checkpoint with
  exact config, code, data, kernel, and seed identities.
- Formal seeds are `1729`, `3407`, and `7919`. A failed seed is recorded and
  excluded without automatic retry or replacement. Valid results report each seed,
  their arithmetic mean, and their sample standard deviation.
- These values are the preregistered Phase 1 starting baseline. They have not been
  shown by training or model selection to be optimal.

Approved `cam16-eval-v1` calculation contract:

- The existing-patch CSV, disk inventory, and immutable prediction ledger fail
  closed on missing, extra, duplicate, conflicting, escaping, or non-finite data.
  No row is screened by labels, annotations, tumor location, quality, or output.
- A slide score is the maximum finite raw float32 patch logit among manifest rows
  for that `slide_id`. It is explicitly manifest-bounded and is not complete WSI
  inference. A generic group cannot be relabeled as a slide or patient.
- Slide AUROC is the sole primary endpoint and patch AUROC is secondary. Both use
  exact Mann-Whitney win/tie counts on raw float32 logits.
- Patch and slide thresholds are independently fit on validation distinct logits.
  Prediction is `z >= t`; maximum exact Youden J wins and the numerically largest
  threshold wins an exact tie. Test contributes no candidate or choice.
- Secondary threshold metrics use exact confusion counts and explicit undefined
  reasons. Sigmoid output is an uncalibrated evaluation score. Calibration-bias,
  Brier, and ten-bin ECE are descriptive only; no calibration transform is fit.
- The primary 95 percent uncertainty interval is a 2000-replicate stratified slide
  bootstrap percentile interval driven by the run seed, using frozen order indices
  49 and 1949. It is not patient-cluster uncertainty.
- Test remains inaccessible until a future final-once authorization names the
  frozen config, code, manifest, checkpoint, and validation-threshold identities.

Approved transfer disposition:

- No transfer dataset, physical-scale adaptation, or transfer protocol belongs to
  the CAM16 Phase 1 starting baseline. Those choices require a later, separate
  preregistration before transfer evaluation and do not block preparation of the
  CAM16 training entry.
- No WSI, candidate-generation, fabrication, SLM, hardware-control, clinical, or
  physical-deployment route is restored.

Rejected alternatives:

- hidden framework defaults, data-adaptive parameter selection, weighted or
  balanced sampling, augmentation, MLP classifiers, learned/normalized optical
  features, test-based choices, automatic failed-run retries, inferred patient
  identities, and implementation-defined metric/threshold arithmetic.

Implementation and evidence:

- The unique formal contract is `configs/phase1_baseline.toml`; the non-formal
  contract is `configs/phase0_dry_run.toml`.
- Normative explanations are `docs/TRAINING_PROTOCOL.md`,
  `docs/EVALUATION_PROTOCOL.md`, and ADR 0010. Source is under
  `src/cg_pipeline/`; public-interface and negative tests are under `tests/`.
- The untracked `docs/CAM16_EVAL_V1_CALCULATION_CONTRACT_DRAFT.md` remains a
  non-normative historical draft and is neither overwritten nor required by this
  decision.

Still externally unresolved:

- The supplied package has `slide_id` but no validated reliable patient-to-slide
  mapping. Slide-ID split isolation is reportable; patient-level isolation remains
  not evaluated and the patient-level Phase 0 gate remains unmet.
- The minimum acceptable external evidence is one complete provenance-bearing CSV
  with exactly `slide_id`, `patient_id`, and `provenance`, covering every in-scope
  slide with no duplicate assignment or patient identity crossing splits.
- `configs/phase0_release.json` therefore remains closed. Formal training must
  continue to fail preflight before its first batch.

Implementation-audit addendum:

- A nonempty `provenance` cell proves only that the mapping CSV is structurally
  complete; it does not prove the source is reliable. The patient gate additionally
  requires an attributable approval artifact, and the release record binds its
  SHA-256 together with the exact mapping and source-manifest SHA-256 values.
  Preflight reads and hashes that separate artifact; a claimed digest without the
  artifact, or any post-approval edit, fails the gate.
- Every preregistered scientific field is value-locked by the config validator.
  Resume accepts only a continuous zero-based sequence of immutable checkpoint and
  epoch-report pairs whose config, code, source/effective manifest, complete fixed
  frontend, model-state checkpoint, seed, and epoch identities match exactly.
- Evaluation compares the prediction ledger against every authorized effective
  manifest row. Results bind config, code, source/effective manifest, fixed frontend,
  checkpoint, seed, and validation-threshold identities. Test access has no Boolean
  shortcut; it requires a final-once authorization object whose identities all
  match the frozen evaluation context.
  The authorization loader likewise reads a separate approval-evidence artifact
  and rechecks both artifact hashes at evaluation time.

## 2026-08-03 — Correct the patient-level gate scope and close Phase 0

Human authorization and scope review:

- The current instruction explicitly freezes `slide_id` as no more than the
  supplied `group_id`, limits the split guarantee to that identifier level, and
  states that the CAM16 study does not require a patient-level performance or
  isolation claim.
- The repository's normative documents, ADRs, README, training/evaluation
  protocols, and the hospital preliminary-cooperation proposal were reviewed for
  an overriding ethics, paper-protocol, or data-use requirement.
- The hospital proposal reserves patient grouping for future hospital/cross-domain
  primary confirmatory analysis and says mapping absence permits only a slide-level
  statement. Its ethics and data-use sections govern approval, de-identification,
  access, and non-reidentification; they do not make a patient mapping a prerequisite
  for the current simulation-only CAM16 Phase 1 baseline.

Approved correction:

- The previous patient-level Phase 0 blocker was an erroneous scope upgrade. It is
  `NOT APPLICABLE`, not `FAIL`, for the current CAM16 Phase 0 acceptance and Phase 1
  formal train/validation preflight.
- The only permitted isolation statement is
  `group_id/slide_id split isolation verified`.
- Machine-readable state is frozen as
  `patient_level_isolation = not_evaluated` and
  `patient_level_claim_allowed = false`.
- No code, configuration, report, filename convention, or identifier syntax may
  infer, fabricate, or parse patient identity. A patient mapping and its approval
  artifact are not preflight or release inputs.
- Any attempt to enable a patient-level claim, mark patient-level isolation as
  verified, or publish that safety statement must fail validation. A future study
  that actually requires a patient-level claim must obtain a separately approved,
  reliable mapping and a new scoped decision; that future requirement does not
  reopen this Phase 0 gate.

Release disposition:

- `configs/phase0_release.json` schema v2 records `phase0_closed = true`,
  `formal_training_authorized = true`, no external blocker, the frozen patient
  state above, and `test_access_authorized = false`.
- With every other acceptance gate passing, Phase 0 is closed and the preregistered
  CAM16 Phase 1 train/validation entry is authorized. No formal training or test
  evaluation is executed by this decision itself.
- This decision supersedes earlier 2026-08-03 statements that patient mapping was
  an external Phase 0 blocker, that the patient-level gate remained unmet, or that
  the release had to remain closed for that reason. Historical text remains as an
  audit trail but is no longer normative.

## 2026-08-04 — Approve Phase 1 formal batch-size revision to 32

Human approval and scope:

- The formal Phase 1 training batch size is revised from 4 to 32 to improve
  RTX 4090 24 GB hardware utilization. The historical batch-size-4 decision above
  remains unchanged as an audit record.
- This revision does not start a formal training run or expand authorization to
  test access, a split or manifest change, or any change to the fixed optical
  frontend or model structure.

Approved direct consequences:

- With 79,570 train rows and `drop_last = false`, a complete epoch contains
  `ceil(79570 / 32) = 2487` optimizer updates and covers every train row exactly
  once. The 20-epoch maximum is 49,740 updates, replacing the historical batch-4
  budgets of 19,893 updates per epoch and 397,860 maximum updates.
- Validation reuses the same configured batch size. Its 18,171 rows therefore
  produce `ceil(18171 / 32) = 568` batches, including the final partial batch; all
  validation rows remain covered.
- The revised formal run identity is `phase1-cam16-baseline-b32-v2`. Its normalized
  config hash is
  `sha256:e44768d80d7c1545138d7d5e1368de4ed53b7b07b71202e2c5bdee6efac7cf3b`,
  superseding batch-4 hash
  `sha256:0653ae0003dac9062b73749e879a9a541a3f9dae18b034bdc1632f8410910e75`.
  The historical `phase0-closed-v1` tag and `configs/phase0_release.json` are not
  overwritten; the active revision is
  `configs/phase1_training_release_b32_v2.json`.

Unchanged contract and deferred empirical question:

- AdamW, learning rate `0.001`, betas `(0.9, 0.999)`, epsilon `0.00000001`, weight
  decay `0.0001`, no scheduler, no class weighting, no augmentation, no gradient
  clipping, at most 20 epochs, patience 5/minimum delta 0, seeds `1729`, `3407`,
  `7919`, immutable per-complete-epoch checkpoints, validation slide-AUROC
  selection, and the final-once test prohibition remain unchanged.
- The learning rate is not changed by a linear-scaling rule. Batch 32 and batch 4
  do not have the same optimizer-step budget. Whether learning rate or epoch budget
  needs a separate future revision will be judged from validation convergence
  curves after an authorized formal run; no suitability claim is made now.

## 2026-08-04 — Publish identity-bound Phase 1 training release v3

Human-approved release correction:

- Release governance identity is fixed as `phase1-training-b32-v3`; its annotated
  tag has the same name. The unchanged Run ID
  `phase1-cam16-baseline-b32-v2` remains the training-configuration identity. The
  two version strings have different roles and do not imply a configuration change.
- Formal code identity is commit
  `e50cc7e0aa16655f132b9cd321bb5a26b41f76bc`. The release commit must have exactly
  one parent, that parent must be this formal code commit, and the annotated tag
  must peel exactly to the release commit.
- The release commit is restricted to exactly
  `configs/phase1_training_release_b32_v3.json`, `docs/DECISIONS.md`, and
  `docs/PHASE1_TRAINING_RUNBOOK.md`. No code, data, split, model, optimizer,
  evaluation, or training-configuration change is permitted in that commit.

Approved data identities and derivation rules:

- Hash algorithm is SHA-256. Source-manifest identity is raw file bytes under rule
  `raw-file-bytes-v1` and is frozen as
  `sha256:23c681a3a338e4df96c2e3443b39349c4758e08009eb47d46928d148f62045ab`.
- Effective split identities use domain rule `cg/cam16-eval-manifest/v1`. Train is
  `sha256:8c54e7f8b1674e4e94c9a46e0d9abf01e4c0c8a88605e7831b2701c0ddbe58c5`;
  validation is
  `sha256:1a6fd51cb6d7ae5da920f06974a871deef2f21147f0df9c4d2c902d30ed3decc`.
- Formal preflight recomputes and compares the source, train, and validation values
  individually. It neither freezes nor computes a test effective-split hash.
  `test_access_authorized = false`, `patient_level_isolation = not_evaluated`, and
  `patient_level_claim_allowed = false` remain unchanged.

Approved preflight consumption contract:

- The standalone `preflight` command runs once and exclusively writes
  `artifacts/preflight/phase1-training-b32-v3/preflight.json`. Formal
  `train --preflight-report` consumes that report and does not repeat the full
  model/optimizer/spectral preflight.
- Except for audit-only `created_at`, the consumed authorization report contains
  only fields that training can reconstruct from current frozen values or live
  identities. Its strict schema,
  canonical content hash, HEAD/tag/release/config/code/source-manifest/train/val
  identities, disk/isolation state, and governance fields are all checked before
  the first batch. A mismatch, injected field, or content corruption fails closed.
- `created_at` is audit-only. It has no age, expiry, or future-time release gate.
  Reuse is allowed only while every live identity still matches exactly.
- Before using the unchanged Run ID, both the release-bound preflight path and
  `artifacts/formal_runs/phase1-cam16-baseline-b32-v2` must be absent. Existing
  output is never overwritten; a collision stops the run and requires a separately
  approved identity-bearing output contract rather than mixing v2/v3 artifacts.

Verification and scope:

- Public CLI and Python API tests cover matching reports and fail-closed cases for
  wrong parent SHA, multiple parents, code changes outside the whitelist,
  lightweight tags, wrong release ID, changed commit, source/effective manifest
  changes, old release/report identity, report tampering including recomputed
  content hashes, and existing preflight/formal-output paths.
- This publication authorizes only the already approved CAM16 train/validation
  entry. It does not execute training, authorize test access, make a patient-level
  leakage claim, alter a dataset split, or advance another phase.

## 2026-08-10 — Replace synthetic dry-run with exploratory/formal training modes

Human-approved workflow change:

- The only real CAM16 training modes are `exploratory_train` and `formal_train`.
  The synthetic dry-run CLI, configuration profile, implementation, acceptance
  dependency, and active documentation are retired.
- `exploratory_train` uses real CAM16 train/validation data for profiling,
  performance optimization, bounded `max_steps`, one-epoch checks, and other
  explicitly non-formal experiments. It may run from a dirty/untracked tree and
  does not require a release, tag, clean-tree audit, formal hash authorization, or
  formal preflight report.
- Exploratory CLI overrides are limited to seed, output/run identity, batch size,
  worker count, runtime device, maximum epochs, and maximum steps. Every exploratory report and
  checkpoint metadata record `formal_experiment=false` and
  `experiment_mode=exploratory_train`; these results cannot be automatically
  promoted or renamed as formal evidence.
- Shared lightweight data safety checks cover the data path, readable manifest,
  legal nonempty train/validation splits, and supplied `group_id`/`slide_id`
  cross-split isolation. Neither mode exposes or constructs a test dataset.

Unchanged formal and scientific contracts:

- `formal_train` retains the approved release, annotated tag, clean source tree,
  release/config/source/manifest hashes, standalone formal preflight, fixed seeds,
  complete epochs, validation checkpoint selection, immutable checkpoints,
  provenance, formal output identity, and non-overwrite gates.
- No dataset split, final-once test rule, isolation claim, fixed optical frontend,
  Morlet definition, spatial-pyramid numerical rule, formal metric, checkpoint
  winner rule, formal seed, provenance rule, or project phase changes.

## 2026-08-11 — Approve workers8 Phase 1 release identity

Human approval and engineering scope:

- The approved release-governance identity is
  `phase1-training-b32-workers8-v1`, superseding
  `phase1-training-b32-v3`. Historical v2/v3 release records and tags remain
  immutable.
- Formal and default exploratory DataLoader execution explicitly use
  `num_workers = 8`. This is a fixed engineering configuration, not runtime or
  hardware-dependent automatic tuning.
- The normalized formal config hash is
  `sha256:a0beda02cd93de04c596f36929ba5aa05c51940e0d82d11297058dc5860666a5`.
  Batch size 32, optimizer, learning rate, epoch budget, early stopping, seeds,
  data splits, model, fixed optical frontend, and evaluation protocol are
  unchanged.

Release topology:

- The formal code commit is
  `340796d4c8fe8916ad7b1d916486207dc4ffd649`.
- The release commit has that commit as its sole parent and is restricted to
  `configs/phase1_training_release_b32_workers8_v1.json`,
  `docs/DECISIONS.md`, and `docs/PHASE1_TRAINING_RUNBOOK.md`.
- The Run ID remains `phase1-cam16-baseline-b32-v2`. Test access remains false,
  patient-level isolation remains `not_evaluated`, and no test effective-split
  identity is computed or frozen.
- This commit prepares the approved release identity but does not publish it. The
  release becomes published only after separate authorization creates an annotated
  `phase1-training-b32-workers8-v1` tag peeling exactly to the release commit.
  No training, test evaluation, benchmark, or phase transition is authorized by
  this decision.
