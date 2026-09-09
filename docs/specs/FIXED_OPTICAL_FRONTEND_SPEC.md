# Fixed optical frontend specification

Normative clauses incorporated by [DEVELOPMENT_SPEC](../DEVELOPMENT_SPEC.md).
These relocated clauses retain their locked scientific meaning and change control.
The highest-level specification and explicitly approved changes in
[DECISIONS](../DECISIONS.md) govern; do not resolve conflicts by file date.
Original section numbers are retained for traceability. Phase2-A exceptions
are scoped by DEVELOPMENT_SPEC section 2 and the 2026-09-05 decisions;
they do not relax the primary Phase1 requirements below.

## Locked stain and Morlet contract (original section 3)

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

The generator must fail when a required field is missing, invalid, or unresolved. It must generate the
bundle once and provide the same immutable tensor reference to the H and E paths.
The interface must expose enough metadata to reproduce channel ordering without
relying on source-code iteration order.

## Phase 0 acceptance clauses (original section 5, lines 1161-1179)

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
## Phase 0 acceptance clauses (original section 5, lines 1265-1316)

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
