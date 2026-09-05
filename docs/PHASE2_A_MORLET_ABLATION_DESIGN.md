# Phase2-A Morlet parameter ablation design (proposal)

Status (2026-09-05): **Phase2 validity implemented; no training started**.

Update: the Phase2-A validity gate supersedes the historical acceptance blocker
below. A0-A3 pass CPU model/frontend preflight with configured theoretical sampled
zero-DC peak checks and a synthetic forward. All original coverage diagnostics
remain recorded. A full cloud CUDA/train-validation preflight is still required
before training; these data-free checks are not that authorization.

The proposal below is retained as design history. Its implementation blocker has
been resolved and all four proposed TOMLs now exist. The bounded Phase2 contract
passes sigma0/xi0/gamma through configuration, model, frontend, generator and
formal construction, and records the values in effective configuration and
Phase2 checkpoint metadata. Phase1 defaults remain numerically identical.

Current acceptance blocker: A1/A2/A3 fail the unchanged spectral-coverage gates.
A1's continuous-reference beta error is about 0.02283; the finite discrete DC
and energy checks pass. Non-default generation records the continuous-reference
error as a diagnostic; the default bank retains its original rejection rule.
The proposed experiments are not yet approved as ready for cloud formal training.

## 1. Current baseline and implementation fact

The frozen reference is `configs/phase1_baseline.toml`, run
`phase1-cam16-baseline-b32-v2`. Its model table declares the values below, and
`src/cg_pipeline/morlet.py::generate_morlet_bundle()` currently hard-codes the
same values.

| Item | Frozen baseline |
|---|---|
| Scale count / indices | `J = 4`, `j = 0, 1, 2, 3` |
| Orientation count / angles | `L = 8`, `theta_l = l*pi/8`, i.e. 0, 22.5, ..., 157.5 degrees clockwise in image coordinates |
| Spatial envelope | `sigma_j = 0.8 * 2^j` pixels |
| Center angular frequency | `xi_j = (3*pi/4) * 2^(-j)` radians/pixel |
| Carrier wavelength | `lambda_j = 2*pi/xi_j` = `8/3, 16/3, 32/3, 64/3` pixels |
| Anisotropy / orientation-bandwidth control | `gamma = 0.5`; spatial standard deviations are `sigma_j` parallel and `sigma_j/gamma` perpendicular |
| Approximate Gaussian spectral standard deviations | parallel `1/sigma_j`; perpendicular `gamma/sigma_j` radians/pixel |
| Common finite support | `105 x 105`, coordinates `-52..52` on both axes |
| DC and energy handling | finite-support discrete zero-DC projection, then per-kernel unit complex L2 normalization |
| Channel order | scale-major, then orientation-major; `c = 8*j + l` |

Per-scale values:

| `j` | `sigma_j` | `xi_j` (rad/pixel) | wavelength (pixels) | approx. spectral std `(parallel, perpendicular)` |
|---:|---:|---:|---:|---:|
| 0 | 0.8 | `3*pi/4` | `8/3` | `(1.25, 0.625)` |
| 1 | 1.6 | `3*pi/8` | `16/3` | `(0.625, 0.3125)` |
| 2 | 3.2 | `3*pi/16` | `32/3` | `(0.3125, 0.15625)` |
| 3 | 6.4 | `3*pi/32` | `64/3` | `(0.15625, 0.078125)` |

Important: these TOML fields are not currently runtime-adjustable. The strict
formal loader rejects any changed Morlet value, and model/preflight construction
does not pass the TOML Morlet values to the generator. Copying the baseline TOML
and editing a value would therefore be rejected or, if validation were bypassed,
would still generate the baseline bank.

## 2. Recommended first-round matrix

Use one Phase2 baseline rerun plus three perturbations. A fresh baseline is needed
because enabling parameter consumption changes the construction path; comparing
only against historical Phase1 output would confound parameter effects with that
code-path change. Phase1 configuration and artifacts remain untouched.

Every row retains the Phase1 train/validation manifests and split contract,
fixed/non-trainable frontend, H/E basis, `J=4`, `L=8`, `105 x 105` support,
electronic backend, batch 32, seeds 1729 and 3407, optimizer, epoch budget, early
stopping, evaluation, determinism, precision, and `allow_test=false`.

| ID | Only Morlet change from baseline | Research question | Recommended run ID |
|---|---|---|---|
| A0 | none | Does the parameterized Phase2 construction reproduce the Phase1 baseline behavior under the same training contract? | `phase2a-cam16-morlet-baseline-b32-v1` |
| A1 | `sigma0: 0.8 -> 0.7` (all `sigma_j` retain dyadic scaling) | Is performance sensitive to a narrower spatial envelope / broader spectral envelope while carrier centers stay fixed? | `phase2a-cam16-morlet-sigma0-0p7-b32-v1` |
| A2 | `xi0: 3*pi/4 -> 2*pi/3` (all `xi_j` retain dyadic scaling) | Does shifting every carrier center downward in frequency (wavelengths become 3, 6, 12, 24 pixels) reduce the advantage? | `phase2a-cam16-morlet-xi0-2pi3-b32-v1` |
| A3 | `gamma: 0.5 -> 0.625` | Is performance sensitive to broader perpendicular spectral bandwidth / reduced spatial elongation while scales and carrier centers stay fixed? | `phase2a-cam16-morlet-gamma-0p625-b32-v1` |

This is a representative sensitivity screen, not a grid search. It does not by
itself estimate an optimum or prove causal dependence on every Morlet parameter.

## 3. Proposed config files

After approval and after the runtime blocker below is resolved, add exactly four
full formal profiles because the current TOML format has no inheritance:

1. `configs/phase2a_morlet_baseline.toml`
2. `configs/phase2a_morlet_sigma0_0p7.toml`
3. `configs/phase2a_morlet_xi0_2pi3.toml`
4. `configs/phase2a_morlet_gamma_0p625.toml`

Each profile should differ from `configs/phase1_baseline.toml` only in
`execution.run_id`, matching `execution.output_root`, a Phase2-A Morlet contract
identity, and the single named Morlet parameter for A1-A3. A0 changes identity
and output only. Do not reuse, rename, overwrite, or edit the Phase1 run/config.

## 4. Deliberately excluded from round one

- `J` or `L`: changes the 32-channel interface and therefore the electronic
  backend shape/parameter budget; it is not a matched single-variable test under
  the stated constraints.
- support: the convolution implementation, padding, crop, and valid-support mask
  are fixed to radius 52. Changing support is a compound execution-boundary
  change, not only a kernel-parameter perturbation.
- simultaneous `sigma0` and `xi0` changes: even if their product were held fixed,
  that is a two-parameter intervention.
- large or symmetric grids: add only if this screen shows a material, consistent
  effect across both seeds.

## 5. Blocker before config creation

No valid perturbation config can be prepared under the current contract without a
separately approved implementation change. The minimum required change is to:

1. define a distinct Phase2-A Morlet contract rather than weakening the frozen
   Phase1 exact-value contract;
2. pass the approved Morlet specification from config through formal training and
   preflight into one parameterized generator;
3. generate new parameter/kernel identities and apply numerical checks appropriate
   to each candidate, without comparing candidates to the frozen Phase1 hash;
4. prove A0 produces the same kernels and frontend behavior as the frozen baseline;
5. keep the 32-channel frontend shape and 9473-scalar electronic backend unchanged.

Until that change is approved, creating the proposed TOMLs would be misleading:
they would not be executable evidence of the named perturbations.
