"""Phase2 validity against the configured, zero-DC, sampled Gaussian spectrum.

Phase1 coverage is deliberately not changed here. Frequency errors are measured
in FFT cells, not against a fitted baseline tolerance. Alias terms are the
Poisson-summation replicas of the continuous Gaussian transform.
"""

from typing import Any

import numpy as np

from .morlet import MorletBundle, validate_morlet_parameters


def _theory(
    k: np.ndarray, sigma: float, xi: float, gamma: float, theta: float, replicas: int
) -> tuple[np.ndarray, np.ndarray]:
    direction = np.array([np.cos(theta), np.sin(theta)])
    transverse = np.array([-np.sin(theta), np.cos(theta)])
    offsets = np.arange(-replicas, replicas + 1) * (2 * np.pi)
    shifts = np.stack(np.meshgrid(offsets, offsets), axis=-1).reshape(-1, 2)

    def gaussian(q):
        return np.exp(-0.5 * sigma**2 * ((q @ direction) ** 2 + (q @ transverse) ** 2 / gamma**2))

    # Sampled zero-DC correction, including aliases, rather than beta_inf.
    beta = gaussian(shifts - xi * direction).sum() / gaussian(shifts).sum()
    main = gaussian(k - xi * direction) - beta * gaussian(k)
    terms = gaussian(k[..., None, :] + shifts - xi * direction) - beta * gaussian(
        k[..., None, :] + shifts
    )
    spectrum = terms.sum(axis=-1)
    replicas_only = np.any(shifts != 0, axis=1)
    alias_amplitude = np.abs(terms[..., replicas_only]).sum(axis=-1)
    return spectrum, alias_amplitude / np.maximum(np.abs(main), np.finfo(float).tiny)


def validate_phase2_morlet(
    bundle: MorletBundle, *, sigma0: str, xi0: str, gamma: str
) -> dict[str, Any]:
    """Recompute hard validity from tensors, never trust cached validation fields."""
    validate_morlet_parameters(sigma0, xi0, gamma)
    checks = {
        "kernel_shape": bundle.kernels128.shape == bundle.kernels64.shape == (32, 105, 105),
        "channel_count": len(bundle.channel_metadata) == 32,
        "dtype": bundle.kernels128.dtype == np.complex128
        and bundle.kernels64.dtype == np.complex64,
        "finite": bool(
            np.isfinite(bundle.kernels128).all() and np.isfinite(bundle.kernels64).all()
        ),
    }
    report: dict[str, Any] = {"schema": "phase2a-morlet-validity-v1", "checks": checks}
    if not all(checks.values()):
        return {**report, "status": "FAIL"}
    residuals = {}
    for name, kernels, tolerance in (
        ("complex128", bundle.kernels128, 1e-12),
        ("complex64", bundle.kernels64.astype(np.complex128), 1e-6),
    ):
        dc = np.abs(kernels.sum(axis=(-2, -1)))
        energy = np.abs((np.abs(kernels) ** 2).sum(axis=(-2, -1)) - 1)
        checks[name + "_dc"] = bool((dc <= tolerance).all())
        checks[name + "_energy"] = bool((energy <= tolerance).all())
        residuals[name] = {
            "dc": dc.tolist(),
            "energy_error": energy.tolist(),
            "threshold": tolerance,
        }
    report["residuals"] = residuals
    if not all(checks.values()):
        return {**report, "status": "FAIL"}

    n = 464
    step = 2 * np.pi / n
    frequencies = 2 * np.pi * np.fft.fftfreq(n)
    power = np.abs(np.fft.fft2(bundle.kernels128, s=(n, n))) ** 2
    power64 = np.abs(np.fft.fft2(bundle.kernels64, s=(n, n))) ** 2
    xi_base = {"3*pi/4": 3 * np.pi / 4, "2*pi/3": 2 * np.pi / 3}[xi0]
    peaks = []
    for channel in range(32):
        sigma = float(sigma0) * 2 ** (channel // 8)
        xi = xi_base * 2 ** (-(channel // 8))
        theta = (channel % 8) * np.pi / 8
        direction = np.array([np.cos(theta), np.sin(theta)])
        # Continuous zero-DC spectrum's positive lobe gives the initial reference.
        beta = np.exp(-(sigma**2) * xi**2 / 2)
        lo, hi = xi, xi + 3 / sigma
        for _ in range(64):
            radius = (lo + hi) / 2
            derivative = -(radius - xi) * np.exp(
                -(sigma**2) * (radius - xi) ** 2 / 2
            ) + beta * radius * np.exp(-(sigma**2) * radius**2 / 2)
            if derivative > 0:
                lo = radius
            else:
                hi = radius
        center = (lo + hi) / 2 * direction
        # A bounded local reference search; hitting its edge fails closed.
        offsets = np.arange(-4, 5)
        x, y = np.meshgrid(np.rint(center[0] / step) + offsets, np.rint(center[1] / step) + offsets)
        grid = np.stack([x, y], axis=-1) * step
        theoretical, _ = _theory(grid, sigma, xi, float(gamma), theta, 2)
        iy, ix = np.unravel_index(np.argmax(theoretical**2), theoretical.shape)
        expected = grid[iy, ix]
        reference, alias_ratio = _theory(expected, sigma, xi, float(gamma), theta, 2)
        converged, _ = _theory(expected, sigma, xi, float(gamma), theta, 3)
        py, px = np.unravel_index(np.argmax(power[channel]), power[channel].shape)
        observed = np.array([frequencies[px], frequencies[py]])
        error_cells = float(np.max(np.abs(observed - expected)) / step)
        py64, px64 = np.unravel_index(np.argmax(power64[channel]), power64[channel].shape)
        observed64 = np.array([frequencies[px64], frequencies[py64]])
        error64 = float(np.max(np.abs(observed64 - expected)) / step)
        conditions = {
            "reference_search_interior": bool(0 < iy < 8 and 0 < ix < 8),
            "reference_converged": bool(
                abs(reference - converged)
                <= 64 * np.finfo(float).eps * max(abs(converged), np.finfo(float).tiny)
            ),
            "peak_matches_theory": error_cells <= 1 + 64 * np.finfo(float).eps,
            "runtime_peak_matches_theory": error64 <= 1 + 64 * np.finfo(float).eps,
            "nyquist_margin": bool(
                (np.abs(expected) < np.pi - step).all()
                and (np.abs(observed) < np.pi - step).all()
                and (np.abs(observed64) < np.pi - step).all()
            ),
            "nonzero_positive_lobe": bool(observed @ direction > step and reference > 0),
            "alias_not_dominant": bool(alias_ratio < 1),
        }
        peaks.append(
            {
                "channel": channel,
                "expected_xy": expected.tolist(),
                "observed_xy": observed.tolist(),
                "error_cells": error_cells,
                "runtime_observed_xy": observed64.tolist(),
                "runtime_error_cells": error64,
                "alias_amplitude_ratio": float(alias_ratio),
                "checks": conditions,
            }
        )
    checks["configured_spectral_peaks"] = all(all(p["checks"].values()) for p in peaks)
    report.update(
        peaks=peaks,
        fft_grid=[n, n],
        peak_tolerance_cells=1,
        angular_cell_radians=step,
        alias_amplitude_ratio_upper_exclusive=1,
        nyquist_margin_cells=1,
        reference="sampled-zero-dc-gaussian-poisson-sum-v1",
        status="PASS" if all(checks.values()) else "FAIL",
    )
    return report
