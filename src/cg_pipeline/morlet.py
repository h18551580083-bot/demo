"""Deterministic first-order complex Morlet kernel generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .identity import domain_hash


@dataclass(frozen=True)
class MorletBundle:
    kernels128: np.ndarray
    kernels64: np.ndarray
    channel_metadata: tuple[tuple[int, int, int, str], ...]
    parameter_hash: str
    canonical_kernel_hash: str
    spatial_execution_hash: str
    validation: dict[str, tuple[float, ...]]


def _theta_name(index: int) -> str:
    if index == 0:
        return "0"
    if index == 1:
        return "pi/8"
    return f"{index}*pi/8"


def _tensor_payload(kernels: np.ndarray) -> tuple[dict[str, Any], bytes]:
    real_imag = np.empty((*kernels.shape, 2), dtype="<f4")
    real_imag[..., 0] = kernels.real.astype("<f4", copy=False)
    real_imag[..., 1] = kernels.imag.astype("<f4", copy=False)
    real_imag[real_imag == 0.0] = 0.0
    real_imag = np.ascontiguousarray(real_imag)
    payload = real_imag.tobytes(order="C")
    header = {
        "axis_semantics": ["channel", "y", "x", "component"],
        "component_order": ["real", "imaginary"],
        "dtype": "float32",
        "endianness": "little",
        "layout": "C-contiguous",
        "payload_length": len(payload),
        "shape": [32, 105, 105, 2],
    }
    return header, payload


def generate_morlet_bundle() -> MorletBundle:
    """Generate the one canonical 32-channel kernel bank from locked constants."""

    axis = np.arange(-52, 53, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    kernels: list[np.ndarray] = []
    metadata: list[tuple[int, int, int, str]] = []
    beta_errors: list[float] = []
    for scale in range(4):
        sigma = np.float64(0.8 * (2**scale))
        xi = np.float64((3.0 * np.pi / 4.0) * (2.0 ** (-scale)))
        beta_inf = np.exp(-(sigma * sigma * xi * xi) / 2.0)
        for orientation in range(8):
            theta = np.float64(orientation * np.pi / 8.0)
            parallel = xx * np.cos(theta) + yy * np.sin(theta)
            perpendicular = -xx * np.sin(theta) + yy * np.cos(theta)
            envelope = np.exp(
                -(parallel * parallel + np.float64(0.25) * perpendicular * perpendicular)
                / (2.0 * sigma * sigma)
            )
            carrier = np.exp(1j * xi * parallel)
            beta_disc = np.sum(envelope * carrier, dtype=np.complex128) / np.sum(
                envelope, dtype=np.float64
            )
            kernel = envelope * (carrier - beta_disc)
            energy = np.sqrt(np.sum(np.abs(kernel) ** 2, dtype=np.float64))
            if not np.isfinite(energy) or energy <= 0.0:
                raise ValueError("Morlet generation produced invalid pre-normalization energy")
            normalized = np.asarray(kernel / energy, dtype=np.complex128)
            kernels.append(normalized)
            channel = 8 * scale + orientation
            metadata.append((channel, scale, orientation, _theta_name(orientation)))
            beta_errors.append(float(abs(beta_disc - beta_inf)))
    kernels128 = np.ascontiguousarray(np.stack(kernels), dtype=np.complex128)
    kernels64 = np.ascontiguousarray(kernels128.astype(np.complex64))
    kernels64.real[kernels64.real == 0.0] = 0.0
    kernels64.imag[kernels64.imag == 0.0] = 0.0
    zero128 = tuple(float(abs(np.sum(kernel, dtype=np.complex128))) for kernel in kernels128)
    energy128 = tuple(
        float(abs(np.sum(np.abs(kernel) ** 2, dtype=np.float64) - 1.0))
        for kernel in kernels128
    )
    zero64 = tuple(
        float(abs(np.sum(kernel.astype(np.complex128), dtype=np.complex128)))
        for kernel in kernels64
    )
    energy64 = tuple(
        float(abs(np.sum(np.abs(kernel.astype(np.complex128)) ** 2, dtype=np.float64) - 1.0))
        for kernel in kernels64
    )
    validation = {
        "complex128_zero_dc_error": zero128,
        "complex128_unit_energy_error": energy128,
        "complex64_zero_dc_error": zero64,
        "complex64_unit_energy_error": energy64,
        "beta_reference_error": tuple(beta_errors),
    }
    if max(zero128) > 1e-12 or max(energy128) > 1e-12:
        raise ValueError("complex128 Morlet kernel validation failed")
    if max(zero64) > 1e-6 or max(energy64) > 1e-6:
        raise ValueError("complex64 Morlet kernel validation failed")
    if max(beta_errors) > 1e-2:
        raise ValueError("discrete/continuous Morlet correction validation failed")
    parameter_header = {
        "angle_convention": "theta_ell=ell*pi/8-clockwise-image-coordinates",
        "boundary_policy": "reflect-no-edge-duplication-radius-52",
        "channel_order": "scale-major-orientation-major-c=8*j+ell",
        "convolution": "true-convolution",
        "coordinate_convention": "ux-right-uy-down-centered",
        "formula_version": "explicit-discrete-morlet-v1",
        "gamma": "0.5",
        "generation_dtype": "complex128-from-float64",
        "modulus": "stable-epsilon-free-v1",
        "morlet_orientations": 8,
        "morlet_scales": 4,
        "normalization": "complex-unit-l2",
        "payload_length": 0,
        "sigma": "0.8*2^j",
        "support": [105, 105],
        "xi": "(3*pi/4)*2^(-j)",
        "zero_dc": "finite-discrete-projection",
    }
    parameter_hash = domain_hash("cg/morlet-param-spec/v1", parameter_header)
    canonical_header, canonical_payload = _tensor_payload(kernels64)
    canonical_header["parameter_hash"] = parameter_hash
    canonical_hash = domain_hash(
        "cg/morlet-kernel-canonical/v1", canonical_header, canonical_payload
    )
    spatial = np.ascontiguousarray(kernels64[:, ::-1, ::-1])
    spatial_header, spatial_payload = _tensor_payload(spatial)
    spatial_header["canonical_kernel_hash"] = canonical_hash
    spatial_header["spatial_transform"] = "flip-y-and-x-for-cross-correlation"
    spatial_hash = domain_hash(
        "cg/morlet-kernel-spatial-exec/v1", spatial_header, spatial_payload
    )
    return MorletBundle(
        kernels128=kernels128,
        kernels64=kernels64,
        channel_metadata=tuple(metadata),
        parameter_hash=parameter_hash,
        canonical_kernel_hash=canonical_hash,
        spatial_execution_hash=spatial_hash,
        validation=validation,
    )


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    numerator = np.sum(first * second, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(first * first, dtype=np.float64) * np.sum(second * second, dtype=np.float64)
    )
    return float(numerator / denominator)


def _periodic_bilinear(array: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    size_y, size_x = array.shape
    y0 = np.floor(y).astype(np.int64) % size_y
    x0 = np.floor(x).astype(np.int64) % size_x
    y1 = (y0 + 1) % size_y
    x1 = (x0 + 1) % size_x
    fy = y - np.floor(y)
    fx = x - np.floor(x)
    return (
        array[y0, x0] * (1.0 - fy) * (1.0 - fx)
        + array[y1, x0] * fy * (1.0 - fx)
        + array[y0, x1] * (1.0 - fy) * fx
        + array[y1, x1] * fy * fx
    )


def validate_spectral_coverage(bundle: MorletBundle) -> dict[str, Any]:
    """Run the locked 464-grid carrier, overlap, and ring-uniformity gates."""

    fft_size = 464
    spectra = np.fft.fft2(bundle.kernels128, s=(fft_size, fft_size), axes=(-2, -1))
    power = np.abs(spectra) ** 2
    negative_indices = (-np.arange(fft_size, dtype=np.int64)) % fft_size
    negative_power = power[:, negative_indices[:, None], negative_indices[None, :]]
    symmetrized = np.sqrt(power + negative_power)
    angular_step = 2.0 * np.pi / fft_size
    frequency = 2.0 * np.pi * np.fft.fftfreq(fft_size)
    direction_errors: list[float] = []
    radial_errors: list[float] = []
    radial_pass: list[bool] = []
    fit_pass: list[bool] = []
    design_rows = []
    for dy in (-1.0, 0.0, 1.0):
        for dx in (-1.0, 0.0, 1.0):
            design_rows.append((dx * dx, dy * dy, dx * dy, dx, dy, 1.0))
    design = np.asarray(design_rows, dtype=np.float64)
    for channel in range(32):
        scale = channel // 8
        orientation = channel % 8
        expected_theta = orientation * np.pi / 8.0
        expected_radius = (3.0 * np.pi / 4.0) * (2.0 ** (-scale))
        carrier_projection = (
            frequency[:, None] * np.sin(expected_theta)
            + frequency[None, :] * np.cos(expected_theta)
        )
        candidate_power = np.where(carrier_projection > 0.0, power[channel], -np.inf)
        peak_linear = int(np.argmax(candidate_power))
        peak_y, peak_x = np.unravel_index(peak_linear, (fft_size, fft_size))
        neighborhood = np.empty((3, 3), dtype=np.float64)
        for row, dy in enumerate((-1, 0, 1)):
            for column, dx in enumerate((-1, 0, 1)):
                neighborhood[row, column] = power[
                    channel, (peak_y + dy) % fft_size, (peak_x + dx) % fft_size
                ]
        logged = np.log(np.maximum(neighborhood.reshape(-1), np.finfo(np.float64).tiny))
        coefficients, _, _, _ = np.linalg.lstsq(design, logged, rcond=None)
        a, b, c, d, e, _ = coefficients
        hessian = np.array(((2.0 * a, c), (c, 2.0 * b)), dtype=np.float64)
        negative_definite = bool(np.all(np.linalg.eigvalsh(hessian) < 0.0))
        try:
            offset_x, offset_y = -np.linalg.solve(hessian, np.array((d, e), dtype=np.float64))
        except np.linalg.LinAlgError:
            offset_x = offset_y = np.nan
        valid_offset = bool(
            negative_definite
            and np.isfinite(offset_x)
            and np.isfinite(offset_y)
            and abs(offset_x) <= 1.0
            and abs(offset_y) <= 1.0
        )
        fit_pass.append(valid_offset)
        if not valid_offset:
            direction_errors.append(float("inf"))
            radial_errors.append(float("inf"))
            radial_pass.append(False)
            continue
        kx = frequency[peak_x] + float(offset_x) * angular_step
        ky = frequency[peak_y] + float(offset_y) * angular_step
        observed_theta = np.arctan2(ky, kx)
        angle_delta = np.arctan2(
            np.sin(observed_theta - expected_theta), np.cos(observed_theta - expected_theta)
        )
        direction_errors.append(float(abs(angle_delta) * 180.0 / np.pi))
        radial_error = float(abs(np.hypot(kx, ky) - expected_radius))
        radial_errors.append(radial_error)
        radial_pass.append(radial_error <= min(0.075, 0.10 * expected_radius))
    orientation_overlaps: list[float] = []
    for scale in range(4):
        for orientation in range(8):
            first = 8 * scale + orientation
            second = 8 * scale + ((orientation + 1) % 8)
            orientation_overlaps.append(_cosine_similarity(symmetrized[first], symmetrized[second]))
    scale_overlaps: list[float] = []
    for scale in range(3):
        for orientation in range(8):
            scale_overlaps.append(
                _cosine_similarity(
                    symmetrized[8 * scale + orientation],
                    symmetrized[8 * (scale + 1) + orientation],
                )
            )
    total_power = np.sum(symmetrized * symmetrized, axis=0, dtype=np.float64)
    radii = [(3.0 * np.pi / 4.0) * (2.0 ** (-scale)) for scale in range(4)]
    radii.extend(np.sqrt(radii[index] * radii[index + 1]) for index in range(3))
    angles = np.arange(1440, dtype=np.float64) * (2.0 * np.pi / 1440.0)
    ring_reports: list[dict[str, float]] = []
    for radius in radii:
        kx = radius * np.cos(angles)
        ky = radius * np.sin(angles)
        x_index = (kx / (2.0 * np.pi) * fft_size) % fft_size
        y_index = (ky / (2.0 * np.pi) * fft_size) % fft_size
        samples = _periodic_bilinear(total_power, y_index, x_index)
        minimum = float(np.min(samples))
        maximum = float(np.max(samples))
        median = float(np.median(samples))
        mean = float(np.mean(samples, dtype=np.float64))
        standard_deviation = float(np.std(samples, dtype=np.float64))
        ring_reports.append(
            {
                "radius": float(radius),
                "min_over_median": minimum / median,
                "min_over_max": minimum / maximum,
                "coefficient_of_variation": standard_deviation / mean,
            }
        )
    passed = (
        all(fit_pass)
        and max(direction_errors) <= 1.0
        and all(radial_pass)
        and all(0.50 <= value <= 0.70 for value in orientation_overlaps)
        and all(0.45 <= value <= 0.60 for value in scale_overlaps)
        and all(item["min_over_median"] >= 0.85 for item in ring_reports)
        and all(item["min_over_max"] >= 0.75 for item in ring_reports)
        and all(item["coefficient_of_variation"] <= 0.10 for item in ring_reports)
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "fft_grid": [fft_size, fft_size],
        "quadratic_fit_pass": fit_pass,
        "carrier_direction_error_degrees": direction_errors,
        "radial_error": radial_errors,
        "radial_pass": radial_pass,
        "adjacent_orientation_overlap": orientation_overlaps,
        "adjacent_scale_overlap": scale_overlaps,
        "ring_uniformity": ring_reports,
    }
