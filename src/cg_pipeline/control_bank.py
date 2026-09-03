"""Frozen envelope-matched random-phase control for frontend comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .identity import domain_hash
from .morlet import _tensor_payload

CONTROL_GENERATOR_VERSION = "frozen-envelope-matched-random-phase-v1"
CONTROL_RNG = "PCG64DXSM"
CONTROL_SEED = 20260901
CONTROL_CONTRACT_ID = "fixed-he-matched-control-linear-v1"


@dataclass(frozen=True)
class MatchedControlBundle:
    kernels128: np.ndarray
    kernels64: np.ndarray
    channel_metadata: tuple[tuple[int, int, int, str], ...]
    specification_hash: str
    canonical_kernel_hash: str
    spatial_execution_hash: str
    validation: dict[str, tuple[float, ...]]


def _theta_name(index: int) -> str:
    if index == 0:
        return "0"
    if index == 1:
        return "pi/8"
    return f"{index}*pi/8"


def generate_matched_control_bundle() -> MatchedControlBundle:
    """Build and verify the one canonical 32-channel matched-control bank."""

    axis = np.arange(-52, 53, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    raw = np.random.PCG64DXSM(CONTROL_SEED).random_raw(32 * 105 * 105)
    phase_codes = np.asarray(raw & np.uint64(3), dtype=np.uint8).reshape(32, 105, 105, order="C")
    phase_lut = np.asarray((1.0, -1.0, 1.0j, -1.0j), dtype=np.complex128)
    phases = phase_lut[phase_codes]
    kernels: list[np.ndarray] = []
    metadata: list[tuple[int, int, int, str]] = []
    for scale in range(4):
        sigma = np.float64(0.8 * (2**scale))
        for orientation in range(8):
            theta = np.float64(orientation * np.pi / 8.0)
            parallel = xx * np.cos(theta) + yy * np.sin(theta)
            perpendicular = -xx * np.sin(theta) + yy * np.cos(theta)
            envelope = np.exp(
                -(parallel * parallel + np.float64(0.25) * perpendicular * perpendicular)
                / (2.0 * sigma * sigma)
            )
            channel = 8 * scale + orientation
            phase = phases[channel]
            beta = np.sum(envelope * phase, dtype=np.complex128) / np.sum(
                envelope, dtype=np.float64
            )
            kernel = np.asarray(envelope * (phase - beta), dtype=np.complex128)
            energy = np.sqrt(np.sum(np.abs(kernel) ** 2, dtype=np.float64))
            if not np.isfinite(energy) or energy <= 0.0:
                raise ValueError("matched-control generation produced invalid energy")
            kernels.append(np.asarray(kernel / energy, dtype=np.complex128))
            metadata.append((channel, scale, orientation, _theta_name(orientation)))

    kernels128 = np.ascontiguousarray(np.stack(kernels), dtype=np.complex128)
    kernels64 = np.ascontiguousarray(kernels128.astype(np.complex64))
    kernels64.real[kernels64.real == 0.0] = 0.0
    kernels64.imag[kernels64.imag == 0.0] = 0.0
    zero128 = tuple(float(abs(np.sum(kernel, dtype=np.complex128))) for kernel in kernels128)
    energy128 = tuple(
        float(abs(np.sum(np.abs(kernel) ** 2, dtype=np.float64) - 1.0)) for kernel in kernels128
    )
    zero64 = tuple(
        float(abs(np.sum(kernel.astype(np.complex128), dtype=np.complex128)))
        for kernel in kernels64
    )
    energy64 = tuple(
        float(abs(np.sum(np.abs(kernel.astype(np.complex128)) ** 2, dtype=np.float64) - 1.0))
        for kernel in kernels64
    )
    if max(zero128) > 1e-12 or max(energy128) > 1e-12:
        raise ValueError("complex128 matched-control kernel validation failed")
    if max(zero64) > 1e-6 or max(energy64) > 1e-6:
        raise ValueError("complex64 matched-control kernel validation failed")

    specification_header = {
        "boundary_policy": "reflect-no-edge-duplication-radius-52",
        "channel_order": "scale-major-orientation-major-c=8*j+ell",
        "control_seed": CONTROL_SEED,
        "dc_projection": "beta=sum(g*z)/sum(g)-complex128",
        "envelope": "exp(-(parallel^2+0.25*perpendicular^2)/(2*sigma^2))",
        "generation_dtype": "complex128-from-float64",
        "generator_version": CONTROL_GENERATOR_VERSION,
        "modulus": "stable-epsilon-free-v1",
        "normalization": "complex-unit-l2-in-complex128",
        "phase_mapping": ["0:1", "1:-1", "2:i", "3:-i"],
        "phase_source": "random_raw-uint64-low-two-bits",
        "payload_length": 0,
        "rng": CONTROL_RNG,
        "rng_traversal": "single-stream-channel-y-x-c-order",
        "sigma": "0.8*2^j",
        "support": [105, 105],
    }
    specification_hash = domain_hash("cg/matched-control-filter-bank-spec/v1", specification_header)
    canonical_header, canonical_payload = _tensor_payload(kernels64)
    canonical_header["filter_bank_specification_hash"] = specification_hash
    canonical_hash = domain_hash(
        "cg/matched-control-kernel-canonical/v1", canonical_header, canonical_payload
    )
    spatial = np.ascontiguousarray(kernels64[:, ::-1, ::-1])
    spatial_header, spatial_payload = _tensor_payload(spatial)
    spatial_header["canonical_kernel_hash"] = canonical_hash
    spatial_header["spatial_transform"] = "flip-y-and-x-for-cross-correlation"
    spatial_hash = domain_hash(
        "cg/matched-control-kernel-spatial-exec/v1", spatial_header, spatial_payload
    )
    return MatchedControlBundle(
        kernels128=kernels128,
        kernels64=kernels64,
        channel_metadata=tuple(metadata),
        specification_hash=specification_hash,
        canonical_kernel_hash=canonical_hash,
        spatial_execution_hash=spatial_hash,
        validation={
            "complex128_zero_dc_error": zero128,
            "complex128_unit_energy_error": energy128,
            "complex64_zero_dc_error": zero64,
            "complex64_unit_energy_error": energy64,
        },
    )
