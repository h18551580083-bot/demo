from __future__ import annotations

import numpy as np
import pytest
import torch

from cg_pipeline.frontend import FixedHEMorletFrontend, FrontendContractError, complex_convolve
from cg_pipeline.morlet import generate_morlet_bundle


def test_morlet_generation_is_deterministic_ordered_and_within_locked_tolerances() -> None:
    first = generate_morlet_bundle()
    second = generate_morlet_bundle()

    assert first.kernels128.shape == (32, 105, 105)
    assert first.kernels128.dtype == np.complex128
    assert first.kernels64.shape == (32, 105, 105)
    assert first.kernels64.dtype == np.complex64
    assert first.kernels64.flags.c_contiguous
    assert first.parameter_hash == second.parameter_hash
    assert first.canonical_kernel_hash == second.canonical_kernel_hash
    assert first.spatial_execution_hash == second.spatial_execution_hash
    assert np.array_equal(first.kernels64, second.kernels64)
    assert first.channel_metadata[0] == (0, 0, 0, "0")
    assert first.channel_metadata[-1] == (31, 3, 7, "7*pi/8")
    assert max(first.validation["complex128_zero_dc_error"]) <= 1e-12
    assert max(first.validation["complex128_unit_energy_error"]) <= 1e-12
    assert max(first.validation["complex64_zero_dc_error"]) <= 1e-6
    assert max(first.validation["complex64_unit_energy_error"]) <= 1e-6
    assert max(first.validation["beta_reference_error"]) <= 1e-2


def test_frontend_separates_white_to_zero_and_has_no_trainable_parameter() -> None:
    frontend = FixedHEMorletFrontend(backend="fft")
    white = torch.full((1, 3, 110, 110), 255, dtype=torch.uint8)

    concentrations = frontend.separate_stains(white)
    output = frontend(white)

    assert concentrations.shape == (1, 2, 110, 110)
    assert concentrations.dtype == torch.float32
    assert torch.equal(concentrations, torch.zeros_like(concentrations))
    assert output.feature_h.shape == (1, 4, 8, 110, 110)
    assert output.feature_e.shape == (1, 4, 8, 110, 110)
    assert output.feature_h.dtype == torch.float32
    assert torch.equal(output.feature_h, torch.zeros_like(output.feature_h))
    assert torch.equal(output.feature_e, torch.zeros_like(output.feature_e))
    assert output.valid_support_mask.shape == (1, 1, 110, 110)
    assert int(output.valid_support_mask.sum()) == 36
    assert list(frontend.parameters()) == []
    assert frontend.shared_kernel_reference_count == 1


def test_frontend_rejects_invalid_input_and_active_autocast() -> None:
    frontend = FixedHEMorletFrontend(backend="fft")

    with pytest.raises(FrontendContractError, match="uint8"):
        frontend(torch.zeros((1, 3, 110, 110), dtype=torch.float32))
    with pytest.raises(FrontendContractError, match="channel"):
        frontend(torch.zeros((1, 1, 110, 110), dtype=torch.uint8))
    with pytest.raises(FrontendContractError, match="larger than 52"):
        frontend(torch.zeros((1, 3, 52, 110), dtype=torch.uint8))
    with (
        torch.autocast(device_type="cpu", dtype=torch.bfloat16),
        pytest.raises(FrontendContractError, match="autocast"),
    ):
        frontend(torch.zeros((1, 3, 110, 110), dtype=torch.uint8))


def test_spatial_and_fft_true_convolution_agree_for_one_complex_kernel() -> None:
    bundle = generate_morlet_bundle()
    image = torch.zeros((1, 1, 53, 53), dtype=torch.float32)
    image[0, 0, 26, 26] = 1.0
    kernel = torch.from_numpy(bundle.kernels64[:1].copy())

    spatial = complex_convolve(image, kernel, backend="spatial")
    fft = complex_convolve(image, kernel, backend="fft")

    assert spatial.shape == fft.shape == (1, 1, 53, 53)
    assert torch.allclose(fft.real, spatial.real, atol=2e-5, rtol=2e-4)
    assert torch.allclose(fft.imag, spatial.imag, atol=2e-5, rtol=2e-4)


def test_frontend_buffer_bytes_and_identities_are_stable() -> None:
    frontend = FixedHEMorletFrontend(backend="fft")
    before = frontend.fixed_state_identity()
    _ = frontend(torch.full((1, 3, 110, 110), 255, dtype=torch.uint8))
    after = frontend.fixed_state_identity()

    assert before == after
    assert set(before) == {
        "stain_spec_hash",
        "morlet_parameter_hash",
        "canonical_kernel_hash",
        "spatial_execution_hash",
        "fixed_state_sha256",
    }
