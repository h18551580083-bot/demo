import torch

from cam16_wavelet.contracts import DetectorConfig, FrontendConfig, KernelSpec
from cam16_wavelet.optics import OpticalBackbone


def _config(mode: str) -> FrontendConfig:
    return FrontendConfig(
        mode=mode,
        kernel=KernelSpec(size=9, scales=(1.5,), orientations=2),
        detector=DetectorConfig(sqrt_after_detection=False, pool_size=1),
    )


def test_spatial_and_fourier_implementations_are_equivalent():
    torch.manual_seed(7)
    image = torch.rand(2, 3, 32, 32)
    spatial = OpticalBackbone(_config("digital_ideal"))
    fourier = OpticalBackbone(_config("fourier_4f"))
    actual = fourier(image)
    expected = spatial(image)
    assert actual.shape == expected.shape
    assert fourier.channel_names == spatial.channel_names
    assert fourier.kernel_bank_hash == spatial.kernel_bank_hash
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-4)


def test_frontend_has_no_trainable_parameters():
    model = OpticalBackbone(_config("digital_ideal"))
    assert sum(parameter.numel() for parameter in model.parameters()) == 0

