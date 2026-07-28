import torch

from cam16_wavelet.contracts import KernelSpec
from cam16_wavelet.optics import WaveletBank


def test_wavelet_bank_is_deterministic_and_normalized():
    spec = KernelSpec(scales=(1.5,), orientations=4)
    first = WaveletBank.build(spec)
    second = WaveletBank.build(spec)
    assert first.kernel_hash == second.kernel_hash
    assert first.channel_names == second.channel_names
    assert torch.equal(first.kernels, second.kernels)
    norms = first.kernels.square().sum(dim=(-2, -1)).sqrt()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_kernel_spec_rejects_even_size():
    try:
        WaveletBank.build(KernelSpec(size=14))
    except ValueError as error:
        assert "odd" in str(error)
    else:
        raise AssertionError("expected invalid kernel size to fail")

