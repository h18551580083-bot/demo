from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from cam16_wavelet.contracts import FrontendConfig
from cam16_wavelet.optics.detector import SquareLawDetector
from cam16_wavelet.optics.wavelet_bank import KernelBundle, WaveletBank


class OpticalBackbone(nn.Module):
    """Frozen fixed-kernel front end with interchangeable numerical implementations."""

    def __init__(self, config: FrontendConfig) -> None:
        super().__init__()
        self.config = config
        bundle: KernelBundle = WaveletBank.build(config.kernel)
        self.register_buffer("kernels", bundle.kernels[:, None])
        self.channel_names = bundle.channel_names
        self.kernel_bank_hash = bundle.kernel_hash
        self.detector = SquareLawDetector(config.detector)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("image must have shape BxCxHxW")
        gray = image.mean(dim=1, keepdim=True)
        if self.config.mode == "digital_ideal":
            field = F.conv2d(gray, self.kernels, padding="same")
        elif self.config.mode == "fourier_4f":
            field = self._fft_convolve_same(gray, self.kernels)
        else:
            raise ValueError(f"unsupported mode: {self.config.mode}")
        return self.detector(field)

    @staticmethod
    def _fft_convolve_same(image: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        kh, kw = kernels.shape[-2:]
        full_shape = (height + kh - 1, width + kw - 1)
        image_fft = torch.fft.rfft2(image[:, 0], s=full_shape)
        # conv2d implements cross-correlation; flip the kernel so FFT convolution
        # has exactly the same contract for future non-centrosymmetric kernels.
        kernel_fft = torch.fft.rfft2(kernels[:, 0].flip((-2, -1)), s=full_shape)
        full = torch.fft.irfft2(image_fft[:, None] * kernel_fft[None], s=full_shape)
        top, left = kh // 2, kw // 2
        return full[..., top : top + height, left : left + width]
