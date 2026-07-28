from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from cam16_wavelet.contracts import DetectorConfig


class SquareLawDetector(nn.Module):
    def __init__(self, config: DetectorConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        intensity = field.abs().square() if field.is_complex() else field.square()
        if self.config.noise_std:
            intensity = intensity + torch.randn_like(intensity) * self.config.noise_std
        intensity = intensity.clamp_min(0)
        if self.config.saturation is not None:
            intensity = intensity.clamp_max(self.config.saturation)
        if self.config.quantization_bits is not None:
            maximum = self.config.saturation or float(intensity.detach().amax().clamp_min(1e-12))
            levels = 2**self.config.quantization_bits - 1
            intensity = torch.round(intensity / maximum * levels) / levels * maximum
        if self.config.sqrt_after_detection:
            intensity = torch.sqrt(intensity + self.config.epsilon)
        if self.config.pool_size > 1:
            intensity = F.avg_pool2d(intensity, self.config.pool_size)
        return intensity
