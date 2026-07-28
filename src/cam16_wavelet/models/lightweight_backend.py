from __future__ import annotations

import torch
from torch import nn


class LightweightBackend(nn.Module):
    def __init__(self, channels: int, classes: int = 2, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

