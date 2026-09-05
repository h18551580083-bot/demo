"""Complete fixed-front-end, trainable-electronic classifier."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn

from .frontend import FixedHEMatchedControlFrontend, FixedHEMorletFrontend, FrontendOutput
from .interaction import HEInteractionBlock, InteractionOutput
from .pooling import PoolOutput, SupportAlignedPool, flatten_pooled


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    classifier_input: torch.Tensor
    frontend: FrontendOutput
    interaction: InteractionOutput
    pooling: PoolOutput


class FixedHEClassifier(nn.Module):
    """The exact 9473-scalar electronic backend after an immutable frontend."""

    def __init__(
        self,
        *,
        frontend_backend: str,
        frontend_variant: str = "morlet",
        sigma0: str = "0.8",
        xi0: str = "3*pi/4",
        gamma: str = "0.5",
    ) -> None:
        super().__init__()
        if frontend_variant == "morlet":
            self.frontend = FixedHEMorletFrontend(
                backend=frontend_backend, sigma0=sigma0, xi0=xi0, gamma=gamma
            )
        elif frontend_variant == "matched_control":
            if (sigma0, xi0, gamma) != ("0.8", "3*pi/4", "0.5"):
                raise ValueError("matched_control requires the Phase1 Morlet envelope")
            self.frontend = FixedHEMatchedControlFrontend(backend=frontend_backend)
        else:
            raise ValueError("frontend_variant must be morlet or matched_control")
        self.interaction = HEInteractionBlock()
        self.pooling = SupportAlignedPool()
        self.classifier = nn.Linear(9408, 1, bias=True, dtype=torch.float32)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        electronic_count = sum(parameter.numel() for parameter in self.electronic_parameters())
        trainable_count = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        if electronic_count != 9473 or trainable_count != 9473:
            raise RuntimeError("trainable electronic parameter budget is not exactly 9473")

    def electronic_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.interaction.parameters()
        yield from self.classifier.parameters()

    def forward(self, rgb: torch.Tensor) -> ModelOutput:
        frontend = self.frontend(rgb)
        interaction = self.interaction(
            frontend.feature_h,
            frontend.feature_e,
            frontend.valid_support_mask,
        )
        pooling = self.pooling(
            interaction.features,
            interaction.valid_support_mask,
            interaction.neighborhood_valid_support_mask,
        )
        classifier_input = flatten_pooled(pooling.pool_float32)
        logits = self.classifier(classifier_input).squeeze(-1)
        if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
            raise ValueError("classifier must emit finite float32 raw logits")
        return ModelOutput(
            logits=logits,
            classifier_input=classifier_input,
            frontend=frontend,
            interaction=interaction,
            pooling=pooling,
        )
