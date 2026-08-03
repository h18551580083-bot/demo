"""Frozen seven-feature electronic H/E interaction contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class InteractionContractError(ValueError):
    """H/E interaction input or output violated the locked contract."""


FEATURE_NAMES = (
    "H_gated",
    "E_gated",
    "HE_product",
    "H_x_E_ring",
    "E_x_H_ring",
    "H_excess",
    "E_excess",
)


@dataclass(frozen=True)
class InteractionOutput:
    features: torch.Tensor
    valid_support_mask: torch.Tensor
    neighborhood_valid_support_mask: torch.Tensor
    feature_names: tuple[str, ...] = FEATURE_NAMES


def _ring_mean(values: torch.Tensor) -> torch.Tensor:
    batch, scales, orientations, height, width = values.shape
    flattened = values.reshape(batch * scales * orientations, 1, height, width)
    kernel = torch.ones((1, 1, 3, 3), dtype=values.dtype, device=values.device)
    kernel[..., 1, 1] = 0.0
    result = F.conv2d(flattened, kernel, padding=1) / 8.0
    return result.reshape(batch, scales, orientations, height, width)


def _neighborhood_mask(mask: torch.Tensor) -> torch.Tensor:
    counts = F.conv2d(
        mask.to(torch.float32),
        torch.ones((1, 1, 3, 3), dtype=torch.float32, device=mask.device),
        padding=1,
    )
    return counts == 9.0


class HEInteractionBlock(nn.Module):
    """Symmetric gating plus three parameter-free interaction families."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_scale = nn.Parameter(torch.zeros((4, 8), dtype=torch.float32))
        self.gate_bias = nn.Parameter(torch.zeros((4, 8), dtype=torch.float32))

    @property
    def trainable_parameter_names(self) -> tuple[str, str]:
        return ("gate_scale", "gate_bias")

    def forward(
        self,
        feature_h: torch.Tensor,
        feature_e: torch.Tensor,
        valid_support_mask: torch.Tensor,
    ) -> InteractionOutput:
        if feature_h.shape != feature_e.shape or feature_h.ndim != 5:
            raise InteractionContractError("F_H and F_E must share shape [B,4,8,H,W]")
        if feature_h.shape[1:3] != (4, 8) or feature_h.dtype != torch.float32:
            raise InteractionContractError("F_H and F_E must be float32 [B,4,8,H,W]")
        batch, _, _, height, width = feature_h.shape
        if (
            valid_support_mask.shape != (batch, 1, height, width)
            or valid_support_mask.dtype != torch.bool
        ):
            raise InteractionContractError("valid_support_mask must be Boolean [B,1,H,W]")
        if not torch.isfinite(feature_h).all() or not torch.isfinite(feature_e).all():
            raise InteractionContractError("non-finite fixed-frontend interaction input")
        scale = self.gate_scale[None, :, :, None, None]
        bias = self.gate_bias[None, :, :, None, None]
        gate_h = 2.0 * torch.sigmoid(scale * feature_e + bias)
        gate_e = 2.0 * torch.sigmoid(scale * feature_h + bias)
        h_gated = feature_h * gate_h
        e_gated = feature_e * gate_e
        product = feature_h * feature_e
        h_e_ring = feature_h * _ring_mean(feature_e)
        e_h_ring = feature_e * _ring_mean(feature_h)
        h_excess = torch.clamp(feature_h - feature_e, min=0.0)
        e_excess = torch.clamp(feature_e - feature_h, min=0.0)
        combined = torch.stack(
            (h_gated, e_gated, product, h_e_ring, e_h_ring, h_excess, e_excess), dim=3
        )
        if combined.dtype != torch.float32 or not torch.isfinite(combined).all():
            raise InteractionContractError("interaction produced a non-finite value")
        return InteractionOutput(
            features=combined,
            valid_support_mask=valid_support_mask,
            neighborhood_valid_support_mask=_neighborhood_mask(valid_support_mask),
        )
