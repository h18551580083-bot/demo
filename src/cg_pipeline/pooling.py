"""Support-aligned pyramid pooling with the frozen balanced reduction tree."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class PoolingContractError(ValueError):
    """Mask, geometry, statistic, or handoff contract failed."""


@dataclass(frozen=True)
class Region:
    index: int
    level: int
    row: int
    column: int
    y_start: int
    y_end: int
    x_start: int
    x_end: int


@dataclass(frozen=True)
class PoolOutput:
    statistics_float64: torch.Tensor
    pool_float32: torch.Tensor
    valid_count: torch.Tensor
    geometry: tuple[Region, ...]
    pooling_support_mask: torch.Tensor


def support_aligned_regions(height: int, width: int) -> tuple[Region, ...]:
    if height < 110 or width < 110:
        raise PoolingContractError("spatial pyramid requires height and width at least 110")
    support_height = height - 106
    support_width = width - 106
    output: list[Region] = []
    offsets = {1: 0, 2: 1, 4: 5}
    for level in (1, 2, 4):
        for row in range(level):
            for column in range(level):
                output.append(
                    Region(
                        index=offsets[level] + row * level + column,
                        level=level,
                        row=row,
                        column=column,
                        y_start=53 + row * support_height // level,
                        y_end=53 + (row + 1) * support_height // level,
                        x_start=53 + column * support_width // level,
                        x_end=53 + (column + 1) * support_width // level,
                    )
                )
    return tuple(output)


def _balanced_sum(values: torch.Tensor) -> torch.Tensor:
    level = values
    while level.shape[-1] > 1:
        pair_count = level.shape[-1] // 2
        parents = level[..., : 2 * pair_count : 2] + level[..., 1 : 2 * pair_count : 2]
        if level.shape[-1] % 2:
            parents = torch.cat((parents, level[..., -1:]), dim=-1)
        level = parents
    return level[..., 0]


class _SafePopulationStd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        leaves: torch.Tensor,
        mean: torch.Tensor,
        variance: torch.Tensor,
        divisor: torch.Tensor,
    ) -> torch.Tensor:
        standard_deviation = torch.sqrt(variance)
        ctx.save_for_backward(leaves, mean, standard_deviation, divisor)
        return standard_deviation

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        leaves, mean, standard_deviation, divisor = ctx.saved_tensors
        gradient = torch.zeros_like(leaves)
        positive = standard_deviation > 0.0
        if torch.any(positive):
            gradient[positive] = (
                grad_output[positive].unsqueeze(-1)
                * (leaves[positive] - mean[positive].unsqueeze(-1))
                / (divisor * standard_deviation[positive].unsqueeze(-1))
            )
        return gradient, None, None, None


class SupportAlignedPool(nn.Module):
    def forward(
        self,
        features: torch.Tensor,
        valid_support_mask: torch.Tensor,
        neighborhood_valid_support_mask: torch.Tensor,
    ) -> PoolOutput:
        if features.ndim != 6 or features.shape[1:4] != (4, 8, 7):
            raise PoolingContractError("features must have shape [B,4,8,7,H,W]")
        if features.dtype != torch.float32:
            raise PoolingContractError("pooling input must be float32")
        batch, _, _, _, height, width = features.shape
        expected_mask_shape = (batch, 1, height, width)
        if (
            valid_support_mask.shape != expected_mask_shape
            or neighborhood_valid_support_mask.shape != expected_mask_shape
            or valid_support_mask.dtype != torch.bool
            or neighborhood_valid_support_mask.dtype != torch.bool
        ):
            raise PoolingContractError("support masks must be Boolean [B,1,H,W]")
        if torch.any(neighborhood_valid_support_mask & ~valid_support_mask):
            raise PoolingContractError("neighborhood mask must be a subset of valid mask")
        geometry = support_aligned_regions(height, width)
        pooling_mask = valid_support_mask & neighborhood_valid_support_mask
        if not torch.equal(pooling_mask, neighborhood_valid_support_mask):
            raise PoolingContractError("pooling support must equal the neighborhood mask")
        flattened = features.reshape(batch, 224, height, width)
        samples: list[torch.Tensor] = []
        count_rows: list[list[int]] = []
        for sample_index in range(batch):
            regions: list[torch.Tensor] = []
            counts: list[int] = []
            for region in geometry:
                selected_mask = pooling_mask[
                    sample_index,
                    0,
                    region.y_start : region.y_end,
                    region.x_start : region.x_end,
                ].reshape(-1)
                selected = flattened[
                    sample_index,
                    :,
                    region.y_start : region.y_end,
                    region.x_start : region.x_end,
                ].reshape(224, -1)[:, selected_mask]
                count = selected.shape[-1]
                if count == 0:
                    raise PoolingContractError(f"empty region before reduction: {region.index}")
                if not torch.isfinite(selected).all():
                    raise PoolingContractError("non-finite selected pooling input")
                leaves = selected.to(torch.float64)
                divisor = torch.tensor(float(count), dtype=torch.float64, device=features.device)
                mean = _balanced_sum(leaves) / divisor
                centered = leaves - mean.unsqueeze(-1)
                variance = _balanced_sum(centered * centered) / divisor
                standard_deviation = _SafePopulationStd.apply(leaves, mean, variance, divisor)
                statistics = torch.stack((mean, standard_deviation), dim=-1)
                if not torch.isfinite(statistics).all():
                    raise PoolingContractError("non-finite pooling intermediate or statistic")
                regions.append(statistics)
                counts.append(count)
            samples.append(torch.stack(regions, dim=1))
            count_rows.append(counts)
        statistics64 = torch.stack(samples).reshape(batch, 4, 8, 7, 21, 2)
        limit = torch.finfo(torch.float32).max
        if torch.any(torch.abs(statistics64) > limit):
            raise PoolingContractError("float64 statistic exceeds the float32 range")
        pool32 = statistics64.to(torch.float32)
        if not torch.isfinite(pool32).all():
            raise PoolingContractError("float32 pooled output is non-finite")
        return PoolOutput(
            statistics_float64=statistics64,
            pool_float32=pool32,
            valid_count=torch.tensor(count_rows, dtype=torch.int64, device=features.device),
            geometry=geometry,
            pooling_support_mask=pooling_mask,
        )


def flatten_pooled(pool: torch.Tensor) -> torch.Tensor:
    if pool.ndim != 6 or pool.shape[1:] != (4, 8, 7, 21, 2):
        raise PoolingContractError("pooled tensor must have shape [B,4,8,7,21,2]")
    if pool.dtype != torch.float32 or not torch.isfinite(pool).all():
        raise PoolingContractError("classifier handoff must be finite float32")
    return pool.reshape(pool.shape[0], 9408)
