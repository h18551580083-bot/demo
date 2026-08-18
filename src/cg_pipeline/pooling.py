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


def _balanced_sum(values: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    level = values
    level_counts = counts
    while level.shape[-1] > 1:
        left = level[..., 0::2]
        right = level[..., 1::2]
        if right.shape[-1] < left.shape[-1]:
            right = torch.cat((right, torch.zeros_like(left[..., :1])), dim=-1)
        pair_counts = level_counts // 2
        parent_positions = torch.arange(left.shape[-1], device=level.device)[None, :]
        paired = parent_positions < pair_counts[:, None]
        parents = torch.where(
            paired[:, None, :],
            left + right,
            torch.zeros((), dtype=level.dtype, device=level.device),
        )
        carried = torch.gather(
            level,
            dim=-1,
            index=(2 * pair_counts).clamp_max(level.shape[-1] - 1)[:, None, None].expand(
                -1, level.shape[1], 1
            ),
        )
        unpaired = (level_counts % 2 == 1)[:, None] & (
            parent_positions == pair_counts[:, None]
        )
        parents = torch.where(unpaired[:, None, :], carried, parents)
        level = parents
        level_counts = (level_counts + 1) // 2
    return level[..., 0]


class _SafePopulationStd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        leaves: torch.Tensor,
        mean: torch.Tensor,
        variance: torch.Tensor,
        divisor: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        standard_deviation = torch.sqrt(variance)
        ctx.save_for_backward(leaves, mean, standard_deviation, divisor, selected_mask)
        return standard_deviation

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        leaves, mean, standard_deviation, divisor, selected_mask = ctx.saved_tensors
        positive = standard_deviation > 0.0
        safe_standard_deviation = torch.where(
            positive,
            standard_deviation,
            torch.ones_like(standard_deviation),
        )
        gradient = (
            grad_output.unsqueeze(-1)
            * (leaves - mean.unsqueeze(-1))
            / (divisor[:, None, None] * safe_standard_deviation.unsqueeze(-1))
        )
        gradient = gradient * selected_mask[:, None, :] * positive.unsqueeze(-1)
        return gradient, None, None, None, None


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
        region_statistics: list[torch.Tensor] = []
        count_columns: list[torch.Tensor] = []
        for region in geometry:
            selected_mask = pooling_mask[
                :,
                0,
                region.y_start : region.y_end,
                region.x_start : region.x_end,
            ].reshape(batch, -1)
            counts = selected_mask.sum(dim=-1)
            if torch.any(counts == 0):
                raise PoolingContractError(f"empty region before reduction: {region.index}")
            region_features = flattened[
                :,
                :,
                region.y_start : region.y_end,
                region.x_start : region.x_end,
            ].reshape(batch, 224, -1)
            if not torch.all(torch.isfinite(region_features) | ~selected_mask[:, None, :]):
                raise PoolingContractError("non-finite selected pooling input")
            selected_first = torch.argsort(
                (~selected_mask).to(torch.uint8), dim=-1, stable=True
            )
            compacted_features = torch.gather(
                region_features.to(torch.float64),
                dim=-1,
                index=selected_first[:, None, :].expand(-1, 224, -1),
            )
            compacted_mask = (
                torch.arange(selected_mask.shape[-1], device=features.device)[None, :]
                < counts[:, None]
            )
            leaves = torch.where(
                compacted_mask[:, None, :],
                compacted_features,
                torch.zeros((), dtype=torch.float64, device=features.device),
            )
            divisor = counts.to(torch.float64)
            mean = _balanced_sum(leaves, counts) / divisor[:, None]
            centered = torch.where(
                compacted_mask[:, None, :],
                leaves - mean.unsqueeze(-1),
                torch.zeros((), dtype=torch.float64, device=features.device),
            )
            variance = _balanced_sum(centered * centered, counts) / divisor[:, None]
            standard_deviation = _SafePopulationStd.apply(
                leaves,
                mean,
                variance,
                divisor,
                compacted_mask,
            )
            statistics = torch.stack((mean, standard_deviation), dim=-1)
            if not torch.isfinite(statistics).all():
                raise PoolingContractError("non-finite pooling intermediate or statistic")
            region_statistics.append(statistics)
            count_columns.append(counts)
        statistics64 = torch.stack(region_statistics, dim=2).reshape(batch, 4, 8, 7, 21, 2)
        limit = torch.finfo(torch.float32).max
        if torch.any(torch.abs(statistics64) > limit):
            raise PoolingContractError("float64 statistic exceeds the float32 range")
        pool32 = statistics64.to(torch.float32)
        if not torch.isfinite(pool32).all():
            raise PoolingContractError("float32 pooled output is non-finite")
        return PoolOutput(
            statistics_float64=statistics64,
            pool_float32=pool32,
            valid_count=torch.stack(count_columns, dim=1),
            geometry=geometry,
            pooling_support_mask=pooling_mask,
        )


def flatten_pooled(pool: torch.Tensor) -> torch.Tensor:
    if pool.ndim != 6 or pool.shape[1:] != (4, 8, 7, 21, 2):
        raise PoolingContractError("pooled tensor must have shape [B,4,8,7,21,2]")
    if pool.dtype != torch.float32 or not torch.isfinite(pool).all():
        raise PoolingContractError("classifier handoff must be finite float32")
    return pool.reshape(pool.shape[0], 9408)
