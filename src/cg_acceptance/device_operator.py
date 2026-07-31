"""Candidate device pooling operator exercised through real PyTorch autograd."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch

OPERATOR_VERSION = "decision30-device-pooling-v2"


class CandidateFault(str, Enum):
    NONE = "none"
    AXIS_PERMUTATION = "axis_permutation"
    SLOT_EXCHANGE = "mean_std_slot_exchange"
    SIGN_INVERSION = "sign_inversion"
    WRONG_REDUCTION_ORDER = "wrong_reduction_order"
    EARLY_FLOAT32 = "early_float32_conversion"
    SPATIAL_MISALIGNMENT = "single_point_spatial_misalignment"
    WRONG_MASK = "wrong_mask_selection"
    ZERO_VARIANCE_BACKWARD = "zero_variance_backward_violation"


@dataclass(frozen=True)
class DevicePoolOutput:
    statistics_float64: torch.Tensor
    pool_float32: torch.Tensor


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _candidate_regions(height: int, width: int) -> tuple[tuple[int, int, int, int], ...]:
    result: list[tuple[int, int, int, int]] = []
    for grid_size in (1, 2, 4):
        for grid_y in range(grid_size):
            for grid_x in range(grid_size):
                y0 = grid_y * height // grid_size
                y1 = (grid_y + 1) * height // grid_size
                x0 = grid_x * width // grid_size
                x1 = (grid_x + 1) * width // grid_size
                result.append((y0, y1, x0, x1))
    return tuple(result)


def _candidate_balanced_sum(values: torch.Tensor) -> torch.Tensor:
    nodes = values
    while nodes.shape[-1] != 1:
        pair_total = nodes.shape[-1] // 2
        parents = nodes[..., : 2 * pair_total : 2] + nodes[..., 1 : 2 * pair_total : 2]
        if nodes.shape[-1] & 1:
            parents = torch.cat((parents, nodes[..., -1:]), dim=-1)
        nodes = parents
    return nodes.squeeze(-1)


def _candidate_sequential_sum(values: torch.Tensor) -> torch.Tensor:
    total = values[..., 0]
    for index in range(1, values.shape[-1]):
        total = total + values[..., index]
    return total


class _PopulationStd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        leaves: torch.Tensor,
        mean: torch.Tensor,
        variance: torch.Tensor,
        count: torch.Tensor,
        wrong_zero_backward: bool,
    ) -> torch.Tensor:
        std = torch.sqrt(variance)
        ctx.save_for_backward(leaves, mean, std, count)
        ctx.wrong_zero_backward = wrong_zero_backward
        return std

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        leaves, mean, std, count = ctx.saved_tensors
        centered = leaves - mean.unsqueeze(-1)
        gradient = torch.zeros_like(leaves)
        positive = std > 0.0
        gradient[positive] = (
            grad_output[positive].unsqueeze(-1)
            * centered[positive]
            / (count * std[positive].unsqueeze(-1))
        )
        if ctx.wrong_zero_backward:
            wrong = (grad_output[~positive] / count).unsqueeze(-1)
            gradient[~positive] = wrong.expand(-1, leaves.shape[-1])
        return gradient, None, None, None, None


def _faulted_spatial_input(z: torch.Tensor) -> torch.Tensor:
    changed = z.clone()
    changed[0, 0, 0, 0, 0, 1] = z[0, 0, 0, 0, 0, 2]
    return changed


def _faulted_mask(mask: torch.Tensor) -> torch.Tensor:
    changed = mask.clone()
    excluded = torch.nonzero(~changed[0], as_tuple=False)[0]
    included = torch.nonzero(changed[0], as_tuple=False)[0]
    changed[0, excluded[0], excluded[1]] = True
    changed[0, included[0], included[1]] = False
    return changed


def pool(
    z: torch.Tensor,
    mask: torch.Tensor,
    valid_counts: torch.Tensor,
    *,
    fault: CandidateFault = CandidateFault.NONE,
) -> DevicePoolOutput:
    """Execute the candidate pooling forward path on the tensor's device."""

    batch, _, _, _, height, width = z.shape
    channels = 4 * 8 * 7
    candidate_z = _faulted_spatial_input(z) if fault is CandidateFault.SPATIAL_MISALIGNMENT else z
    candidate_mask = _faulted_mask(mask) if fault is CandidateFault.WRONG_MASK else mask
    flattened = candidate_z.reshape(batch, channels, height, width)
    sample_results: list[torch.Tensor] = []
    reducer = (
        _candidate_sequential_sum
        if fault is CandidateFault.WRONG_REDUCTION_ORDER
        else _candidate_balanced_sum
    )
    for sample_index in range(batch):
        region_results: list[torch.Tensor] = []
        for region_index, bounds in enumerate(_candidate_regions(height, width)):
            y0, y1, x0, x1 = bounds
            selected_mask = candidate_mask[sample_index, y0:y1, x0:x1].reshape(-1)
            selected = flattened[
                sample_index, :, y0:y1, x0:x1
            ].reshape(channels, -1)[:, selected_mask]
            expected_count = int(valid_counts[sample_index, region_index])
            if selected.shape[-1] != expected_count:
                raise ValueError("candidate valid_count mismatch")
            leaves = selected if fault is CandidateFault.EARLY_FLOAT32 else selected.to(torch.float64)
            divisor = valid_counts[sample_index, region_index].to(leaves.dtype)
            mean = reducer(leaves) / divisor
            centered = leaves - mean.unsqueeze(-1)
            variance = reducer(centered * centered) / divisor
            std = _PopulationStd.apply(
                leaves,
                mean,
                variance,
                divisor,
                fault is CandidateFault.ZERO_VARIANCE_BACKWARD,
            )
            region_results.append(torch.stack((mean, std), dim=-1))
        sample_results.append(torch.stack(region_results, dim=1))
    statistics = torch.stack(sample_results).reshape(batch, 4, 8, 7, 21, 2)
    statistics64 = statistics.to(torch.float64)
    if fault is CandidateFault.AXIS_PERMUTATION:
        statistics64 = torch.roll(statistics64, shifts=1, dims=1)
    elif fault is CandidateFault.SLOT_EXCHANGE:
        statistics64 = statistics64.flip(-1)
    elif fault is CandidateFault.SIGN_INVERSION:
        statistics64 = -statistics64
    return DevicePoolOutput(
        statistics_float64=statistics64,
        pool_float32=statistics64.to(torch.float32),
    )
