"""Independent normative CPU reference for Decision 30 pooling."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

REFERENCE_VERSION = "decision30-cpu-reference-v2"


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _reference_regions(height: int, width: int) -> tuple[tuple[int, int, int, int], ...]:
    output: list[tuple[int, int, int, int]] = []
    for divisions in (1, 2, 4):
        for region_row in range(divisions):
            for region_column in range(divisions):
                output.append(
                    (
                        region_row * height // divisions,
                        (region_row + 1) * height // divisions,
                        region_column * width // divisions,
                        (region_column + 1) * width // divisions,
                    )
                )
    return tuple(output)


def _reference_balanced_sum(leaves: torch.Tensor) -> torch.Tensor:
    level = leaves
    while level.shape[-1] > 1:
        pairs = level.shape[-1] // 2
        next_level = level[..., 0 : 2 * pairs : 2] + level[..., 1 : 2 * pairs : 2]
        if level.shape[-1] % 2:
            next_level = torch.cat((next_level, level[..., -1:]), dim=-1)
        level = next_level
    return level[..., 0]


def forward(
    z: torch.Tensor,
    mask: torch.Tensor,
    valid_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute normative float64 statistics and the sole float32 conversion."""

    if z.device.type != "cpu":
        raise ValueError("normative reference executes on CPU")
    batch, _, _, _, height, width = z.shape
    channels = 4 * 8 * 7
    z_channels = z.reshape(batch, channels, height, width)
    samples: list[torch.Tensor] = []
    for sample_index in range(batch):
        regions: list[torch.Tensor] = []
        for region_index, bounds in enumerate(_reference_regions(height, width)):
            y_start, y_end, x_start, x_end = bounds
            selected_mask = mask[sample_index, y_start:y_end, x_start:x_end].reshape(-1)
            selected = z_channels[
                sample_index, :, y_start:y_end, x_start:x_end
            ].reshape(channels, -1)[:, selected_mask]
            count = int(valid_counts[sample_index, region_index])
            if selected.shape[-1] != count:
                raise ValueError("reference valid_count mismatch")
            leaves = selected.to(torch.float64)
            divisor = valid_counts[sample_index, region_index].to(torch.float64)
            mean = _reference_balanced_sum(leaves) / divisor
            centered = leaves - mean.unsqueeze(-1)
            variance = _reference_balanced_sum(centered * centered) / divisor
            std = torch.sqrt(variance)
            regions.append(torch.stack((mean, std), dim=-1))
        samples.append(torch.stack(regions, dim=1))
    statistics = torch.stack(samples).reshape(batch, 4, 8, 7, 21, 2)
    return statistics, statistics.to(torch.float32)


def backward(
    z: torch.Tensor,
    mask: torch.Tensor,
    valid_counts: torch.Tensor,
    statistics: torch.Tensor,
    upstream: torch.Tensor,
) -> torch.Tensor:
    """Compute the independent normative analytic VJP on CPU."""

    batch, _, _, _, height, width = z.shape
    channels = 4 * 8 * 7
    z_channels = z.reshape(batch, channels, height, width)
    stats = statistics.reshape(batch, channels, 21, 2)
    gradient = upstream.reshape(batch, channels, 21, 2)
    result = torch.zeros((batch, channels, height, width), dtype=torch.float64)
    for sample_index in range(batch):
        for region_index, bounds in enumerate(_reference_regions(height, width)):
            y_start, y_end, x_start, x_end = bounds
            region_mask = mask[sample_index, y_start:y_end, x_start:x_end]
            coordinates = torch.nonzero(region_mask, as_tuple=False)
            ys = coordinates[:, 0] + y_start
            xs = coordinates[:, 1] + x_start
            leaves = z_channels[sample_index, :, ys, xs].to(torch.float64)
            count = int(valid_counts[sample_index, region_index])
            if leaves.shape[-1] != count:
                raise ValueError("reference backward valid_count mismatch")
            divisor = valid_counts[sample_index, region_index].to(torch.float64)
            mean = stats[sample_index, :, region_index, 0]
            std = stats[sample_index, :, region_index, 1]
            g_mean = gradient[sample_index, :, region_index, 0].to(torch.float64)
            g_std = gradient[sample_index, :, region_index, 1].to(torch.float64)
            contribution = (g_mean / divisor).unsqueeze(-1).expand_as(leaves).clone()
            positive = std > 0.0
            contribution[positive] += (
                g_std[positive].unsqueeze(-1)
                * (leaves[positive] - mean[positive].unsqueeze(-1))
                / (divisor * std[positive].unsqueeze(-1))
            )
            previous = result[sample_index, :, ys, xs]
            result[sample_index, :, ys, xs] = previous + contribution
    return result.to(torch.float32).reshape(batch, 4, 8, 7, height, width)
