"""Deterministic and pre-registered fixtures for Decision 30 calibration."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum

import torch


class CalibrationMode(str, Enum):
    """Permitted calibration scopes."""

    LOCAL_SMOKE = "local_smoke"
    FORMAL_ACCEPTANCE = "formal_acceptance"


@dataclass(frozen=True)
class CalibrationFixture:
    """Serialized formal inputs and optional pre-registration identity."""

    name: str
    z: torch.Tensor
    mask: torch.Tensor
    valid_counts: torch.Tensor
    registered_input_hashes: Mapping[str, str] | None = None
    registered_formal_shape: tuple[int, ...] | None = None

    @property
    def is_preregistered(self) -> bool:
        return (
            self.registered_input_hashes is not None
            and self.registered_formal_shape is not None
        )

    def preregister(self) -> CalibrationFixture:
        """Return an immutable fixture with its current content identity frozen."""

        return replace(
            self,
            registered_input_hashes=fixture_input_hashes(self),
            registered_formal_shape=tuple(self.z.shape),
        )


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash dtype, shape, and canonical contiguous CPU payload."""

    cpu = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(size) for size in cpu.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def fixture_input_hashes(fixture: CalibrationFixture) -> dict[str, str]:
    return {
        "z": tensor_sha256(fixture.z),
        "mask": tensor_sha256(fixture.mask),
        "valid_counts": tensor_sha256(fixture.valid_counts),
    }


def _fixture_regions(height: int, width: int) -> tuple[tuple[int, int, int, int], ...]:
    regions: list[tuple[int, int, int, int]] = []
    for level in (1, 2, 4):
        for row in range(level):
            for column in range(level):
                regions.append(
                    (
                        row * height // level,
                        (row + 1) * height // level,
                        column * width // level,
                        (column + 1) * width // level,
                    )
                )
    return tuple(regions)


def build_deterministic_fixture(
    *,
    name: str,
    batch: int = 1,
    height: int = 8,
    width: int = 8,
    preregister: bool = False,
) -> CalibrationFixture:
    """Build a small exact float32 fixture with zero-variance and adversarial channels."""

    if batch < 1 or height < 4 or width < 4:
        raise ValueError("fixture requires batch >= 1 and height/width >= 4")
    channels = 4 * 8 * 7
    channel = torch.arange(channels, dtype=torch.int64).view(1, channels, 1, 1)
    batch_axis = torch.arange(batch, dtype=torch.int64).view(batch, 1, 1, 1)
    y = torch.arange(height, dtype=torch.int64).view(1, 1, height, 1)
    x = torch.arange(width, dtype=torch.int64).view(1, 1, 1, width)
    integer_values = ((17 * batch_axis + 13 * channel + 5 * y + 3 * x) % 127) + 1
    values = integer_values.to(torch.float32) / 128.0
    values[:, 0, :, :] = 0.5
    values[:, 1, :, :] = 1.0

    grid_y = torch.arange(height, dtype=torch.int64).view(height, 1)
    grid_x = torch.arange(width, dtype=torch.int64).view(1, width)
    masks = []
    for batch_index in range(batch):
        masks.append((grid_y + 2 * grid_x + batch_index) % 7 != 0)
    mask = torch.stack(masks, dim=0)
    for batch_index in range(batch):
        first_valid = torch.nonzero(mask[batch_index], as_tuple=False)[0]
        values[batch_index, 1, first_valid[0], first_valid[1]] = float.fromhex("0x1p+53")

    z = values.reshape(batch, 4, 8, 7, height, width).contiguous()
    counts = []
    for y_start, y_end, x_start, x_end in _fixture_regions(height, width):
        counts.append(mask[:, y_start:y_end, x_start:x_end].sum(dim=(-2, -1)))
    valid_counts = torch.stack(counts, dim=1).to(torch.int64)
    if torch.any(valid_counts == 0):
        raise RuntimeError("calibration fixture contains an empty region")
    fixture = CalibrationFixture(
        name=name,
        z=z,
        mask=mask,
        valid_counts=valid_counts,
    )
    return fixture.preregister() if preregister else fixture


def validate_fixture(fixture: CalibrationFixture) -> None:
    z, mask, counts = fixture.z, fixture.mask, fixture.valid_counts
    if z.device.type != "cpu" or mask.device.type != "cpu" or counts.device.type != "cpu":
        raise ValueError("fixture inputs must be serialized CPU tensors")
    if z.dtype != torch.float32 or z.ndim != 6 or z.shape[1:4] != (4, 8, 7):
        raise ValueError("fixture z must be float32 [B,4,8,7,H,W]")
    if mask.dtype != torch.bool or tuple(mask.shape) != (z.shape[0], z.shape[4], z.shape[5]):
        raise ValueError("fixture mask shape or dtype is invalid")
    if counts.dtype != torch.int64 or tuple(counts.shape) != (z.shape[0], 21):
        raise ValueError("fixture valid_counts must be int64 [B,21]")
    if not torch.isfinite(z).all():
        raise ValueError("fixture z must be finite")
