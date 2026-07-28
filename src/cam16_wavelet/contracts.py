from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    root: Path
    manifest: Path
    class_mapping: dict[str, int]
    patient_id_column: str = "slide_id"
    sample_id_column: str = "patch_id"
    path_column: str = "patch_path"
    split_column: str = "split"
    label_column: str = "label"
    label_name_column: str = "label_name"
    allowed_splits: tuple[str, ...] = ("train", "val", "test")
    license_note: str = "TBD"
    preprocessing_note: str = "TBD"


@dataclass(frozen=True)
class KernelSpec:
    families: tuple[Literal["log", "gabor"], ...] = ("log", "gabor")
    size: int = 15
    scales: tuple[float, ...] = (1.5, 3.0)
    orientations: int = 4
    wavelength_factor: float = 4.0
    gamma: float = 0.5
    zero_mean: bool = True
    normalization: Literal["l1", "l2"] = "l2"

    def validate(self) -> None:
        if self.size < 3 or self.size % 2 == 0:
            raise ValueError("kernel size must be odd and >= 3")
        if not self.scales or any(scale <= 0 for scale in self.scales):
            raise ValueError("scales must contain positive values")
        if self.orientations < 1:
            raise ValueError("orientations must be positive")


@dataclass(frozen=True)
class DetectorConfig:
    sqrt_after_detection: bool = True
    epsilon: float = 1e-6
    noise_std: float = 0.0
    quantization_bits: int | None = None
    saturation: float | None = None
    pool_size: int = 4

    def validate(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if self.quantization_bits is not None and self.quantization_bits < 1:
            raise ValueError("quantization_bits must be >= 1")
        if self.saturation is not None and self.saturation <= 0:
            raise ValueError("saturation must be positive")
        if self.pool_size < 1:
            raise ValueError("pool_size must be >= 1")


@dataclass(frozen=True)
class FrontendConfig:
    mode: Literal["digital_ideal", "fourier_4f"] = "digital_ideal"
    kernel: KernelSpec = field(default_factory=KernelSpec)
    detector: DetectorConfig = field(default_factory=DetectorConfig)


def dataclass_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
