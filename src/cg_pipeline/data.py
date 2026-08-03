"""Existing-patch CAM16 manifest, isolation, and PNG contracts."""

from __future__ import annotations

import csv
import hashlib
import io
import random
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, Sampler

from .identity import domain_hash
from .training import hash_epoch_order, worker_seed


class DataContractError(ValueError):
    """The supplied patch package cannot be used without silent repair."""


_REQUIRED_COLUMNS = (
    "patch_id",
    "patch_path",
    "split",
    "slide_id",
    "label",
    "label_name",
    "patch_label",
    "slide_label",
)
_SPLITS = ("train", "val", "test")
_CLASS_MAP = {"normal": 0, "tumor": 1}
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class ManifestRow:
    patch_id: str
    patch_path: str
    split: str
    slide_id: str
    patch_target: int
    slide_target: int


@dataclass(frozen=True)
class IsolationReport:
    identity_level: str
    identity_column: str
    cross_split_conflicts: int
    patient_level_status: str
    patient_mapping_evidence: str


@dataclass(frozen=True)
class ManifestBundle:
    root: Path
    manifest: Path
    rows: tuple[ManifestRow, ...]
    split_counts: dict[str, int]
    label_counts: dict[str, int]
    source_manifest_sha256: str
    effective_split_hashes: dict[str, str]
    isolation: IsolationReport
    disk_inventory: dict[str, int]

    def rows_for(self, split: str) -> tuple[ManifestRow, ...]:
        if split not in _SPLITS:
            raise DataContractError(f"unknown split: {split}")
        return tuple(row for row in self.rows if row.split == split)


def _relative_patch_path(value: str) -> PurePosixPath:
    if not value or value != value.strip() or "\\" in value:
        raise DataContractError("patch_path must be a nonempty forward-slash relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DataContractError("patch_path must be a normalized relative path")
    if path.suffix != ".png":
        raise DataContractError("patch_path must name a lowercase .png file")
    return path


def _resolve_inside(root: Path, path: PurePosixPath) -> Path:
    resolved = (root / Path(*path.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise DataContractError("patch_path resolves outside the approved package root") from error
    return resolved


def _effective_hash(rows: tuple[ManifestRow, ...], split: str) -> str:
    effective_rows: list[dict[str, Any]] = []
    for row in sorted((item for item in rows if item.split == split), key=lambda item: item.patch_id.encode("utf-8")):
        effective_rows.append(
            {
                "patch_id": row.patch_id,
                "patch_path": row.patch_path,
                "patch_target": row.patch_target,
                "slide_id": row.slide_id,
                "slide_target": row.slide_target,
                "split": row.split,
            }
        )
    header = {
        "contract_id": "cam16-existing-patch-v1",
        "payload_length": 0,
        "row_count": len(effective_rows),
        "rows": effective_rows,
        "split": split,
    }
    return domain_hash("cg/cam16-eval-manifest/v1", header)


def _load_csv(manifest: Path) -> tuple[bytes, list[dict[str, str]]]:
    try:
        raw = manifest.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise DataContractError(f"cannot read UTF-8 source manifest: {error}") from error
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        records = list(reader)
    except csv.Error as error:
        raise DataContractError(f"invalid RFC 4180 CSV: {error}") from error
    if not records:
        raise DataContractError("manifest is empty")
    header = records[0]
    if len(header) != len(set(header)):
        raise DataContractError("manifest contains duplicate column names")
    missing = set(_REQUIRED_COLUMNS).difference(header)
    if missing:
        raise DataContractError(f"manifest missing required columns: {sorted(missing)}")
    rows: list[dict[str, str]] = []
    for line_number, values in enumerate(records[1:], start=2):
        if len(values) != len(header):
            raise DataContractError(f"manifest row {line_number} has the wrong column count")
        rows.append(dict(zip(header, values)))
    if not rows:
        raise DataContractError("manifest has no data rows")
    return raw, rows


def validate_manifest(
    root: Path | str,
    manifest: Path | str,
    *,
    check_files: bool,
    reconcile_disk: bool,
) -> ManifestBundle:
    """Validate an immutable existing-patch manifest without reading patch pixels."""

    package_root = Path(root).resolve()
    manifest_path = Path(manifest).resolve()
    if not package_root.is_dir():
        raise DataContractError(f"package root is not a directory: {package_root}")
    if not manifest_path.is_file():
        raise DataContractError(f"manifest is not a file: {manifest_path}")
    raw, source_rows = _load_csv(manifest_path)
    seen_patch_ids: set[str] = set()
    seen_paths: set[str] = set()
    slide_splits: dict[str, str] = {}
    slide_targets: dict[str, int] = {}
    rows: list[ManifestRow] = []
    for source in source_rows:
        for column in _REQUIRED_COLUMNS:
            value = source[column]
            if not value or value != value.strip():
                raise DataContractError(f"missing {column} or prohibited whitespace repair")
        patch_id = source["patch_id"]
        if patch_id in seen_patch_ids:
            raise DataContractError(f"duplicate patch_id: {patch_id}")
        seen_patch_ids.add(patch_id)
        path = _relative_patch_path(source["patch_path"])
        normalized_path = path.as_posix()
        if normalized_path in seen_paths:
            raise DataContractError(f"duplicate patch_path: {normalized_path}")
        seen_paths.add(normalized_path)
        split = source["split"]
        if split not in _SPLITS:
            raise DataContractError(f"invalid split: {split}")
        slide_id = source["slide_id"]
        if not slide_id:
            raise DataContractError("missing slide_id")
        previous_split = slide_splits.setdefault(slide_id, split)
        if previous_split != split:
            raise DataContractError(f"slide_id crosses splits: {slide_id}")
        label_name = source["label_name"]
        if label_name not in _CLASS_MAP:
            raise DataContractError(f"unknown label_name: {label_name}")
        try:
            patch_target = int(source["label"])
        except ValueError as error:
            raise DataContractError("label conflict: label is not integer 0 or 1") from error
        if (
            patch_target != _CLASS_MAP[label_name]
            or source["patch_label"] != label_name
            or patch_target not in (0, 1)
        ):
            raise DataContractError(f"label conflict for patch_id: {patch_id}")
        slide_label = source["slide_label"]
        if slide_label not in _CLASS_MAP:
            raise DataContractError(f"invalid slide_label for patch_id: {patch_id}")
        slide_target = _CLASS_MAP[slide_label]
        previous_target = slide_targets.setdefault(slide_id, slide_target)
        if previous_target != slide_target:
            raise DataContractError(f"conflicting slide labels for slide_id: {slide_id}")
        resolved_patch = _resolve_inside(package_root, path)
        if check_files and (not resolved_patch.is_file() or resolved_patch.is_symlink()):
            raise DataContractError(f"missing patch file for manifest row: {normalized_path}")
        rows.append(
            ManifestRow(
                patch_id=patch_id,
                patch_path=normalized_path,
                split=split,
                slide_id=slide_id,
                patch_target=patch_target,
                slide_target=slide_target,
            )
        )
    disk_inventory = {"manifest_png_count": len(seen_paths), "disk_png_count": -1}
    if reconcile_disk:
        patch_root = package_root / "patches"
        disk_paths = {
            path.relative_to(package_root).as_posix()
            for path in patch_root.rglob("*.png")
            if path.is_file() and not path.is_symlink()
        }
        disk_inventory["disk_png_count"] = len(disk_paths)
        missing_files = seen_paths.difference(disk_paths)
        extra_files = disk_paths.difference(seen_paths)
        if missing_files or extra_files:
            raise DataContractError(
                "disk inventory mismatch: "
                f"missing={len(missing_files)}, extra={len(extra_files)}"
            )
    frozen_rows = tuple(rows)
    split_counts_counter = Counter(row.split for row in frozen_rows)
    label_counts_counter = Counter("tumor" if row.patch_target else "normal" for row in frozen_rows)
    return ManifestBundle(
        root=package_root,
        manifest=manifest_path,
        rows=frozen_rows,
        split_counts={split: split_counts_counter[split] for split in _SPLITS},
        label_counts={name: label_counts_counter[name] for name in _CLASS_MAP},
        source_manifest_sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
        effective_split_hashes={split: _effective_hash(frozen_rows, split) for split in _SPLITS},
        isolation=IsolationReport(
            identity_level="slide_id",
            identity_column="slide_id",
            cross_split_conflicts=0,
            patient_level_status="not_evaluated",
            patient_mapping_evidence="not_available",
        ),
        disk_inventory=disk_inventory,
    )


def decode_png_rgb(data: bytes) -> np.ndarray:
    """Decode the exact approved PNG subset to C-contiguous RGB uint8."""

    if len(data) < 33 or not data.startswith(_PNG_SIGNATURE):
        raise DataContractError("patch is not a valid PNG container")
    length = struct.unpack(">I", data[8:12])[0]
    if length != 13 or data[12:16] != b"IHDR":
        raise DataContractError("PNG does not begin with a canonical IHDR")
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", data[16:29]
    )
    if (width, height) != (256, 256):
        raise DataContractError(f"PNG shape must be [256,256,3], got [{height},{width},?]")
    if (bit_depth, color_type, compression, filter_method, interlace) != (8, 2, 0, 0, 0):
        raise DataContractError("PNG must be non-interlaced 8-bit truecolor without alpha")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (256, 256):
                raise DataContractError("decoded PNG mode or shape violates the RGB contract")
            array = np.array(image, dtype=np.uint8, copy=True)
    except (OSError, UnidentifiedImageError) as error:
        raise DataContractError(f"PNG decoding failed: {error}") from error
    if array.shape != (256, 256, 3) or array.dtype != np.uint8:
        raise DataContractError("decoded PNG shape or dtype violates the contract")
    return np.ascontiguousarray(array)


def validate_patient_mapping(bundle: ManifestBundle, path: Path | str) -> dict[str, Any]:
    """Validate mapping structure and isolation, without asserting provenance reliability."""

    mapping_path = Path(path)
    try:
        raw = mapping_path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise DataContractError(f"cannot read patient mapping: {error}") from error
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        records = list(reader)
    except csv.Error as error:
        raise DataContractError(f"invalid patient mapping CSV: {error}") from error
    if not records:
        raise DataContractError("patient mapping is empty")
    header = records[0]
    if len(header) != len(set(header)) or set(header) != {"slide_id", "patient_id", "provenance"}:
        raise DataContractError(
            "patient mapping must have exactly slide_id, patient_id, provenance columns"
        )
    index = {name: header.index(name) for name in header}
    slide_to_patient: dict[str, str] = {}
    provenance_values: set[str] = set()
    for line_number, values in enumerate(records[1:], start=2):
        if len(values) != len(header):
            raise DataContractError(f"patient mapping row {line_number} has wrong column count")
        slide_id = values[index["slide_id"]]
        patient_id = values[index["patient_id"]]
        provenance = values[index["provenance"]]
        if any(not value or value != value.strip() for value in (slide_id, patient_id, provenance)):
            raise DataContractError("patient mapping contains missing or repairable values")
        if slide_id in slide_to_patient:
            raise DataContractError("patient mapping contains duplicate slide_id")
        slide_to_patient[slide_id] = patient_id
        provenance_values.add(provenance)
    required_slides = {row.slide_id for row in bundle.rows}
    supplied_slides = set(slide_to_patient)
    if required_slides != supplied_slides:
        raise DataContractError(
            "patient mapping coverage mismatch: "
            f"missing={len(required_slides - supplied_slides)}, "
            f"extra={len(supplied_slides - required_slides)}"
        )
    slide_split = {row.slide_id: row.split for row in bundle.rows}
    patient_split: dict[str, str] = {}
    for slide_id, patient_id in slide_to_patient.items():
        split = slide_split[slide_id]
        previous = patient_split.setdefault(patient_id, split)
        if previous != split:
            raise DataContractError("patient identity crosses splits")
    return {
        "status": "structure_validated",
        "provenance_reliability": "not_assessed",
        "mapping_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "mapping_coverage": len(supplied_slides),
        "slide_count": len(required_slides),
        "patient_count": len(patient_split),
        "provenance_value_count": len(provenance_values),
        "cross_split_conflicts": 0,
    }


class PatchDataset(Dataset[dict[str, Any]]):
    def __init__(self, bundle: ManifestBundle, split: str) -> None:
        self.bundle = bundle
        self.split = split
        self.rows = bundle.rows_for(split)
        if not self.rows:
            raise DataContractError(f"requested split has no rows: {split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = self.bundle.root / Path(*PurePosixPath(row.patch_path).parts)
        try:
            data = path.read_bytes()
        except OSError as error:
            raise DataContractError(f"cannot read patch file for patch_id {row.patch_id}") from error
        rgb = decode_png_rgb(data)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        return {
            "rgb": tensor,
            "target": torch.tensor(row.patch_target, dtype=torch.float32),
            "patch_id": row.patch_id,
            "slide_id": row.slide_id,
            "slide_target": row.slide_target,
            "split": row.split,
        }


class _HashOrderSampler(Sampler[int]):
    def __init__(self, dataset: PatchDataset, seed: int, epoch: int) -> None:
        identifiers = tuple(row.patch_id for row in dataset.rows)
        ordered = hash_epoch_order(identifiers, seed=seed, epoch=epoch)
        index = {row.patch_id: position for position, row in enumerate(dataset.rows)}
        self._indices = tuple(index[identifier] for identifier in ordered)

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


@dataclass(frozen=True)
class _WorkerSeeder:
    base_seed: int
    epoch: int

    def __call__(self, worker_id: int) -> None:
        seed = worker_seed(self.base_seed, self.epoch, worker_id)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def build_dataloader(
    dataset: PatchDataset,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    num_workers: int,
) -> DataLoader:
    if batch_size < 1 or num_workers < 0:
        raise DataContractError("batch_size and num_workers are illegal")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(worker_seed(seed, epoch, 0))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=_HashOrderSampler(dataset, seed, epoch),
        num_workers=num_workers,
        drop_last=False,
        worker_init_fn=_WorkerSeeder(seed, epoch),
        generator=generator,
        pin_memory=False,
    )
