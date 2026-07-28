from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from cam16_wavelet.contracts import DatasetSpec
from cam16_wavelet.hashing import stable_hash


@dataclass(frozen=True)
class DatasetBundle:
    spec: DatasetSpec
    rows: tuple[dict[str, str], ...]
    split_counts: dict[str, int]
    label_counts: dict[str, int]
    manifest_hash: str

    def rows_for(self, split: str) -> tuple[dict[str, str], ...]:
        return tuple(row for row in self.rows if row[self.spec.split_column] == split)


class DatasetRegistry:
    """Loads a manifest and enforces the experiment's data-isolation contract."""

    @classmethod
    def load(cls, spec: DatasetSpec, check_files: bool = False) -> DatasetBundle:
        manifest = spec.manifest if spec.manifest.is_absolute() else spec.root / spec.manifest
        if not manifest.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest}")
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                spec.sample_id_column,
                spec.path_column,
                spec.split_column,
                spec.label_column,
                spec.label_name_column,
                spec.patient_id_column,
            }
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"manifest missing columns: {sorted(missing)}")
            rows = tuple(dict(row) for row in reader)
        if not rows:
            raise ValueError("manifest is empty")
        cls._validate_rows(spec, rows, check_files)
        cls._validate_isolation(spec, rows)
        split_counts = {split: 0 for split in spec.allowed_splits}
        label_counts = {name: 0 for name in spec.class_mapping}
        for row in rows:
            split_counts[row[spec.split_column]] += 1
            label_counts[row[spec.label_name_column]] += 1
        # Hash all manifest content in column-independent canonical form. Any change
        # to paths, patient ownership, physical metadata, or labels changes identity.
        return DatasetBundle(spec, rows, split_counts, label_counts, stable_hash(rows))

    @classmethod
    def _validate_rows(
        cls, spec: DatasetSpec, rows: tuple[dict[str, str], ...], check_files: bool
    ) -> None:
        seen: set[str] = set()
        for row in rows:
            sample_id = row[spec.sample_id_column]
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            seen.add(sample_id)
            if row[spec.split_column] not in spec.allowed_splits:
                raise ValueError(f"invalid split for {sample_id}: {row[spec.split_column]}")
            label_name = row[spec.label_name_column]
            if label_name not in spec.class_mapping:
                raise ValueError(f"unknown label_name: {label_name}")
            if int(row[spec.label_column]) != spec.class_mapping[label_name]:
                raise ValueError(f"inconsistent label mapping for {sample_id}")
            if check_files and not (spec.root / row[spec.path_column]).is_file():
                raise FileNotFoundError(spec.root / row[spec.path_column])

    @staticmethod
    def _validate_isolation(spec: DatasetSpec, rows: tuple[dict[str, str], ...]) -> None:
        ownership: dict[str, str] = {}
        for row in rows:
            patient_id = row[spec.patient_id_column]
            split = row[spec.split_column]
            previous = ownership.setdefault(patient_id, split)
            if previous != split:
                raise ValueError(
                    f"patient/WSI leakage: {patient_id!r} occurs in {previous!r} and {split!r}"
                )
