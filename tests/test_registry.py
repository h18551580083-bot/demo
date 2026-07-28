import csv
from pathlib import Path

import pytest

from cam16_wavelet.contracts import DatasetSpec
from cam16_wavelet.data import DatasetRegistry


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["patch_id", "patch_path", "split", "label", "label_name", "slide_id"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _spec(root: Path) -> DatasetSpec:
    return DatasetSpec("fixture", root, Path("manifest.csv"), {"normal": 0, "tumor": 1})


def test_registry_loads_valid_patient_isolated_manifest(tmp_path):
    rows = [
        {"patch_id": "a", "patch_path": "a.png", "split": "train", "label": "0",
         "label_name": "normal", "slide_id": "p1"},
        {"patch_id": "b", "patch_path": "b.png", "split": "val", "label": "1",
         "label_name": "tumor", "slide_id": "p2"},
    ]
    _write_manifest(tmp_path / "manifest.csv", rows)
    bundle = DatasetRegistry.load(_spec(tmp_path))
    assert bundle.split_counts == {"train": 1, "val": 1, "test": 0}
    assert bundle.label_counts == {"normal": 1, "tumor": 1}


def test_registry_rejects_patient_leakage(tmp_path):
    rows = [
        {"patch_id": "a", "patch_path": "a.png", "split": "train", "label": "0",
         "label_name": "normal", "slide_id": "p1"},
        {"patch_id": "b", "patch_path": "b.png", "split": "test", "label": "1",
         "label_name": "tumor", "slide_id": "p1"},
    ]
    _write_manifest(tmp_path / "manifest.csv", rows)
    with pytest.raises(ValueError, match="leakage"):
        DatasetRegistry.load(_spec(tmp_path))


def test_manifest_hash_changes_when_patient_metadata_changes(tmp_path):
    row = {"patch_id": "a", "patch_path": "a.png", "split": "train", "label": "0",
           "label_name": "normal", "slide_id": "p1"}
    _write_manifest(tmp_path / "manifest.csv", [row])
    first = DatasetRegistry.load(_spec(tmp_path)).manifest_hash
    row["slide_id"] = "p2"
    _write_manifest(tmp_path / "manifest.csv", [row])
    assert DatasetRegistry.load(_spec(tmp_path)).manifest_hash != first
