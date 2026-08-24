from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cg_pipeline.data import (
    DataContractError,
    PatchDataset,
    build_dataloader,
    decode_png_rgb,
    validate_manifest,
    validate_patient_mapping,
)

FIELDNAMES = [
    "patch_id",
    "patch_path",
    "split",
    "slide_id",
    "label",
    "label_name",
    "patch_label",
    "slide_label",
]


def _png(path: Path, value: int = 127) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((256, 256, 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path, format="PNG")


def _rows() -> list[dict[str, str]]:
    return [
        {
            "patch_id": "p-train-normal",
            "patch_path": "patches/train/normal/p-train-normal.png",
            "split": "train",
            "slide_id": "slide-train-normal",
            "label": "0",
            "label_name": "normal",
            "patch_label": "normal",
            "slide_label": "normal",
        },
        {
            "patch_id": "p-val-tumor",
            "patch_path": "patches/val/tumor/p-val-tumor.png",
            "split": "val",
            "slide_id": "slide-val-tumor",
            "label": "1",
            "label_name": "tumor",
            "patch_label": "tumor",
            "slide_label": "tumor",
        },
    ]


def _manifest(root: Path, rows: list[dict[str, str]]) -> Path:
    path = root / "metadata" / "training_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _valid_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "package"
    rows = _rows()
    for index, row in enumerate(rows):
        _png(root / Path(row["patch_path"]), value=100 + index)
    return root, _manifest(root, rows)


def test_manifest_contract_reports_exact_slide_level_and_disk_identity(tmp_path: Path) -> None:
    root, manifest = _valid_root(tmp_path)

    bundle = validate_manifest(root, manifest, check_files=True, reconcile_disk=True)

    assert len(bundle.rows) == 2
    assert bundle.split_counts == {"train": 1, "val": 1, "test": 0}
    assert bundle.label_counts == {"normal": 1, "tumor": 1}
    assert bundle.isolation.identity_level == "slide_id"
    assert bundle.isolation.cross_split_conflicts == 0
    assert bundle.isolation.patient_level_isolation == "not_evaluated"
    assert bundle.isolation.patient_level_claim_allowed is False
    assert bundle.isolation.isolation_claim == (
        "group_id/slide_id split isolation verified"
    )
    assert bundle.isolation.patient_mapping_evidence == "not_available"
    assert bundle.source_manifest_sha256.startswith("sha256:")
    assert set(bundle.effective_split_hashes) == {"train", "val", "test"}
    assert bundle.disk_inventory == {"manifest_png_count": 2, "disk_png_count": 2}


def test_manifest_does_not_infer_patient_identity_from_identifier_syntax(
    tmp_path: Path,
) -> None:
    root = tmp_path / "package"
    rows = _rows()
    rows[0]["slide_id"] = "patient-like-prefix-slide-a"
    rows[1]["slide_id"] = "patient-like-prefix-slide-b"
    rows[0]["patch_id"] = "patient-like-prefix-patch-a"
    rows[1]["patch_id"] = "patient-like-prefix-patch-b"
    rows[0]["patch_path"] = "patches/train/normal/patient-like-prefix-patch-a.png"
    rows[1]["patch_path"] = "patches/val/tumor/patient-like-prefix-patch-b.png"
    for row in rows:
        _png(root / Path(row["patch_path"]))

    bundle = validate_manifest(
        root,
        _manifest(root, rows),
        check_files=True,
        reconcile_disk=True,
    )

    assert bundle.isolation.cross_split_conflicts == 0
    assert bundle.isolation.patient_level_isolation == "not_evaluated"
    assert bundle.isolation.patient_level_claim_allowed is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.__setitem__(1, {**rows[1], "patch_id": rows[0]["patch_id"]}), "duplicate patch_id"),
        (lambda rows: rows.__setitem__(1, {**rows[1], "patch_path": rows[0]["patch_path"]}), "duplicate patch_path"),
        (lambda rows: rows.__setitem__(1, {**rows[1], "slide_id": rows[0]["slide_id"]}), "crosses splits"),
        (lambda rows: rows.__setitem__(1, {**rows[1], "slide_id": ""}), "missing slide_id"),
        (lambda rows: rows.__setitem__(1, {**rows[1], "label": "0"}), "label conflict"),
        (lambda rows: rows.__setitem__(1, {**rows[1], "patch_path": "../escape.png"}), "relative path"),
    ],
)
def test_manifest_contract_fails_closed_on_identity_label_and_path_errors(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    root = tmp_path / "package"
    rows = _rows()
    mutate(rows)
    manifest = _manifest(root, rows)

    with pytest.raises(DataContractError, match=message):
        validate_manifest(root, manifest, check_files=False, reconcile_disk=False)


def test_manifest_contract_rejects_missing_and_extra_files(tmp_path: Path) -> None:
    root, manifest = _valid_root(tmp_path)
    (root / "patches/val/tumor/p-val-tumor.png").unlink()

    with pytest.raises(DataContractError, match="missing patch file"):
        validate_manifest(root, manifest, check_files=True, reconcile_disk=True)

    _png(root / "patches/val/tumor/p-val-tumor.png")
    _png(root / "patches/train/normal/extra.png")
    with pytest.raises(DataContractError, match="disk inventory mismatch"):
        validate_manifest(root, manifest, check_files=True, reconcile_disk=True)


def test_png_decoder_returns_canonical_rgb_and_rejects_wrong_shape(tmp_path: Path) -> None:
    valid = tmp_path / "valid.png"
    invalid = tmp_path / "invalid.png"
    _png(valid, value=23)
    Image.fromarray(np.zeros((255, 256, 3), dtype=np.uint8), mode="RGB").save(invalid)

    decoded = decode_png_rgb(valid.read_bytes())

    assert decoded.shape == (256, 256, 3)
    assert decoded.dtype == np.uint8
    assert decoded.flags.c_contiguous
    assert int(decoded[0, 0, 0]) == 23
    with pytest.raises(DataContractError, match="shape"):
        decode_png_rgb(invalid.read_bytes())


def test_patient_mapping_requires_complete_provenance_and_patient_isolation(tmp_path: Path) -> None:
    root, manifest = _valid_root(tmp_path)
    bundle = validate_manifest(root, manifest, check_files=True, reconcile_disk=True)
    mapping = root / "patient_mapping.csv"
    mapping.write_text(
        "slide_id,patient_id,provenance\n"
        "slide-train-normal,patient-a,registry-v1\n"
        "slide-val-tumor,patient-b,registry-v1\n",
        encoding="utf-8",
    )

    evidence = validate_patient_mapping(bundle, mapping)

    assert evidence["status"] == "structure_validated"
    assert evidence["provenance_reliability"] == "not_assessed"
    assert evidence["mapping_coverage"] == 2
    assert evidence["slide_count"] == 2
    assert evidence["patient_count"] == 2
    assert evidence["mapping_sha256"].startswith("sha256:")
    mapping.write_text(
        "slide_id,patient_id,provenance\n"
        "slide-train-normal,patient-a,registry-v1\n"
        "slide-val-tumor,patient-a,registry-v1\n",
        encoding="utf-8",
    )
    with pytest.raises(DataContractError, match="patient identity crosses splits"):
        validate_patient_mapping(bundle, mapping)


def test_dataloader_uses_hash_order_once_without_drop_or_repeat(tmp_path: Path) -> None:
    root, manifest = _valid_root(tmp_path)
    bundle = validate_manifest(root, manifest, check_files=True, reconcile_disk=True)
    dataset = PatchDataset(bundle, "train")
    loader = build_dataloader(
        dataset,
        batch_size=32,
        seed=1729,
        epoch=0,
        num_workers=0,
    )

    batches = list(loader)

    assert len(batches) == 1
    assert loader.drop_last is False
    assert batches[0]["rgb"].shape == (1, 3, 256, 256)
    assert str(batches[0]["rgb"].dtype) == "torch.uint8"
    assert batches[0]["patch_id"] == ["p-train-normal"]


class _CountingDataset:
    def __init__(self, row_count: int) -> None:
        self.rows = tuple(
            SimpleNamespace(patch_id=f"patch-{index:06d}") for index in range(row_count)
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return {"patch_id": self.rows[index].patch_id}


@pytest.mark.parametrize(
    ("row_count", "expected_batch_count"),
    [(79_570, 4_974), (18_171, 1_136)],
)
def test_batch16_retains_every_train_and_validation_row_exactly_once(
    row_count: int, expected_batch_count: int
) -> None:
    dataset = _CountingDataset(row_count)
    loader = build_dataloader(
        dataset,  # type: ignore[arg-type]
        batch_size=16,
        seed=1729,
        epoch=0,
        num_workers=0,
    )

    observed = [patch_id for batch in loader for patch_id in batch["patch_id"]]

    assert loader.batch_size == 16
    assert loader.drop_last is False
    assert len(loader) == expected_batch_count
    assert len(observed) == row_count
    assert len(set(observed)) == row_count
    assert set(observed) == {row.patch_id for row in dataset.rows}
