from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from cg_pipeline.config import load_experiment_config
from cg_pipeline.data import expected_batch_count, validate_manifest
from cg_pipeline.pipeline import (
    Phase0BlockedError,
    _formal_datasets,
    run_dry_run,
    run_preflight,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def _config_copy(source: str, target: Path, *, kind: str, output_root: str) -> Path:
    text = (REPOSITORY / "configs" / source).read_text(encoding="utf-8")
    original_kind = "dry_run" if source == "phase0_dry_run.toml" else "formal_train"
    text = text.replace(f'kind = "{original_kind}"', f'kind = "{kind}"')
    del output_root
    target.write_text(text, encoding="utf-8")
    return target


def _synthetic_release(config_path: Path, data_root: Path, target: Path) -> Path:
    config = load_experiment_config(config_path)
    release = json.loads(
        (REPOSITORY / "configs" / "phase1_training_release_b32_v2.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = data_root / Path(*str(config.data["manifest_relpath"]).split("/"))
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        counts = Counter(row["split"] for row in csv.DictReader(handle))
    batch_size = int(config.training["batch_size"])
    release.update(
        {
            "release_id": "synthetic-phase1-training-release-test-only",
            "config_hash": config.sha256,
            "normalized_config_sha256": config.sha256,
            "run_id": config.execution["run_id"],
            "batch_size": batch_size,
            "expected_train_rows": counts["train"],
            "expected_train_batch_count": expected_batch_count(
                counts["train"], batch_size, drop_last=False
            ),
            "maximum_optimizer_updates": expected_batch_count(
                counts["train"], batch_size, drop_last=False
            )
            * int(config.training["max_epochs"]),
            "expected_validation_rows": counts["val"],
            "expected_validation_batch_count": expected_batch_count(
                counts["val"], batch_size, drop_last=False
            ),
        }
    )
    target.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    return target


def test_repository_formal_and_dry_configs_are_machine_validated() -> None:
    formal = load_experiment_config(REPOSITORY / "configs" / "phase1_baseline.toml")
    dry = load_experiment_config(REPOSITORY / "configs" / "phase0_dry_run.toml")
    release = json.loads(
        (REPOSITORY / "configs" / "phase1_training_release_b32_v2.json").read_text(
            encoding="utf-8"
        )
    )

    assert formal.execution_kind == "formal_train"
    assert dry.execution_kind == "dry_run"
    assert formal.execution["allow_test"] is False
    assert dry.execution["allow_test"] is False
    assert formal.training["batch_size"] == 32
    assert dry.training["batch_size"] == 32
    assert formal.execution["run_id"] == "phase1-cam16-baseline-b32-v2"
    assert formal.sha256 == (
        "sha256:e44768d80d7c1545138d7d5e1368de4ed53b7b07b71202e2c5bdee6efac7cf3b"
    )
    assert release["config_hash"] == formal.sha256
    assert release["normalized_config_sha256"] == formal.sha256
    assert release["expected_train_batch_count"] == 2_487
    assert release["maximum_optimizer_updates"] == 49_740
    assert release["expected_validation_batch_count"] == 568
    assert formal.sha256 != dry.sha256


def test_complete_synthetic_dry_run_is_repeatable_and_covers_failure(tmp_path: Path) -> None:
    config_path = _config_copy(
        "phase0_dry_run.toml",
        tmp_path / "dry.toml",
        kind="dry_run",
        output_root="dry-output",
    )

    report = run_dry_run(config_path, workspace_root=tmp_path)

    assert report["status"] == "PASS"
    assert report["schema"] == "phase0-dry-run-report-v1"
    assert report["formal_experiment"] is False
    assert report["test_split_accessed"] is False
    assert report["isolation_claim"] == "group_id/slide_id split isolation verified"
    assert report["patient_level_isolation"] == "not_evaluated"
    assert report["patient_level_claim_allowed"] is False
    assert all(report["steps"].values())
    assert report["fixed_frontend_unchanged"] is True
    assert report["repeatability"]["exact_match"] is True
    assert report["negative_control"]["detected"] is True
    assert report["negative_control"]["name"] == "duplicate_patch_id"
    assert report["checkpoint_restore"]["passed"] is True
    output = tmp_path / "artifacts" / "phase0_dry_run_v1"
    assert (output / "report.json").is_file()
    assert (output / "checkpoint" / "epoch-0000.pt").is_file()
    parsed = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert parsed == report


def test_preflight_marks_patient_isolation_not_applicable_and_validates_training_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dry_config = _config_copy(
        "phase0_dry_run.toml",
        tmp_path / "fixture-config.toml",
        kind="dry_run",
        output_root="fixture-output",
    )
    dry_report = run_dry_run(dry_config, workspace_root=tmp_path)
    assert dry_report["status"] == "PASS"
    formal_config = _config_copy(
        "phase1_baseline.toml",
        tmp_path / "formal.toml",
        kind="formal_train",
        output_root="formal-output",
    )
    data_root = tmp_path / "artifacts" / "phase0_dry_run_v1" / "synthetic_package"
    nested_metadata = data_root / "cam16_class_quota" / "metadata"
    nested_metadata.mkdir(parents=True)
    shutil.copy2(data_root / "metadata" / "training_manifest.csv", nested_metadata)
    report_path = tmp_path / "preflight.json"
    release_path = _synthetic_release(
        formal_config, data_root, tmp_path / "training-release.json"
    )

    def _unexpected_training_step(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("preflight must not call train_one_step")

    monkeypatch.setattr("cg_pipeline.pipeline.train_one_step", _unexpected_training_step)

    report = run_preflight(
        formal_config,
        data_root=data_root,
        release_path=release_path,
        output_path=report_path,
    )

    assert report["status"] == "PASS"
    assert report["schema"] == "phase1-training-preflight-report-v1"
    assert report["passed_gates"] == [
        "configuration",
        "manifest_and_disk",
        "slide_id_isolation",
        "fixed_frontend",
        "morlet_spectral_coverage",
        "optimizer_ownership",
        "precision_and_determinism",
        "test_access_disabled",
        "phase1_training_release",
    ]
    assert report["not_applicable_gates"] == ["patient_level_isolation"]
    assert report["blocking_gates"] == []
    assert report["patient_level_isolation"] == "not_evaluated"
    assert report["patient_level_claim_allowed"] is False
    assert report["isolation_claim"] == "group_id/slide_id split isolation verified"
    assert report["normalized_config_sha256"] == load_experiment_config(
        formal_config
    ).sha256
    assert report["release_id"] == "synthetic-phase1-training-release-test-only"
    assert report["batch_contract"] == {
        "batch_size": 32,
        "drop_last": False,
        "train_rows": 2,
        "train_batch_count": 1,
        "maximum_optimizer_updates": 20,
        "validation_rows": 2,
        "validation_batch_count": 1,
    }
    assert report["training_started"] is False
    assert report["test_split_accessed"] is False
    bundle = validate_manifest(
        data_root,
        nested_metadata / "training_manifest.csv",
        check_files=True,
        reconcile_disk=True,
    )
    train_dataset, validation_dataset = _formal_datasets(bundle)
    assert train_dataset.split == "train"
    assert validation_dataset.split == "val"
    assert {train_dataset.split, validation_dataset.split} == {"train", "val"}
    assert report["patient_mapping"] == {
        "status": "not_evaluated",
        "reason": "patient identity is outside the CAM16 Phase 1 claim scope",
    }
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert not (tmp_path / "artifacts" / "formal_runs").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("patient_level_claim_allowed", True, "patient-level claim"),
        ("patient_level_isolation", "verified", "patient-level isolation"),
    ],
)
def test_preflight_rejects_any_patient_level_safety_claim(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    dry_config = _config_copy(
        "phase0_dry_run.toml", tmp_path / "dry.toml", kind="dry_run", output_root="ignored"
    )
    run_dry_run(dry_config, workspace_root=tmp_path)
    data_root = tmp_path / "artifacts" / "phase0_dry_run_v1" / "synthetic_package"
    nested_metadata = data_root / "cam16_class_quota" / "metadata"
    nested_metadata.mkdir(parents=True)
    source_manifest = data_root / "metadata" / "training_manifest.csv"
    nested_manifest = nested_metadata / "training_manifest.csv"
    shutil.copy2(source_manifest, nested_manifest)
    formal_config = tmp_path / "formal.toml"
    formal_config.write_text(
        (REPOSITORY / "configs" / "phase1_baseline.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    release_path = _synthetic_release(
        formal_config, data_root, tmp_path / "training-release.json"
    )
    release_document = json.loads(release_path.read_text(encoding="utf-8"))
    release_document[field] = value
    release = tmp_path / "release.json"
    release.write_text(json.dumps(release_document), encoding="utf-8")

    with pytest.raises(Phase0BlockedError, match=message):
        run_preflight(
            formal_config,
            data_root=data_root,
            release_path=release,
            output_path=tmp_path / "rejected.json",
        )


def test_preflight_rejects_stale_config_hash_before_training(tmp_path: Path) -> None:
    dry_config = _config_copy(
        "phase0_dry_run.toml", tmp_path / "dry.toml", kind="dry_run", output_root="ignored"
    )
    run_dry_run(dry_config, workspace_root=tmp_path)
    data_root = tmp_path / "artifacts" / "phase0_dry_run_v1" / "synthetic_package"
    nested_metadata = data_root / "cam16_class_quota" / "metadata"
    nested_metadata.mkdir(parents=True)
    shutil.copy2(
        data_root / "metadata" / "training_manifest.csv",
        nested_metadata / "training_manifest.csv",
    )
    formal_config = _config_copy(
        "phase1_baseline.toml",
        tmp_path / "formal.toml",
        kind="formal_train",
        output_root="ignored",
    )
    release_path = _synthetic_release(
        formal_config, data_root, tmp_path / "stale-release.json"
    )
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["config_hash"] = (
        "sha256:0653ae0003dac9062b73749e879a9a541a3f9dae18b034bdc1632f8410910e75"
    )
    release_path.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(Phase0BlockedError, match="config hash"):
        run_preflight(
            formal_config,
            data_root=data_root,
            release_path=release_path,
            output_path=tmp_path / "must-not-exist.json",
        )

    assert not (tmp_path / "artifacts" / "formal_runs").exists()
