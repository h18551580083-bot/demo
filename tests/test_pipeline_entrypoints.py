from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from cg_pipeline.config import load_experiment_config
from cg_pipeline.pipeline import (
    Phase0BlockedError,
    run_dry_run,
    run_formal_training,
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


def test_repository_formal_and_dry_configs_are_machine_validated() -> None:
    formal = load_experiment_config(REPOSITORY / "configs" / "phase1_baseline.toml")
    dry = load_experiment_config(REPOSITORY / "configs" / "phase0_dry_run.toml")

    assert formal.execution_kind == "formal_train"
    assert dry.execution_kind == "dry_run"
    assert formal.execution["allow_test"] is False
    assert dry.execution["allow_test"] is False
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
    assert report["formal_experiment"] is False
    assert report["test_split_accessed"] is False
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


def test_preflight_reports_only_release_and_patient_mapping_blockers(tmp_path: Path) -> None:
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

    report = run_preflight(
        formal_config,
        data_root=data_root,
        release_path=REPOSITORY / "configs" / "phase0_release.json",
        output_path=report_path,
        patient_mapping_path=None,
    )

    assert report["status"] == "FAIL"
    assert report["passed_gates"] == [
        "configuration",
        "manifest_and_disk",
        "slide_id_isolation",
        "fixed_frontend",
        "morlet_spectral_coverage",
        "optimizer_ownership",
        "precision_and_determinism",
        "test_access_disabled",
    ]
    assert report["blocking_gates"] == ["patient_level_isolation", "phase0_release"]
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    with pytest.raises(Phase0BlockedError, match="preflight"):
        run_formal_training(
            formal_config,
            data_root=data_root,
            release_path=REPOSITORY / "configs" / "phase0_release.json",
            patient_mapping_path=None,
        )
    assert not (tmp_path / "artifacts" / "formal_runs").exists()


def test_patient_gate_requires_actual_hash_bound_provenance_approval(tmp_path: Path) -> None:
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
    mapping = tmp_path / "patient-mapping.csv"
    mapping.write_text(
        "slide_id,patient_id,provenance\n"
        "dry-slide-train-normal,p0,registry-v1\n"
        "dry-slide-train-tumor,p1,registry-v1\n"
        "dry-slide-val-normal,p2,registry-v1\n"
        "dry-slide-val-tumor,p3,registry-v1\n",
        encoding="utf-8",
    )
    mapping_hash = "sha256:" + hashlib.sha256(mapping.read_bytes()).hexdigest()
    source_hash = "sha256:" + hashlib.sha256(nested_manifest.read_bytes()).hexdigest()
    formal_config = tmp_path / "formal.toml"
    formal_config.write_text(
        (REPOSITORY / "configs" / "phase1_baseline.toml")
        .read_text(encoding="utf-8")
        .replace('patient_mapping_evidence = "not_available"', f'patient_mapping_evidence = "{mapping_hash}"'),
        encoding="utf-8",
    )
    approval_document = {
        "schema": "patient-mapping-provenance-approval-v1",
        "mapping_sha256": mapping_hash,
        "source_manifest_sha256": source_hash,
        "approved_by": "test-approver",
        "approved_at": "2026-08-03T00:00:00Z",
        "provenance_reliability_approved": True,
    }
    approval_artifact = tmp_path / "mapping-approval.json"
    approval_artifact.write_text(json.dumps(approval_document), encoding="utf-8")
    approval_hash = "sha256:" + hashlib.sha256(approval_artifact.read_bytes()).hexdigest()
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "schema": "phase0-release-v1",
                "phase0_closed": True,
                "formal_training_authorized": True,
                "external_blockers": [],
                "minimum_external_input": "satisfied by approved synthetic test fixture",
                "patient_mapping_approval": {
                    "schema": "patient-mapping-approval-v1",
                    "mapping_sha256": mapping_hash,
                    "source_manifest_sha256": source_hash,
                    "approval_evidence_sha256": approval_hash,
                    "approved_by": "test-approver",
                    "approved_at": "2026-08-03T00:00:00Z",
                    "provenance_reliability_approved": True,
                },
                "test_access_authorized": False,
            }
        ),
        encoding="utf-8",
    )

    without_artifact = run_preflight(
        formal_config,
        data_root=data_root,
        release_path=release,
        output_path=tmp_path / "without-artifact.json",
        patient_mapping_path=mapping,
    )
    assert without_artifact["blocking_gates"] == ["patient_level_isolation", "phase0_release"]
    passed = run_preflight(
        formal_config,
        data_root=data_root,
        release_path=release,
        output_path=tmp_path / "approved.json",
        patient_mapping_path=mapping,
        patient_mapping_approval_path=approval_artifact,
    )
    assert passed["status"] == "PASS"
    assert passed["patient_mapping"]["status"] == "validated_reliable"
    approval_document["approved_by"] = "tampered"
    approval_artifact.write_text(json.dumps(approval_document), encoding="utf-8")
    tampered = run_preflight(
        formal_config,
        data_root=data_root,
        release_path=release,
        output_path=tmp_path / "tampered.json",
        patient_mapping_path=mapping,
        patient_mapping_approval_path=approval_artifact,
    )
    assert tampered["blocking_gates"] == ["patient_level_isolation", "phase0_release"]
