from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cg_pipeline.config import load_experiment_config
from cg_pipeline.pipeline import Phase0BlockedError, run_dry_run, run_preflight

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


def test_preflight_marks_patient_isolation_not_applicable_and_releases_phase0(
    tmp_path: Path,
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

    report = run_preflight(
        formal_config,
        data_root=data_root,
        release_path=REPOSITORY / "configs" / "phase0_release.json",
        output_path=report_path,
    )

    assert report["status"] == "PASS"
    assert report["passed_gates"] == [
        "configuration",
        "manifest_and_disk",
        "slide_id_isolation",
        "fixed_frontend",
        "morlet_spectral_coverage",
        "optimizer_ownership",
        "precision_and_determinism",
        "test_access_disabled",
        "phase0_release",
    ]
    assert report["not_applicable_gates"] == ["patient_level_isolation"]
    assert report["blocking_gates"] == []
    assert report["patient_level_isolation"] == "not_evaluated"
    assert report["patient_level_claim_allowed"] is False
    assert report["isolation_claim"] == "group_id/slide_id split isolation verified"
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
    release_document = json.loads(
        (REPOSITORY / "configs" / "phase0_release.json").read_text(encoding="utf-8")
    )
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
