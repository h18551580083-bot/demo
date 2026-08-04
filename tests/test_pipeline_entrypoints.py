from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cg_pipeline.config import load_experiment_config
from cg_pipeline.pipeline import Phase0BlockedError, run_dry_run, run_preflight

REPOSITORY = Path(__file__).resolve().parents[1]


def _config_copy(source: str, target: Path) -> Path:
    target.write_text(
        (REPOSITORY / "configs" / source).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return target


def test_repository_formal_and_dry_configs_are_machine_validated() -> None:
    formal = load_experiment_config(REPOSITORY / "configs" / "phase1_baseline.toml")
    dry = load_experiment_config(REPOSITORY / "configs" / "phase0_dry_run.toml")
    historical_release = json.loads(
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
    assert historical_release["release_id"] == "phase1-training-b32-v2"
    assert historical_release["config_hash"] == formal.sha256
    assert formal.sha256 != dry.sha256


def test_complete_synthetic_dry_run_is_repeatable_and_covers_failure(tmp_path: Path) -> None:
    config_path = _config_copy("phase0_dry_run.toml", tmp_path / "dry.toml")

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
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == report


def test_preflight_rejects_legacy_release_without_bound_identities(tmp_path: Path) -> None:
    dry_config = _config_copy("phase0_dry_run.toml", tmp_path / "dry.toml")
    run_dry_run(dry_config, workspace_root=tmp_path)
    data_root = tmp_path / "artifacts" / "phase0_dry_run_v1" / "synthetic_package"
    nested_metadata = data_root / "cam16_class_quota" / "metadata"
    nested_metadata.mkdir(parents=True)
    shutil.copy2(
        data_root / "metadata" / "training_manifest.csv",
        nested_metadata / "training_manifest.csv",
    )
    formal_config = _config_copy("phase1_baseline.toml", tmp_path / "formal.toml")
    legacy_release = tmp_path / "legacy-release.json"
    shutil.copy2(
        REPOSITORY / "configs" / "phase1_training_release_b32_v2.json",
        legacy_release,
    )

    with pytest.raises(Phase0BlockedError, match="phase1-training-release-v2"):
        run_preflight(
            formal_config,
            data_root=data_root,
            release_path=legacy_release,
            output_path=tmp_path / "must-not-exist.json",
        )
