from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import cg_pipeline.pipeline as pipeline_module
from cg_pipeline.__main__ import main
from cg_pipeline.config import load_experiment_config
from cg_pipeline.data import expected_batch_count, validate_manifest
from cg_pipeline.pipeline import (
    Phase0BlockedError,
    run_dry_run,
    run_formal_training,
    run_preflight,
)

REPOSITORY = Path(__file__).resolve().parents[1]
RELEASE_ID = "phase1-training-b32-v3"
RELEASE_PATH = "configs/phase1_training_release_b32_v3.json"
RELEASE_COMMIT_ALLOWED_PATHS = [
    RELEASE_PATH,
    "docs/DECISIONS.md",
    "docs/PHASE1_TRAINING_RUNBOOK.md",
]


def _report_path(repository: Path) -> Path:
    return repository / "artifacts" / "preflight" / RELEASE_ID / "preflight.json"


def _recompute_report_identity(report: dict[str, object]) -> str:
    material = {key: value for key, value in report.items() if key != "report_identity"}
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _synthetic_data(tmp_path: Path) -> Path:
    config_path = tmp_path / "dry.toml"
    config_path.write_text(
        (REPOSITORY / "configs" / "phase0_dry_run.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    run_dry_run(config_path, workspace_root=tmp_path)
    data_root = tmp_path / "artifacts" / "phase0_dry_run_v1" / "synthetic_package"
    nested = data_root / "cam16_class_quota" / "metadata"
    nested.mkdir(parents=True)
    shutil.copy2(data_root / "metadata" / "training_manifest.csv", nested)
    return data_root


def _release_document(
    config_path: Path, data_root: Path, *, formal_code_commit: str
) -> dict[str, object]:
    config = load_experiment_config(config_path)
    manifest = data_root / "cam16_class_quota" / "metadata" / "training_manifest.csv"
    bundle = validate_manifest(data_root, manifest, check_files=True, reconcile_disk=True)
    batch_size = int(config.training["batch_size"])
    train_batches = expected_batch_count(
        bundle.split_counts["train"], batch_size, drop_last=False
    )
    return {
        "schema": "phase1-training-release-v2",
        "release_id": RELEASE_ID,
        "supersedes_release_id": "phase1-training-b32-v2",
        "phase0_release_tag": "phase0-closed-v1",
        "release_id_role": "release-governance-identity",
        "annotated_tag": RELEASE_ID,
        "formal_code_commit": formal_code_commit,
        "release_commit_allowed_paths": RELEASE_COMMIT_ALLOWED_PATHS,
        "run_id": config.execution["run_id"],
        "run_id_role": "unchanged-training-config-identity",
        "config_hash": config.sha256,
        "normalized_config_sha256": config.sha256,
        "manifest_relpath": config.data["manifest_relpath"],
        "manifest_hash_algorithm": "sha256",
        "source_manifest_hash_rule": "raw-file-bytes-v1",
        "effective_split_hash_rule": "cg/cam16-eval-manifest/v1",
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "effective_split_hashes": {
            "train": bundle.effective_split_hashes["train"],
            "val": bundle.effective_split_hashes["val"],
        },
        "batch_size": batch_size,
        "drop_last": False,
        "expected_train_rows": bundle.split_counts["train"],
        "expected_train_batch_count": train_batches,
        "maximum_optimizer_updates": train_batches * int(config.training["max_epochs"]),
        "expected_validation_rows": bundle.split_counts["val"],
        "expected_validation_batch_count": expected_batch_count(
            bundle.split_counts["val"], batch_size, drop_last=False
        ),
        "phase0_closed": True,
        "formal_training_authorized": True,
        "external_blockers": [],
        "patient_level_isolation": "not_evaluated",
        "patient_level_claim_allowed": False,
        "test_access_authorized": False,
    }


def _released_repository(tmp_path: Path, data_root: Path) -> tuple[Path, Path, Path]:
    repository = tmp_path / "released-repository"
    (repository / "configs").mkdir(parents=True)
    (repository / "src" / "cg_pipeline").mkdir(parents=True)
    (repository / "docs").mkdir()
    config_path = repository / "configs" / "phase1_baseline.toml"
    shutil.copy2(REPOSITORY / "configs" / "phase1_baseline.toml", config_path)
    (repository / "src" / "cg_pipeline" / "marker.py").write_text(
        'IDENTITY = "synthetic-code-commit"\n', encoding="utf-8"
    )
    (repository / "docs" / "DECISIONS.md").write_text("code baseline\n", encoding="utf-8")
    (repository / "docs" / "PHASE1_TRAINING_RUNBOOK.md").write_text(
        "code baseline\n", encoding="utf-8"
    )
    _git(repository, "init")
    _git(repository, "config", "user.name", "Release Test")
    _git(repository, "config", "user.email", "release-test@example.invalid")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "code commit")
    code_commit = _git(repository, "rev-parse", "HEAD")

    release_path = repository / RELEASE_PATH
    release_path.write_text(
        json.dumps(
            _release_document(config_path, data_root, formal_code_commit=code_commit),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (repository / "docs" / "DECISIONS.md").open("a", encoding="utf-8") as handle:
        handle.write("release decision\n")
    with (repository / "docs" / "PHASE1_TRAINING_RUNBOOK.md").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("release runbook\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "release commit")
    _git(repository, "tag", "-a", RELEASE_ID, "-m", "synthetic formal release")
    return repository, config_path, release_path


def _retag_annotated(repository: Path) -> None:
    _git(repository, "tag", "-d", RELEASE_ID)
    _git(repository, "tag", "-a", RELEASE_ID, "-m", "synthetic formal release")


def _amend_release_document(repository: Path, release_path: Path, **changes: object) -> None:
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release.update(changes)
    release_path.write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _git(repository, "add", RELEASE_PATH)
    _git(repository, "commit", "--amend", "--no-edit")
    _retag_annotated(repository)


def test_preflight_accepts_exact_two_commit_release_identity(tmp_path: Path) -> None:
    data_root = _synthetic_data(tmp_path)
    repository, config_path, release_path = _released_repository(tmp_path, data_root)
    report_path = _report_path(repository)

    report = run_preflight(
        config_path,
        data_root=data_root,
        release_path=release_path,
        output_path=report_path,
    )

    assert report["status"] == "PASS"
    assert report["release_id"] == RELEASE_ID
    assert report["release_identity"]["annotated_tag"] == RELEASE_ID
    assert report["release_identity"]["release_commit"] == _git(
        repository, "rev-parse", "HEAD"
    )
    assert report["release_identity"]["formal_code_commit"] == _git(
        repository, "rev-parse", "HEAD^"
    )
    assert report["release_identity"]["release_commit_allowed_paths"] == (
        RELEASE_COMMIT_ALLOWED_PATHS
    )
    assert report["manifest_identity"] == {
        "manifest_relpath": "cam16_class_quota/metadata/training_manifest.csv",
        "manifest_hash_algorithm": "sha256",
        "source_manifest_hash_rule": "raw-file-bytes-v1",
        "effective_split_hash_rule": "cg/cam16-eval-manifest/v1",
        "source_manifest_sha256": report["source_manifest_sha256"],
        "effective_split_hashes": {
            "train": report["effective_split_hashes"]["train"],
            "val": report["effective_split_hashes"]["val"],
        },
    }
    assert set(report["effective_split_hashes"]) == {"train", "val"}
    assert "test" not in report["manifest_identity"]["effective_split_hashes"]


def test_train_rejects_tampered_preflight_report_before_training(tmp_path: Path) -> None:
    data_root = _synthetic_data(tmp_path)
    repository, config_path, release_path = _released_repository(tmp_path, data_root)
    report_path = _report_path(repository)
    run_preflight(
        config_path,
        data_root=data_root,
        release_path=release_path,
        output_path=report_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["release_id"] = "phase1-training-b32-v2"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(Phase0BlockedError, match="preflight report identity"):
        run_formal_training(
            config_path,
            data_root=data_root,
            release_path=release_path,
            preflight_report_path=report_path,
        )

    assert not (repository / "artifacts" / "formal_runs").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_parent", "parent does not match"),
        ("lightweight_tag", "annotated tag"),
        ("unauthorized_path", "outside the approved whitelist"),
        ("untracked_code", "untracked or missing Python code"),
        ("wrong_release_id", "release_id"),
        ("switched_commit", "parent does not match"),
    ],
)
def test_preflight_rejects_invalid_git_release_identity(
    tmp_path: Path, mutation: str, message: str
) -> None:
    data_root = _synthetic_data(tmp_path)
    repository, config_path, release_path = _released_repository(tmp_path, data_root)

    if mutation == "wrong_parent":
        _amend_release_document(repository, release_path, formal_code_commit="0" * 40)
    elif mutation == "lightweight_tag":
        _git(repository, "tag", "-d", RELEASE_ID)
        _git(repository, "tag", RELEASE_ID)
    elif mutation == "unauthorized_path":
        unexpected = repository / "src" / "cg_pipeline" / "unauthorized.py"
        unexpected.write_text("UNAUTHORIZED = True\n", encoding="utf-8")
        _git(repository, "add", unexpected.relative_to(repository).as_posix())
        _git(repository, "commit", "--amend", "--no-edit")
        _retag_annotated(repository)
    elif mutation == "untracked_code":
        (repository / "src" / "cg_pipeline" / "injected.py").write_text(
            "INJECTED = True\n", encoding="utf-8"
        )
    elif mutation == "wrong_release_id":
        _amend_release_document(
            repository, release_path, release_id="phase1-training-b32-v2"
        )
    elif mutation == "switched_commit":
        _git(repository, "commit", "--allow-empty", "-m", "unexpected commit after release")
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(mutation)

    with pytest.raises(Phase0BlockedError, match=message):
        run_preflight(
            config_path,
            data_root=data_root,
            release_path=release_path,
            output_path=repository / "rejected.json",
        )


def test_preflight_rejects_merge_release_commit(tmp_path: Path) -> None:
    data_root = _synthetic_data(tmp_path)
    repository, config_path, release_path = _released_repository(tmp_path, data_root)
    primary_branch = _git(repository, "branch", "--show-current")
    code_commit = _git(repository, "rev-parse", "HEAD^")
    _git(repository, "switch", "-c", "side", code_commit)
    (repository / "side.txt").write_text("side parent\n", encoding="utf-8")
    _git(repository, "add", "side.txt")
    _git(repository, "commit", "-m", "side parent")
    _git(repository, "switch", primary_branch)
    _git(repository, "merge", "--no-ff", "side", "-m", "invalid merge release")
    _retag_annotated(repository)

    with pytest.raises(Phase0BlockedError, match="exactly one parent"):
        run_preflight(
            config_path,
            data_root=data_root,
            release_path=release_path,
            output_path=repository / "rejected.json",
        )


def test_preflight_rejects_replaced_source_manifest(tmp_path: Path) -> None:
    data_root = _synthetic_data(tmp_path)
    repository, config_path, release_path = _released_repository(tmp_path, data_root)
    manifest = data_root / "cam16_class_quota" / "metadata" / "training_manifest.csv"
    raw = manifest.read_bytes()
    replacement = raw.replace(b"\r\n", b"\n")
    assert replacement != raw
    manifest.write_bytes(replacement)

    with pytest.raises(Phase0BlockedError, match="source manifest identity"):
        run_preflight(
            config_path,
            data_root=data_root,
            release_path=release_path,
            output_path=repository / "rejected.json",
        )


def test_preflight_rejects_changed_effective_train_identity(tmp_path: Path) -> None:
    data_root = _synthetic_data(tmp_path)
    repository, config_path, release_path = _released_repository(tmp_path, data_root)
    manifest = data_root / "cam16_class_quota" / "metadata" / "training_manifest.csv"
    text = manifest.read_text(encoding="utf-8")
    changed = text.replace("dry-train-normal", "dry-train-replaced", 1)
    assert changed != text
    manifest.write_text(changed, encoding="utf-8", newline="")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["source_manifest_sha256"] = "sha256:" + hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    release_path.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(Phase0BlockedError, match="effective split identity"):
        run_preflight(
            config_path,
            data_root=data_root,
            release_path=release_path,
            output_path=repository / "rejected.json",
        )


def test_cli_and_api_report_same_release_id_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = _synthetic_data(tmp_path)
    repository, config_path, release_path = _released_repository(tmp_path, data_root)
    _amend_release_document(
        repository, release_path, release_id="phase1-training-b32-v2"
    )

    with pytest.raises(Phase0BlockedError, match="release_id"):
        run_preflight(
            config_path,
            data_root=data_root,
            release_path=release_path,
            output_path=repository / "api-rejected.json",
        )

    exit_code = main(
        [
            "preflight",
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--release",
            str(release_path),
            "--output",
            str(repository / "cli-rejected.json"),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "BLOCKED:" in captured.err
    assert "release_id" in captured.err


def test_cli_and_api_preflight_reports_have_consistent_contents(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = _synthetic_data(tmp_path)
    repository, config_path, release_path = _released_repository(tmp_path, data_root)
    report_path = _report_path(repository)
    api_report = run_preflight(
        config_path,
        data_root=data_root,
        release_path=release_path,
        output_path=report_path,
    )
    report_path.unlink()

    exit_code = main(
        [
            "preflight",
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--release",
            str(release_path),
            "--output",
            str(report_path),
        ]
    )
    captured = capsys.readouterr()
    cli_report = json.loads(report_path.read_text(encoding="utf-8"))
    for report in (api_report, cli_report):
        report.pop("created_at")
        report.pop("report_identity")

    assert exit_code == 0
    assert json.loads(captured.out)["status"] == "PASS"
    assert api_report == cli_report


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("patient_level_isolation", "verified", "patient-level isolation"),
        ("patient_level_claim_allowed", True, "patient-level claim"),
        (
            "config_hash",
            "sha256:0653ae0003dac9062b73749e879a9a541a3f9dae18b034bdc1632f8410910e75",
            "config hash",
        ),
    ],
)
def test_preflight_rejects_unsafe_claim_or_stale_config_through_public_api(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    data_root = _synthetic_data(tmp_path)
    repository, config_path, release_path = _released_repository(tmp_path, data_root)
    _amend_release_document(repository, release_path, **{field: value})

    with pytest.raises(Phase0BlockedError, match=message):
        run_preflight(
            config_path,
            data_root=data_root,
            release_path=release_path,
            output_path=repository / "rejected.json",
        )


def test_train_rejects_preflight_report_from_previous_release_commit(tmp_path: Path) -> None:
    data_root = _synthetic_data(tmp_path)
    repository, config_path, release_path = _released_repository(tmp_path, data_root)
    report_path = _report_path(repository)
    run_preflight(
        config_path,
        data_root=data_root,
        release_path=release_path,
        output_path=report_path,
    )
    with (repository / "docs" / "DECISIONS.md").open("a", encoding="utf-8") as handle:
        handle.write("amended release evidence\n")
    _git(repository, "add", "docs/DECISIONS.md")
    _git(repository, "commit", "--amend", "--no-edit")
    _retag_annotated(repository)

    with pytest.raises(Phase0BlockedError, match="current formal identities"):
        run_formal_training(
            config_path,
            data_root=data_root,
            release_path=release_path,
            preflight_report_path=report_path,
        )


def test_train_cli_and_api_reject_same_tampered_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = _synthetic_data(tmp_path)
    repository, config_path, release_path = _released_repository(tmp_path, data_root)
    report_path = _report_path(repository)
    run_preflight(
        config_path,
        data_root=data_root,
        release_path=release_path,
        output_path=report_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["config_hash"] = "sha256:" + "0" * 64
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(Phase0BlockedError, match="preflight report identity"):
        run_formal_training(
            config_path,
            data_root=data_root,
            release_path=release_path,
            preflight_report_path=report_path,
        )

    exit_code = main(
        [
            "train",
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--release",
            str(release_path),
            "--preflight-report",
            str(report_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "BLOCKED:" in captured.err
    assert "preflight report identity" in captured.err


def test_train_rejects_governance_tamper_even_if_report_hash_is_recomputed(
    tmp_path: Path,
) -> None:
    data_root = _synthetic_data(tmp_path)
    repository, config_path, release_path = _released_repository(tmp_path, data_root)
    report_path = _report_path(repository)
    run_preflight(
        config_path,
        data_root=data_root,
        release_path=release_path,
        output_path=report_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["patient_level_isolation"] = "verified"
    report["report_identity"] = _recompute_report_identity(report)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(Phase0BlockedError, match="patient_level_isolation"):
        run_formal_training(
            config_path,
            data_root=data_root,
            release_path=release_path,
            preflight_report_path=report_path,
        )


def test_train_rejects_injected_internal_evidence_even_if_hash_is_recomputed(
    tmp_path: Path,
) -> None:
    data_root = _synthetic_data(tmp_path)
    repository, config_path, release_path = _released_repository(tmp_path, data_root)
    report_path = _report_path(repository)
    run_preflight(
        config_path,
        data_root=data_root,
        release_path=release_path,
        output_path=report_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["fixed_frontend_identity"] = {"forged": "sha256:" + "0" * 64}
    report["report_identity"] = _recompute_report_identity(report)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(Phase0BlockedError, match="frozen schema"):
        run_formal_training(
            config_path,
            data_root=data_root,
            release_path=release_path,
            preflight_report_path=report_path,
        )


def test_train_treats_created_at_as_audit_only_without_expiry(tmp_path: Path) -> None:
    data_root = _synthetic_data(tmp_path)
    repository, config_path, release_path = _released_repository(tmp_path, data_root)
    report_path = _report_path(repository)
    run_preflight(
        config_path,
        data_root=data_root,
        release_path=release_path,
        output_path=report_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["created_at"] = "1900-01-01T00:00:00Z"
    report["report_identity"] = _recompute_report_identity(report)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_root = repository / "artifacts" / "formal_runs" / "phase1-cam16-baseline-b32-v2"
    output_root.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        run_formal_training(
            config_path,
            data_root=data_root,
            release_path=release_path,
            preflight_report_path=report_path,
        )


def test_preflight_api_and_cli_reject_existing_release_bound_report_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = _synthetic_data(tmp_path)
    repository, config_path, release_path = _released_repository(tmp_path, data_root)
    report_path = _report_path(repository)
    report_path.parent.mkdir(parents=True)
    report_path.write_text("old report\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_preflight(
            config_path,
            data_root=data_root,
            release_path=release_path,
            output_path=report_path,
        )
    exit_code = main(
        [
            "preflight",
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--release",
            str(release_path),
            "--output",
            str(report_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 3
    assert "FileExistsError" in captured.err


def test_train_api_and_cli_reject_existing_formal_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = _synthetic_data(tmp_path)
    repository, config_path, release_path = _released_repository(tmp_path, data_root)
    report_path = _report_path(repository)
    run_preflight(
        config_path,
        data_root=data_root,
        release_path=release_path,
        output_path=report_path,
    )
    output_root = repository / "artifacts" / "formal_runs" / "phase1-cam16-baseline-b32-v2"
    output_root.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        run_formal_training(
            config_path,
            data_root=data_root,
            release_path=release_path,
            preflight_report_path=report_path,
        )
    exit_code = main(
        [
            "train",
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--release",
            str(release_path),
            "--preflight-report",
            str(report_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 3
    assert "FileExistsError" in captured.err


def test_train_cli_and_api_consume_same_report_without_repeating_preflight(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = _synthetic_data(tmp_path)
    repository, config_path, release_path = _released_repository(tmp_path, data_root)
    report_path = _report_path(repository)
    run_preflight(
        config_path,
        data_root=data_root,
        release_path=release_path,
        output_path=report_path,
    )

    def fake_seed(*args: object, seed: int, **kwargs: object) -> dict[str, object]:
        return {
            "seed": seed,
            "best_epoch": 1,
            "best_validation_slide_auroc": 0.5,
            "epochs_completed": 1,
            "status": "complete",
        }

    monkeypatch.setattr(pipeline_module, "_run_formal_seed", fake_seed)
    monkeypatch.setattr(
        pipeline_module,
        "_perform_preflight",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("training repeated standalone preflight")
        ),
    )
    api_summary = run_formal_training(
        config_path,
        data_root=data_root,
        release_path=release_path,
        preflight_report_path=report_path,
    )
    output_root = repository / "artifacts" / "formal_runs" / "phase1-cam16-baseline-b32-v2"
    shutil.rmtree(output_root)

    exit_code = main(
        [
            "train",
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--release",
            str(release_path),
            "--preflight-report",
            str(report_path),
        ]
    )
    captured = capsys.readouterr()
    cli_summary = json.loads((output_root / "training_summary.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert json.loads(captured.out)["status"] == "PASS"
    assert api_summary == cli_summary
