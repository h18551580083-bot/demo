from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import cg_pipeline.pipeline as pipeline_module
from cg_pipeline.config import ConfigError, load_experiment_config
from cg_pipeline.pipeline import Phase0BlockedError, run_formal_training, run_preflight

REPOSITORY = Path(__file__).resolve().parents[1]


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    (repository / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=Pipeline Test",
        "-c",
        "user.email=pipeline@example.invalid",
        "commit",
        "-m",
        "base",
    )
    return repository


def _formal_config(
    tmp_path: Path, *, manifest_relpath: str = "package/training_manifest.csv"
) -> Path:
    text = (REPOSITORY / "configs" / "phase1_baseline.toml").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("manifest_relpath = "):
            text = text.replace(line, f'manifest_relpath = "{manifest_relpath}"')
            break
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _data_root(tmp_path: Path, *, conflict: bool = False) -> Path:
    root = tmp_path / "data"
    rows: list[dict[str, str]] = []
    for split, label_name, label in (
        ("train", "normal", 0),
        ("train", "tumor", 1),
        ("val", "normal", 0),
        ("val", "tumor", 1),
        ("test", "normal", 0),
    ):
        patch_id = f"{split}-{label_name}"
        relative = f"patches/{split}/{label_name}/{patch_id}.png"
        if split != "test":
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((256, 256, 3), label * 255, dtype=np.uint8)).save(path)
        rows.append(
            {
                "patch_id": patch_id,
                "patch_path": relative,
                "split": split,
                "slide_id": (
                    "shared"
                    if conflict and split != "test" and label_name == "normal"
                    else patch_id
                ),
                "label": str(label),
                "label_name": label_name,
                "patch_label": label_name,
                "slide_label": label_name,
            }
        )
    manifest = root / "package" / "training_manifest.csv"
    manifest.parent.mkdir(parents=True)
    with manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def _authorization(tmp_path: Path) -> Path:
    path = tmp_path / "authorization.json"
    path.write_text(
        json.dumps(
            {
                "schema": "formal-training-authorization-v1",
                "formal_training_authorized": True,
                "test_access_authorized": False,
                "external_blockers": [],
                "patient_level_isolation": "not_evaluated",
                "patient_level_claim_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def _stub_model_audits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "cg_pipeline.preflight._model_audits",
        lambda config, device: {
            "fixed_frontend_identity": {"status": "fixture"},
            "morlet_identity_audit": {"status": "PASS"},
            "morlet_spectral_coverage": {"status": "PASS"},
            "optimizer_ownership": {"all_electronic_exactly_once": True},
        },
    )


def test_config_edit_remains_legal_and_preflight_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _formal_config(tmp_path)
    data = _data_root(tmp_path)
    authorization = _authorization(tmp_path)
    report_path = tmp_path / "arbitrary" / "preflight.json"
    monkeypatch.setattr(pipeline_module.torch.cuda, "is_available", lambda: True)
    _stub_model_audits(monkeypatch)

    report = run_preflight(
        config,
        data_root=data,
        authorization_path=authorization,
        output_path=report_path,
    )

    assert (
        load_experiment_config(config).data["manifest_relpath"] == "package/training_manifest.csv"
    )
    assert report["status"] == "PASS"
    assert report["blocking_gates"] == []
    assert report_path.exists()
    assert report["schema"] == "formal-training-preflight-v1"
    assert "fixed_frontend_identity" in report


def test_dirty_git_worktree_does_not_block_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _git_repository(tmp_path)
    config = _formal_config(repository)
    data = _data_root(repository)
    authorization = _authorization(repository)
    (repository / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    assert "M tracked.txt" in _git(repository, "status", "--porcelain")
    monkeypatch.setattr(pipeline_module.torch.cuda, "is_available", lambda: True)
    _stub_model_audits(monkeypatch)

    report = run_preflight(
        config,
        data_root=data,
        authorization_path=authorization,
        output_path=repository / "preflight.json",
    )

    assert report["status"] == "PASS"


def test_no_annotated_tag_does_not_block_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _git_repository(tmp_path)
    assert _git(repository, "tag") == ""
    monkeypatch.setattr(pipeline_module.torch.cuda, "is_available", lambda: True)
    _stub_model_audits(monkeypatch)

    report = run_preflight(
        _formal_config(repository),
        data_root=_data_root(repository),
        authorization_path=_authorization(repository),
        output_path=repository / "preflight.json",
    )

    assert report["status"] == "PASS"


def test_train_validation_cross_split_conflict_still_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _formal_config(tmp_path)
    monkeypatch.setattr(pipeline_module.torch.cuda, "is_available", lambda: True)
    _stub_model_audits(monkeypatch)

    with pytest.raises(ValueError, match="crosses splits"):
        run_preflight(
            config,
            data_root=_data_root(tmp_path, conflict=True),
            authorization_path=_authorization(tmp_path),
            output_path=tmp_path / "preflight.json",
        )


def test_training_config_cannot_enable_test_access(tmp_path: Path) -> None:
    config = _formal_config(tmp_path)
    text = config.read_text(encoding="utf-8").replace("allow_test = false", "allow_test = true")
    config.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="test access"):
        load_experiment_config(config)


def test_cuda_unavailable_blocks_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_module.torch.cuda, "is_available", lambda: False)
    report = run_preflight(
        _formal_config(tmp_path),
        data_root=_data_root(tmp_path),
        authorization_path=_authorization(tmp_path),
        output_path=tmp_path / "preflight.json",
    )

    assert report["status"] == "FAIL"
    assert "configured_device" in report["blocking_gates"]


def test_preflight_report_allows_formal_training_to_reach_seed_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _git_repository(tmp_path)
    config = _formal_config(repository)
    data = _data_root(repository)
    authorization = _authorization(repository)
    report_path = repository / "preflight.json"
    monkeypatch.setattr(pipeline_module.torch.cuda, "is_available", lambda: True)
    _stub_model_audits(monkeypatch)
    run_preflight(
        config,
        data_root=data,
        authorization_path=authorization,
        output_path=report_path,
    )
    untracked = repository / "untracked-before-train.py"
    untracked.write_text("unrelated = True\n", encoding="utf-8")
    assert "?? untracked-before-train.py" in _git(repository, "status", "--porcelain")
    observed: list[int] = []

    def fake_seed(*args: object, seed: int, **kwargs: object) -> dict[str, object]:
        observed.append(seed)
        return {
            "seed": seed,
            "best_epoch": 0,
            "best_validation_slide_auroc": 0.5,
            "epochs_completed": 1,
            "status": "complete",
        }

    monkeypatch.setattr(pipeline_module, "run_formal_seed", fake_seed)
    summary = run_formal_training(
        config,
        data_root=data,
        authorization_path=authorization,
        preflight_report_path=report_path,
    )

    assert observed == [1729, 3407, 7919]
    assert summary["phase1_training_preflight"] == "PASS"
    assert summary["test_split_accessed"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (("status", "FAIL"), ("test_split_accessed", True)),
)
def test_formal_training_rejects_failed_or_test_accessing_preflight(
    field: str, value: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _formal_config(tmp_path)
    data = _data_root(tmp_path)
    authorization = _authorization(tmp_path)
    monkeypatch.setattr(pipeline_module.torch.cuda, "is_available", lambda: True)
    _stub_model_audits(monkeypatch)
    report_path = tmp_path / f"{field}.json"
    run_preflight(
        config,
        data_root=data,
        authorization_path=authorization,
        output_path=report_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report[field] = value
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(Phase0BlockedError, match="preflight"):
        run_formal_training(
            config,
            data_root=data,
            authorization_path=authorization,
            preflight_report_path=report_path,
        )
