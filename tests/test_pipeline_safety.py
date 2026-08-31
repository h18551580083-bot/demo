from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import cg_pipeline.pipeline as pipeline_module
import cg_pipeline.runtime as runtime_module
from cg_pipeline.config import ConfigError, load_experiment_config
from cg_pipeline.pipeline import Phase0BlockedError, run_formal_training, run_preflight
from cg_pipeline.runtime import prediction_ledger, validate_training_data

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


def _formal_config(tmp_path: Path) -> Path:
    text = (REPOSITORY / "configs" / "phase1_baseline.toml").read_text(encoding="utf-8")
    text = text.replace(
        'train_manifest_relpath = "metadata/training_manifest_train.csv"',
        'train_manifest_relpath = "package/training_manifest_train.csv"',
    ).replace(
        'validation_manifest_relpath = "metadata/training_manifest_val.csv"',
        'validation_manifest_relpath = "package/training_manifest_val.csv"',
    )
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
    ):
        patch_id = f"{split}-{label_name}"
        relative = f"patches/{split}/{label_name}/{patch_id}.png"
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
                    if conflict and label_name == "normal"
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
    for split in ("train", "val"):
        split_manifest = manifest.with_name(f"training_manifest_{split}.csv")
        with split_manifest.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(row for row in rows if row["split"] == split)
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

    loaded = load_experiment_config(config)
    assert loaded.data["train_manifest_relpath"] == "package/training_manifest_train.csv"
    assert (
        loaded.data["validation_manifest_relpath"]
        == "package/training_manifest_val.csv"
    )
    assert report["status"] == "PASS"
    assert report["blocking_gates"] == []
    assert report_path.exists()
    assert report["schema"] == "formal-training-preflight-v1"
    assert "fixed_frontend_identity" in report


def test_training_data_prefers_train_validation_manifests_without_reading_combined(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(_formal_config(tmp_path))
    data = _data_root(tmp_path)
    combined = data / "package" / "training_manifest.csv"
    combined.write_text("prohibited combined manifest must remain unread\n", encoding="utf-8")

    bundle = validate_training_data(config, data_root=data)

    assert bundle.split_counts == {"train": 2, "val": 2, "test": 0}
    assert bundle.effective_split_hashes.keys() == {"train", "val"}
    assert tuple(path.name for path in bundle.manifests) == (
        "training_manifest_train.csv",
        "training_manifest_val.csv",
    )


def test_training_data_fails_closed_when_one_split_manifest_is_missing(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(_formal_config(tmp_path))
    data = _data_root(tmp_path)
    (data / "package" / "training_manifest_val.csv").unlink()

    with pytest.raises(ValueError, match="explicit train and validation manifests"):
        validate_training_data(config, data_root=data)


def test_prediction_ledger_uses_inference_mode_and_one_host_logit_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = (
        {
            "rgb": torch.tensor([1, 2], dtype=torch.uint8).reshape(2, 1, 1, 1).expand(-1, 3, -1, -1),
            "patch_id": ["p1", "p2"],
            "slide_id": ["s1", "s2"],
            "split": ["val", "val"],
            "target": torch.tensor([0, 1]),
            "slide_target": torch.tensor([0, 1]),
        },
        {
            "rgb": torch.tensor([3], dtype=torch.uint8).reshape(1, 1, 1, 1).expand(-1, 3, -1, -1),
            "patch_id": ["p3"],
            "slide_id": ["s3"],
            "split": ["val"],
            "target": torch.tensor([1]),
            "slide_target": torch.tensor([1]),
        },
    )

    class SyntheticValidation:
        def __len__(self) -> int:
            return 3

    class InferenceModel(torch.nn.Module):
        def forward(self, rgb: torch.Tensor) -> SimpleNamespace:
            assert torch.is_inference_mode_enabled()
            return SimpleNamespace(logits=rgb[:, 0, 0, 0].to(torch.float32))

    monkeypatch.setattr(runtime_module, "build_dataloader", lambda *_args, **_kwargs: batches)
    original_transfer = runtime_module._concatenated_logits_to_host
    transfer_batch_counts: list[int] = []

    def observed_transfer(logit_batches: list[torch.Tensor]) -> list[float]:
        transfer_batch_counts.append(len(logit_batches))
        return original_transfer(logit_batches)

    monkeypatch.setattr(runtime_module, "_concatenated_logits_to_host", observed_transfer)
    config = SimpleNamespace(training={"batch_size": 2, "num_workers": 0})

    predictions = prediction_ledger(
        InferenceModel(),
        SyntheticValidation(),
        config,
        torch.device("cpu"),
        seed=1729,
        epoch=0,
    )

    assert transfer_batch_counts == [2]
    assert [(row.patch_id, row.slide_id, row.split, row.logit) for row in predictions] == [
        ("p1", "s1", "val", 1.0),
        ("p2", "s2", "val", 2.0),
        ("p3", "s3", "val", 3.0),
    ]


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


def test_formal_training_runs_only_the_selected_seed_under_an_existing_baseline(
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
    destination = repository / "artifacts" / "formal_runs" / "phase1-cam16-baseline-b32-v2"
    completed = destination / "seed-1729"
    completed.mkdir(parents=True)
    completion = {
        "run_id": "wrong-formal-baseline",
        "seed": 1729,
        "status": "complete",
        "best_validation_slide_auroc": 0.5,
    }
    (completed / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid completion record for seed 1729"):
        run_formal_training(
            config,
            data_root=data,
            authorization_path=authorization,
            preflight_report_path=report_path,
            seed=3407,
        )
    assert observed == []
    completion["run_id"] = "phase1-cam16-baseline-b32-v2"
    (completed / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
    summary = run_formal_training(
        config,
        data_root=data,
        authorization_path=authorization,
        preflight_report_path=report_path,
        seed=3407,
    )

    assert observed == [3407]
    assert summary["seed_status"] == {
        "1729": "completed",
        "3407": "completed",
        "7919": "pending",
    }
    assert summary["phase1_training_preflight"] == "PASS"
    assert summary["test_split_accessed"] is False
    generated_completion = json.loads(
        (destination / "seed-3407" / "completion.json").read_text(encoding="utf-8")
    )
    assert generated_completion["run_id"] == "phase1-cam16-baseline-b32-v2"
    assert generated_completion["automatic_retry"] is False
    original_completion = (completed / "completion.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_formal_training(
            config,
            data_root=data,
            authorization_path=authorization,
            preflight_report_path=report_path,
            seed=1729,
        )
    assert (completed / "completion.json").read_bytes() == original_completion

    def failed_seed(*args: object, seed: int, **kwargs: object) -> dict[str, object]:
        raise RuntimeError(f"fixture failure for seed {seed}")

    monkeypatch.setattr(pipeline_module, "run_formal_seed", failed_seed)
    failed = run_formal_training(
        config,
        data_root=data,
        authorization_path=authorization,
        preflight_report_path=report_path,
        seed=7919,
    )
    failure_record = json.loads(
        (destination / "seed-7919" / "failure.json").read_text(encoding="utf-8")
    )
    assert failed["status"] == "failed"
    assert failure_record["run_id"] == "phase1-cam16-baseline-b32-v2"


def test_formal_training_rejects_unapproved_seed_before_preflight(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seed 1 is not approved"):
        run_formal_training(
            _formal_config(tmp_path),
            data_root=tmp_path / "data",
            authorization_path=tmp_path / "authorization.json",
            preflight_report_path=tmp_path / "preflight.json",
            seed=1,
        )


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
            seed=1729,
        )
