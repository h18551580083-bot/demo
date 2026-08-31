from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import cg_pipeline.__main__ as cli_module
import cg_pipeline.pipeline as pipeline_module
from cg_pipeline.__main__ import main
from cg_pipeline.config import load_experiment_config
from cg_pipeline.pipeline import run_exploratory_training

REPOSITORY = Path(__file__).resolve().parents[1]


def _manifest_package(root: Path) -> Path:
    rows = []
    for split, label_name, label in (
        ("train", "normal", 0),
        ("train", "tumor", 1),
        ("val", "normal", 0),
        ("val", "tumor", 1),
    ):
        relative = f"patches/{split}/{label_name}/patch-{split}-{label_name}.png"
        rows.append(
            {
                "patch_id": f"patch-{split}-{label_name}",
                "patch_path": relative,
                "split": split,
                "slide_id": f"slide-{split}-{label_name}",
                "label": str(label),
                "label_name": label_name,
                "patch_label": label_name,
                "slide_label": label_name,
            }
        )
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    manifest = root / "cam16_class_quota" / "metadata" / "training_manifest.csv"
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


def test_repository_formal_and_exploratory_configs_are_machine_validated() -> None:
    formal = load_experiment_config(REPOSITORY / "configs" / "phase1_baseline.toml")
    exploratory = load_experiment_config(REPOSITORY / "configs" / "exploratory_train.toml")
    assert formal.execution_kind == "formal_train"
    assert exploratory.execution_kind == "exploratory_train"
    assert formal.execution["allow_test"] is exploratory.execution["allow_test"] is False
    assert formal.training["seeds"] == (1729, 3407, 7919)
    assert formal.training["batch_size"] == 32
    assert exploratory.training["seeds"] == (1729,)
    assert formal.training["num_workers"] == exploratory.training["num_workers"] == 8


def test_exploratory_training_uses_only_lightweight_checks_and_records_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "exploratory.toml"
    config_path.write_text(
        (REPOSITORY / "configs" / "exploratory_train.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    data_root = _manifest_package(tmp_path / "cam16")
    observed_datasets: dict[str, object] = {}

    def fake_seed(*args: object, seed: int, **kwargs: object) -> dict[str, object]:
        observed_datasets["splits"] = (args[2].split, args[3].split)
        observed_datasets["row_splits"] = {
            row.split for dataset in (args[2], args[3]) for row in dataset.rows
        }
        return {
            "formal_experiment": False,
            "experiment_mode": "exploratory_train",
            "seed": seed,
            "best_epoch": 0,
            "best_validation_slide_auroc": 0.5,
            "epochs_completed": 1,
            "steps_completed": 2,
            "status": "complete",
        }

    monkeypatch.setattr(pipeline_module, "run_exploratory_seed", fake_seed)
    monkeypatch.setattr(pipeline_module.torch.cuda, "is_available", lambda: True)
    report = run_exploratory_training(
        config_path,
        data_root=data_root,
        device="cuda:1",
        seed=41,
        output="artifacts/exploratory_runs/profile-41",
        run_id="profile-41",
        batch_size=8,
        num_workers=3,
        max_epochs=1,
        max_steps=2,
    )

    assert report["formal_experiment"] is False
    assert report["experiment_mode"] == "exploratory_train"
    assert report["requested_overrides"] == {
        "device": "cuda:1",
        "seed": 41,
        "output": "artifacts/exploratory_runs/profile-41",
        "run_id": "profile-41",
        "batch_size": 8,
        "num_workers": 3,
        "max_epochs": 1,
        "max_steps": 2,
    }
    assert report["effective_config"]["training"]["seeds"] == [41]
    assert report["effective_config"]["execution"]["device"] == "cuda:1"
    assert report["lightweight_safety_checks"]["train_validation_splits_valid"] is True
    assert report["test_split_accessed"] is False
    assert observed_datasets == {
        "splits": ("train", "val"),
        "row_splits": {"train", "val"},
    }
    summary = tmp_path / "artifacts" / "exploratory_runs" / "profile-41" / "training_summary.json"
    assert json.loads(summary.read_text(encoding="utf-8")) == report


def test_cli_exposes_only_explicit_exploratory_and_formal_training_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_exploratory(config: Path, **kwargs: object) -> dict[str, object]:
        captured.update({"config": config, **kwargs})
        return {"run": {"status": "complete"}}

    monkeypatch.setattr(cli_module, "run_exploratory_training", fake_exploratory)
    exit_code = main(
        [
            "exploratory-train",
            "--config",
            str(tmp_path / "config.toml"),
            "--data-root",
            str(tmp_path / "data"),
            "--device",
            "cpu",
            "--seed",
            "7",
            "--output",
            "artifacts/exploratory_runs/cli-7",
            "--run-id",
            "cli-7",
            "--batch-size",
            "4",
            "--num-workers",
            "2",
            "--max-epochs",
            "1",
            "--max-steps",
            "3",
        ]
    )

    assert exit_code == 0
    assert captured["seed"] == 7
    assert captured["device"] == "cpu"
    assert captured["max_steps"] == 3
    with pytest.raises(SystemExit):
        main(["dry-run"])
    with pytest.raises(SystemExit):
        main(["train"])


def test_formal_cli_requires_and_forwards_one_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_formal(config: Path, **kwargs: object) -> dict[str, object]:
        captured.update({"config": config, **kwargs})
        status = "complete" if kwargs["seed"] == 3407 else "failed"
        return {"status": status, "runs": [{"seed": kwargs["seed"], "status": status}]}

    monkeypatch.setattr(cli_module, "run_formal_training", fake_formal)
    arguments = [
        "formal-train",
        "--config",
        str(tmp_path / "config.toml"),
        "--data-root",
        str(tmp_path / "data"),
        "--authorization",
        str(tmp_path / "authorization.json"),
        "--preflight-report",
        str(tmp_path / "preflight.json"),
    ]

    assert main([*arguments, "--seed", "3407"]) == 0
    assert captured["seed"] == 3407
    assert main([*arguments, "--seed", "7919"]) == 3
    with pytest.raises(SystemExit):
        main(arguments)
