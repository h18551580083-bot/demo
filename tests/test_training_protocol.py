from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from cg_pipeline.model import FixedHEClassifier
from cg_pipeline.training import (
    TrainingContractError,
    aggregate_seed_results,
    audit_optimizer_ownership,
    build_adamw,
    configure_determinism,
    hash_epoch_order,
    load_checkpoint,
    model_state_identity,
    optimizer_state_identity,
    save_checkpoint,
    train_one_step,
)
from cg_pipeline.training_runs import load_complete_epoch_history


def _state_metadata(
    model: FixedHEClassifier, optimizer: torch.optim.Optimizer, **extra
) -> dict[str, object]:
    return {
        **extra,
        "checkpoint_identity": model_state_identity(model),
        "optimizer_state_identity": optimizer_state_identity(optimizer),
    }


def test_seed_and_hash_epoch_order_are_repeatable_complete_and_epoch_specific() -> None:
    first_audit = configure_determinism(1729)
    first = torch.rand(4)
    second_audit = configure_determinism(1729)
    second = torch.rand(4)
    identifiers = ("p3", "p1", "p2", "p0")

    order0 = hash_epoch_order(identifiers, seed=1729, epoch=0)
    repeat0 = hash_epoch_order(identifiers, seed=1729, epoch=0)
    order1 = hash_epoch_order(identifiers, seed=1729, epoch=1)

    assert first_audit == second_audit
    assert torch.equal(first, second)
    assert order0 == repeat0
    assert sorted(order0) == sorted(identifiers)
    assert len(set(order0)) == len(identifiers)
    assert order0 != order1
    assert first_audit["python_seed"] == 1729
    assert first_audit["numpy_seed"] == 1729
    assert first_audit["torch_cpu_seed"] == 1729
    assert first_audit["deterministic_algorithms"] is True
    assert first_audit["tf32"] is False
    assert first_audit["cudnn_benchmark"] is False


def test_optimizer_owns_every_backend_parameter_once_and_no_frontend_parameter() -> None:
    model = FixedHEClassifier(frontend_backend="fft")
    optimizer = build_adamw(
        model,
        learning_rate="0.001",
        beta1="0.9",
        beta2="0.999",
        epsilon="0.00000001",
        weight_decay="0.0001",
    )

    audit = audit_optimizer_ownership(model, optimizer)

    assert audit == {
        "electronic_parameter_count": 9473,
        "electronic_tensor_count": 4,
        "optimizer_unique_tensor_count": 4,
        "optical_parameter_count": 0,
        "all_electronic_exactly_once": True,
        "optical_in_optimizer": False,
    }
    optimizer.param_groups[0]["params"].append(model.classifier.bias)
    with pytest.raises(TrainingContractError, match="exactly once"):
        audit_optimizer_ownership(model, optimizer)


def test_real_training_step_changes_only_backend_and_checkpoint_restores(tmp_path: Path) -> None:
    configure_determinism(3407)
    model = FixedHEClassifier(frontend_backend="fft")
    optimizer = build_adamw(
        model,
        learning_rate="0.001",
        beta1="0.9",
        beta2="0.999",
        epsilon="0.00000001",
        weight_decay="0.0001",
    )
    rgb = torch.full((1, 3, 110, 110), 255, dtype=torch.uint8)
    target = torch.ones((1,), dtype=torch.float32)
    fixed_before = model.frontend.fixed_state_identity()

    step = train_one_step(model, optimizer, rgb, target)

    assert step.loss > 0.0
    assert step.changed_backend_parameters
    assert step.fixed_frontend_unchanged is True
    assert model.frontend.fixed_state_identity() == fixed_before
    assert step.optimizer_state_precision == "float32"
    checkpoint = tmp_path / "epoch-0000.pt"
    metadata = _state_metadata(
        model,
        optimizer,
        source_manifest_sha256="sha256:" + "1" * 64,
        manifest_hash="sha256:" + "2" * 64,
        kernel_hash=fixed_before["canonical_kernel_hash"],
        seed=3407,
        epoch=0,
    )
    save_checkpoint(checkpoint, model, optimizer, metadata)
    restored_model = FixedHEClassifier(frontend_backend="fft")
    restored_optimizer = build_adamw(
        restored_model,
        learning_rate="0.001",
        beta1="0.9",
        beta2="0.999",
        epsilon="0.00000001",
        weight_decay="0.0001",
    )
    restored_metadata = load_checkpoint(
        checkpoint,
        restored_model,
        restored_optimizer,
        expected_metadata=metadata,
    )

    assert restored_metadata == metadata
    assert torch.equal(model.classifier.bias, restored_model.classifier.bias)
    assert model.frontend.fixed_state_identity() == restored_model.frontend.fixed_state_identity()
    with pytest.raises(FileExistsError):
        save_checkpoint(checkpoint, model, optimizer, metadata)


def test_checkpoint_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    model = FixedHEClassifier(frontend_backend="fft")
    optimizer = build_adamw(
        model,
        learning_rate="0.001",
        beta1="0.9",
        beta2="0.999",
        epsilon="0.00000001",
        weight_decay="0.0001",
    )
    path = tmp_path / "checkpoint.pt"
    metadata = _state_metadata(model, optimizer, source_manifest_sha256="sha256:" + "a" * 64)
    save_checkpoint(path, model, optimizer, metadata)

    with pytest.raises(TrainingContractError, match="metadata identity mismatch"):
        load_checkpoint(
            path,
            model,
            optimizer,
            expected_metadata={**metadata, "source_manifest_sha256": "sha256:" + "b" * 64},
        )


def test_checkpoint_cannot_replace_fixed_frontend_buffers(tmp_path: Path) -> None:
    model = FixedHEClassifier(frontend_backend="fft")
    optimizer = build_adamw(
        model,
        learning_rate="0.001",
        beta1="0.9",
        beta2="0.999",
        epsilon="0.00000001",
        weight_decay="0.0001",
    )
    original = tmp_path / "original.pt"
    metadata = _state_metadata(model, optimizer, source_manifest_sha256="sha256:" + "a" * 64)
    save_checkpoint(original, model, optimizer, metadata)
    payload = torch.load(original, map_location="cpu")
    fixed_key = next(key for key in payload["model_state"] if key.startswith("frontend."))
    payload["model_state"][fixed_key] = payload["model_state"][fixed_key].clone()
    payload["model_state"][fixed_key].view(-1)[0] += 1
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)

    restored_model = FixedHEClassifier(frontend_backend="fft")
    restored_optimizer = build_adamw(
        restored_model,
        learning_rate="0.001",
        beta1="0.9",
        beta2="0.999",
        epsilon="0.00000001",
        weight_decay="0.0001",
    )
    with pytest.raises(TrainingContractError, match="canonical fixed frontend"):
        load_checkpoint(
            tampered,
            restored_model,
            restored_optimizer,
            expected_metadata=metadata,
        )


def test_checkpoint_rejects_tampered_electronic_and_optimizer_state(tmp_path: Path) -> None:
    model = FixedHEClassifier(frontend_backend="fft")
    optimizer = build_adamw(
        model,
        learning_rate="0.001",
        beta1="0.9",
        beta2="0.999",
        epsilon="0.00000001",
        weight_decay="0.0001",
    )
    rgb = torch.full((1, 3, 110, 110), 255, dtype=torch.uint8)
    train_one_step(model, optimizer, rgb, torch.ones((1,), dtype=torch.float32))
    metadata = _state_metadata(model, optimizer, source_manifest_sha256="sha256:" + "a" * 64)
    original = tmp_path / "original.pt"
    save_checkpoint(original, model, optimizer, metadata)
    payload = torch.load(original, map_location="cpu")

    electronic = tmp_path / "electronic-tampered.pt"
    electronic_payload = {**payload, "model_state": dict(payload["model_state"])}
    electronic_payload["model_state"]["classifier.bias"] = payload["model_state"][
        "classifier.bias"
    ].clone()
    electronic_payload["model_state"]["classifier.bias"].view(-1)[0] += 1
    torch.save(electronic_payload, electronic)

    def restored_pair():
        restored_model = FixedHEClassifier(frontend_backend="fft")
        restored_optimizer = build_adamw(
            restored_model,
            learning_rate="0.001",
            beta1="0.9",
            beta2="0.999",
            epsilon="0.00000001",
            weight_decay="0.0001",
        )
        return restored_model, restored_optimizer

    restored_model, restored_optimizer = restored_pair()
    with pytest.raises(TrainingContractError, match="model state identity"):
        load_checkpoint(
            electronic,
            restored_model,
            restored_optimizer,
            expected_metadata=metadata,
        )

    optimizer_tampered = tmp_path / "optimizer-tampered.pt"
    optimizer_payload = torch.load(original, map_location="cpu")
    first_state = next(iter(optimizer_payload["optimizer_state"]["state"].values()))
    first_state["exp_avg"] = first_state["exp_avg"].clone()
    first_state["exp_avg"].view(-1)[0] += 1
    torch.save(optimizer_payload, optimizer_tampered)
    restored_model, restored_optimizer = restored_pair()
    with pytest.raises(TrainingContractError, match="optimizer state identity"):
        load_checkpoint(
            optimizer_tampered,
            restored_model,
            restored_optimizer,
            expected_metadata=metadata,
        )


def test_resume_requires_continuous_checkpoint_report_pairs_and_exact_identities(
    tmp_path: Path,
) -> None:
    base = {"seed": 1729, "source_manifest_sha256": "sha256:" + "a" * 64}
    checkpoint_identity = "sha256:" + "b" * 64
    optimizer_identity = "sha256:" + "c" * 64
    identities = {
        **base,
        "checkpoint_identity": checkpoint_identity,
        "optimizer_state_identity": optimizer_identity,
        "epoch": 0,
    }
    checkpoint_bytes = b"immutable-checkpoint-placeholder"
    (tmp_path / "epoch-0000.pt").write_bytes(checkpoint_bytes)
    (tmp_path / "epoch-0000.json").write_text(
        json.dumps(
            {
                "schema": "formal-train-epoch-v1",
                "epoch": 0,
                "seed": 1729,
                "checkpoint": "epoch-0000.pt",
                "checkpoint_file_sha256": "sha256:" + hashlib.sha256(checkpoint_bytes).hexdigest(),
                "identities": identities,
            }
        ),
        encoding="utf-8",
    )

    history, latest, metadata = load_complete_epoch_history(tmp_path, base)

    assert len(history) == 1
    assert latest == tmp_path / "epoch-0000.pt"
    assert metadata == identities
    (tmp_path / "epoch-0000.pt").write_bytes(checkpoint_bytes + b"tampered")
    with pytest.raises(ValueError, match="checkpoint/report pair"):
        load_complete_epoch_history(tmp_path, base)
    (tmp_path / "epoch-0000.pt").write_bytes(checkpoint_bytes)
    (tmp_path / "epoch-0001.pt").write_bytes(b"partial")
    with pytest.raises(ValueError, match="checkpoint/report pair"):
        load_complete_epoch_history(tmp_path, base)


def test_multi_seed_aggregation_excludes_failed_runs_and_reports_individuals() -> None:
    result = aggregate_seed_results(
        (
            {"seed": 1729, "status": "complete", "best_validation_slide_auroc": 1.0},
            {"seed": 3407, "status": "failed", "failure_reason": "device_error"},
            {"seed": 7919, "status": "complete", "best_validation_slide_auroc": 0.5},
        )
    )

    assert result["valid_seed_count"] == 2
    assert result["failed_seed_count"] == 1
    assert result["mean"] == 0.75
    assert result["sample_standard_deviation"] == pytest.approx(2**-1.5)
    assert result["individual"] == [{"seed": 1729, "value": 1.0}, {"seed": 7919, "value": 0.5}]
