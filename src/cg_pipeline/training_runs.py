"""Exploratory and formal seed loops for the train/validation-only pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from .artifacts import read_json_object, write_json_exclusive
from .claims import isolation_claim_fields
from .config import ExperimentConfig
from .data import ManifestBundle, PatchDataset, build_dataloader
from .evaluation import evaluate_predictions
from .identity import raw_sha256
from .model import FixedHEClassifier
from .runtime import build_optimizer, evaluation_context, prediction_ledger
from .training import (
    configure_determinism,
    load_checkpoint,
    model_state_identity,
    optimizer_state_identity,
    save_checkpoint,
    train_one_step,
)


def _train_epoch(
    model: FixedHEClassifier,
    optimizer: torch.optim.Optimizer,
    dataset: PatchDataset,
    config: ExperimentConfig,
    device: torch.device,
    *,
    seed: int,
    epoch: int,
    remaining_steps: int | None = None,
) -> list[float]:
    model.train()
    loader = build_dataloader(
        dataset,
        batch_size=int(config.training["batch_size"]),
        seed=seed,
        epoch=epoch,
        num_workers=int(config.training["num_workers"]),
    )
    losses: list[float] = []
    for batch in loader:
        if remaining_steps is not None and len(losses) >= remaining_steps:
            break
        step = train_one_step(
            model,
            optimizer,
            batch["rgb"].to(device),
            batch["target"].to(device),
        )
        losses.append(step.loss)
    return losses


def _validation_report(
    model: FixedHEClassifier,
    dataset: PatchDataset,
    bundle: ManifestBundle,
    config: ExperimentConfig,
    device: torch.device,
    *,
    seed: int,
    epoch: int,
) -> dict[str, Any]:
    predictions = prediction_ledger(model, dataset, config, device, seed=seed, epoch=epoch)
    return evaluate_predictions(
        predictions,
        context=evaluation_context(bundle, dataset, model, seed=seed),
        fit_thresholds=True,
        ci_seed=seed,
    )


def _state_metadata(
    base: dict[str, Any],
    model: FixedHEClassifier,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    steps_completed: int | None = None,
) -> dict[str, Any]:
    metadata = {
        **base,
        "checkpoint_identity": model_state_identity(model),
        "optimizer_state_identity": optimizer_state_identity(optimizer),
        "epoch": epoch,
    }
    if steps_completed is not None:
        metadata["steps_completed"] = steps_completed
    return metadata


def _write_epoch_artifacts(
    seed_dir: Path,
    model: FixedHEClassifier,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
    report: dict[str, Any],
) -> None:
    epoch = int(metadata["epoch"])
    checkpoint = seed_dir / f"epoch-{epoch:04d}.pt"
    save_checkpoint(checkpoint, model, optimizer, metadata)
    report.update(
        {
            "checkpoint": checkpoint.name,
            "checkpoint_file_sha256": raw_sha256(checkpoint.read_bytes()),
            "identities": metadata,
        }
    )
    write_json_exclusive(seed_dir / f"epoch-{epoch:04d}.json", report)


def _exploratory_metadata(
    config: ExperimentConfig,
    bundle: ManifestBundle,
    model: FixedHEClassifier,
    *,
    seed: int,
    requested_overrides: dict[str, Any],
) -> dict[str, Any]:
    return {
        "formal_experiment": False,
        "experiment_mode": "exploratory_train",
        "run_id": str(config.execution["run_id"]),
        "requested_overrides": requested_overrides,
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "effective_split_hashes": bundle.effective_split_hashes,
        "fixed_frontend_identity": model.frontend.fixed_state_identity(),
        "frontend_artifact_identity": model.frontend.artifact_identity(),
        **isolation_claim_fields(),
        "seed": seed,
    }


def run_exploratory_seed(
    config: ExperimentConfig,
    bundle: ManifestBundle,
    train_dataset: PatchDataset,
    val_dataset: PatchDataset,
    device: torch.device,
    output_base: Path,
    *,
    seed: int,
    requested_overrides: dict[str, Any],
) -> dict[str, Any]:
    configure_determinism(seed)
    seed_dir = output_base / f"seed-{seed}"
    seed_dir.mkdir(parents=True, exist_ok=False)
    model = FixedHEClassifier(
        frontend_backend=str(config.model["frontend_backend"]),
        frontend_variant=config.frontend_variant,
    ).to(device)
    optimizer = build_optimizer(config, model)
    base = _exploratory_metadata(
        config, bundle, model, seed=seed, requested_overrides=requested_overrides
    )
    history: list[dict[str, Any]] = []
    best_metric, best_epoch, total_steps, no_improvement = -float("inf"), -1, 0, 0
    max_steps = int(config.execution["max_steps"])
    for epoch in range(int(config.training["max_epochs"])):
        remaining = max_steps - total_steps if max_steps else None
        losses = _train_epoch(
            model,
            optimizer,
            train_dataset,
            config,
            device,
            seed=seed,
            epoch=epoch,
            remaining_steps=remaining,
        )
        if not losses:
            break
        total_steps += len(losses)
        evaluation = _validation_report(
            model, val_dataset, bundle, config, device, seed=seed, epoch=epoch
        )
        metric = float(evaluation["slide_auroc"]["value"])
        if metric > best_metric:
            best_metric, best_epoch, no_improvement = metric, epoch, 0
        else:
            no_improvement += 1
        metadata = _state_metadata(base, model, optimizer, epoch=epoch, steps_completed=total_steps)
        epoch_report = {
            "schema": "exploratory-train-epoch-v1",
            "formal_experiment": False,
            "experiment_mode": "exploratory_train",
            "run_id": str(config.execution["run_id"]),
            "requested_overrides": requested_overrides,
            "epoch": epoch,
            "seed": seed,
            "batch_count": len(losses),
            "steps_completed": total_steps,
            "max_steps": max_steps,
            "mean_training_loss": sum(losses) / len(losses),
            "evaluation": evaluation,
            "best_epoch": best_epoch,
            "test_split_accessed": False,
        }
        _write_epoch_artifacts(seed_dir, model, optimizer, metadata, epoch_report)
        history.append(epoch_report)
        if max_steps and total_steps >= max_steps:
            break
        if no_improvement >= int(config.training["early_stopping_patience"]):
            break
    if not history:
        raise RuntimeError("exploratory training completed no optimizer steps")
    return {
        "formal_experiment": False,
        "experiment_mode": "exploratory_train",
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_slide_auroc": best_metric,
        "epochs_completed": len(history),
        "steps_completed": total_steps,
        "early_stopping_triggered": no_improvement
        >= int(config.training["early_stopping_patience"]),
        "status": "complete",
    }


def _epoch_number(path: Path) -> int:
    suffix = path.stem.removeprefix("epoch-")
    if len(suffix) != 4 or not suffix.isdigit() or path.stem != f"epoch-{suffix}":
        raise ValueError(f"invalid epoch artifact name: {path.name}")
    return int(suffix)


def _validated_history_item(
    report: dict[str, Any],
    checkpoint: Path,
    base_metadata: dict[str, Any],
    epoch: int,
) -> dict[str, Any]:
    identities = report.get("identities", {})
    expected = {
        **base_metadata,
        "checkpoint_identity": identities.get("checkpoint_identity"),
        "optimizer_state_identity": identities.get("optimizer_state_identity"),
        "epoch": epoch,
    }
    checks = (
        report.get("schema") == "formal-train-epoch-v1",
        report.get("epoch") == epoch,
        report.get("seed") == base_metadata["seed"],
        report.get("checkpoint") == checkpoint.name,
        report.get("checkpoint_file_sha256") == raw_sha256(checkpoint.read_bytes()),
        identities == expected,
    )
    if not all(checks):
        raise ValueError(f"epoch {epoch} checkpoint/report pair is incomplete or mismatched")
    return expected


def load_complete_epoch_history(
    seed_dir: Path, base_metadata: dict[str, Any]
) -> tuple[list[dict[str, Any]], Path | None, dict[str, Any] | None]:
    checkpoints = {_epoch_number(path): path for path in seed_dir.glob("epoch-*.pt")}
    reports = {_epoch_number(path): path for path in seed_dir.glob("epoch-*.json")}
    if set(checkpoints) != set(reports):
        raise ValueError("resume requires a checkpoint/report pair for every epoch")
    if not checkpoints:
        return [], None, None
    indices = sorted(checkpoints)
    if indices != list(range(indices[-1] + 1)):
        raise ValueError("resume epoch history is not continuous from epoch zero")
    history: list[dict[str, Any]] = []
    latest_metadata: dict[str, Any] | None = None
    for epoch in indices:
        report = read_json_object(reports[epoch])
        latest_metadata = _validated_history_item(report, checkpoints[epoch], base_metadata, epoch)
        history.append(report)
    return history, checkpoints[indices[-1]], latest_metadata


def _history_state(history: list[dict[str, Any]]) -> tuple[float, int, int]:
    metrics = [float(item["evaluation"]["slide_auroc"]["value"]) for item in history]
    if not metrics:
        return -float("inf"), -1, 0
    best_metric = max(metrics)
    best_epoch = metrics.index(best_metric)
    return best_metric, best_epoch, len(metrics) - best_epoch - 1


def run_formal_seed(
    config: ExperimentConfig,
    bundle: ManifestBundle,
    train_dataset: PatchDataset,
    val_dataset: PatchDataset,
    device: torch.device,
    output_base: Path,
    *,
    seed: int,
    resume: bool,
) -> dict[str, Any]:
    configure_determinism(seed)
    seed_dir = output_base / f"seed-{seed}"
    seed_dir.mkdir(parents=True, exist_ok=resume)
    model = FixedHEClassifier(
        frontend_backend=str(config.model["frontend_backend"]),
        frontend_variant=config.frontend_variant,
    ).to(device)
    optimizer = build_optimizer(config, model)
    base = {
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "effective_split_hashes": bundle.effective_split_hashes,
        "fixed_frontend_identity": model.frontend.fixed_state_identity(),
        **isolation_claim_fields(),
        "seed": seed,
    }
    if config.frontend_variant == "matched_control":
        base["frontend_artifact_identity"] = model.frontend.artifact_identity()
    history, latest, expected = (
        load_complete_epoch_history(seed_dir, base) if resume else ([], None, None)
    )
    start_epoch = 0
    if latest is not None and expected is not None:
        load_checkpoint(latest, model, optimizer, expected_metadata=expected)
        start_epoch = int(expected["epoch"]) + 1
    best_metric, best_epoch, no_improvement = _history_state(history)
    for epoch in range(start_epoch, int(config.training["max_epochs"])):
        epoch_started_at = time.perf_counter()
        losses = _train_epoch(
            model, optimizer, train_dataset, config, device, seed=seed, epoch=epoch
        )
        evaluation = _validation_report(
            model, val_dataset, bundle, config, device, seed=seed, epoch=epoch
        )
        epoch_duration_seconds = time.perf_counter() - epoch_started_at
        mean_training_loss = sum(losses) / len(losses)
        metric = float(evaluation["slide_auroc"]["value"])
        if metric > best_metric:
            best_metric, best_epoch, no_improvement = metric, epoch, 0
        else:
            no_improvement += 1
        metadata = _state_metadata(base, model, optimizer, epoch=epoch)
        report = {
            "schema": "formal-train-epoch-v1",
            "epoch": epoch,
            "seed": seed,
            "batch_count": len(losses),
            "mean_training_loss": mean_training_loss,
            "epoch_duration_seconds": epoch_duration_seconds,
            "evaluation": evaluation,
            "best_epoch": best_epoch,
            "test_split_accessed": False,
        }
        validation_slide_accuracy = float(evaluation["slide_metrics"]["accuracy"])
        print(
            f"[FORMAL] seed={seed} epoch={epoch} "
            f"time={epoch_duration_seconds:.1f}s ({epoch_duration_seconds / 60.0:.2f}m) "
            f"loss={mean_training_loss:.4f} val_slide_auc={metric:.4f} "
            f"val_slide_acc={validation_slide_accuracy:.4f} "
            f"best_epoch={best_epoch} best_auc={best_metric:.4f}",
            flush=True,
        )
        _write_epoch_artifacts(seed_dir, model, optimizer, metadata, report)
        history.append(report)
        if no_improvement >= int(config.training["early_stopping_patience"]):
            break
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_slide_auroc": best_metric,
        "epochs_completed": len(history),
        "status": "complete",
    }
