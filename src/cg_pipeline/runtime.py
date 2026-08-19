"""Shared train/validation runtime preparation without repository governance."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import TypedDict

import torch

from .config import ExperimentConfig
from .data import (
    DataContractError,
    ManifestBundle,
    PatchDataset,
    build_dataloader,
    expected_batch_count,
    validate_train_validation_manifests,
)
from .evaluation import AuthorizedEvaluationRow, EvaluationContext, Prediction
from .model import FixedHEClassifier
from .training import build_adamw, model_state_identity


class BatchContract(TypedDict):
    batch_size: int
    drop_last: bool
    train_rows: int
    train_batch_count: int
    maximum_optimizer_updates: int
    validation_rows: int
    validation_batch_count: int


def batch_contract(config: ExperimentConfig, bundle: ManifestBundle) -> BatchContract:
    batch_size = int(config.training["batch_size"])
    train_rows = int(bundle.split_counts["train"])
    validation_rows = int(bundle.split_counts["val"])
    train_batches = expected_batch_count(train_rows, batch_size, drop_last=False)
    return {
        "batch_size": batch_size,
        "drop_last": False,
        "train_rows": train_rows,
        "train_batch_count": train_batches,
        "maximum_optimizer_updates": train_batches * int(config.training["max_epochs"]),
        "validation_rows": validation_rows,
        "validation_batch_count": expected_batch_count(
            validation_rows, batch_size, drop_last=False
        ),
    }


def validate_training_data(config: ExperimentConfig, *, data_root: Path) -> ManifestBundle:
    train_manifest = data_root / Path(
        *PurePosixPath(str(config.data["train_manifest_relpath"])).parts
    )
    validation_manifest = data_root / Path(
        *PurePosixPath(str(config.data["validation_manifest_relpath"])).parts
    )
    if not train_manifest.is_file() or not validation_manifest.is_file():
        raise DataContractError(
            "training requires explicit train and validation manifests; "
            "no combined manifest fallback is allowed"
        )
    bundle = validate_train_validation_manifests(
        data_root,
        train_manifest,
        validation_manifest,
        check_files=True,
    )
    if bundle.split_counts["train"] < 1 or bundle.split_counts["val"] < 1:
        raise ValueError("training data must contain nonempty train and validation splits")
    if bundle.isolation.cross_split_conflicts != 0:
        raise ValueError("slide_id/group_id isolation check failed")
    return bundle


def training_datasets(bundle: ManifestBundle) -> tuple[PatchDataset, PatchDataset]:
    return PatchDataset(bundle, "train"), PatchDataset(bundle, "val")


def build_optimizer(config: ExperimentConfig, model: FixedHEClassifier) -> torch.optim.AdamW:
    training = config.training
    return build_adamw(
        model,
        learning_rate=str(training["learning_rate"]),
        beta1=str(training["beta1"]),
        beta2=str(training["beta2"]),
        epsilon=str(training["epsilon"]),
        weight_decay=str(training["weight_decay"]),
    )


def evaluation_context(
    bundle: ManifestBundle,
    dataset: PatchDataset,
    model: FixedHEClassifier,
    *,
    seed: int,
) -> EvaluationContext:
    return EvaluationContext(
        split=dataset.split,
        authorized_rows=tuple(
            AuthorizedEvaluationRow(
                patch_id=row.patch_id,
                slide_id=row.slide_id,
                split=row.split,
                patch_target=row.patch_target,
                slide_target=row.slide_target,
            )
            for row in dataset.rows
        ),
        source_manifest_sha256=bundle.source_manifest_sha256,
        effective_manifest_sha256=bundle.effective_split_hashes[dataset.split],
        fixed_frontend_identity=model.frontend.fixed_state_identity(),
        checkpoint_identity=model_state_identity(model),
        seed=seed,
    )


def _concatenated_logits_to_host(logit_batches: list[torch.Tensor]) -> list[float]:
    if not logit_batches:
        return []
    return torch.cat(logit_batches, dim=0).cpu().tolist()


def prediction_ledger(
    model: FixedHEClassifier,
    dataset: PatchDataset,
    config: ExperimentConfig,
    device: torch.device,
    *,
    seed: int,
    epoch: int,
) -> tuple[Prediction, ...]:
    loader = build_dataloader(
        dataset,
        batch_size=int(config.training["batch_size"]),
        seed=seed,
        epoch=epoch,
        num_workers=int(config.training["num_workers"]),
    )
    logit_batches: list[torch.Tensor] = []
    metadata: list[tuple[str, str, str, int, int]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["rgb"].to(device)).logits.detach()
            if logits.ndim != 1:
                raise RuntimeError("prediction logits must have shape [B]")
            batch_metadata = list(
                zip(
                    batch["patch_id"],
                    batch["slide_id"],
                    batch["split"],
                    batch["target"].tolist(),
                    batch["slide_target"].tolist(),
                    strict=True,
                )
            )
            if len(batch_metadata) != logits.shape[0]:
                raise RuntimeError("prediction metadata and logits have different lengths")
            logit_batches.append(logits)
            metadata.extend(
                (patch_id, slide_id, split, int(patch_target), int(slide_target))
                for patch_id, slide_id, split, patch_target, slide_target in batch_metadata
            )
    host_logits = _concatenated_logits_to_host(logit_batches)
    if len(host_logits) != len(metadata):
        raise RuntimeError("prediction metadata and logits have different lengths")
    predictions = [
        Prediction(
            patch_id=patch_id,
            slide_id=slide_id,
            split=split,
            patch_target=patch_target,
            slide_target=slide_target,
            logit=logit,
        )
        for (patch_id, slide_id, split, patch_target, slide_target), logit in zip(
            metadata,
            host_logits,
            strict=True,
        )
    ]
    if len(predictions) != len(dataset) or len({row.patch_id for row in predictions}) != len(
        dataset
    ):
        raise RuntimeError("prediction ledger is incomplete or duplicated")
    return tuple(predictions)


def runtime_path(base: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("runtime output path must be a normalized relative path")
    target = (base / Path(*path.parts)).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError as error:
        raise ValueError("runtime output path escapes its workspace") from error
    return target


def output_base(config: ExperimentConfig) -> Path:
    parent = config.source.parent
    base = parent.parent if parent.name == "configs" else parent
    return runtime_path(base, str(config.execution["output_root"]))
