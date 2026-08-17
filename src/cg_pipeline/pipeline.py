"""Public orchestration for exploratory, preflight, and formal training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .artifacts import PipelineBlockedError, write_json_exclusive
from .claims import isolation_claim_fields
from .config import load_experiment_config
from .preflight import consume_preflight_report, run_preflight_report
from .runtime import output_base, training_datasets, validate_training_data
from .training import aggregate_seed_results
from .training_runs import run_exploratory_seed, run_formal_seed

Phase0BlockedError = PipelineBlockedError


def run_preflight(
    config_path: Path | str,
    *,
    data_root: Path | str,
    authorization_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    return run_preflight_report(
        config,
        data_root=Path(data_root).resolve(),
        authorization_path=Path(authorization_path).resolve(),
        output_path=Path(output_path).resolve(),
    )


def _requested_overrides(**values: Any) -> dict[str, Any]:
    return {
        name: str(value).replace("\\", "/") if name == "output" else value
        for name, value in values.items()
        if value is not None
    }


def run_exploratory_training(
    config_path: Path | str,
    *,
    data_root: Path | str,
    device: str | None = None,
    seed: int | None = None,
    output: Path | str | None = None,
    run_id: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    max_epochs: int | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    overrides = _requested_overrides(
        device=device,
        seed=seed,
        output=output,
        run_id=run_id,
        batch_size=batch_size,
        num_workers=num_workers,
        max_epochs=max_epochs,
        max_steps=max_steps,
    )
    config = load_experiment_config(config_path, exploratory_overrides=overrides)
    if config.execution_kind != "exploratory_train":
        raise ValueError("exploratory training requires execution.kind=exploratory_train")
    bundle = validate_training_data(config, data_root=Path(data_root).resolve())
    training_device = torch.device(str(config.execution["device"]))
    if training_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured exploratory CUDA device is unavailable")
    destination = output_base(config)
    destination.mkdir(parents=True, exist_ok=False)
    train_dataset, val_dataset = training_datasets(bundle)
    result = run_exploratory_seed(
        config,
        bundle,
        train_dataset,
        val_dataset,
        training_device,
        destination,
        seed=int(config.training["seeds"][0]),
        requested_overrides=overrides,
    )
    summary = {
        "schema": "exploratory-training-summary-v1",
        "status": "complete",
        "formal_experiment": False,
        "experiment_mode": "exploratory_train",
        "run_id": str(config.execution["run_id"]),
        "requested_overrides": overrides,
        "effective_config": config.as_dict(),
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "effective_split_hashes": bundle.effective_split_hashes,
        **isolation_claim_fields(),
        "lightweight_safety_checks": {
            "manifest_readable": True,
            "train_validation_paths_exist": True,
            "train_validation_splits_valid": True,
            "slide_id_group_id_cross_split_conflicts": 0,
        },
        "run": result,
        "test_split_accessed": False,
    }
    write_json_exclusive(destination / "training_summary.json", summary)
    return summary


def _failed_seed(seed: int, error: Exception) -> dict[str, Any]:
    return {
        "seed": seed,
        "status": "failed",
        "failure_reason": f"{type(error).__name__}: {error}",
        "automatic_retry": False,
        **isolation_claim_fields(),
    }


def run_formal_training(
    config_path: Path | str,
    *,
    data_root: Path | str,
    authorization_path: Path | str,
    preflight_report_path: Path | str,
    resume: bool = False,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    if config.execution_kind != "formal_train":
        raise ValueError("formal training requires execution.kind=formal_train")
    _, bundle = consume_preflight_report(
        config,
        data_root=Path(data_root).resolve(),
        authorization_path=Path(authorization_path).resolve(),
        preflight_report_path=Path(preflight_report_path).resolve(),
    )
    destination = output_base(config)
    if destination.exists() and not resume:
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=resume)
    device = torch.device(str(config.execution["device"]))
    train_dataset, val_dataset = training_datasets(bundle)
    results: list[dict[str, Any]] = []
    for seed_value in config.training["seeds"]:
        seed = int(seed_value)
        try:
            result = run_formal_seed(
                config,
                bundle,
                train_dataset,
                val_dataset,
                device,
                destination,
                seed=seed,
                resume=resume,
            )
        except (OSError, RuntimeError, ValueError) as error:
            result = _failed_seed(seed, error)
            failure_path = destination / f"seed-{seed}-failure.json"
            if not failure_path.exists():
                write_json_exclusive(failure_path, result)
        results.append(result)
    valid = [item for item in results if item["status"] == "complete"]
    aggregation = (
        aggregate_seed_results(tuple(results))
        if valid
        else {"status": "undefined", "reason": "no_valid_seed_run"}
    )
    summary = {
        "schema": "formal-training-summary-v1",
        "effective_config": config.as_dict(),
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "effective_split_hashes": bundle.effective_split_hashes,
        **isolation_claim_fields(),
        "runs": results,
        "multi_seed_validation_slide_auroc": aggregation,
        "test_split_accessed": False,
        "phase1_training_preflight": "PASS",
    }
    write_json_exclusive(destination / "training_summary.json", summary)
    return summary
