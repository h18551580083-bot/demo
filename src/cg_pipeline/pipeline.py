"""Public preflight, synthetic dry-run, and guarded formal-training entry points."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from PIL import Image

from .claims import (
    PATIENT_LEVEL_CLAIM_ALLOWED,
    PATIENT_LEVEL_ISOLATION,
    audit_isolation_claim_payload,
    isolation_claim_fields,
)
from .config import ExperimentConfig, load_experiment_config
from .data import (
    DataContractError,
    ManifestBundle,
    PatchDataset,
    build_dataloader,
    validate_manifest,
)
from .evaluation import (
    AuthorizedEvaluationRow,
    EvaluationContext,
    Prediction,
    evaluate_predictions,
)
from .identity import raw_sha256
from .model import FixedHEClassifier
from .morlet import generate_morlet_bundle, validate_spectral_coverage
from .training import (
    aggregate_seed_results,
    audit_optimizer_ownership,
    build_adamw,
    configure_determinism,
    load_checkpoint,
    model_state_identity,
    optimizer_state_identity,
    save_checkpoint,
    train_one_step,
)


class Phase0BlockedError(RuntimeError):
    """Formal training was prevented before its first batch."""


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    claim_audit = audit_isolation_claim_payload(value)
    if claim_audit["status"] != "PASS":
        raise ValueError(f"unsafe patient-level claim in report: {claim_audit['forbidden_claims']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _read_json_strict(path: Path) -> dict[str, Any]:
    def no_duplicates(pairs):
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise Phase0BlockedError(f"cannot validate JSON record {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise Phase0BlockedError("release record must be a JSON object")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _validate_release_record(value: dict[str, Any]) -> None:
    expected = {
        "schema",
        "phase0_closed",
        "formal_training_authorized",
        "external_blockers",
        "patient_level_isolation",
        "patient_level_claim_allowed",
        "test_access_authorized",
    }
    if set(value) != expected:
        raise Phase0BlockedError("release record fields do not match phase0-release-v2")
    if value["schema"] != "phase0-release-v2":
        raise Phase0BlockedError("release record schema is not phase0-release-v2")
    if any(
        not isinstance(value[key], bool)
        for key in (
            "phase0_closed",
            "formal_training_authorized",
            "patient_level_claim_allowed",
            "test_access_authorized",
        )
    ):
        raise Phase0BlockedError("release authorization fields must be Boolean")
    if not isinstance(value["external_blockers"], list) or any(
        not isinstance(item, str) or not item for item in value["external_blockers"]
    ):
        raise Phase0BlockedError("release external_blockers must be an array of nonempty strings")
    if value["patient_level_isolation"] != PATIENT_LEVEL_ISOLATION:
        raise Phase0BlockedError("patient-level isolation must remain not_evaluated")
    if value["patient_level_claim_allowed"] is not PATIENT_LEVEL_CLAIM_ALLOWED:
        raise Phase0BlockedError("patient-level claim is not allowed")


def _runtime_path(base: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("runtime output path must be a normalized relative path")
    target = (base / Path(*path.parts)).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError as error:
        raise ValueError("runtime output path escapes its workspace") from error
    return target


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_code_identity() -> str:
    digest = hashlib.sha256()
    for root in (_REPOSITORY_ROOT / "src",):
        for path in sorted(root.rglob("*.py"), key=lambda item: item.as_posix()):
            relative = path.relative_to(_REPOSITORY_ROOT).as_posix().encode("utf-8")
            data = path.read_bytes()
            digest.update(len(relative).to_bytes(8, "big") + relative)
            digest.update(len(data).to_bytes(8, "big") + data)
    return "sha256:" + digest.hexdigest()


def _evaluation_context(
    config: ExperimentConfig,
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
        config_hash=config.sha256,
        code_identity=_source_code_identity(),
        source_manifest_sha256=bundle.source_manifest_sha256,
        effective_manifest_sha256=bundle.effective_split_hashes[dataset.split],
        fixed_frontend_identity=model.frontend.fixed_state_identity(),
        checkpoint_identity=model_state_identity(model),
        seed=seed,
    )


def _optimizer(config: ExperimentConfig, model: FixedHEClassifier) -> torch.optim.AdamW:
    training = config.training
    return build_adamw(
        model,
        learning_rate=str(training["learning_rate"]),
        beta1=str(training["beta1"]),
        beta2=str(training["beta2"]),
        epsilon=str(training["epsilon"]),
        weight_decay=str(training["weight_decay"]),
    )


def _synthetic_pattern(kind: int) -> np.ndarray:
    y, x = np.indices((256, 256), dtype=np.uint16)
    if kind == 0:
        channels = (x % 256, y % 256, (x + y) % 256)
    elif kind == 1:
        checker = ((x // 8 + y // 8) % 2) * 190 + 32
        channels = (checker, (checker + 31) % 256, (255 - checker) % 256)
    elif kind == 2:
        channels = ((255 - x) % 256, (2 * y) % 256, (x // 2 + y) % 256)
    else:
        stripe = ((x // 5) % 2) * 210 + 20
        channels = ((stripe + y) % 256, stripe, (255 - stripe) % 256)
    return np.ascontiguousarray(np.stack(channels, axis=-1).astype(np.uint8))


def _create_synthetic_package(root: Path) -> Path:
    rows = [
        ("dry-train-normal", "train", "normal", 0, "dry-slide-train-normal", 0),
        ("dry-train-tumor", "train", "tumor", 1, "dry-slide-train-tumor", 1),
        ("dry-val-normal", "val", "normal", 0, "dry-slide-val-normal", 0),
        ("dry-val-tumor", "val", "tumor", 1, "dry-slide-val-tumor", 1),
    ]
    manifest_rows: list[dict[str, str]] = []
    for index, (patch_id, split, label_name, label, slide_id, slide_label) in enumerate(rows):
        relative = f"patches/{split}/{label_name}/{patch_id}.png"
        path = root / Path(*PurePosixPath(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(_synthetic_pattern(index), mode="RGB").save(path, format="PNG")
        manifest_rows.append(
            {
                "patch_id": patch_id,
                "patch_path": relative,
                "split": split,
                "slide_id": slide_id,
                "label": str(label),
                "label_name": label_name,
                "patch_label": label_name,
                "slide_label": "tumor" if slide_label else "normal",
            }
        )
    manifest = root / "metadata" / "training_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patch_id",
        "patch_path",
        "split",
        "slide_id",
        "label",
        "label_name",
        "patch_label",
        "slide_label",
    ]
    with manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    duplicate = root / "metadata" / "negative_duplicate_patch_id.csv"
    with duplicate.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(manifest_rows[0])
        writer.writerow(manifest_rows[0])
    return manifest


def _prediction_ledger(
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
    predictions: list[Prediction] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(batch["rgb"].to(device)).logits.detach().cpu()
            for index, logit in enumerate(output.tolist()):
                predictions.append(
                    Prediction(
                        patch_id=batch["patch_id"][index],
                        slide_id=batch["slide_id"][index],
                        split=batch["split"][index],
                        patch_target=int(batch["target"][index].item()),
                        slide_target=int(batch["slide_target"][index].item()),
                        logit=logit,
                    )
                )
    if len(predictions) != len(dataset) or len({row.patch_id for row in predictions}) != len(dataset):
        raise RuntimeError("prediction ledger is incomplete or duplicated")
    return tuple(predictions)


def _core_dry_execution(
    config: ExperimentConfig,
    bundle: ManifestBundle,
    device: torch.device,
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    seed = int(config.training["seeds"][0])
    seed_audit = configure_determinism(seed)
    model = FixedHEClassifier(frontend_backend=str(config.model["frontend_backend"])).to(device)
    optimizer = _optimizer(config, model)
    train_dataset = PatchDataset(bundle, "train")
    train_loader = build_dataloader(
        train_dataset,
        batch_size=int(config.training["batch_size"]),
        seed=seed,
        epoch=0,
        num_workers=int(config.training["num_workers"]),
    )
    batch = next(iter(train_loader))
    step = train_one_step(
        model,
        optimizer,
        batch["rgb"].to(device),
        batch["target"].to(device),
    )
    fixed_identity = model.frontend.fixed_state_identity()
    metadata = {
        "code_revision": _git_revision(),
        "code_identity": _source_code_identity(),
        "config_hash": config.sha256,
        "checkpoint_identity": model_state_identity(model),
        "optimizer_state_identity": optimizer_state_identity(optimizer),
        "epoch": 0,
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "effective_split_hashes": bundle.effective_split_hashes,
        "fixed_frontend_identity": fixed_identity,
        "seed": seed,
    }
    checkpoint_restore = {"passed": False}
    if checkpoint_path is not None:
        state_before_save = model_state_identity(model)
        save_checkpoint(checkpoint_path, model, optimizer, metadata)
        restored = FixedHEClassifier(frontend_backend=str(config.model["frontend_backend"])).to(
            device
        )
        restored_optimizer = _optimizer(config, restored)
        restored_metadata = load_checkpoint(
            checkpoint_path,
            restored,
            restored_optimizer,
            expected_metadata=metadata,
        )
        if model_state_identity(restored) != state_before_save:
            raise RuntimeError("checkpoint restore changed model state identity")
        model = restored
        optimizer = restored_optimizer
        checkpoint_restore = {"passed": True, "metadata": restored_metadata}
    val_dataset = PatchDataset(bundle, "val")
    val_predictions = _prediction_ledger(
        model,
        val_dataset,
        config,
        device,
        seed=seed,
        epoch=0,
    )
    evaluation = evaluate_predictions(
        val_predictions,
        context=_evaluation_context(config, bundle, val_dataset, model, seed=seed),
        fit_thresholds=True,
        ci_seed=seed,
    )
    step_payload = asdict(step)
    step_payload["changed_backend_parameters"] = list(step.changed_backend_parameters)
    prediction_bits = [
        torch.tensor(row.logit, dtype=torch.float32).view(torch.int32).item()
        for row in sorted(val_predictions, key=lambda item: item.patch_id)
    ]
    return {
        "seed_audit": seed_audit,
        "training_step": step_payload,
        "fixed_identity": fixed_identity,
        "fixed_frontend_unchanged": step.fixed_frontend_unchanged,
        "model_state_identity": model_state_identity(model),
        "prediction_float32_bits": prediction_bits,
        "evaluation": evaluation,
        "checkpoint_restore": checkpoint_restore,
        "optimizer_ownership": audit_optimizer_ownership(model, optimizer),
    }


def run_dry_run(
    config_path: Path | str,
    *,
    workspace_root: Path | str,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    if config.execution_kind != "dry_run":
        raise ValueError("dry-run entry requires execution.kind=dry_run")
    base = Path(workspace_root).resolve()
    output = _runtime_path(base, str(config.execution["output_root"]))
    output.mkdir(parents=True, exist_ok=False)
    package_root = output / "synthetic_package"
    manifest = _create_synthetic_package(package_root)
    bundle = validate_manifest(package_root, manifest, check_files=True, reconcile_disk=True)
    device = torch.device(str(config.execution["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured dry-run CUDA device is unavailable")
    checkpoint = output / "checkpoint" / "epoch-0000.pt"
    first = _core_dry_execution(config, bundle, device, checkpoint)
    second = _core_dry_execution(config, bundle, device, None)
    exact_match = all(
        first[key] == second[key]
        for key in ("model_state_identity", "prediction_float32_bits", "evaluation")
    )
    if not exact_match:
        raise RuntimeError("repeat dry-run executions are not exactly reproducible")
    negative_path = package_root / "metadata" / "negative_duplicate_patch_id.csv"
    negative_detected = False
    try:
        validate_manifest(
            package_root,
            negative_path,
            check_files=False,
            reconcile_disk=False,
        )
    except DataContractError as error:
        negative_detected = "duplicate patch_id" in str(error)
    if not negative_detected:
        raise RuntimeError("duplicate-patch negative control was not detected")
    spectral_coverage = validate_spectral_coverage(generate_morlet_bundle())
    if spectral_coverage["status"] != "PASS":
        raise RuntimeError("Morlet spectral coverage gate failed")
    steps = {
        "config_load": True,
        "config_schema_validation": True,
        "manifest_validation": True,
        "dataset_build": True,
        "dataloader_build": True,
        "model_build": True,
        "forward": True,
        "backward": True,
        "optimizer_update": True,
        "allowed_parameter_change": bool(first["training_step"]["changed_backend_parameters"]),
        "fixed_frontend_identity": first["fixed_frontend_unchanged"],
        "morlet_spectral_coverage": True,
        "checkpoint_save": checkpoint.is_file(),
        "checkpoint_restore": first["checkpoint_restore"]["passed"],
        "metric_calculation": True,
        "result_generation": True,
        "identity_hashes": True,
        "repeatability": exact_match,
        "negative_path": negative_detected,
    }
    if not all(steps.values()):
        raise RuntimeError("one or more dry-run steps did not pass")
    report = {
        "schema": "phase0-dry-run-report-v1",
        "status": "PASS",
        "formal_experiment": False,
        "performance_claim_permitted": False,
        "test_split_accessed": False,
        **isolation_claim_fields(),
        "config_hash": config.sha256,
        "normalized_config_sha256": config.sha256,
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "effective_split_hashes": bundle.effective_split_hashes,
        "code_revision": _git_revision(),
        "code_identity": _source_code_identity(),
        "device": str(device),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "platform": platform.platform(),
        },
        "steps": steps,
        "fixed_frontend_unchanged": first["fixed_frontend_unchanged"],
        "fixed_frontend_identity": first["fixed_identity"],
        "morlet_spectral_coverage": spectral_coverage,
        "optimizer_ownership": first["optimizer_ownership"],
        "training_step": first["training_step"],
        "checkpoint_restore": first["checkpoint_restore"],
        "evaluation": first["evaluation"],
        "repeatability": {
            "exact_match": exact_match,
            "model_state_identity": first["model_state_identity"],
            "prediction_float32_bits": first["prediction_float32_bits"],
        },
        "negative_control": {
            "name": "duplicate_patch_id",
            "detected": negative_detected,
        },
        "limitations": [
            "synthetic fixture only",
            "not a formal experiment",
            "no CAM16 test split access",
            "no performance claim",
        ],
    }
    _write_json_exclusive(output / "report.json", report)
    return report


def _perform_preflight(
    config: ExperimentConfig,
    *,
    data_root: Path,
    release_path: Path,
) -> tuple[dict[str, Any], ManifestBundle]:
    if config.execution_kind != "formal_train":
        raise ValueError("preflight requires execution.kind=formal_train")
    passed = ["configuration"]
    not_applicable = ["patient_level_isolation"]
    blocked: list[str] = []
    manifest = data_root / Path(*PurePosixPath(str(config.data["manifest_relpath"])).parts)
    bundle = validate_manifest(data_root, manifest, check_files=True, reconcile_disk=True)
    passed.extend(["manifest_and_disk", "slide_id_isolation"])
    release = _read_json_strict(release_path)
    _validate_release_record(release)
    mapping_evidence: dict[str, Any] = {
        "status": PATIENT_LEVEL_ISOLATION,
        "reason": "patient identity is outside the CAM16 Phase 1 claim scope",
    }
    seed_audit = configure_determinism(int(config.training["seeds"][0]))
    device = torch.device(str(config.execution["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        blocked.append("configured_device")
        device = torch.device("cpu")
    model = FixedHEClassifier(frontend_backend=str(config.model["frontend_backend"])).to(device)
    fixed_identity = model.frontend.fixed_state_identity()
    passed.append("fixed_frontend")
    spectral_coverage = validate_spectral_coverage(generate_morlet_bundle())
    if spectral_coverage["status"] == "PASS":
        passed.append("morlet_spectral_coverage")
    else:
        blocked.append("morlet_spectral_coverage")
    optimizer = _optimizer(config, model)
    ownership = audit_optimizer_ownership(model, optimizer)
    passed.extend(["optimizer_ownership", "precision_and_determinism"])
    if config.execution["allow_test"] is False and config.evaluation["test_access"] == (
        "final-once-authorization-required"
    ):
        passed.append("test_access_disabled")
    else:
        blocked.append("test_access_disabled")
    if (
        release["phase0_closed"] is True
        and release["formal_training_authorized"] is True
        and release["test_access_authorized"] is False
        and release["external_blockers"] == []
        and release["patient_level_isolation"] == PATIENT_LEVEL_ISOLATION
        and release["patient_level_claim_allowed"] is PATIENT_LEVEL_CLAIM_ALLOWED
    ):
        passed.append("phase0_release")
    else:
        blocked.append("phase0_release")
    report = {
        "schema": "phase0-preflight-report-v1",
        "status": "PASS" if not blocked else "FAIL",
        "config_hash": config.sha256,
        "code_revision": _git_revision(),
        "code_identity": _source_code_identity(),
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "effective_split_hashes": bundle.effective_split_hashes,
        "split_counts": bundle.split_counts,
        "label_counts": bundle.label_counts,
        "disk_inventory": bundle.disk_inventory,
        "isolation": asdict(bundle.isolation),
        **isolation_claim_fields(),
        "patient_mapping": mapping_evidence,
        "fixed_frontend_identity": fixed_identity,
        "morlet_spectral_coverage": spectral_coverage,
        "optimizer_ownership": ownership,
        "determinism": seed_audit,
        "configured_device": str(config.execution["device"]),
        "effective_preflight_device": str(device),
        "passed_gates": passed,
        "not_applicable_gates": not_applicable,
        "blocking_gates": blocked,
        "training_started": False,
        "test_split_accessed": False,
    }
    return report, bundle


def run_preflight(
    config_path: Path | str,
    *,
    data_root: Path | str,
    release_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    report, _ = _perform_preflight(
        config,
        data_root=Path(data_root).resolve(),
        release_path=Path(release_path).resolve(),
    )
    _write_json_exclusive(Path(output_path), report)
    return report


def _formal_output_base(config: ExperimentConfig) -> Path:
    source_parent = config.source.parent
    base = source_parent.parent if source_parent.name == "configs" else source_parent
    return _runtime_path(base, str(config.execution["output_root"]))


def _epoch_number(path: Path) -> int:
    suffix = path.stem.removeprefix("epoch-")
    if len(suffix) != 4 or not suffix.isdigit() or path.stem != f"epoch-{suffix}":
        raise ValueError(f"invalid epoch artifact name: {path.name}")
    return int(suffix)


def _load_complete_epoch_history(
    seed_dir: Path, base_metadata: dict[str, Any]
) -> tuple[list[dict[str, Any]], Path | None, dict[str, Any] | None]:
    checkpoints = {_epoch_number(path): path for path in seed_dir.glob("epoch-*.pt")}
    reports = {_epoch_number(path): path for path in seed_dir.glob("epoch-*.json")}
    if set(checkpoints) != set(reports):
        raise ValueError("resume requires an immutable checkpoint/report pair for every epoch")
    if not checkpoints:
        return [], None, None
    indices = sorted(checkpoints)
    if indices != list(range(indices[-1] + 1)):
        raise ValueError("resume epoch history is not continuous from epoch zero")
    history: list[dict[str, Any]] = []
    latest_metadata: dict[str, Any] | None = None
    for epoch in indices:
        report = _read_json_strict(reports[epoch])
        checkpoint_identity = report.get("identities", {}).get("checkpoint_identity")
        optimizer_identity = report.get("identities", {}).get("optimizer_state_identity")
        expected_metadata = {
            **base_metadata,
            "checkpoint_identity": checkpoint_identity,
            "optimizer_state_identity": optimizer_identity,
            "epoch": epoch,
        }
        if (
            report.get("schema") != "formal-train-epoch-v1"
            or report.get("epoch") != epoch
            or report.get("seed") != base_metadata["seed"]
            or report.get("checkpoint") != checkpoints[epoch].name
            or report.get("checkpoint_file_sha256")
            != raw_sha256(checkpoints[epoch].read_bytes())
            or report.get("identities") != expected_metadata
            or not _is_sha256(checkpoint_identity)
            or not _is_sha256(optimizer_identity)
        ):
            raise ValueError(f"epoch {epoch} report identity is incomplete or mismatched")
        history.append(report)
        latest_metadata = expected_metadata
    return history, checkpoints[indices[-1]], latest_metadata


def _run_formal_seed(
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
    model = FixedHEClassifier(frontend_backend=str(config.model["frontend_backend"])).to(device)
    optimizer = _optimizer(config, model)
    fixed_identity = model.frontend.fixed_state_identity()
    base_metadata = {
        "code_revision": _git_revision(),
        "code_identity": _source_code_identity(),
        "config_hash": config.sha256,
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "effective_split_hashes": bundle.effective_split_hashes,
        "fixed_frontend_identity": fixed_identity,
        **isolation_claim_fields(),
        "seed": seed,
    }
    start_epoch = 0
    history: list[dict[str, Any]] = []
    if resume:
        history, latest, expected = _load_complete_epoch_history(seed_dir, base_metadata)
    else:
        latest, expected = None, None
    if latest is not None and expected is not None:
        latest_epoch = int(expected["epoch"])
        load_checkpoint(latest, model, optimizer, expected_metadata=expected)
        start_epoch = latest_epoch + 1
    best_metric = max(
        (float(item["evaluation"]["slide_auroc"]["value"]) for item in history),
        default=-float("inf"),
    )
    best_epoch = min(
        (
            int(item["epoch"])
            for item in history
            if float(item["evaluation"]["slide_auroc"]["value"]) == best_metric
        ),
        default=-1,
    )
    no_improvement = next(
        (
            offset
            for offset, item in enumerate(reversed(history))
            if int(item["epoch"]) == best_epoch
        ),
        len(history),
    )
    for epoch in range(start_epoch, int(config.training["max_epochs"])):
        model.train()
        loader = build_dataloader(
            train_dataset,
            batch_size=int(config.training["batch_size"]),
            seed=seed,
            epoch=epoch,
            num_workers=int(config.training["num_workers"]),
        )
        losses: list[float] = []
        for batch in loader:
            step = train_one_step(
                model,
                optimizer,
                batch["rgb"].to(device),
                batch["target"].to(device),
            )
            losses.append(step.loss)
        predictions = _prediction_ledger(
            model, val_dataset, config, device, seed=seed, epoch=epoch
        )
        evaluation = evaluate_predictions(
            predictions,
            context=_evaluation_context(config, bundle, val_dataset, model, seed=seed),
            fit_thresholds=True,
            ci_seed=seed,
        )
        metric = float(evaluation["slide_auroc"]["value"])
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1
        metadata = {
            **base_metadata,
            "checkpoint_identity": model_state_identity(model),
            "optimizer_state_identity": optimizer_state_identity(optimizer),
            "epoch": epoch,
        }
        checkpoint_path = seed_dir / f"epoch-{epoch:04d}.pt"
        save_checkpoint(checkpoint_path, model, optimizer, metadata)
        checkpoint_file_sha256 = raw_sha256(checkpoint_path.read_bytes())
        epoch_report = {
            "schema": "formal-train-epoch-v1",
            "epoch": epoch,
            "seed": seed,
            "batch_count": len(losses),
            "mean_training_loss": sum(losses) / len(losses),
            "evaluation": evaluation,
            "checkpoint": checkpoint_path.name,
            "checkpoint_file_sha256": checkpoint_file_sha256,
            "identities": metadata,
            "best_epoch": best_epoch,
            "test_split_accessed": False,
        }
        _write_json_exclusive(seed_dir / f"epoch-{epoch:04d}.json", epoch_report)
        history.append(epoch_report)
        if no_improvement >= int(config.training["early_stopping_patience"]):
            break
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_slide_auroc": best_metric,
        "epochs_completed": len(history),
        "status": "complete",
    }


def run_formal_training(
    config_path: Path | str,
    *,
    data_root: Path | str,
    release_path: Path | str,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the frozen train/validation protocol only after every preflight gate passes."""

    config = load_experiment_config(config_path)
    report, bundle = _perform_preflight(
        config,
        data_root=Path(data_root).resolve(),
        release_path=Path(release_path).resolve(),
    )
    if report["status"] != "PASS":
        raise Phase0BlockedError(
            "formal training preflight failed before training: "
            + ", ".join(report["blocking_gates"])
        )
    output_base = _formal_output_base(config)
    if output_base.exists() and not resume:
        raise FileExistsError(output_base)
    output_base.mkdir(parents=True, exist_ok=resume)
    normalized_path = output_base / "normalized_config.json"
    if resume:
        try:
            if normalized_path.read_bytes() != config.normalized_bytes + b"\n":
                raise Phase0BlockedError("resume normalized configuration identity mismatch")
        except OSError as error:
            raise Phase0BlockedError("resume normalized configuration is missing") from error
    else:
        with normalized_path.open("xb") as handle:
            handle.write(config.normalized_bytes + b"\n")
    device = torch.device(str(config.execution["device"]))
    train_dataset = PatchDataset(bundle, "train")
    val_dataset = PatchDataset(bundle, "val")
    run_results: list[dict[str, Any]] = []
    for seed_value in config.training["seeds"]:
        seed = int(seed_value)
        try:
            result = _run_formal_seed(
                config,
                bundle,
                train_dataset,
                val_dataset,
                device,
                output_base,
                seed=seed,
                resume=resume,
            )
        except (OSError, RuntimeError, ValueError) as error:
            result = {
                "seed": seed,
                "status": "failed",
                "failure_reason": f"{type(error).__name__}: {error}",
                "automatic_retry": False,
                **isolation_claim_fields(),
            }
            failure_path = output_base / f"seed-{seed}-failure.json"
            if not failure_path.exists():
                _write_json_exclusive(failure_path, result)
        run_results.append(result)
    valid_results = [item for item in run_results if item["status"] == "complete"]
    aggregation = (
        aggregate_seed_results(tuple(run_results))
        if valid_results
        else {"status": "undefined", "reason": "no_valid_seed_run"}
    )
    final_report = {
        "schema": "formal-training-summary-v1",
        "config_hash": config.sha256,
        "code_identity": _source_code_identity(),
        "source_manifest_sha256": bundle.source_manifest_sha256,
        **isolation_claim_fields(),
        "runs": run_results,
        "multi_seed_validation_slide_auroc": aggregation,
        "test_split_accessed": False,
        "phase0_preflight": "PASS",
    }
    _write_json_exclusive(output_base / "training_summary.json", final_report)
    return final_report
