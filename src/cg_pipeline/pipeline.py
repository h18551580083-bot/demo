"""Public preflight, synthetic dry-run, and guarded formal-training entry points."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, TypedDict, cast

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
    expected_batch_count,
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


class BatchContract(TypedDict):
    batch_size: int
    drop_last: bool
    train_rows: int
    train_batch_count: int
    maximum_optimizer_updates: int
    validation_rows: int
    validation_batch_count: int


class ReleaseIdentity(TypedDict):
    release_id: str
    annotated_tag: str
    formal_code_commit: str
    release_commit: str
    release_commit_parent_count: int
    release_commit_allowed_paths: list[str]


class ManifestIdentity(TypedDict):
    manifest_relpath: str
    manifest_hash_algorithm: str
    source_manifest_hash_rule: str
    effective_split_hash_rule: str
    source_manifest_sha256: str
    effective_split_hashes: dict[str, str]


class PreflightReport(TypedDict):
    schema: str
    created_at: str
    status: str
    config_hash: str
    normalized_config_sha256: str
    release_id: str
    batch_contract: BatchContract
    code_revision: str
    code_identity: str
    release_identity: ReleaseIdentity
    source_manifest_sha256: str
    effective_split_hashes: dict[str, str]
    manifest_identity: ManifestIdentity
    split_counts: dict[str, int]
    label_counts: dict[str, int]
    disk_inventory: dict[str, Any]
    isolation: dict[str, Any]
    isolation_claim: str
    patient_level_isolation: str
    patient_level_claim_allowed: bool
    patient_mapping: dict[str, Any]
    configured_device: str
    effective_preflight_device: str
    passed_gates: list[str]
    not_applicable_gates: list[str]
    blocking_gates: list[str]
    training_started: bool
    test_split_accessed: bool
    report_identity: str


class Phase0BlockedError(RuntimeError):
    """Formal training was prevented before its first batch."""


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_APPROVED_RELEASE_ID = "phase1-training-b32-v3"
_APPROVED_RELEASE_TAG = "phase1-training-b32-v3"
_RELEASE_COMMIT_ALLOWED_PATHS = (
    "configs/phase1_training_release_b32_v3.json",
    "docs/DECISIONS.md",
    "docs/PHASE1_TRAINING_RUNBOOK.md",
)
_PREFLIGHT_REPORT_FIELDS = frozenset(PreflightReport.__required_keys__)
_PREFLIGHT_INTERNAL_EVIDENCE_FIELDS = frozenset(
    {
        "fixed_frontend_identity",
        "morlet_spectral_coverage",
        "optimizer_ownership",
        "determinism",
    }
)
_PREFLIGHT_PASSED_GATES = [
    "configuration",
    "manifest_and_disk",
    "slide_id_isolation",
    "fixed_frontend",
    "morlet_spectral_coverage",
    "optimizer_ownership",
    "precision_and_determinism",
    "test_access_disabled",
    "phase1_training_release",
]


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


def _batch_contract(config: ExperimentConfig, bundle: ManifestBundle) -> BatchContract:
    batch_size = int(config.training["batch_size"])
    drop_last = False
    train_rows = int(bundle.split_counts["train"])
    validation_rows = int(bundle.split_counts["val"])
    train_batch_count = expected_batch_count(
        train_rows, batch_size, drop_last=drop_last
    )
    return {
        "batch_size": batch_size,
        "drop_last": drop_last,
        "train_rows": train_rows,
        "train_batch_count": train_batch_count,
        "maximum_optimizer_updates": train_batch_count
        * int(config.training["max_epochs"]),
        "validation_rows": validation_rows,
        "validation_batch_count": expected_batch_count(
            validation_rows, batch_size, drop_last=drop_last
        ),
    }


def _git_output(repository_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Phase0BlockedError(f"cannot validate Git release identity: {error}") from error
    return result.stdout.strip()


def _repository_root_for_config(config: ExperimentConfig) -> Path:
    candidate = config.source.parent
    root = Path(_git_output(candidate, "rev-parse", "--show-toplevel")).resolve()
    try:
        config.source.relative_to(root)
    except ValueError as error:
        raise Phase0BlockedError("formal configuration is outside the release repository") from error
    return root


def _validate_source_tree_membership(repository_root: Path) -> None:
    tracked = {
        PurePosixPath(line).as_posix()
        for line in _git_output(repository_root, "ls-files", "--", "src").splitlines()
        if line.endswith(".py")
    }
    observed = {
        path.relative_to(repository_root).as_posix()
        for path in (repository_root / "src").rglob("*.py")
    }
    if observed != tracked:
        raise Phase0BlockedError("release source tree contains untracked or missing Python code")


def _validate_git_release_identity(
    value: dict[str, Any], *, repository_root: Path, release_path: Path
) -> ReleaseIdentity:
    expected_release_path = repository_root / _RELEASE_COMMIT_ALLOWED_PATHS[0]
    if release_path.resolve() != expected_release_path.resolve():
        raise Phase0BlockedError("release path is not the approved Phase 1 release path")
    if value["release_id"] != _APPROVED_RELEASE_ID:
        raise Phase0BlockedError("release_id does not match the approved release identity")
    if value["annotated_tag"] != _APPROVED_RELEASE_TAG:
        raise Phase0BlockedError("annotated tag does not match the approved release identity")
    if value["release_commit_allowed_paths"] != list(_RELEASE_COMMIT_ALLOWED_PATHS):
        raise Phase0BlockedError("release commit whitelist does not match the approved paths")

    head = _git_output(repository_root, "rev-parse", "HEAD")
    parents = _git_output(repository_root, "rev-list", "--parents", "-n", "1", "HEAD").split()
    if not parents or parents[0] != head or len(parents) != 2:
        raise Phase0BlockedError("release commit must have exactly one parent")
    parent = parents[1]
    if parent != value["formal_code_commit"]:
        raise Phase0BlockedError("release commit parent does not match formal_code_commit")

    tag_type = _git_output(repository_root, "cat-file", "-t", value["annotated_tag"])
    if tag_type != "tag":
        raise Phase0BlockedError("formal release tag must be an annotated tag")
    tag_commit = _git_output(
        repository_root, "rev-parse", f"{value['annotated_tag']}^{{}}"
    )
    if tag_commit != head:
        raise Phase0BlockedError("annotated tag does not resolve to the release commit")

    changed_paths = tuple(
        line.replace("\\", "/")
        for line in _git_output(
            repository_root,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "--no-renames",
            "-r",
            parent,
            head,
        ).splitlines()
        if line
    )
    if changed_paths != _RELEASE_COMMIT_ALLOWED_PATHS:
        raise Phase0BlockedError("release commit changed paths outside the approved whitelist")
    if _git_output(repository_root, "status", "--porcelain", "--untracked-files=no"):
        raise Phase0BlockedError("tracked release worktree is not clean")
    _validate_source_tree_membership(repository_root)
    return {
        "release_id": value["release_id"],
        "annotated_tag": value["annotated_tag"],
        "formal_code_commit": parent,
        "release_commit": head,
        "release_commit_parent_count": 1,
        "release_commit_allowed_paths": list(changed_paths),
    }


def _validate_release_record(
    value: dict[str, Any],
    config: ExperimentConfig,
    batch_contract: BatchContract,
    bundle: ManifestBundle,
    *,
    repository_root: Path,
    release_path: Path,
) -> ReleaseIdentity:
    expected = {
        "schema",
        "release_id",
        "supersedes_release_id",
        "phase0_release_tag",
        "release_id_role",
        "annotated_tag",
        "formal_code_commit",
        "release_commit_allowed_paths",
        "config_hash",
        "normalized_config_sha256",
        "run_id",
        "run_id_role",
        "manifest_relpath",
        "manifest_hash_algorithm",
        "source_manifest_hash_rule",
        "effective_split_hash_rule",
        "source_manifest_sha256",
        "effective_split_hashes",
        "batch_size",
        "drop_last",
        "expected_train_rows",
        "expected_train_batch_count",
        "maximum_optimizer_updates",
        "expected_validation_rows",
        "expected_validation_batch_count",
        "phase0_closed",
        "formal_training_authorized",
        "external_blockers",
        "patient_level_isolation",
        "patient_level_claim_allowed",
        "test_access_authorized",
    }
    if set(value) != expected:
        raise Phase0BlockedError(
            "release record fields do not match phase1-training-release-v2"
        )
    if value["schema"] != "phase1-training-release-v2":
        raise Phase0BlockedError(
            "release record schema is not phase1-training-release-v2"
        )
    if value["release_id"] != _APPROVED_RELEASE_ID:
        raise Phase0BlockedError("release_id does not match phase1-training-b32-v3")
    if value["supersedes_release_id"] != "phase1-training-b32-v2":
        raise Phase0BlockedError("release does not supersede the approved v2 release")
    if value["phase0_release_tag"] != "phase0-closed-v1":
        raise Phase0BlockedError("training release must bind phase0-closed-v1")
    for key in ("config_hash", "normalized_config_sha256", "source_manifest_sha256"):
        if not _is_sha256(value[key]):
            raise Phase0BlockedError(f"release {key} is not a canonical SHA-256")
    if value["config_hash"] != config.sha256 or value["normalized_config_sha256"] != (
        config.sha256
    ):
        raise Phase0BlockedError("training release config hash does not match normalized config")
    if value["run_id"] != config.execution["run_id"]:
        raise Phase0BlockedError("training release run_id does not match configuration")
    if value["release_id_role"] != "release-governance-identity" or value[
        "run_id_role"
    ] != "unchanged-training-config-identity":
        raise Phase0BlockedError("release and run identity roles are not explicit")
    if value["manifest_relpath"] != config.data["manifest_relpath"]:
        raise Phase0BlockedError("release manifest path does not match configuration")
    if value["manifest_hash_algorithm"] != "sha256":
        raise Phase0BlockedError("release manifest hash algorithm is not sha256")
    if value["source_manifest_hash_rule"] != "raw-file-bytes-v1":
        raise Phase0BlockedError("release source manifest hash rule is invalid")
    if value["effective_split_hash_rule"] != "cg/cam16-eval-manifest/v1":
        raise Phase0BlockedError("release effective split hash rule is invalid")
    effective = value["effective_split_hashes"]
    if not isinstance(effective, dict) or set(effective) != {"train", "val"}:
        raise Phase0BlockedError("release must bind only train and val effective split hashes")
    if any(not _is_sha256(effective[split]) for split in ("train", "val")):
        raise Phase0BlockedError("release effective split hash is not a canonical SHA-256")
    if value["source_manifest_sha256"] != bundle.source_manifest_sha256:
        raise Phase0BlockedError("source manifest identity does not match the approved release")
    if any(
        effective[split] != bundle.effective_split_hashes[split]
        for split in ("train", "val")
    ):
        raise Phase0BlockedError("effective split identity does not match the approved release")
    for key in (
        "batch_size",
        "expected_train_rows",
        "expected_train_batch_count",
        "maximum_optimizer_updates",
        "expected_validation_rows",
        "expected_validation_batch_count",
    ):
        if not isinstance(value[key], int) or isinstance(value[key], bool):
            raise Phase0BlockedError(f"release {key} must be an integer")
    expected_batch_fields = {
        "batch_size": batch_contract["batch_size"],
        "drop_last": batch_contract["drop_last"],
        "expected_train_rows": batch_contract["train_rows"],
        "expected_train_batch_count": batch_contract["train_batch_count"],
        "maximum_optimizer_updates": batch_contract["maximum_optimizer_updates"],
        "expected_validation_rows": batch_contract["validation_rows"],
        "expected_validation_batch_count": batch_contract["validation_batch_count"],
    }
    if any(
        value[key] != expected_value
        for key, expected_value in expected_batch_fields.items()
    ):
        raise Phase0BlockedError("training release batch contract does not match manifest/config")
    if any(
        not isinstance(value[key], bool)
        for key in (
            "phase0_closed",
            "formal_training_authorized",
            "patient_level_claim_allowed",
            "test_access_authorized",
            "drop_last",
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
    return _validate_git_release_identity(
        value, repository_root=repository_root, release_path=release_path
    )


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


def _git_revision(repository_root: Path = _REPOSITORY_ROOT) -> str:
    return _git_output(repository_root, "rev-parse", "HEAD")


def _source_code_identity(repository_root: Path = _REPOSITORY_ROOT) -> str:
    digest = hashlib.sha256()
    for root in (repository_root / "src",):
        for path in sorted(root.rglob("*.py"), key=lambda item: item.as_posix()):
            relative = path.relative_to(repository_root).as_posix().encode("utf-8")
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
    repository_root: Path = _REPOSITORY_ROOT,
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
        code_identity=_source_code_identity(repository_root),
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
    repository_root = _repository_root_for_config(config)
    passed = ["configuration"]
    not_applicable = ["patient_level_isolation"]
    blocked: list[str] = []
    manifest = data_root / Path(*PurePosixPath(str(config.data["manifest_relpath"])).parts)
    bundle = validate_manifest(
        data_root,
        manifest,
        check_files=True,
        reconcile_disk=True,
        effective_hash_splits=("train", "val"),
    )
    passed.extend(["manifest_and_disk", "slide_id_isolation"])
    release = _read_json_strict(release_path)
    batch_contract = _batch_contract(config, bundle)
    release_identity = _validate_release_record(
        release,
        config,
        batch_contract,
        bundle,
        repository_root=repository_root,
        release_path=release_path,
    )
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
        passed.append("phase1_training_release")
    else:
        blocked.append("phase1_training_release")
    report = {
        "schema": "phase1-training-preflight-report-v2",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS" if not blocked else "FAIL",
        "config_hash": config.sha256,
        "normalized_config_sha256": config.sha256,
        "release_id": release["release_id"],
        "batch_contract": batch_contract,
        "code_revision": _git_revision(repository_root),
        "code_identity": _source_code_identity(repository_root),
        "release_identity": release_identity,
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "effective_split_hashes": bundle.effective_split_hashes,
        "manifest_identity": {
            "manifest_relpath": release["manifest_relpath"],
            "manifest_hash_algorithm": release["manifest_hash_algorithm"],
            "source_manifest_hash_rule": release["source_manifest_hash_rule"],
            "effective_split_hash_rule": release["effective_split_hash_rule"],
            "source_manifest_sha256": bundle.source_manifest_sha256,
            "effective_split_hashes": {
                "train": bundle.effective_split_hashes["train"],
                "val": bundle.effective_split_hashes["val"],
            },
        },
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
) -> PreflightReport:
    config = load_experiment_config(config_path)
    detailed_report, _ = _perform_preflight(
        config,
        data_root=Path(data_root).resolve(),
        release_path=Path(release_path).resolve(),
    )
    repository_root = _repository_root_for_config(config)
    expected_output = (
        repository_root / "artifacts" / "preflight" / _APPROVED_RELEASE_ID / "preflight.json"
    ).resolve()
    if Path(output_path).resolve() != expected_output:
        raise Phase0BlockedError(
            "preflight report path must be artifacts/preflight/"
            f"{_APPROVED_RELEASE_ID}/preflight.json"
        )
    report = {
        key: value
        for key, value in detailed_report.items()
        if key not in _PREFLIGHT_INTERNAL_EVIDENCE_FIELDS
    }
    report["report_identity"] = _preflight_report_identity(report)
    _write_json_exclusive(Path(output_path), report)
    return cast(PreflightReport, report)


def _preflight_report_identity(report: dict[str, Any]) -> str:
    material = {key: value for key, value in report.items() if key != "report_identity"}
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return raw_sha256(encoded)


def _validate_preflight_report_for_training(
    config: ExperimentConfig,
    *,
    data_root: Path,
    release_path: Path,
    preflight_report_path: Path,
) -> tuple[PreflightReport, ManifestBundle, ReleaseIdentity, Path]:
    report = _read_json_strict(preflight_report_path)
    if set(report) != _PREFLIGHT_REPORT_FIELDS:
        raise Phase0BlockedError("preflight report fields do not match the frozen schema")
    if report.get("report_identity") != _preflight_report_identity(report):
        raise Phase0BlockedError("preflight report identity is missing or mismatched")
    repository_root = _repository_root_for_config(config)
    expected_report_path = (
        repository_root / "artifacts" / "preflight" / _APPROVED_RELEASE_ID / "preflight.json"
    ).resolve()
    if preflight_report_path.resolve() != expected_report_path:
        raise Phase0BlockedError("preflight report path does not match the approved release identity")
    manifest = data_root / Path(*PurePosixPath(str(config.data["manifest_relpath"])).parts)
    bundle = validate_manifest(
        data_root,
        manifest,
        check_files=True,
        reconcile_disk=True,
        effective_hash_splits=("train", "val"),
    )
    release = _read_json_strict(release_path)
    batch_contract = _batch_contract(config, bundle)
    release_identity = _validate_release_record(
        release,
        config,
        batch_contract,
        bundle,
        repository_root=repository_root,
        release_path=release_path,
    )
    expected_manifest_identity: ManifestIdentity = {
        "manifest_relpath": release["manifest_relpath"],
        "manifest_hash_algorithm": release["manifest_hash_algorithm"],
        "source_manifest_hash_rule": release["source_manifest_hash_rule"],
        "effective_split_hash_rule": release["effective_split_hash_rule"],
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "effective_split_hashes": dict(bundle.effective_split_hashes),
    }
    checks = {
        "schema": report.get("schema") == "phase1-training-preflight-report-v2",
        "status": report.get("status") == "PASS",
        "blocking_gates": report.get("blocking_gates") == [],
        "training_started": report.get("training_started") is False,
        "test_split_accessed": report.get("test_split_accessed") is False,
        "config_hash": report.get("config_hash") == config.sha256,
        "normalized_config_sha256": report.get("normalized_config_sha256") == config.sha256,
        "release_id": report.get("release_id") == release["release_id"],
        "release_identity": report.get("release_identity") == release_identity,
        "code_revision": report.get("code_revision") == _git_revision(repository_root),
        "code_identity": report.get("code_identity") == _source_code_identity(repository_root),
        "source_manifest_sha256": report.get("source_manifest_sha256")
        == bundle.source_manifest_sha256,
        "effective_split_hashes": report.get("effective_split_hashes")
        == bundle.effective_split_hashes,
        "manifest_identity": report.get("manifest_identity") == expected_manifest_identity,
        "batch_contract": report.get("batch_contract") == batch_contract,
        "split_counts": report.get("split_counts") == bundle.split_counts,
        "label_counts": report.get("label_counts") == bundle.label_counts,
        "disk_inventory": report.get("disk_inventory") == bundle.disk_inventory,
        "isolation": report.get("isolation") == asdict(bundle.isolation),
        "isolation_claim": report.get("isolation_claim")
        == isolation_claim_fields()["isolation_claim"],
        "patient_level_isolation": report.get("patient_level_isolation")
        == PATIENT_LEVEL_ISOLATION,
        "patient_level_claim_allowed": report.get("patient_level_claim_allowed")
        is PATIENT_LEVEL_CLAIM_ALLOWED,
        "patient_mapping": report.get("patient_mapping")
        == {
            "status": PATIENT_LEVEL_ISOLATION,
            "reason": "patient identity is outside the CAM16 Phase 1 claim scope",
        },
        "configured_device": report.get("configured_device")
        == str(config.execution["device"]),
        "effective_preflight_device": report.get("effective_preflight_device")
        == str(config.execution["device"]),
        "passed_gates": report.get("passed_gates") == _PREFLIGHT_PASSED_GATES,
        "not_applicable_gates": report.get("not_applicable_gates")
        == ["patient_level_isolation"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise Phase0BlockedError(
            "preflight report does not match current formal identities: " + ", ".join(failed)
        )
    return cast(PreflightReport, report), bundle, release_identity, repository_root


def _formal_output_base(config: ExperimentConfig) -> Path:
    source_parent = config.source.parent
    base = source_parent.parent if source_parent.name == "configs" else source_parent
    return _runtime_path(base, str(config.execution["output_root"]))


def _formal_datasets(bundle: ManifestBundle) -> tuple[PatchDataset, PatchDataset]:
    """Build only the train and validation datasets used by formal training."""

    return PatchDataset(bundle, "train"), PatchDataset(bundle, "val")


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
    repository_root: Path,
    release_identity: dict[str, Any],
    preflight_report_identity: str,
) -> dict[str, Any]:
    configure_determinism(seed)
    seed_dir = output_base / f"seed-{seed}"
    seed_dir.mkdir(parents=True, exist_ok=resume)
    model = FixedHEClassifier(frontend_backend=str(config.model["frontend_backend"])).to(device)
    optimizer = _optimizer(config, model)
    fixed_identity = model.frontend.fixed_state_identity()
    base_metadata = {
        "code_revision": _git_revision(repository_root),
        "code_identity": _source_code_identity(repository_root),
        "config_hash": config.sha256,
        "release_identity": release_identity,
        "preflight_report_identity": preflight_report_identity,
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
            context=_evaluation_context(
                config,
                bundle,
                val_dataset,
                model,
                seed=seed,
                repository_root=repository_root,
            ),
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
    preflight_report_path: Path | str,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the frozen train/validation protocol only after every preflight gate passes."""

    config = load_experiment_config(config_path)
    report, bundle, release_identity, repository_root = _validate_preflight_report_for_training(
        config,
        data_root=Path(data_root).resolve(),
        release_path=Path(release_path).resolve(),
        preflight_report_path=Path(preflight_report_path).resolve(),
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
    train_dataset, val_dataset = _formal_datasets(bundle)
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
                repository_root=repository_root,
                release_identity=release_identity,
                preflight_report_identity=str(report["report_identity"]),
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
        "code_revision": _git_revision(repository_root),
        "code_identity": _source_code_identity(repository_root),
        "release_identity": release_identity,
        "preflight_report_identity": report["report_identity"],
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "effective_split_hashes": bundle.effective_split_hashes,
        **isolation_claim_fields(),
        "runs": run_results,
        "multi_seed_validation_slide_auroc": aggregation,
        "test_split_accessed": False,
        "phase1_training_preflight": "PASS",
    }
    _write_json_exclusive(output_base / "training_summary.json", final_report)
    return final_report
