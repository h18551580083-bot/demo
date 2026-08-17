"""Manifest-bounded, validation-only evaluation arithmetic."""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .evaluation_metrics import (
    EvaluationMetricError,
    Prediction,
    aggregate_slides,
    binary_metrics,
    exact_auroc,
    select_youden_threshold,
    stratified_slide_bootstrap_auroc,
)
from .evaluation_metrics import calibration as _calibration
from .identity import domain_hash, raw_sha256

EvaluationContractError = EvaluationMetricError


@dataclass(frozen=True)
class AuthorizedEvaluationRow:
    patch_id: str
    slide_id: str
    split: str
    patch_target: int
    slide_target: int


@dataclass(frozen=True)
class FrozenThresholds:
    patch_threshold: float
    slide_threshold: float
    checkpoint_identity: str
    effective_validation_manifest_sha256: str
    seed: int
    identity: str


@dataclass(frozen=True)
class FinalTestAuthorization:
    schema: str
    test_access_authorized: bool
    source_manifest_sha256: str
    effective_test_manifest_sha256: str
    checkpoint_identity: str
    validation_threshold_identity: str
    approval_evidence_sha256: str
    approved_by: str
    approved_at: str
    authorization_artifact_sha256: str
    authorization_path: Path
    approval_evidence_path: Path


@dataclass(frozen=True)
class EvaluationContext:
    split: str
    authorized_rows: tuple[AuthorizedEvaluationRow, ...]
    source_manifest_sha256: str
    effective_manifest_sha256: str
    fixed_frontend_identity: dict[str, str]
    checkpoint_identity: str
    seed: int
    test_authorization: FinalTestAuthorization | None = None


def _read_json_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=lambda pairs: _unique_pairs(pairs),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise EvaluationContractError(
            f"cannot validate final-test authorization: {error}"
        ) from error
    if not isinstance(value, dict):
        raise EvaluationContractError("final-test authorization must be a JSON object")
    return raw, value


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def load_final_test_authorization(
    authorization_path: Path | str, approval_evidence_path: Path | str
) -> FinalTestAuthorization:
    authorization_file = Path(authorization_path).resolve()
    evidence_file = Path(approval_evidence_path).resolve()
    raw, document = _read_json_object(authorization_file)
    expected_fields = {
        "schema",
        "test_access_authorized",
        "source_manifest_sha256",
        "effective_test_manifest_sha256",
        "checkpoint_identity",
        "validation_threshold_identity",
        "approval_evidence_sha256",
        "approved_by",
        "approved_at",
    }
    if set(document) != expected_fields:
        raise EvaluationContractError("final-test authorization fields are incomplete or unknown")
    if document["schema"] != "cam16-final-test-authorization-v1":
        raise EvaluationContractError("final-test authorization schema is invalid")
    if document["test_access_authorized"] is not True:
        raise EvaluationContractError("final-test access is not authorized")
    identity_fields = (
        "source_manifest_sha256",
        "effective_test_manifest_sha256",
        "checkpoint_identity",
        "validation_threshold_identity",
        "approval_evidence_sha256",
    )
    if any(
        not isinstance(document[key], str) or not _is_sha256(document[key])
        for key in identity_fields
    ):
        raise EvaluationContractError("final-test authorization contains an invalid identity")
    if any(
        not isinstance(document[key], str) or not document[key]
        for key in ("approved_by", "approved_at")
    ):
        raise EvaluationContractError("final-test authorization attribution is missing")
    try:
        evidence_hash = raw_sha256(evidence_file.read_bytes())
    except OSError as error:
        raise EvaluationContractError("final-test approval evidence is unavailable") from error
    if evidence_hash != document["approval_evidence_sha256"]:
        raise EvaluationContractError("final-test approval evidence identity mismatch")
    return FinalTestAuthorization(
        **document,
        authorization_artifact_sha256=raw_sha256(raw),
        authorization_path=authorization_file,
        approval_evidence_path=evidence_file,
    )


def _identity_material(value: Any) -> Any:
    if isinstance(value, float):
        return "float64:" + struct.pack(">d", value).hex()
    if isinstance(value, dict):
        return {key: _identity_material(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_identity_material(item) for item in value]
    return value


def _is_sha256(value: str) -> bool:
    return (
        value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _validate_ledger(predictions: tuple[Prediction, ...], context: EvaluationContext) -> None:
    expected = context.authorized_rows
    if not expected:
        raise EvaluationContractError("authorized evaluation ledger is empty")
    if any(row.split != context.split for row in expected):
        raise EvaluationContractError("authorized row split conflicts with evaluation context")
    expected_keys = [
        (row.patch_id, row.slide_id, row.split, row.patch_target, row.slide_target)
        for row in expected
    ]
    actual_keys = [
        (row.patch_id, row.slide_id, row.split, row.patch_target, row.slide_target)
        for row in predictions
    ]
    if len({row.patch_id for row in expected}) != len(expected):
        raise EvaluationContractError("authorized ledger contains duplicate patch_id")
    if sorted(actual_keys) != sorted(expected_keys):
        raise EvaluationContractError(
            "prediction ledger is incomplete, extra, or identity-mismatched"
        )
    identities = (
        context.source_manifest_sha256,
        context.effective_manifest_sha256,
        context.checkpoint_identity,
        *context.fixed_frontend_identity.values(),
    )
    if not context.fixed_frontend_identity or any(not _is_sha256(value) for value in identities):
        raise EvaluationContractError("evaluation context contains an invalid identity")


def _validate_test_authorization(
    context: EvaluationContext, thresholds: FrozenThresholds | None
) -> None:
    authorization = context.test_authorization
    if authorization is None or thresholds is None:
        raise EvaluationContractError("test split requires identity-bound final-once authorization")
    recomputed_threshold_identity = _threshold_identity_values(
        checkpoint_identity=thresholds.checkpoint_identity,
        effective_manifest_sha256=thresholds.effective_validation_manifest_sha256,
        seed=thresholds.seed,
        patch_threshold=thresholds.patch_threshold,
        slide_threshold=thresholds.slide_threshold,
    )
    if (
        thresholds.identity != recomputed_threshold_identity
        or thresholds.checkpoint_identity != context.checkpoint_identity
    ):
        raise EvaluationContractError("frozen threshold value or identity mismatch")
    reloaded = load_final_test_authorization(
        authorization.authorization_path, authorization.approval_evidence_path
    )
    if reloaded != authorization:
        raise EvaluationContractError("test authorization artifact changed after loading")
    expected_values = {
        "schema": "cam16-final-test-authorization-v1",
        "test_access_authorized": True,
        "source_manifest_sha256": context.source_manifest_sha256,
        "effective_test_manifest_sha256": context.effective_manifest_sha256,
        "checkpoint_identity": context.checkpoint_identity,
        "validation_threshold_identity": thresholds.identity,
    }
    if any(getattr(authorization, key) != value for key, value in expected_values.items()):
        raise EvaluationContractError("test authorization identity mismatch")


def _threshold_identity(
    context: EvaluationContext, patch_threshold: float, slide_threshold: float
) -> str:
    return _threshold_identity_values(
        checkpoint_identity=context.checkpoint_identity,
        effective_manifest_sha256=context.effective_manifest_sha256,
        seed=context.seed,
        patch_threshold=patch_threshold,
        slide_threshold=slide_threshold,
    )


def _threshold_identity_values(
    *,
    checkpoint_identity: str,
    effective_manifest_sha256: str,
    seed: int,
    patch_threshold: float,
    slide_threshold: float,
) -> str:
    header = {
        "checkpoint_identity": checkpoint_identity,
        "effective_manifest_sha256": effective_manifest_sha256,
        "patch_threshold": _identity_material(patch_threshold),
        "payload_length": 0,
        "seed": seed,
        "slide_threshold": _identity_material(slide_threshold),
        "split": "val",
    }
    return domain_hash("cg/cam16-validation-thresholds/v1", header)


def _validated_split(
    predictions: tuple[Prediction, ...],
    context: EvaluationContext,
    *,
    fit_thresholds: bool,
    ci_seed: int,
    frozen_thresholds: FrozenThresholds | None,
) -> str:
    if not predictions:
        raise EvaluationContractError("prediction ledger is empty")
    if ci_seed != context.seed:
        raise EvaluationContractError("bootstrap seed conflicts with evaluation context")
    if len({row.patch_id for row in predictions}) != len(predictions):
        raise EvaluationContractError("duplicate prediction ledger row")
    splits = {row.split for row in predictions}
    if len(splits) != 1:
        raise EvaluationContractError("one evaluation call must contain exactly one split")
    split = splits.pop()
    if split != context.split:
        raise EvaluationContractError("prediction split conflicts with evaluation context")
    _validate_ledger(predictions, context)
    if split == "test":
        if fit_thresholds:
            raise EvaluationContractError("test split cannot fit thresholds")
        _validate_test_authorization(context, frozen_thresholds)
    if fit_thresholds and split != "val":
        raise EvaluationContractError("thresholds may be fit on validation only")
    if split != "test" and not fit_thresholds:
        raise EvaluationContractError("non-test evaluation must fit validation thresholds")
    return split


def _base_evaluation_report(
    ordered: tuple[Prediction, ...], context: EvaluationContext, *, split: str, ci_seed: int
) -> tuple[dict[str, Any], tuple[float, ...], tuple[int, ...], tuple[float, ...], tuple[int, ...]]:
    patch_scores = tuple(row.logit for row in ordered)
    patch_targets = tuple(row.patch_target for row in ordered)
    slides = aggregate_slides(ordered)
    slide_scores = tuple(row.logit for row in slides)
    slide_targets = tuple(row.target for row in slides)
    report: dict[str, Any] = {
        "contract_id": "cam16-eval-v1",
        "split": split,
        "primary_metric": "slide_auroc",
        "patch_count": len(ordered),
        "slide_count": len(slides),
        "patch_auroc": asdict(exact_auroc(patch_scores, patch_targets)),
        "slide_auroc": asdict(exact_auroc(slide_scores, slide_targets)),
        "slide_auroc_ci": stratified_slide_bootstrap_auroc(
            slide_scores, slide_targets, seed=ci_seed, replicates=2000
        ),
        "patch_calibration": _calibration(patch_scores, patch_targets),
        "slide_calibration": _calibration(slide_scores, slide_targets),
        "identities": {
            "source_manifest_sha256": context.source_manifest_sha256,
            "effective_manifest_sha256": context.effective_manifest_sha256,
            "fixed_frontend_identity": context.fixed_frontend_identity,
            "checkpoint_identity": context.checkpoint_identity,
            "seed": context.seed,
        },
    }
    return report, patch_scores, patch_targets, slide_scores, slide_targets


def _apply_thresholds(
    report: dict[str, Any],
    context: EvaluationContext,
    scores: tuple[tuple[float, ...], tuple[int, ...], tuple[float, ...], tuple[int, ...]],
    *,
    fit_thresholds: bool,
    frozen_thresholds: FrozenThresholds | None,
) -> None:
    patch_scores, patch_targets, slide_scores, slide_targets = scores
    if fit_thresholds:
        patch = select_youden_threshold(patch_scores, patch_targets)
        slide = select_youden_threshold(slide_scores, slide_targets)
        report["patch_threshold"] = asdict(patch)
        report["slide_threshold"] = asdict(slide)
        report["patch_metrics"] = binary_metrics(
            patch_scores, patch_targets, threshold=patch.threshold
        )
        report["slide_metrics"] = binary_metrics(
            slide_scores, slide_targets, threshold=slide.threshold
        )
        report["threshold_identity"] = _threshold_identity(
            context, patch.threshold, slide.threshold
        )
        return
    assert frozen_thresholds is not None
    report["applied_thresholds"] = {
        "patch": frozen_thresholds.patch_threshold,
        "slide": frozen_thresholds.slide_threshold,
    }
    report["threshold_identity"] = frozen_thresholds.identity
    report["patch_metrics"] = binary_metrics(
        patch_scores, patch_targets, threshold=frozen_thresholds.patch_threshold
    )
    report["slide_metrics"] = binary_metrics(
        slide_scores, slide_targets, threshold=frozen_thresholds.slide_threshold
    )


def evaluate_predictions(
    predictions: tuple[Prediction, ...],
    *,
    context: EvaluationContext,
    fit_thresholds: bool,
    ci_seed: int,
    frozen_thresholds: FrozenThresholds | None = None,
) -> dict[str, Any]:
    split = _validated_split(
        predictions,
        context,
        fit_thresholds=fit_thresholds,
        ci_seed=ci_seed,
        frozen_thresholds=frozen_thresholds,
    )
    ordered = tuple(sorted(predictions, key=lambda row: row.patch_id.encode("utf-8")))
    report, patch_scores, patch_targets, slide_scores, slide_targets = _base_evaluation_report(
        ordered, context, split=split, ci_seed=ci_seed
    )
    if split == "test":
        assert context.test_authorization is not None
        report["identities"]["test_authorization_artifact_sha256"] = (
            context.test_authorization.authorization_artifact_sha256
        )
        report["identities"]["test_approval_evidence_sha256"] = (
            context.test_authorization.approval_evidence_sha256
        )
    _apply_thresholds(
        report,
        context,
        (patch_scores, patch_targets, slide_scores, slide_targets),
        fit_thresholds=fit_thresholds,
        frozen_thresholds=frozen_thresholds,
    )
    identity_header = {
        "payload_length": 0,
        "report": _identity_material(report),
        "schema": "cam16-result-v1",
    }
    report["result_identity"] = domain_hash("cg/cam16-result/v1", identity_header)
    return report
