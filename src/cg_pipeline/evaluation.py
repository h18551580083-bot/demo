"""Manifest-bounded, validation-only evaluation arithmetic."""

from __future__ import annotations

import json
import math
import random
import struct
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .identity import domain_hash, raw_sha256


class EvaluationContractError(ValueError):
    """Prediction completeness, split access, or metric arithmetic failed."""


def _float32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", value))[0]


@dataclass(frozen=True)
class Prediction:
    patch_id: str
    slide_id: str
    split: str
    patch_target: int
    slide_target: int
    logit: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "logit", _float32(self.logit))


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
    config_hash: str
    checkpoint_identity: str
    effective_validation_manifest_sha256: str
    seed: int
    identity: str


@dataclass(frozen=True)
class FinalTestAuthorization:
    schema: str
    test_access_authorized: bool
    config_hash: str
    code_identity: str
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
    config_hash: str
    code_identity: str
    source_manifest_sha256: str
    effective_manifest_sha256: str
    fixed_frontend_identity: dict[str, str]
    checkpoint_identity: str
    seed: int
    test_authorization: FinalTestAuthorization | None = None


@dataclass(frozen=True)
class SlidePrediction:
    slide_id: str
    split: str
    target: int
    logit: float
    source_patch_id: str


@dataclass(frozen=True)
class AurocResult:
    value: float
    winning_pairs: int
    tied_pairs: int
    positive_count: int
    negative_count: int


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    youden_numerator: int
    youden_denominator: int


def _read_json_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=lambda pairs: _unique_pairs(pairs),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise EvaluationContractError(f"cannot validate final-test authorization: {error}") from error
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
        "config_hash",
        "code_identity",
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
        "config_hash",
        "code_identity",
        "source_manifest_sha256",
        "effective_test_manifest_sha256",
        "checkpoint_identity",
        "validation_threshold_identity",
        "approval_evidence_sha256",
    )
    if any(not isinstance(document[key], str) or not _is_sha256(document[key]) for key in identity_fields):
        raise EvaluationContractError("final-test authorization contains an invalid identity")
    if any(not isinstance(document[key], str) or not document[key] for key in ("approved_by", "approved_at")):
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


def _validate_scores_targets(scores: tuple[float, ...], targets: tuple[int, ...]) -> None:
    if not scores or len(scores) != len(targets):
        raise EvaluationContractError("scores and targets must be nonempty and aligned")
    if any(not math.isfinite(score) for score in scores):
        raise EvaluationContractError("non-finite score or logit")
    if any(target not in (0, 1) for target in targets):
        raise EvaluationContractError("binary targets must be integer 0 or 1")


def exact_auroc(scores: tuple[float, ...], targets: tuple[int, ...]) -> AurocResult:
    _validate_scores_targets(scores, targets)
    positive_count = sum(targets)
    negative_count = len(targets) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise EvaluationContractError("AUROC requires both classes")
    ordered = sorted(zip(scores, targets), key=lambda item: item[0])
    wins = 0
    ties = 0
    negatives_before = 0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group = ordered[index:end]
        group_positive = sum(target for _, target in group)
        group_negative = len(group) - group_positive
        wins += group_positive * negatives_before
        ties += group_positive * group_negative
        negatives_before += group_negative
        index = end
    denominator = 2 * positive_count * negative_count
    value = (2 * wins + ties) / denominator
    return AurocResult(value, wins, ties, positive_count, negative_count)


def stratified_slide_bootstrap_auroc(
    scores: tuple[float, ...],
    targets: tuple[int, ...],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    _validate_scores_targets(scores, targets)
    if replicates != 2000:
        raise EvaluationContractError("the frozen confidence interval requires 2000 replicates")
    positive = tuple(score for score, target in zip(scores, targets) if target == 1)
    negative = tuple(score for score, target in zip(scores, targets) if target == 0)
    if not positive or not negative:
        raise EvaluationContractError("stratified bootstrap requires both slide classes")
    generator = random.Random(seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled_positive = tuple(generator.choice(positive) for _ in positive)
        sampled_negative = tuple(generator.choice(negative) for _ in negative)
        sampled_scores = sampled_negative + sampled_positive
        sampled_targets = (0,) * len(sampled_negative) + (1,) * len(sampled_positive)
        values.append(exact_auroc(sampled_scores, sampled_targets).value)
    ordered = sorted(values)
    lower_index = math.ceil(0.025 * replicates) - 1
    upper_index = math.ceil(0.975 * replicates) - 1
    return {
        "method": "stratified-slide-bootstrap-percentile-python-mt19937-v1",
        "confidence_level": "0.95",
        "replicates": replicates,
        "seed": seed,
        "lower": ordered[lower_index],
        "upper": ordered[upper_index],
        "lower_order_index": lower_index,
        "upper_order_index": upper_index,
    }


def aggregate_slides(predictions: tuple[Prediction, ...]) -> tuple[SlidePrediction, ...]:
    if not predictions:
        raise EvaluationContractError("prediction ledger is empty")
    groups: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        if not prediction.patch_id or not prediction.slide_id:
            raise EvaluationContractError("prediction identity is missing")
        if not math.isfinite(prediction.logit):
            raise EvaluationContractError("non-finite prediction logit")
        groups[prediction.slide_id].append(prediction)
    output: list[SlidePrediction] = []
    for slide_id in sorted(groups, key=lambda value: value.encode("utf-8")):
        rows = groups[slide_id]
        targets = {row.slide_target for row in rows}
        splits = {row.split for row in rows}
        if len(targets) != 1:
            raise EvaluationContractError(f"inconsistent slide target: {slide_id}")
        if len(splits) != 1:
            raise EvaluationContractError(f"slide crosses ledger splits: {slide_id}")
        maximum = max(row.logit for row in rows)
        provenance = min(
            (row.patch_id for row in rows if row.logit == maximum),
            key=lambda value: value.encode("utf-8"),
        )
        output.append(
            SlidePrediction(slide_id, rows[0].split, targets.pop(), maximum, provenance)
        )
    return tuple(output)


def _counts(
    scores: tuple[float, ...], targets: tuple[int, ...], threshold: float
) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for score, target in zip(scores, targets):
        predicted = int(score >= threshold)
        if predicted and target:
            tp += 1
        elif predicted:
            fp += 1
        elif target:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def select_youden_threshold(
    scores: tuple[float, ...], targets: tuple[int, ...]
) -> ThresholdResult:
    _validate_scores_targets(scores, targets)
    positive_count = sum(targets)
    negative_count = len(targets) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise EvaluationContractError("Youden threshold requires both classes")
    best: ThresholdResult | None = None
    for threshold in sorted(set(scores)):
        tp, fp, tn, fn = _counts(scores, targets, threshold)
        numerator = tp * negative_count + tn * positive_count - positive_count * negative_count
        candidate = ThresholdResult(
            threshold,
            tp,
            fp,
            tn,
            fn,
            numerator,
            positive_count * negative_count,
        )
        if best is None or (candidate.youden_numerator, candidate.threshold) > (
            best.youden_numerator,
            best.threshold,
        ):
            best = candidate
    if best is None:  # pragma: no cover - guarded by nonempty score validation
        raise EvaluationContractError("Youden candidate set is empty")
    return best


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def binary_metrics(
    scores: tuple[float, ...], targets: tuple[int, ...], *, threshold: float
) -> dict[str, Any]:
    _validate_scores_targets(scores, targets)
    if not math.isfinite(threshold):
        raise EvaluationContractError("threshold must be finite")
    tp, fp, tn, fn = _counts(scores, targets, threshold)
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    ppv = _ratio(tp, tp + fp)
    npv = _ratio(tn, tn + fn)
    accuracy = _ratio(tp + tn, tp + fp + tn + fn)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    balanced = None if sensitivity is None or specificity is None else (sensitivity + specificity) / 2
    youden = None if sensitivity is None or specificity is None else sensitivity + specificity - 1
    output: dict[str, Any] = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "youden_j": youden,
    }
    reasons = {
        "sensitivity": "no_positive_targets",
        "specificity": "no_negative_targets",
        "ppv": "no_predicted_positives",
        "npv": "no_predicted_negatives",
        "f1": "no_positive_or_predicted_positive",
    }
    for name, reason in reasons.items():
        if output[name] is None:
            output[f"{name}_undefined_reason"] = reason
    return output


def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _calibration(scores: tuple[float, ...], targets: tuple[int, ...]) -> dict[str, float]:
    probabilities = tuple(_stable_sigmoid(value) for value in scores)
    count = len(scores)
    mean_score = math.fsum(probabilities) / count
    prevalence = sum(targets) / count
    brier = math.fsum((score - target) ** 2 for score, target in zip(probabilities, targets)) / count
    ece_terms: list[float] = []
    for index in range(10):
        selected = [
            position
            for position, score in enumerate(probabilities)
            if (index / 10 <= score < (index + 1) / 10) or (index == 9 and score == 1.0)
        ]
        if selected:
            bin_score = math.fsum(probabilities[position] for position in selected) / len(selected)
            event_rate = sum(targets[position] for position in selected) / len(selected)
            ece_terms.append((len(selected) / count) * abs(bin_score - event_rate))
        else:
            ece_terms.append(0.0)
    return {
        "mean_score": mean_score,
        "prevalence": prevalence,
        "calibration_bias": mean_score - prevalence,
        "brier": brier,
        "ece10": math.fsum(ece_terms),
    }


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
        raise EvaluationContractError("prediction ledger is incomplete, extra, or identity-mismatched")
    identities = (
        context.config_hash,
        context.code_identity,
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
        config_hash=thresholds.config_hash,
        checkpoint_identity=thresholds.checkpoint_identity,
        effective_manifest_sha256=thresholds.effective_validation_manifest_sha256,
        seed=thresholds.seed,
        patch_threshold=thresholds.patch_threshold,
        slide_threshold=thresholds.slide_threshold,
    )
    if (
        thresholds.identity != recomputed_threshold_identity
        or thresholds.config_hash != context.config_hash
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
        "config_hash": context.config_hash,
        "code_identity": context.code_identity,
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
        config_hash=context.config_hash,
        checkpoint_identity=context.checkpoint_identity,
        effective_manifest_sha256=context.effective_manifest_sha256,
        seed=context.seed,
        patch_threshold=patch_threshold,
        slide_threshold=slide_threshold,
    )


def _threshold_identity_values(
    *,
    config_hash: str,
    checkpoint_identity: str,
    effective_manifest_sha256: str,
    seed: int,
    patch_threshold: float,
    slide_threshold: float,
) -> str:
    header = {
        "checkpoint_identity": checkpoint_identity,
        "config_hash": config_hash,
        "effective_manifest_sha256": effective_manifest_sha256,
        "patch_threshold": _identity_material(patch_threshold),
        "payload_length": 0,
        "seed": seed,
        "slide_threshold": _identity_material(slide_threshold),
        "split": "val",
    }
    return domain_hash("cg/cam16-validation-thresholds/v1", header)


def evaluate_predictions(
    predictions: tuple[Prediction, ...],
    *,
    context: EvaluationContext,
    fit_thresholds: bool,
    ci_seed: int,
    frozen_thresholds: FrozenThresholds | None = None,
) -> dict[str, Any]:
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
    ordered = tuple(sorted(predictions, key=lambda row: row.patch_id.encode("utf-8")))
    patch_scores = tuple(row.logit for row in ordered)
    patch_targets = tuple(row.patch_target for row in ordered)
    slides = aggregate_slides(ordered)
    slide_scores = tuple(row.logit for row in slides)
    slide_targets = tuple(row.target for row in slides)
    patch_auroc = exact_auroc(patch_scores, patch_targets)
    slide_auroc = exact_auroc(slide_scores, slide_targets)
    report: dict[str, Any] = {
        "contract_id": "cam16-eval-v1",
        "split": split,
        "primary_metric": "slide_auroc",
        "patch_count": len(ordered),
        "slide_count": len(slides),
        "patch_auroc": asdict(patch_auroc),
        "slide_auroc": asdict(slide_auroc),
        "slide_auroc_ci": stratified_slide_bootstrap_auroc(
            slide_scores, slide_targets, seed=ci_seed, replicates=2000
        ),
        "patch_calibration": _calibration(patch_scores, patch_targets),
        "slide_calibration": _calibration(slide_scores, slide_targets),
        "identities": {
            "config_hash": context.config_hash,
            "code_identity": context.code_identity,
            "source_manifest_sha256": context.source_manifest_sha256,
            "effective_manifest_sha256": context.effective_manifest_sha256,
            "fixed_frontend_identity": context.fixed_frontend_identity,
            "checkpoint_identity": context.checkpoint_identity,
            "seed": context.seed,
        },
    }
    if split == "test":
        assert context.test_authorization is not None
        report["identities"]["test_authorization_artifact_sha256"] = (
            context.test_authorization.authorization_artifact_sha256
        )
        report["identities"]["test_approval_evidence_sha256"] = (
            context.test_authorization.approval_evidence_sha256
        )
    if fit_thresholds:
        patch_threshold = select_youden_threshold(patch_scores, patch_targets)
        slide_threshold = select_youden_threshold(slide_scores, slide_targets)
        report["patch_threshold"] = asdict(patch_threshold)
        report["slide_threshold"] = asdict(slide_threshold)
        report["patch_metrics"] = binary_metrics(
            patch_scores, patch_targets, threshold=patch_threshold.threshold
        )
        report["slide_metrics"] = binary_metrics(
            slide_scores, slide_targets, threshold=slide_threshold.threshold
        )
        report["threshold_identity"] = _threshold_identity(
            context, patch_threshold.threshold, slide_threshold.threshold
        )
    else:
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
    identity_header = {
        "payload_length": 0,
        "report": _identity_material(report),
        "schema": "cam16-result-v1",
    }
    report["result_identity"] = domain_hash("cg/cam16-result/v1", identity_header)
    return report
