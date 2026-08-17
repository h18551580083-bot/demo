"""Pure manifest-bounded validation metrics and threshold selection."""

from __future__ import annotations

import math
import random
import struct
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


class EvaluationMetricError(ValueError):
    """Metric inputs are empty, inconsistent, or numerically invalid."""


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


def _validate_scores_targets(scores: tuple[float, ...], targets: tuple[int, ...]) -> None:
    if not scores or len(scores) != len(targets):
        raise EvaluationMetricError("scores and targets must be nonempty and aligned")
    if any(not math.isfinite(score) for score in scores):
        raise EvaluationMetricError("non-finite score or logit")
    if any(target not in (0, 1) for target in targets):
        raise EvaluationMetricError("binary targets must be integer 0 or 1")


def exact_auroc(scores: tuple[float, ...], targets: tuple[int, ...]) -> AurocResult:
    _validate_scores_targets(scores, targets)
    positive_count = sum(targets)
    negative_count = len(targets) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise EvaluationMetricError("AUROC requires both classes")
    ordered = sorted(zip(scores, targets), key=lambda item: item[0])
    wins = ties = negatives_before = index = 0
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
    return AurocResult((2 * wins + ties) / denominator, wins, ties, positive_count, negative_count)


def stratified_slide_bootstrap_auroc(
    scores: tuple[float, ...], targets: tuple[int, ...], *, seed: int, replicates: int
) -> dict[str, Any]:
    _validate_scores_targets(scores, targets)
    if replicates != 2000:
        raise EvaluationMetricError("the frozen confidence interval requires 2000 replicates")
    positive = tuple(score for score, target in zip(scores, targets) if target == 1)
    negative = tuple(score for score, target in zip(scores, targets) if target == 0)
    if not positive or not negative:
        raise EvaluationMetricError("stratified bootstrap requires both slide classes")
    generator = random.Random(seed)
    values = []
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
        raise EvaluationMetricError("prediction ledger is empty")
    groups: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        if not prediction.patch_id or not prediction.slide_id:
            raise EvaluationMetricError("prediction identity is missing")
        if not math.isfinite(prediction.logit):
            raise EvaluationMetricError("non-finite prediction logit")
        groups[prediction.slide_id].append(prediction)
    output: list[SlidePrediction] = []
    for slide_id in sorted(groups, key=lambda value: value.encode("utf-8")):
        rows = groups[slide_id]
        targets = {row.slide_target for row in rows}
        splits = {row.split for row in rows}
        if len(targets) != 1:
            raise EvaluationMetricError(f"inconsistent slide target: {slide_id}")
        if len(splits) != 1:
            raise EvaluationMetricError(f"slide crosses ledger splits: {slide_id}")
        maximum = max(row.logit for row in rows)
        provenance = min(
            (row.patch_id for row in rows if row.logit == maximum),
            key=lambda value: value.encode("utf-8"),
        )
        output.append(SlidePrediction(slide_id, rows[0].split, targets.pop(), maximum, provenance))
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


def select_youden_threshold(scores: tuple[float, ...], targets: tuple[int, ...]) -> ThresholdResult:
    _validate_scores_targets(scores, targets)
    positive_count = sum(targets)
    negative_count = len(targets) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise EvaluationMetricError("Youden threshold requires both classes")
    best: ThresholdResult | None = None
    for threshold in sorted(set(scores)):
        tp, fp, tn, fn = _counts(scores, targets, threshold)
        numerator = tp * negative_count + tn * positive_count - positive_count * negative_count
        candidate = ThresholdResult(
            threshold, tp, fp, tn, fn, numerator, positive_count * negative_count
        )
        if best is None or (candidate.youden_numerator, candidate.threshold) > (
            best.youden_numerator,
            best.threshold,
        ):
            best = candidate
    if best is None:  # pragma: no cover
        raise EvaluationMetricError("Youden candidate set is empty")
    return best


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def binary_metrics(
    scores: tuple[float, ...], targets: tuple[int, ...], *, threshold: float
) -> dict[str, Any]:
    _validate_scores_targets(scores, targets)
    if not math.isfinite(threshold):
        raise EvaluationMetricError("threshold must be finite")
    tp, fp, tn, fn = _counts(scores, targets, threshold)
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    output: dict[str, Any] = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": _ratio(tp + tn, tp + fp + tn + fn),
        "balanced_accuracy": (
            None if sensitivity is None or specificity is None else (sensitivity + specificity) / 2
        ),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "youden_j": (
            None if sensitivity is None or specificity is None else sensitivity + specificity - 1
        ),
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


def calibration(scores: tuple[float, ...], targets: tuple[int, ...]) -> dict[str, float]:
    probabilities = tuple(
        1.0 / (1.0 + math.exp(-value)) if value >= 0 else math.exp(value) / (1.0 + math.exp(value))
        for value in scores
    )
    count = len(scores)
    mean_score = math.fsum(probabilities) / count
    prevalence = sum(targets) / count
    brier = (
        math.fsum((score - target) ** 2 for score, target in zip(probabilities, targets)) / count
    )
    ece_terms: list[float] = []
    for index in range(10):
        selected = [
            position
            for position, score in enumerate(probabilities)
            if index / 10 <= score < (index + 1) / 10 or (index == 9 and score == 1.0)
        ]
        if not selected:
            ece_terms.append(0.0)
            continue
        bin_score = math.fsum(probabilities[position] for position in selected) / len(selected)
        event_rate = sum(targets[position] for position in selected) / len(selected)
        ece_terms.append((len(selected) / count) * abs(bin_score - event_rate))
    return {
        "mean_score": mean_score,
        "prevalence": prevalence,
        "calibration_bias": mean_score - prevalence,
        "brier": brier,
        "ece10": math.fsum(ece_terms),
    }
