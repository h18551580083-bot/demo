from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from cg_pipeline.evaluation import (
    AuthorizedEvaluationRow,
    EvaluationContext,
    EvaluationContractError,
    FrozenThresholds,
    Prediction,
    _threshold_identity_values,
    aggregate_slides,
    binary_metrics,
    evaluate_predictions,
    exact_auroc,
    load_final_test_authorization,
    select_youden_threshold,
)

_HASH_A = "sha256:" + "a" * 64
_HASH_C = "sha256:" + "c" * 64
_HASH_D = "sha256:" + "d" * 64
_HASH_E = "sha256:" + "e" * 64


def _frozen_thresholds(
    patch_threshold: float = 0.5, slide_threshold: float = 1.0
) -> FrozenThresholds:
    identity = _threshold_identity_values(
        checkpoint_identity=_HASH_E,
        effective_manifest_sha256=_HASH_D,
        seed=1729,
        patch_threshold=patch_threshold,
        slide_threshold=slide_threshold,
    )
    return FrozenThresholds(
        patch_threshold=patch_threshold,
        slide_threshold=slide_threshold,
        checkpoint_identity=_HASH_E,
        effective_validation_manifest_sha256=_HASH_D,
        seed=1729,
        identity=identity,
    )


def _context(split: str = "val", *, test_authorization=None) -> EvaluationContext:
    predictions = _predictions(split)
    return EvaluationContext(
        split=split,
        authorized_rows=tuple(
            AuthorizedEvaluationRow(
                row.patch_id,
                row.slide_id,
                row.split,
                row.patch_target,
                row.slide_target,
            )
            for row in predictions
        ),
        source_manifest_sha256=_HASH_C,
        effective_manifest_sha256=_HASH_D,
        fixed_frontend_identity={"canonical_kernel_hash": _HASH_A},
        checkpoint_identity=_HASH_E,
        seed=1729,
        test_authorization=test_authorization,
    )


def _test_authorization(tmp_path: Path, thresholds: FrozenThresholds):
    evidence = tmp_path / "test-approval-evidence.txt"
    evidence_bytes = b"approved final-once CAM16 test access\n"
    evidence.write_bytes(evidence_bytes)
    document = {
        "schema": "cam16-final-test-authorization-v1",
        "test_access_authorized": True,
        "source_manifest_sha256": _HASH_C,
        "effective_test_manifest_sha256": _HASH_D,
        "checkpoint_identity": _HASH_E,
        "validation_threshold_identity": thresholds.identity,
        "approval_evidence_sha256": "sha256:" + hashlib.sha256(evidence_bytes).hexdigest(),
        "approved_by": "test-approver",
        "approved_at": "2026-08-03T00:00:00Z",
    }
    authorization = tmp_path / "test-authorization.json"
    authorization.write_text(json.dumps(document), encoding="utf-8")
    return load_final_test_authorization(authorization, evidence)


def _predictions(split: str = "val") -> tuple[Prediction, ...]:
    return (
        Prediction("p0", "s0", split, 0, 0, -2.0),
        Prediction("p1", "s0", split, 0, 0, -1.0),
        Prediction("p2", "s1", split, 1, 1, 0.5),
        Prediction("p3", "s1", split, 1, 1, 1.0),
    )


def test_exact_auroc_ties_and_undefined_classes() -> None:
    value = exact_auroc((0.0, 1.0, 1.0, 2.0), (0, 0, 1, 1))

    assert value.value == 0.875
    assert value.winning_pairs == 3
    assert value.tied_pairs == 1
    assert value.positive_count == 2
    assert value.negative_count == 2
    with pytest.raises(EvaluationContractError, match="both classes"):
        exact_auroc((0.0, 1.0), (1, 1))
    with pytest.raises(EvaluationContractError, match="non-finite"):
        exact_auroc((0.0, math.nan), (0, 1))


def test_manifest_bounded_max_slide_aggregation_and_consistent_label() -> None:
    slides = aggregate_slides(_predictions())

    assert [(item.slide_id, item.logit, item.target, item.source_patch_id) for item in slides] == [
        ("s0", -1.0, 0, "p1"),
        ("s1", 1.0, 1, "p3"),
    ]
    bad = list(_predictions())
    bad[1] = Prediction("p1", "s0", "val", 0, 1, -1.0)
    with pytest.raises(EvaluationContractError, match="inconsistent slide target"):
        aggregate_slides(tuple(bad))


def test_youden_uses_distinct_validation_logits_and_largest_threshold_tie() -> None:
    selected = select_youden_threshold((-2.0, -1.0, 0.5, 1.0), (0, 0, 1, 1))

    assert selected.threshold == 0.5
    assert selected.tp == 2 and selected.tn == 2
    tie = select_youden_threshold((0.0, 1.0, 2.0, 3.0), (0, 1, 0, 1))
    assert tie.threshold == 3.0


def test_metrics_zero_denominator_is_explicitly_undefined() -> None:
    result = binary_metrics((1.0, 2.0), (1, 1), threshold=0.0)

    assert result["sensitivity"] == 1.0
    assert result["specificity"] is None
    assert result["specificity_undefined_reason"] == "no_negative_targets"
    assert result["ppv"] == 1.0
    assert result["npv"] is None


def test_evaluation_fits_validation_only_and_hashes_result_identity() -> None:
    report = evaluate_predictions(
        _predictions(), context=_context(), fit_thresholds=True, ci_seed=1729
    )

    assert report["split"] == "val"
    assert report["primary_metric"] == "slide_auroc"
    assert report["slide_auroc"]["value"] == 1.0
    assert report["patch_auroc"]["value"] == 1.0
    assert report["patch_threshold"]["threshold"] == 0.5
    assert report["slide_threshold"]["threshold"] == 1.0
    assert report["slide_auroc_ci"]["replicates"] == 2000
    assert report["slide_auroc_ci"]["lower"] == 1.0
    assert report["slide_auroc_ci"]["upper"] == 1.0
    assert report["result_identity"].startswith("sha256:")
    assert report["threshold_identity"].startswith("sha256:")
    assert report["identities"]["checkpoint_identity"] == _HASH_E
    test_rows = _predictions(split="test")
    with pytest.raises(EvaluationContractError, match="authorization"):
        evaluate_predictions(
            test_rows,
            context=_context("test"),
            fit_thresholds=False,
            ci_seed=1729,
            frozen_thresholds=_frozen_thresholds(),
        )


def test_prediction_ledger_requires_every_authorized_row_and_test_gate_is_identity_bound(
    tmp_path: Path,
) -> None:
    with pytest.raises(EvaluationContractError, match="incomplete"):
        evaluate_predictions(
            _predictions()[:-1], context=_context(), fit_thresholds=True, ci_seed=1729
        )
    thresholds = _frozen_thresholds()
    authorization = _test_authorization(tmp_path, thresholds)
    report = evaluate_predictions(
        _predictions("test"),
        context=_context("test", test_authorization=authorization),
        fit_thresholds=False,
        ci_seed=1729,
        frozen_thresholds=thresholds,
    )
    assert report["split"] == "test"
    assert report["threshold_identity"] == thresholds.identity
    mismatched = FrozenThresholds(
        patch_threshold=0.75,
        slide_threshold=thresholds.slide_threshold,
        checkpoint_identity=thresholds.checkpoint_identity,
        effective_validation_manifest_sha256=thresholds.effective_validation_manifest_sha256,
        seed=thresholds.seed,
        identity=thresholds.identity,
    )
    with pytest.raises(EvaluationContractError, match="threshold value"):
        evaluate_predictions(
            _predictions("test"),
            context=_context("test", test_authorization=authorization),
            fit_thresholds=False,
            ci_seed=1729,
            frozen_thresholds=mismatched,
        )


def test_final_test_authorization_rechecks_independent_approval_artifact(tmp_path: Path) -> None:
    thresholds = _frozen_thresholds()
    authorization = _test_authorization(tmp_path, thresholds)
    authorization.approval_evidence_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(EvaluationContractError, match="approval evidence identity mismatch"):
        evaluate_predictions(
            _predictions("test"),
            context=_context("test", test_authorization=authorization),
            fit_thresholds=False,
            ci_seed=1729,
            frozen_thresholds=thresholds,
        )


def test_bootstrap_seed_must_equal_recorded_run_seed() -> None:
    with pytest.raises(EvaluationContractError, match="bootstrap seed"):
        evaluate_predictions(_predictions(), context=_context(), fit_thresholds=True, ci_seed=3407)
