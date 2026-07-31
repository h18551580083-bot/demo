from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cg_acceptance import (
    CalibrationMode,
    FormalObjectKind,
    Verdict,
    build_deterministic_fixture,
    run_calibration_gate,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a real CUDA device",
)


@pytest.fixture(scope="module")
def dynamic_local_report(tmp_path_factory: pytest.TempPathFactory):
    fixture = build_deterministic_fixture(
        name="batch2-non8x8",
        batch=2,
        height=9,
        width=11,
    )
    output_path = tmp_path_factory.mktemp("decision30") / "local-smoke.json"
    return run_calibration_gate(
        torch.device("cuda:0"),
        run_id="autograd-local-smoke",
        mode=CalibrationMode.LOCAL_SMOKE,
        fixture=fixture,
        output_path=output_path,
    )


def test_device_backward_uses_three_real_autograd_runs_and_z_grad(dynamic_local_report) -> None:
    evidence = dynamic_local_report.backward_execution

    assert evidence == {
        "api": "torch.autograd.backward(pool_float32, G)",
        "backward_calls": 3,
        "fresh_leaf_per_fixture": True,
        "gradient_source": "z.grad",
        "manual_device_dz_generation": False,
    }
    assert [item.audit.fixture_name for item in dynamic_local_report.object_reports[2:]] == [
        "mean_only",
        "std_only",
        "joint_signed",
    ]


def test_dynamic_batch_and_non8x8_shape_are_formally_compared(dynamic_local_report) -> None:
    assert dynamic_local_report.formal_input_shape == (2, 4, 8, 7, 9, 11)
    assert [item.object_kind for item in dynamic_local_report.object_reports] == [
        FormalObjectKind.PRECAST_STATS,
        FormalObjectKind.POSTCAST_POOL,
        FormalObjectKind.DZ,
        FormalObjectKind.DZ,
        FormalObjectKind.DZ,
    ]
    assert all(item.verdict is Verdict.PASSED for item in dynamic_local_report.object_reports)
    assert all(item.quarter_margin_pass for item in dynamic_local_report.object_reports)


def test_zero_variance_rule_is_real_backward_evidence(dynamic_local_report) -> None:
    assert dynamic_local_report.zero_variance_evidence == {
        "forward_std_exact_zero": True,
        "all_backward_results_finite": True,
        "mean_only_dz_has_nonzero": True,
        "std_only_zero_variance_dz_exact_zero": True,
        "joint_dz_has_nonzero": True,
    }


def test_cpu_reference_device_operator_comparator_and_orchestrator_are_independent(
    dynamic_local_report,
) -> None:
    identities = dynamic_local_report.code_identities

    assert set(identities) == {
        "cpu_reference",
        "device_operator",
        "comparator",
        "calibration_orchestrator",
    }
    assert all(len(identity["sha256"]) == 64 for identity in identities.values())
    assert len({identity["sha256"] for identity in identities.values()}) == 4
    assert len({identity["module"] for identity in identities.values()}) == 4


def test_local_smoke_cannot_close_formal_acceptance(dynamic_local_report) -> None:
    assert dynamic_local_report.mode is CalibrationMode.LOCAL_SMOKE
    assert dynamic_local_report.smoke_pass is True
    assert dynamic_local_report.formal_gate_pass is None
    assert dynamic_local_report.overall_pass is True


def test_formal_acceptance_requires_preregistered_fixture_and_environment() -> None:
    fixture = build_deterministic_fixture(
        name="not-preregistered",
        batch=1,
        height=9,
        width=11,
    )

    with pytest.raises(ValueError, match="pre-registered fixture"):
        run_calibration_gate(
            torch.device("cuda:0"),
            run_id="must-not-run-formal",
            mode=CalibrationMode.FORMAL_ACCEPTANCE,
            fixture=fixture,
            expected_environment={},
        )


def test_negative_controls_execute_faults_inside_candidate_operator(
    dynamic_local_report,
) -> None:
    controls = {control.name: control for control in dynamic_local_report.negative_controls}

    assert {
        "wrong_reduction_order",
        "early_float32_conversion",
        "single_point_spatial_misalignment",
        "zero_variance_backward_violation",
    } <= set(controls)
    for control in controls.values():
        assert control.execution_path == "candidate_device_operator"
        assert control.detected is True
        assert control.comparison.verdict is Verdict.DEVICE_FAILED


def test_report_contains_full_environment_and_registered_input_identity(
    dynamic_local_report,
) -> None:
    identity = dynamic_local_report.device_identity

    assert {
        "gpu_name",
        "gpu_uuid",
        "driver_version",
        "torch_version",
        "cuda_runtime",
        "cudnn_version",
        "cpu_identity",
        "python_version",
        "system",
    } <= set(identity)
    assert dynamic_local_report.input_identity_pass is True
    assert dynamic_local_report.input_hashes == dynamic_local_report.fixture_registration[
        "input_hashes"
    ]
    assert tuple(dynamic_local_report.fixture_registration["formal_shape"]) == (
        2,
        4,
        8,
        7,
        9,
        11,
    )


def test_existing_report_is_never_overwritten(tmp_path: Path) -> None:
    output_path = tmp_path / "existing.json"
    output_path.write_text("sentinel", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_calibration_gate(
            torch.device("cuda:0"),
            run_id="no-overwrite",
            mode=CalibrationMode.LOCAL_SMOKE,
            output_path=output_path,
        )
    assert output_path.read_text(encoding="utf-8") == "sentinel"
