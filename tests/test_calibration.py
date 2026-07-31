from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cg_acceptance import FormalObjectKind, Verdict, run_calibration_gate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
def test_real_cuda_device_passes_all_formal_objects_with_quarter_margin(tmp_path: Path) -> None:
    output_path = tmp_path / "decision30-calibration.json"
    report = run_calibration_gate(
        torch.device("cuda:0"),
        run_id="cuda-integration",
        output_path=output_path,
    )

    assert report.device_identity["device_type"] == "cuda"
    assert report.device_identity["device_name"] != "cpu"
    assert [item.object_kind for item in report.object_reports] == [
        FormalObjectKind.PRECAST_STATS,
        FormalObjectKind.POSTCAST_POOL,
        FormalObjectKind.DZ,
        FormalObjectKind.DZ,
        FormalObjectKind.DZ,
    ]
    assert [item.audit.fixture_name for item in report.object_reports[2:]] == [
        "mean_only",
        "std_only",
        "joint_signed",
    ]
    assert all(item.verdict is Verdict.PASSED for item in report.object_reports)
    assert all(item.quarter_margin_pass for item in report.object_reports)
    assert all(
        item.minimum_margin_factor is None or item.minimum_margin_factor >= 4.0
        for item in report.object_reports
    )
    assert {control.name for control in report.negative_controls} == {
        "axis_permutation",
        "mean_std_slot_exchange",
        "sign_inversion",
        "wrong_reduction_order",
        "early_float32_conversion",
        "single_point_spatial_misalignment",
        "wrong_mask_selection",
        "zero_variance_backward_violation",
    }
    assert all(control.detected for control in report.negative_controls)
    assert report.overall_pass is True

    payload = report.to_dict()
    assert payload["gate_version"] == "decision30-calibration-gate-v2"
    assert len(payload["gate_code_sha256"]) == 64
    assert payload["cpu_reference_version"] == "decision30-cpu-reference-v2"
    assert payload["cpu_reference_code_sha256"] != payload["gate_code_sha256"]
    assert payload["input_identity_pass"] is True
    assert payload["environment_pass"] is True
    assert payload["zero_variance_evidence"] == {
        "forward_std_exact_zero": True,
        "all_backward_results_finite": True,
        "mean_only_dz_has_nonzero": True,
        "std_only_zero_variance_dz_exact_zero": True,
        "joint_dz_has_nonzero": True,
    }
    assert all(item["quarter_margin_pass"] for item in payload["object_reports"])
    assert all(
        item["audit"]["cpu_input_hashes"] == item["audit"]["device_input_hashes"]
        for item in payload["object_reports"]
    )
    assert all(
        "upstream_g" in item["audit"]["cpu_input_hashes"]
        for item in payload["object_reports"][2:]
    )
    assert json.loads(json.dumps(payload, allow_nan=False))["overall_pass"] is True
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
