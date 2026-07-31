from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest
import torch

from cg_acceptance import AuditContext, FormalObjectKind, Verdict, compare_object


def _audit() -> AuditContext:
    return AuditContext(
        run_id="boundary-test",
        cpu_reference_id="cpu-reference-v1",
        tested_device_id="test-device",
        fixture_name="forward-deterministic",
        cpu_input_hashes={"fixture": "sha256:1234"},
        device_input_hashes={"fixture": "sha256:1234"},
    )


def _stats64(fill: float = 0.0) -> torch.Tensor:
    return torch.full((1, 4, 8, 7, 21, 2), fill, dtype=torch.float64)


def _pool32(fill: float = 0.0) -> torch.Tensor:
    return torch.full((1, 4, 8, 7, 21, 2), fill, dtype=torch.float32)


def _dz32(fill: float = 0.0) -> torch.Tensor:
    return torch.full((1, 4, 8, 7, 3, 5), fill, dtype=torch.float32)


def test_postcast_pool_closed_boundary_passes_and_next_float32_fails() -> None:
    cpu = _pool32(1.0)
    at_bound = cpu.clone()
    at_bound[0, 0, 0, 0, 0, 0] = float.fromhex("0x1.00000cp+0")
    above_bound = at_bound.clone()
    above_bound[0, 0, 0, 0, 0, 0] = torch.nextafter(
        at_bound[0, 0, 0, 0, 0, 0],
        torch.tensor(float("inf")),
    )

    passed = compare_object(cpu, at_bound, FormalObjectKind.POSTCAST_POOL, _audit())
    failed = compare_object(cpu, above_bound, FormalObjectKind.POSTCAST_POOL, _audit())

    assert passed.verdict is Verdict.PASSED
    assert failed.verdict is Verdict.DEVICE_FAILED
    assert failed.first_failure is not None
    assert failed.first_failure.logical_index == (0, 0, 0, 0, 0, 0)


def test_nonfinite_values_are_classified_by_origin() -> None:
    finite = _stats64()
    cpu_nan = finite.clone()
    cpu_nan[0, 0, 0, 0, 0, 0] = float("nan")
    device_inf = finite.clone()
    device_inf[0, 0, 0, 0, 0, 0] = float("inf")
    cpu_overflow = finite.clone()
    device_overflow = finite.clone()
    cpu_overflow[0, 0, 0, 0, 0, 0] = -torch.finfo(torch.float64).max
    device_overflow[0, 0, 0, 0, 0, 0] = torch.finfo(torch.float64).max

    reference_invalid = compare_object(
        cpu_nan,
        finite,
        FormalObjectKind.PRECAST_STATS,
        _audit(),
    )
    device_failed = compare_object(
        finite,
        device_inf,
        FormalObjectKind.PRECAST_STATS,
        _audit(),
    )
    comparator_invalid = compare_object(
        cpu_overflow,
        device_overflow,
        FormalObjectKind.PRECAST_STATS,
        _audit(),
    )

    assert reference_invalid.verdict is Verdict.REFERENCE_INVALID
    assert reference_invalid.first_failure is not None
    assert reference_invalid.first_failure.nonfinite_type == "nan"
    assert device_failed.verdict is Verdict.DEVICE_FAILED
    assert device_failed.first_failure is not None
    assert device_failed.first_failure.nonfinite_type == "+inf"
    assert comparator_invalid.verdict is Verdict.COMPARATOR_INVALID


def test_comparison_report_audits_code_inputs_environment_and_margin() -> None:
    cpu = _pool32(1.0)
    at_bound = cpu.clone()
    at_bound[0, 0, 0, 0, 0, 0] = float.fromhex("0x1.00000cp+0")

    report = compare_object(cpu, at_bound, FormalObjectKind.POSTCAST_POOL, _audit())

    assert len(report.audit.comparator_code_sha256) == 64
    source_path = Path(inspect.getsourcefile(compare_object) or "")
    expected_code_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert report.audit.comparator_code_sha256 == expected_code_hash
    assert report.audit.cpu_input_hashes == {"fixture": "sha256:1234"}
    assert report.audit.cpu_formal_object_sha256 != report.audit.device_formal_object_sha256
    assert report.audit.formal_object_shape == (1, 4, 8, 7, 21, 2)
    assert report.audit.formal_object_dtype == "torch.float32"
    assert report.audit.formal_object_element_count == 9408
    assert report.audit.tolerance_constants == {
        "atol": "0x1p-22",
        "rtol": "0x1p-21",
        "near_zero_threshold": "0x1p-16",
    }
    assert report.audit.floating_point_environment["float_rounds"] == 1
    assert report.maximum_error is not None
    assert report.maximum_error.logical_index == (0, 0, 0, 0, 0, 0)
    assert report.maximum_error.error == float.fromhex("0x1.8p-21")
    assert report.quarter_margin_pass is False
    assert report.minimum_margin_factor == 1.0


def test_wrong_formal_shape_is_rejected() -> None:
    wrong = torch.zeros((1, 4, 8, 7, 20, 2), dtype=torch.float64)

    with pytest.raises(ValueError, match="formal shape"):
        compare_object(wrong, wrong.clone(), FormalObjectKind.PRECAST_STATS, _audit())


@pytest.mark.parametrize(
    ("object_kind", "factory", "boundary_hex"),
    [
        (FormalObjectKind.PRECAST_STATS, _stats64, "0x1.0000000000030p+0"),
        (FormalObjectKind.POSTCAST_POOL, _pool32, "0x1.00000cp+0"),
        (FormalObjectKind.DZ, _dz32, "0x1.000028p+0"),
    ],
)
def test_each_formal_object_uses_its_exact_closed_boundary(
    object_kind: FormalObjectKind,
    factory,
    boundary_hex: str,
) -> None:
    cpu = factory(1.0)
    at_bound = cpu.clone()
    at_bound[(0,) * 6] = float.fromhex(boundary_hex)
    above_bound = at_bound.clone()
    above_bound[(0,) * 6] = torch.nextafter(
        at_bound[(0,) * 6],
        torch.tensor(float("inf"), dtype=at_bound.dtype),
    )

    passed = compare_object(cpu, at_bound, object_kind, _audit())
    failed = compare_object(cpu, above_bound, object_kind, _audit())

    assert passed.verdict is Verdict.PASSED
    assert failed.verdict is Verdict.DEVICE_FAILED


def test_near_zero_uses_only_atol_and_normative_zero_is_exact() -> None:
    near_cpu = _stats64()
    near_atol = near_cpu.clone()
    near_above = near_cpu.clone()
    near_cpu[(0,) * 6] = float.fromhex("0x1p-40")
    near_atol[(0,) * 6] = float.fromhex("0x1.01p-40")
    near_above[(0,) * 6] = torch.nextafter(
        near_atol[(0,) * 6],
        torch.tensor(float("inf"), dtype=torch.float64),
    )
    signed_zero = _pool32()
    signed_zero[(0,) * 6] = -0.0
    smallest_nonzero = _pool32()
    smallest_nonzero[(0,) * 6] = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float32),
        torch.tensor(float("inf"), dtype=torch.float32),
    )

    near_pass = compare_object(near_cpu, near_atol, FormalObjectKind.PRECAST_STATS, _audit())
    near_fail = compare_object(near_cpu, near_above, FormalObjectKind.PRECAST_STATS, _audit())
    zero_pass = compare_object(_pool32(), signed_zero, FormalObjectKind.POSTCAST_POOL, _audit())
    zero_fail = compare_object(
        _pool32(),
        smallest_nonzero,
        FormalObjectKind.POSTCAST_POOL,
        _audit(),
    )

    assert near_pass.verdict is Verdict.PASSED
    assert near_fail.verdict is Verdict.DEVICE_FAILED
    assert near_fail.first_failure is not None
    assert near_fail.first_failure.active_rule == "near_zero"
    assert zero_pass.verdict is Verdict.PASSED
    assert zero_fail.verdict is Verdict.DEVICE_FAILED
    assert zero_fail.first_failure is not None
    assert zero_fail.first_failure.active_rule == "exact_zero"


def test_input_identity_mismatch_blocks_before_tolerance_comparison() -> None:
    mismatched = AuditContext(
        run_id="identity-test",
        cpu_reference_id="cpu-reference-v1",
        tested_device_id="test-device",
        fixture_name="forward-deterministic",
        cpu_input_hashes={"z": "sha256:cpu"},
        device_input_hashes={"z": "sha256:device"},
    )

    report = compare_object(
        _pool32(),
        _pool32(),
        FormalObjectKind.POSTCAST_POOL,
        mismatched,
    )

    assert report.verdict is Verdict.INPUT_IDENTITY_MISMATCH
    assert report.failing_element_count == 0
