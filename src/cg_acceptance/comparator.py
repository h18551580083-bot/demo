"""Independent CPU binary64 comparator for Decision 30 formal objects."""

from __future__ import annotations

import hashlib
import math
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch


class FormalObjectKind(str, Enum):
    """Formal cross-device comparison objects."""

    PRECAST_STATS = "precast_stats_float64"
    POSTCAST_POOL = "postcast_pool_float32"
    DZ = "dz_float32"


class Verdict(str, Enum):
    """Object-level acceptance verdict."""

    PASSED = "passed"
    DEVICE_FAILED = "device_failed"
    REFERENCE_INVALID = "reference_invalid"
    COMPARATOR_INVALID = "comparator_invalid"
    INPUT_IDENTITY_MISMATCH = "input_identity_mismatch"


@dataclass(frozen=True)
class AuditContext:
    """Identity information shared by the CPU reference and tested device."""

    run_id: str
    cpu_reference_id: str
    tested_device_id: str
    fixture_name: str
    cpu_input_hashes: Mapping[str, str]
    device_input_hashes: Mapping[str, str]


@dataclass(frozen=True)
class FailureDiagnostic:
    """First elementwise failure in canonical row-major order."""

    logical_index: tuple[int, ...]
    cpu_value: float
    device_value: float
    difference: float | None
    error: float | None
    relative_term: float | None
    bound: float | None
    active_rule: str
    nonfinite_type: str | None = None


@dataclass(frozen=True)
class MaximumErrorDiagnostic:
    """Maximum finite absolute error, with canonical first-index tie breaking."""

    logical_index: tuple[int, ...]
    error: float


@dataclass(frozen=True)
class ComparisonAuditRecord:
    """Comparator, input, formal-object, and floating-point identities."""

    run_id: str
    comparator_version: str
    comparator_code_sha256: str
    tolerance_profile_version: str
    cpu_reference_id: str
    tested_device_id: str
    fixture_name: str
    cpu_input_hashes: Mapping[str, str]
    device_input_hashes: Mapping[str, str]
    cpu_formal_object_sha256: str
    device_formal_object_sha256: str
    formal_object_shape: tuple[int, ...]
    formal_object_dtype: str
    formal_object_element_count: int
    tolerance_constants: Mapping[str, str]
    floating_point_environment: Mapping[str, object]


@dataclass(frozen=True)
class ComparisonReport:
    """Result returned by :func:`compare_object`."""

    object_kind: FormalObjectKind
    verdict: Verdict
    fixture_name: str
    failing_element_count: int
    first_failure: FailureDiagnostic | None
    maximum_error: MaximumErrorDiagnostic | None
    quarter_margin_pass: bool
    minimum_margin_factor: float | None
    audit: ComparisonAuditRecord


@dataclass(frozen=True)
class _Tolerance:
    atol_hex: str
    rtol_hex: str
    near_zero_threshold_hex: str
    atol: float
    rtol: float
    near_zero_threshold: float
    dtype: torch.dtype


_TOLERANCES = {
    FormalObjectKind.PRECAST_STATS: _Tolerance(
        atol_hex="0x1p-48",
        rtol_hex="0x1p-47",
        near_zero_threshold_hex="0x1p-40",
        atol=float.fromhex("0x1p-48"),
        rtol=float.fromhex("0x1p-47"),
        near_zero_threshold=float.fromhex("0x1p-40"),
        dtype=torch.float64,
    ),
    FormalObjectKind.POSTCAST_POOL: _Tolerance(
        atol_hex="0x1p-22",
        rtol_hex="0x1p-21",
        near_zero_threshold_hex="0x1p-16",
        atol=float.fromhex("0x1p-22"),
        rtol=float.fromhex("0x1p-21"),
        near_zero_threshold=float.fromhex("0x1p-16"),
        dtype=torch.float32,
    ),
    FormalObjectKind.DZ: _Tolerance(
        atol_hex="0x1p-21",
        rtol_hex="0x1p-19",
        near_zero_threshold_hex="0x1p-16",
        atol=float.fromhex("0x1p-21"),
        rtol=float.fromhex("0x1p-19"),
        near_zero_threshold=float.fromhex("0x1p-16"),
        dtype=torch.float32,
    ),
}

_COMPARATOR_VERSION = "decision30-comparator-v1"
_TOLERANCE_PROFILE_VERSION = "decision30-tolerances-v1"


def _logical_index(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coordinates: list[int] = []
    remaining = flat_index
    for size in reversed(shape):
        coordinates.append(remaining % size)
        remaining //= size
    return tuple(reversed(coordinates))


def _is_formal_shape(object_kind: FormalObjectKind, shape: tuple[int, ...]) -> bool:
    if len(shape) != 6 or shape[0] < 1:
        return False
    if object_kind in (FormalObjectKind.PRECAST_STATS, FormalObjectKind.POSTCAST_POOL):
        return shape[1:] == (4, 8, 7, 21, 2)
    return shape[1:4] == (4, 8, 7) and shape[4] >= 1 and shape[5] >= 1


def _nonfinite_type(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return "+inf" if value > 0 else "-inf"


def _tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(size) for size in cpu.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _comparator_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _floating_point_environment() -> dict[str, object]:
    smallest_subnormal = float.fromhex("0x0.0000000000001p-1022")
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "float_rounds": sys.float_info.rounds,
        "binary64_mantissa_bits": sys.float_info.mant_dig,
        "binary64_ftz_probe_preserved": smallest_subnormal * 1.0 == smallest_subnormal,
        "binary64_daz_probe_preserved": smallest_subnormal + 0.0 == smallest_subnormal,
        "operation_policy": "separate_binary64_operations_no_fma",
    }


def _audit_record(
    cpu_reference: torch.Tensor,
    device_result: torch.Tensor,
    tolerance: _Tolerance,
    audit: AuditContext,
) -> ComparisonAuditRecord:
    return ComparisonAuditRecord(
        run_id=audit.run_id,
        comparator_version=_COMPARATOR_VERSION,
        comparator_code_sha256=_comparator_code_sha256(),
        tolerance_profile_version=_TOLERANCE_PROFILE_VERSION,
        cpu_reference_id=audit.cpu_reference_id,
        tested_device_id=audit.tested_device_id,
        fixture_name=audit.fixture_name,
        cpu_input_hashes=dict(audit.cpu_input_hashes),
        device_input_hashes=dict(audit.device_input_hashes),
        cpu_formal_object_sha256=_tensor_sha256(cpu_reference),
        device_formal_object_sha256=_tensor_sha256(device_result),
        formal_object_shape=tuple(cpu_reference.shape),
        formal_object_dtype=str(cpu_reference.dtype),
        formal_object_element_count=cpu_reference.numel(),
        tolerance_constants={
            "atol": tolerance.atol_hex,
            "rtol": tolerance.rtol_hex,
            "near_zero_threshold": tolerance.near_zero_threshold_hex,
        },
        floating_point_environment=_floating_point_environment(),
    )


def _formal_nonfinite_report(
    *,
    object_kind: FormalObjectKind,
    audit: AuditContext,
    shape: tuple[int, ...],
    cpu_values: list[float],
    device_values: list[float],
    inspect_cpu: bool,
    audit_record: ComparisonAuditRecord,
) -> ComparisonReport | None:
    selected = cpu_values if inspect_cpu else device_values
    invalid_indices = [index for index, value in enumerate(selected) if not math.isfinite(value)]
    if not invalid_indices:
        return None
    first_index = invalid_indices[0]
    cpu_value = float(cpu_values[first_index])
    device_value = float(device_values[first_index])
    return ComparisonReport(
        object_kind=object_kind,
        verdict=Verdict.REFERENCE_INVALID if inspect_cpu else Verdict.DEVICE_FAILED,
        fixture_name=audit.fixture_name,
        failing_element_count=len(invalid_indices),
        first_failure=FailureDiagnostic(
            logical_index=_logical_index(first_index, shape),
            cpu_value=cpu_value,
            device_value=device_value,
            difference=None,
            error=None,
            relative_term=None,
            bound=None,
            active_rule="formal_nonfinite",
            nonfinite_type=_nonfinite_type(cpu_value if inspect_cpu else device_value),
        ),
        maximum_error=None,
        quarter_margin_pass=False,
        minimum_margin_factor=None,
        audit=audit_record,
    )


def compare_object(
    cpu_reference: torch.Tensor,
    device_result: torch.Tensor,
    object_kind: FormalObjectKind,
    audit: AuditContext,
) -> ComparisonReport:
    """Compare one formal object on CPU using normative binary64 arithmetic."""

    tolerance = _TOLERANCES[object_kind]
    if cpu_reference.device.type != "cpu" or device_result.device.type != "cpu":
        raise ValueError("formal objects must be returned to CPU before comparison")
    if cpu_reference.dtype != tolerance.dtype or device_result.dtype != tolerance.dtype:
        raise TypeError(f"{object_kind.value} requires {tolerance.dtype}")
    if cpu_reference.shape != device_result.shape:
        raise ValueError("formal object shapes differ")
    if not _is_formal_shape(object_kind, tuple(cpu_reference.shape)):
        raise ValueError(f"{object_kind.value} has an invalid formal shape")

    audit_record = _audit_record(cpu_reference, device_result, tolerance, audit)
    environment = audit_record.floating_point_environment
    if not (
        environment["float_rounds"] == 1
        and environment["binary64_mantissa_bits"] == 53
        and environment["binary64_ftz_probe_preserved"] is True
        and environment["binary64_daz_probe_preserved"] is True
    ):
        return ComparisonReport(
            object_kind=object_kind,
            verdict=Verdict.COMPARATOR_INVALID,
            fixture_name=audit.fixture_name,
            failing_element_count=0,
            first_failure=None,
            maximum_error=None,
            quarter_margin_pass=False,
            minimum_margin_factor=None,
            audit=audit_record,
        )
    if dict(audit.cpu_input_hashes) != dict(audit.device_input_hashes):
        return ComparisonReport(
            object_kind=object_kind,
            verdict=Verdict.INPUT_IDENTITY_MISMATCH,
            fixture_name=audit.fixture_name,
            failing_element_count=0,
            first_failure=None,
            maximum_error=None,
            quarter_margin_pass=False,
            minimum_margin_factor=None,
            audit=audit_record,
        )

    cpu_values = cpu_reference.detach().contiguous().view(-1).tolist()
    device_values = device_result.detach().contiguous().view(-1).tolist()
    shape = tuple(cpu_reference.shape)
    reference_invalid = _formal_nonfinite_report(
        object_kind=object_kind,
        audit=audit,
        shape=shape,
        cpu_values=cpu_values,
        device_values=device_values,
        inspect_cpu=True,
        audit_record=audit_record,
    )
    if reference_invalid is not None:
        return reference_invalid
    device_invalid = _formal_nonfinite_report(
        object_kind=object_kind,
        audit=audit,
        shape=shape,
        cpu_values=cpu_values,
        device_values=device_values,
        inspect_cpu=False,
        audit_record=audit_record,
    )
    if device_invalid is not None:
        return device_invalid

    failures: list[FailureDiagnostic] = []
    maximum_error: MaximumErrorDiagnostic | None = None
    quarter_margin_pass = True
    minimum_margin_factor: float | None = None

    for flat_index, (cpu_value, tested_value) in enumerate(zip(cpu_values, device_values)):
        cpu_binary64 = float(cpu_value)
        device_binary64 = float(tested_value)
        difference = device_binary64 - cpu_binary64
        error = abs(difference)
        relative_term = tolerance.rtol * abs(cpu_binary64)
        bound = tolerance.atol + relative_term
        if not all(math.isfinite(value) for value in (difference, relative_term, bound)):
            return ComparisonReport(
                object_kind=object_kind,
                verdict=Verdict.COMPARATOR_INVALID,
                fixture_name=audit.fixture_name,
                failing_element_count=1,
                first_failure=FailureDiagnostic(
                    logical_index=_logical_index(flat_index, shape),
                    cpu_value=cpu_binary64,
                    device_value=device_binary64,
                    difference=difference,
                    error=error,
                    relative_term=relative_term,
                    bound=bound,
                    active_rule="comparator_invalid",
                ),
                maximum_error=None,
                quarter_margin_pass=False,
                minimum_margin_factor=None,
                audit=audit_record,
            )
        if cpu_binary64 == 0.0:
            passed = device_binary64 == 0.0
            active_rule = "exact_zero"
            active_limit = 0.0
        elif abs(cpu_binary64) <= tolerance.near_zero_threshold:
            passed = error <= tolerance.atol
            active_rule = "near_zero"
            active_limit = tolerance.atol
        else:
            passed = error <= bound
            active_rule = "general"
            active_limit = bound
        if maximum_error is None or error > maximum_error.error:
            maximum_error = MaximumErrorDiagnostic(
                logical_index=_logical_index(flat_index, shape),
                error=error,
            )
        if error > active_limit / 4.0:
            quarter_margin_pass = False
        if error > 0.0:
            margin_factor = 0.0 if active_limit == 0.0 else active_limit / error
            if minimum_margin_factor is None or margin_factor < minimum_margin_factor:
                minimum_margin_factor = margin_factor
        if not passed:
            failures.append(
                FailureDiagnostic(
                    logical_index=_logical_index(flat_index, shape),
                    cpu_value=cpu_binary64,
                    device_value=device_binary64,
                    difference=difference,
                    error=error,
                    relative_term=relative_term,
                    bound=bound,
                    active_rule=active_rule,
                )
            )

    return ComparisonReport(
        object_kind=object_kind,
        verdict=Verdict.PASSED if not failures else Verdict.DEVICE_FAILED,
        fixture_name=audit.fixture_name,
        failing_element_count=len(failures),
        first_failure=failures[0] if failures else None,
        maximum_error=maximum_error,
        quarter_margin_pass=quarter_margin_pass,
        minimum_margin_factor=minimum_margin_factor,
        audit=audit_record,
    )
