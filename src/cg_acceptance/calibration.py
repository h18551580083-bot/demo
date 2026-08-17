"""Calibration orchestration kept independent from reference and candidate math."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path

import torch

from . import calibration_environment, comparator, cpu_reference, device_operator
from .comparator import AuditContext, ComparisonReport, FormalObjectKind, Verdict
from .fixture import (
    CalibrationFixture,
    CalibrationMode,
    build_deterministic_fixture,
    fixture_input_hashes,
    tensor_sha256,
    validate_fixture,
)


@dataclass(frozen=True)
class NegativeControlResult:
    """Evidence that a fault injected into the candidate path was detected."""

    name: str
    execution_path: str
    detected: bool
    comparison: ComparisonReport


@dataclass(frozen=True)
class CalibrationReport:
    """Complete result of one local-smoke or formal-acceptance run."""

    run_id: str
    mode: CalibrationMode
    gate_version: str
    gate_code_sha256: str
    cpu_reference_version: str
    cpu_reference_code_sha256: str
    code_identities: Mapping[str, Mapping[str, str]]
    device_identity: Mapping[str, object]
    formal_input_shape: tuple[int, ...]
    input_hashes: Mapping[str, str]
    fixture_registration: Mapping[str, object]
    input_identity_pass: bool
    environment_pass: bool
    backward_execution: Mapping[str, object]
    zero_variance_evidence: Mapping[str, bool]
    object_reports: tuple[ComparisonReport, ...]
    negative_controls: tuple[NegativeControlResult, ...]
    smoke_pass: bool | None
    formal_gate_pass: bool | None
    overall_pass: bool

    def to_dict(self) -> dict[str, object]:
        return _json_value(self)


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "+inf" if value > 0 else "-inf"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _orchestrator_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _gradient_fixtures(batch: int) -> dict[str, torch.Tensor]:
    mean_only = torch.zeros((batch, 4, 8, 7, 21, 2), dtype=torch.float32)
    mean_only[..., 0] = 1.0
    std_only = torch.zeros_like(mean_only)
    std_only[..., 1] = 1.0
    r = torch.arange(4704, dtype=torch.int64).view(1, 4, 8, 7, 21)
    b = torch.arange(batch, dtype=torch.int64).view(batch, 1, 1, 1, 1)
    u = b * 4704 + r
    mean_sign = torch.where(u % 2 == 0, 1, -1)
    std_sign = torch.where((u // 4) % 2 == 0, 1, -1)
    joint = torch.empty_like(mean_only)
    joint[..., 0] = (mean_sign * (1 + u % 4)).to(torch.float32) / 4.0
    joint[..., 1] = (std_sign * (1 + ((3 * u + 1) % 4))).to(torch.float32) / 8.0
    return {
        "mean_only": mean_only,
        "std_only": std_only,
        "joint_signed": joint,
    }


def _audit_context(
    *,
    run_id: str,
    tested_device_id: str,
    fixture_name: str,
    cpu_hashes: Mapping[str, str],
    device_hashes: Mapping[str, str],
) -> AuditContext:
    return AuditContext(
        run_id=run_id,
        cpu_reference_id=cpu_reference.REFERENCE_VERSION,
        tested_device_id=tested_device_id,
        fixture_name=fixture_name,
        cpu_input_hashes=cpu_hashes,
        device_input_hashes=device_hashes,
    )


def _run_device_backward(
    *,
    device: torch.device,
    fixture: CalibrationFixture,
    upstream: torch.Tensor,
    fault: device_operator.CandidateFault = device_operator.CandidateFault.NONE,
) -> tuple[torch.Tensor, device_operator.DevicePoolOutput]:
    z_leaf = fixture.z.to(device).detach().clone().requires_grad_(True)
    output = device_operator.pool(
        z_leaf,
        fixture.mask.to(device),
        fixture.valid_counts.to(device),
        fault=fault,
    )
    torch.autograd.backward(output.pool_float32, upstream.to(device))
    if z_leaf.grad is None:
        raise RuntimeError("candidate backward did not populate z.grad")
    return z_leaf.grad.detach().to("cpu"), output


def _negative_control(
    *,
    name: str,
    cpu_reference_value: torch.Tensor,
    candidate_value: torch.Tensor,
    object_kind: FormalObjectKind,
    audit: AuditContext,
) -> NegativeControlResult:
    comparison = comparator.compare_object(
        cpu_reference_value,
        candidate_value,
        object_kind,
        audit,
    )
    return NegativeControlResult(
        name=name,
        execution_path="candidate_device_operator",
        detected=comparison.verdict is Verdict.DEVICE_FAILED,
        comparison=comparison,
    )


def _run_negative_controls(
    *,
    device: torch.device,
    fixture: CalibrationFixture,
    cpu_statistics: torch.Tensor,
    cpu_pool: torch.Tensor,
    cpu_dz: Mapping[str, torch.Tensor],
    gradients: Mapping[str, torch.Tensor],
    forward_audit: AuditContext,
    backward_audits: Mapping[str, AuditContext],
) -> tuple[NegativeControlResult, ...]:
    controls: list[NegativeControlResult] = []
    forward_faults = (
        device_operator.CandidateFault.AXIS_PERMUTATION,
        device_operator.CandidateFault.SLOT_EXCHANGE,
        device_operator.CandidateFault.SIGN_INVERSION,
        device_operator.CandidateFault.WRONG_REDUCTION_ORDER,
        device_operator.CandidateFault.EARLY_FLOAT32,
        device_operator.CandidateFault.WRONG_MASK,
    )
    for fault in forward_faults:
        candidate = device_operator.pool(
            fixture.z.to(device),
            fixture.mask.to(device),
            fixture.valid_counts.to(device),
            fault=fault,
        )
        if fault in (
            device_operator.CandidateFault.WRONG_REDUCTION_ORDER,
            device_operator.CandidateFault.EARLY_FLOAT32,
        ):
            reference_value = cpu_statistics
            candidate_value = candidate.statistics_float64.to("cpu")
            object_kind = FormalObjectKind.PRECAST_STATS
        else:
            reference_value = cpu_pool
            candidate_value = candidate.pool_float32.to("cpu")
            object_kind = FormalObjectKind.POSTCAST_POOL
        controls.append(
            _negative_control(
                name=fault.value,
                cpu_reference_value=reference_value,
                candidate_value=candidate_value,
                object_kind=object_kind,
                audit=forward_audit,
            )
        )
    backward_faults = (
        (
            device_operator.CandidateFault.SPATIAL_MISALIGNMENT,
            "joint_signed",
        ),
        (
            device_operator.CandidateFault.ZERO_VARIANCE_BACKWARD,
            "std_only",
        ),
    )
    for fault, fixture_name in backward_faults:
        candidate_dz, _ = _run_device_backward(
            device=device,
            fixture=fixture,
            upstream=gradients[fixture_name],
            fault=fault,
        )
        controls.append(
            _negative_control(
                name=fault.value,
                cpu_reference_value=cpu_dz[fixture_name],
                candidate_value=candidate_dz,
                object_kind=FormalObjectKind.DZ,
                audit=backward_audits[fixture_name],
            )
        )
    return tuple(controls)


def run_calibration_gate(
    device: torch.device,
    *,
    run_id: str,
    output_path: Path | None = None,
    mode: CalibrationMode | str = CalibrationMode.LOCAL_SMOKE,
    fixture: CalibrationFixture | None = None,
    expected_environment: Mapping[str, object] | None = None,
) -> CalibrationReport:
    """Run one local smoke or pre-registered formal device acceptance."""

    normalized_mode = CalibrationMode(mode)
    selected_fixture = fixture or build_deterministic_fixture(
        name="builtin-small",
        height=9,
        width=11,
    )
    validate_fixture(selected_fixture)
    calibration_environment.validate_scope(normalized_mode, selected_fixture, expected_environment)
    if output_path is not None and output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output_path}")
    if device.type == "cpu":
        raise ValueError("calibration requires a real non-CPU device")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this calibration implementation requires CUDA")

    cpu_hashes = fixture_input_hashes(selected_fixture)
    cpu_statistics, cpu_pool = cpu_reference.forward(
        selected_fixture.z,
        selected_fixture.mask,
        selected_fixture.valid_counts,
    )
    gradients = _gradient_fixtures(selected_fixture.z.shape[0])
    cpu_dz = {
        name: cpu_reference.backward(
            selected_fixture.z,
            selected_fixture.mask,
            selected_fixture.valid_counts,
            cpu_statistics,
            upstream,
        )
        for name, upstream in gradients.items()
    }

    with calibration_environment.protected_device_environment(device):
        device_hashes = {
            "z": tensor_sha256(selected_fixture.z.to(device)),
            "mask": tensor_sha256(selected_fixture.mask.to(device)),
            "valid_counts": tensor_sha256(selected_fixture.valid_counts.to(device)),
        }
        identity = calibration_environment.device_identity(device)
        calibration_environment.validate_expected_environment(
            normalized_mode, expected_environment, identity
        )
        tested_device_id = f"{identity['gpu_name']}|{identity['gpu_uuid']}"
        forward_output = device_operator.pool(
            selected_fixture.z.to(device),
            selected_fixture.mask.to(device),
            selected_fixture.valid_counts.to(device),
        )
        forward_audit = _audit_context(
            run_id=run_id,
            tested_device_id=tested_device_id,
            fixture_name=selected_fixture.name,
            cpu_hashes=cpu_hashes,
            device_hashes=device_hashes,
        )
        reports = [
            comparator.compare_object(
                cpu_statistics,
                forward_output.statistics_float64.to("cpu"),
                FormalObjectKind.PRECAST_STATS,
                forward_audit,
            ),
            comparator.compare_object(
                cpu_pool,
                forward_output.pool_float32.to("cpu"),
                FormalObjectKind.POSTCAST_POOL,
                forward_audit,
            ),
        ]
        device_dz: dict[str, torch.Tensor] = {}
        backward_audits: dict[str, AuditContext] = {}
        for fixture_name, upstream in gradients.items():
            fixture_cpu_hashes = dict(cpu_hashes)
            fixture_cpu_hashes["upstream_g"] = tensor_sha256(upstream)
            fixture_device_hashes = dict(device_hashes)
            fixture_device_hashes["upstream_g"] = tensor_sha256(upstream.to(device))
            audit = _audit_context(
                run_id=run_id,
                tested_device_id=tested_device_id,
                fixture_name=fixture_name,
                cpu_hashes=fixture_cpu_hashes,
                device_hashes=fixture_device_hashes,
            )
            dz, _ = _run_device_backward(
                device=device,
                fixture=selected_fixture,
                upstream=upstream,
            )
            device_dz[fixture_name] = dz
            backward_audits[fixture_name] = audit
            reports.append(
                comparator.compare_object(
                    cpu_dz[fixture_name],
                    dz,
                    FormalObjectKind.DZ,
                    audit,
                )
            )
        negative_controls = _run_negative_controls(
            device=device,
            fixture=selected_fixture,
            cpu_statistics=cpu_statistics,
            cpu_pool=cpu_pool,
            cpu_dz=cpu_dz,
            gradients=gradients,
            forward_audit=forward_audit,
            backward_audits=backward_audits,
        )
        torch.cuda.synchronize(device)

    object_reports = tuple(reports)
    input_identity_pass = all(
        dict(report.audit.cpu_input_hashes) == dict(report.audit.device_input_hashes)
        for report in object_reports
    )
    environment_pass = calibration_environment.environment_pass(identity)
    zero_channel_cpu_std = cpu_statistics[:, 0, 0, 0, :, 1]
    zero_channel_device_std = forward_output.statistics_float64.to("cpu")[:, 0, 0, 0, :, 1]
    zero_variance_evidence = {
        "forward_std_exact_zero": bool(
            torch.all(zero_channel_cpu_std == 0.0) and torch.all(zero_channel_device_std == 0.0)
        ),
        "all_backward_results_finite": all(
            bool(torch.isfinite(cpu_dz[name]).all() and torch.isfinite(device_dz[name]).all())
            for name in gradients
        ),
        "mean_only_dz_has_nonzero": bool(
            torch.any(cpu_dz["mean_only"][:, 0, 0, 0] != 0.0)
            and torch.any(device_dz["mean_only"][:, 0, 0, 0] != 0.0)
        ),
        "std_only_zero_variance_dz_exact_zero": bool(
            torch.all(cpu_dz["std_only"][:, 0, 0, 0] == 0.0)
            and torch.all(device_dz["std_only"][:, 0, 0, 0] == 0.0)
        ),
        "joint_dz_has_nonzero": bool(
            torch.any(cpu_dz["joint_signed"][:, 0, 0, 0] != 0.0)
            and torch.any(device_dz["joint_signed"][:, 0, 0, 0] != 0.0)
        ),
    }
    run_pass = (
        all(
            report.verdict is Verdict.PASSED and report.quarter_margin_pass
            for report in object_reports
        )
        and all(control.detected for control in negative_controls)
        and input_identity_pass
        and environment_pass
        and all(zero_variance_evidence.values())
    )
    orchestrator_hash = _orchestrator_sha256()
    code_identities = {
        "cpu_reference": {
            "module": "cg_acceptance.cpu_reference",
            "version": cpu_reference.REFERENCE_VERSION,
            "sha256": cpu_reference.code_sha256(),
        },
        "device_operator": {
            "module": "cg_acceptance.device_operator",
            "version": device_operator.OPERATOR_VERSION,
            "sha256": device_operator.code_sha256(),
        },
        "comparator": {
            "module": "cg_acceptance.comparator",
            "version": object_reports[0].audit.comparator_version,
            "sha256": object_reports[0].audit.comparator_code_sha256,
        },
        "calibration_orchestrator": {
            "module": "cg_acceptance.calibration",
            "version": "decision30-calibration-gate-v2",
            "sha256": orchestrator_hash,
        },
    }
    registration_hashes = (
        dict(selected_fixture.registered_input_hashes)
        if selected_fixture.registered_input_hashes is not None
        else cpu_hashes
    )
    registration_shape = (
        selected_fixture.registered_formal_shape
        if selected_fixture.registered_formal_shape is not None
        else tuple(selected_fixture.z.shape)
    )
    report = CalibrationReport(
        run_id=run_id,
        mode=normalized_mode,
        gate_version="decision30-calibration-gate-v2",
        gate_code_sha256=orchestrator_hash,
        cpu_reference_version=cpu_reference.REFERENCE_VERSION,
        cpu_reference_code_sha256=cpu_reference.code_sha256(),
        code_identities=code_identities,
        device_identity=identity,
        formal_input_shape=tuple(selected_fixture.z.shape),
        input_hashes=cpu_hashes,
        fixture_registration={
            "fixture_name": selected_fixture.name,
            "input_hashes": registration_hashes,
            "formal_shape": registration_shape,
            "preregistered": selected_fixture.is_preregistered,
        },
        input_identity_pass=input_identity_pass,
        environment_pass=environment_pass,
        backward_execution={
            "api": "torch.autograd.backward(pool_float32, G)",
            "backward_calls": 3,
            "fresh_leaf_per_fixture": True,
            "gradient_source": "z.grad",
            "manual_device_dz_generation": False,
        },
        zero_variance_evidence=zero_variance_evidence,
        object_reports=object_reports,
        negative_controls=negative_controls,
        smoke_pass=run_pass if normalized_mode is CalibrationMode.LOCAL_SMOKE else None,
        formal_gate_pass=run_pass if normalized_mode is CalibrationMode.FORMAL_ACCEPTANCE else None,
        overall_pass=run_pass,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as output:
            json.dump(report.to_dict(), output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
    return report
