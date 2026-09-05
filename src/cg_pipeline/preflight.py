"""Experiment-correctness preflight without Git, code, config, or report identity gates."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .artifacts import PipelineBlockedError, read_json_object, write_json_exclusive
from .claims import PATIENT_LEVEL_CLAIM_ALLOWED, PATIENT_LEVEL_ISOLATION, isolation_claim_fields
from .config import PHASE2_MORLET_CONTRACT, ExperimentConfig
from .data import ManifestBundle
from .model import FixedHEClassifier
from .morlet import audit_morlet_identity, generate_morlet_bundle, validate_spectral_coverage
from .morlet_validity import validate_phase2_morlet
from .runtime import BatchContract, batch_contract, build_optimizer, validate_training_data
from .training import audit_optimizer_ownership, configure_determinism

_REQUIRED_PASSED_GATES = {
    "configuration",
    "manifest_and_disk",
    "slide_id_isolation",
    "fixed_frontend",
    "optimizer_ownership",
    "precision_and_determinism",
    "test_access_disabled",
    "formal_training_authorization",
}


def _frontend_gate(config: ExperimentConfig) -> str:
    if config.model["contract_id"] == PHASE2_MORLET_CONTRACT:
        return "phase2a_morlet_validity"
    return (
        "morlet_spectral_coverage"
        if config.frontend_variant == "morlet"
        else "matched_control_numerical"
    )


def validate_training_authorization(path: Path) -> dict[str, Any]:
    value = read_json_object(path)
    checks = {
        "schema": value.get("schema") == "formal-training-authorization-v1",
        "formal_training_authorized": value.get("formal_training_authorized") is True,
        "test_access_authorized": value.get("test_access_authorized") is False,
        "external_blockers": value.get("external_blockers") == [],
        "patient_level_isolation": value.get("patient_level_isolation") == PATIENT_LEVEL_ISOLATION,
        "patient_level_claim_allowed": value.get("patient_level_claim_allowed")
        is PATIENT_LEVEL_CLAIM_ALLOWED,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise PipelineBlockedError("formal training authorization failed: " + ", ".join(failed))
    return value


def _model_audits(config: ExperimentConfig, device: torch.device) -> dict[str, Any]:
    model = FixedHEClassifier(
        frontend_backend=str(config.model["frontend_backend"]),
        frontend_variant=config.frontend_variant,
        **{key: config.model[key] for key in ("sigma0", "xi0", "gamma")},
    ).to(device)
    fixed_identity = model.frontend.fixed_state_identity()
    if config.frontend_variant == "morlet":
        parameters = {key: config.model[key] for key in ("sigma0", "xi0", "gamma")}
        morlet_bundle = generate_morlet_bundle(**parameters)
        spectral = validate_spectral_coverage(morlet_bundle, xi0=parameters["xi0"])
        phase2 = config.model["contract_id"] == PHASE2_MORLET_CONTRACT
        morlet_audit = (
            validate_phase2_morlet(morlet_bundle, **parameters)
            if phase2
            else audit_morlet_identity(morlet_bundle, spectral_coverage=spectral)
        )
        fixed_matches = all(
            (
                fixed_identity.get("morlet_parameter_hash") == morlet_bundle.parameter_hash,
                fixed_identity.get("canonical_kernel_hash") == morlet_bundle.canonical_kernel_hash,
                fixed_identity.get("spatial_execution_hash")
                == morlet_bundle.spatial_execution_hash,
            )
        )
        if morlet_audit["status"] != "PASS" or not fixed_matches:
            raise PipelineBlockedError("fixed frontend Morlet audit failed")
        if not phase2 and spectral["status"] != "PASS":
            raise PipelineBlockedError("Morlet spectral coverage failed")
        frontend_audit = {
            "morlet_identity_audit": morlet_audit,
            "morlet_spectral_coverage": spectral,
        }
        if phase2:
            # Synthetic forward validates the actual frontend/backend interface; no data IO.
            with torch.no_grad():
                output = model(
                    torch.zeros(
                        (1, 3, int(config.model["input_height"]), int(config.model["input_width"])),
                        dtype=torch.uint8,
                        device=device,
                    )
                )
            interface_checks = {
                "he_shape": output.frontend.feature_h.shape
                == output.frontend.feature_e.shape
                == (1, 4, 8, int(config.model["input_height"]), int(config.model["input_width"])),
                "classifier_input": tuple(output.classifier_input.shape) == (1, 9408),
                "backend_parameters": sum(p.numel() for p in model.electronic_parameters()) == 9473,
                "fixed_frontend": not list(model.frontend.parameters()),
                "finite_output": bool(
                    torch.isfinite(output.classifier_input).all()
                    and torch.isfinite(output.logits).all()
                ),
            }
            if not all(interface_checks.values()):
                raise PipelineBlockedError("Phase2-A frontend/backend interface failed")
            frontend_audit = {
                "phase2a_morlet_validity": {**morlet_audit, "interface_checks": interface_checks},
                "morlet_spectral_diagnostics": spectral,
            }
    else:
        # Construction validates DC and unit energy; hashes are provenance only.
        frontend_audit = {
            "matched_control_numerical_audit": {"status": "PASS"},
            "frontend_artifact_identity": model.frontend.artifact_identity(),
        }
    if config.model["contract_id"] == PHASE2_MORLET_CONTRACT:
        frontend_audit["morlet_parameters"] = parameters
        frontend_audit["morlet_numerical_diagnostics"] = morlet_bundle.validation
    optimizer = build_optimizer(config, model)
    return {
        "fixed_frontend_identity": fixed_identity,
        **frontend_audit,
        "optimizer_ownership": audit_optimizer_ownership(model, optimizer),
    }


def _base_report(
    bundle: ManifestBundle,
    contract: BatchContract,
    determinism: dict[str, Any],
    configured_device: str,
) -> dict[str, Any]:
    return {
        "schema": "formal-training-preflight-v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "batch_contract": contract,
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "effective_split_hashes": bundle.effective_split_hashes,
        "split_counts": bundle.split_counts,
        "label_counts": bundle.label_counts,
        "disk_inventory": bundle.disk_inventory,
        "isolation": asdict(bundle.isolation),
        **isolation_claim_fields(),
        "patient_mapping": {
            "status": PATIENT_LEVEL_ISOLATION,
            "reason": "patient identity is outside the CAM16 Phase 1 claim scope",
        },
        "determinism": determinism,
        "configured_device": configured_device,
        "training_started": False,
        "test_split_accessed": False,
    }


def perform_preflight(
    config: ExperimentConfig,
    *,
    data_root: Path,
    authorization_path: Path,
) -> tuple[dict[str, Any], ManifestBundle]:
    if config.execution_kind != "formal_train":
        raise ValueError("preflight requires execution.kind=formal_train")
    bundle = validate_training_data(config, data_root=data_root)
    validate_training_authorization(authorization_path)
    determinism = configure_determinism(int(config.training["seeds"][0]))
    configured = str(config.execution["device"])
    device = torch.device(configured)
    passed = {
        "configuration",
        "manifest_and_disk",
        "slide_id_isolation",
        "precision_and_determinism",
        "test_access_disabled",
        "formal_training_authorization",
    }
    blocked: list[str] = []
    not_applicable = ["patient_level_isolation"]
    if (
        config.frontend_variant == "matched_control"
        or config.model["contract_id"] == PHASE2_MORLET_CONTRACT
    ):
        not_applicable.append("morlet_spectral_coverage")
    evidence: dict[str, Any] = {}
    if device.type == "cuda" and not torch.cuda.is_available():
        blocked.append("configured_device")
    else:
        evidence = _model_audits(config, device)
        passed.update({"fixed_frontend", _frontend_gate(config), "optimizer_ownership"})
    report = _base_report(bundle, batch_contract(config, bundle), determinism, configured)
    report.update(evidence)
    report.update(
        {
            "status": "PASS" if not blocked else "FAIL",
            "effective_preflight_device": configured if not blocked else "unavailable",
            "passed_gates": sorted(passed),
            "not_applicable_gates": not_applicable,
            "blocking_gates": blocked,
        }
    )
    return report, bundle


def run_preflight_report(
    config: ExperimentConfig,
    *,
    data_root: Path,
    authorization_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    report, _ = perform_preflight(
        config, data_root=data_root, authorization_path=authorization_path
    )
    write_json_exclusive(output_path, report)
    return report


def consume_preflight_report(
    config: ExperimentConfig,
    *,
    data_root: Path,
    authorization_path: Path,
    preflight_report_path: Path,
) -> tuple[dict[str, Any], ManifestBundle]:
    report = read_json_object(preflight_report_path)
    required_gates = _REQUIRED_PASSED_GATES | {_frontend_gate(config)}
    checks = {
        "status": report.get("status") == "PASS",
        "blocking_gates": report.get("blocking_gates") == [],
        "training_started": report.get("training_started") is False,
        "test_split_accessed": report.get("test_split_accessed") is False,
        "required_gates": required_gates.issubset(set(report.get("passed_gates", []))),
    }
    if config.model["contract_id"] == PHASE2_MORLET_CONTRACT:
        checks["morlet_parameters"] = report.get("morlet_parameters") == {
            key: config.model[key] for key in ("sigma0", "xi0", "gamma")
        }
        checks["phase2a_validity_status"] = (
            report.get("phase2a_morlet_validity", {}).get("status") == "PASS"
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise PipelineBlockedError("preflight report is not safe to consume: " + ", ".join(failed))
    validate_training_authorization(authorization_path)
    if str(config.execution["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise PipelineBlockedError("configured CUDA device is unavailable")
    bundle = validate_training_data(config, data_root=data_root)
    return report, bundle
