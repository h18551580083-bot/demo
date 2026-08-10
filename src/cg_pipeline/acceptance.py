"""Consolidated Phase 0 acceptance evidence and external-blocker classification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .claims import audit_isolation_claim_text
from .config import load_experiment_config
from .pipeline import _perform_preflight, _write_json_exclusive


class AcceptanceError(ValueError):
    """An acceptance artifact is missing, tampered, or internally inconsistent."""


_DECISION30_SHA256 = "fe0c0a3d704a2ae458e26e894ef82be0fddcdf13ad430ac5f483bf72b1836117"
def _read_json(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs):
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise AcceptanceError(f"duplicate JSON key in {path.name}: {key}")
            value[key] = item
        return value

    try:
        result = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda token: (_ for _ in ()).throw(AcceptanceError(token)),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise AcceptanceError(f"cannot read strict JSON {path}: {error}") from error
    if not isinstance(result, dict):
        raise AcceptanceError(f"JSON artifact must be an object: {path}")
    return result


def audit_decision30_report(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    try:
        data = source.read_bytes()
    except OSError as error:
        raise AcceptanceError(f"cannot read Decision 30 report: {error}") from error
    digest = hashlib.sha256(data).hexdigest()
    if digest != _DECISION30_SHA256:
        raise AcceptanceError("Decision 30 report SHA-256 does not match the approved artifact")
    report = _read_json(source)
    required = {
        "mode": "formal_acceptance",
        "formal_gate_pass": True,
        "overall_pass": True,
        "input_identity_pass": True,
        "environment_pass": True,
    }
    if any(report.get(key) != expected for key, expected in required.items()):
        raise AcceptanceError("Decision 30 formal gate fields do not match the approved result")
    if len(report.get("object_reports", [])) != 5 or len(report.get("negative_controls", [])) != 8:
        raise AcceptanceError("Decision 30 formal object or negative-control count is invalid")
    return {
        "status": "PASS",
        "sha256": "sha256:" + digest,
        "mode": report["mode"],
        "run_id": report["run_id"],
        "formal_gate_pass": report["formal_gate_pass"],
        "overall_pass": report["overall_pass"],
        "formal_object_count": len(report["object_reports"]),
        "negative_control_count": len(report["negative_controls"]),
    }


def audit_tracked_files(repository: Path | str) -> dict[str, Any]:
    root = Path(repository).resolve()
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = [item.decode("utf-8") for item in result.stdout.split(b"\x00") if item]
    forbidden_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".pt",
        ".pth",
        ".ckpt",
        ".tar",
        ".gz",
        ".zip",
        ".env",
    }
    forbidden = [
        path
        for path in paths
        if path.startswith(("artifacts/", "cam16_patch/"))
        or Path(path).suffix.lower() in forbidden_extensions
    ]
    if forbidden:
        raise AcceptanceError(f"forbidden tracked files: {forbidden}")
    return {"status": "PASS", "tracked_file_count": len(paths), "forbidden_count": 0}


def _run_command(command: list[str], repository: Path, environment: dict[str, str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": subprocess.list2cmdline(command),
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "status": "PASS" if result.returncode == 0 else "FAIL",
    }


def _test_evidence(repository: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    source_path = str(repository / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_path if not existing else source_path + os.pathsep + existing
    commands = [
        [sys.executable, "-m", "pytest", "tests", "-q"],
        [sys.executable, "-m", "compileall", "-q", "src", "tests"],
        [sys.executable, "-m", "ruff", "check", "."],
        ["git", "diff", "--check"],
    ]
    reports = [_run_command(command, repository, environment) for command in commands]
    return {
        "status": "PASS" if all(item["status"] == "PASS" for item in reports) else "FAIL",
        "commands": reports,
    }


def _documentation_audit(repository: Path) -> dict[str, Any]:
    required = [
        "AGENTS.md",
        "CONTEXT.md",
        "README.md",
        "docs/DEVELOPMENT_SPEC.md",
        "docs/DECISIONS.md",
        "docs/PHASE0_ACCEPTANCE_MATRIX.md",
        "docs/PHASE0_GAP_REGISTER.md",
        "docs/TRAINING_PROTOCOL.md",
        "docs/PHASE1_TRAINING_RUNBOOK.md",
        "docs/EVALUATION_PROTOCOL.md",
        "docs/adr/0010-phase1-preregistered-baseline.md",
    ]
    missing = [path for path in required if not (repository / path).is_file()]
    claim_paths = {
        repository / "AGENTS.md",
        repository / "CONTEXT.md",
        repository / "README.md",
        *(repository / "docs").rglob("*.md"),
    }
    forbidden_claims = [
        str(path.relative_to(repository)).replace("\\", "/")
        for path in sorted(claim_paths)
        if path.is_file()
        and audit_isolation_claim_text(path.read_text(encoding="utf-8"))["status"] == "FAIL"
    ]
    specification = (repository / "docs" / "DEVELOPMENT_SPEC.md").read_text(encoding="utf-8")
    blocking_section = specification.split("## 6. Blocking unresolved decisions", 1)[-1].split(
        "## 7. Prohibited actions", 1
    )[0]
    active_tbd = (
        "The following values remain `TBD`" in blocking_section
        or "must not be inferred, defaulted, or selected" in blocking_section
    )
    return {
        "status": (
            "PASS" if not missing and not active_tbd and not forbidden_claims else "FAIL"
        ),
        "required_document_count": len(required),
        "missing": missing,
        "active_blocking_tbd": active_tbd,
        "forbidden_patient_level_claims": forbidden_claims,
    }


def run_phase0_acceptance(
    *,
    repository: Path | str,
    config_path: Path | str,
    data_root: Path | str,
    release_path: Path | str,
    decision30_report_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    root = Path(repository).resolve()
    config = load_experiment_config(config_path)
    preflight, _ = _perform_preflight(
        config,
        data_root=Path(data_root).resolve(),
        release_path=Path(release_path).resolve(),
    )
    decision30 = audit_decision30_report(decision30_report_path)
    tracked = audit_tracked_files(root)
    tests = _test_evidence(root)
    documents = _documentation_audit(root)
    internal_preflight_failures = list(preflight["blocking_gates"])
    gates = [
        {
            "gate": "documentation_and_decision_freeze",
            "result": documents["status"],
            "command": "internal documentation audit",
            "evidence": "docs/DEVELOPMENT_SPEC.md; docs/DECISIONS.md; protocol docs",
            "blocker": None if documents["status"] == "PASS" else documents,
        },
        {
            "gate": "configuration_contract",
            "result": "PASS",
            "command": "python -m cg_pipeline formal-preflight ...",
            "evidence": preflight["config_hash"],
            "blocker": None,
        },
        {
            "gate": "patch_manifest_and_disk_contract",
            "result": "PASS" if "manifest_and_disk" in preflight["passed_gates"] else "FAIL",
            "command": "python -m cg_pipeline formal-preflight ...",
            "evidence": preflight["source_manifest_sha256"],
            "blocker": None,
        },
        {
            "gate": "slide_id_split_isolation",
            "result": "PASS" if "slide_id_isolation" in preflight["passed_gates"] else "FAIL",
            "command": "python -m cg_pipeline formal-preflight ...",
            "evidence": preflight["isolation"],
            "blocker": None,
        },
        {
            "gate": "patient_level_split_isolation",
            "result": "NOT APPLICABLE",
            "command": "not applicable to the CAM16 Phase 1 claim scope",
            "evidence": preflight["patient_mapping"],
            "blocker": None,
        },
        {
            "gate": "fixed_frontend_and_morlet_contract",
            "result": (
                "PASS"
                if {"fixed_frontend", "morlet_spectral_coverage"}.issubset(
                    preflight["passed_gates"]
                )
                else "FAIL"
            ),
            "command": "python -m pytest tests/test_morlet_frontend.py tests/test_morlet_spectral.py -q",
            "evidence": preflight["fixed_frontend_identity"],
            "blocker": None,
        },
        {
            "gate": "electronic_model_and_optimizer_contract",
            "result": "PASS" if "optimizer_ownership" in preflight["passed_gates"] else "FAIL",
            "command": "python -m pytest tests/test_model_contract.py tests/test_training_protocol.py -q",
            "evidence": preflight["optimizer_ownership"],
            "blocker": None,
        },
        {
            "gate": "determinism_and_precision_contract",
            "result": (
                "PASS" if "precision_and_determinism" in preflight["passed_gates"] else "FAIL"
            ),
            "command": "python -m pytest tests/test_training_protocol.py -q",
            "evidence": preflight["determinism"],
            "blocker": None,
        },
        {
            "gate": "decision30_formal_cuda_equivalence",
            "result": decision30["status"],
            "command": "audit approved ignored Decision 30 report",
            "evidence": decision30,
            "blocker": None,
        },
        {
            "gate": "training_and_evaluation_protocol",
            "result": "PASS" if not internal_preflight_failures else "FAIL",
            "command": "python -m pytest tests/test_training_protocol.py tests/test_evaluation_protocol.py -q",
            "evidence": "configs/phase1_baseline.toml; protocol docs",
            "blocker": internal_preflight_failures or None,
        },
        {
            "gate": "full_tests_static_and_forbidden_file_audit",
            "result": "PASS" if tests["status"] == tracked["status"] == "PASS" else "FAIL",
            "command": "pytest; compileall; ruff; git diff --check; git ls-files audit",
            "evidence": {"tests": tests, "tracked": tracked},
            "blocker": None if tests["status"] == tracked["status"] == "PASS" else "verification failed",
        },
        {
            "gate": "formal_training_release",
            "result": (
                "PASS" if "phase1_training_release" in preflight["passed_gates"] else "FAIL"
            ),
            "command": "python -m cg_pipeline formal-preflight ...",
            "evidence": str(Path(release_path)),
            "blocker": (
                None
                if "phase1_training_release" in preflight["passed_gates"]
                else "Phase 1 training release did not pass"
            ),
        },
    ]
    failed = [gate["gate"] for gate in gates if gate["result"] == "FAIL"]
    external_blockers: list[dict[str, Any]] = []
    report = {
        "schema": "phase0-total-acceptance-report-v1",
        "status": "PASS" if not failed else "FAIL",
        "phase0_closed": not failed,
        "formal_training_authorized": not failed,
        "isolation_claim": preflight["isolation_claim"],
        "patient_level_isolation": preflight["patient_level_isolation"],
        "patient_level_claim_allowed": preflight["patient_level_claim_allowed"],
        "test_split_accessed": False,
        "config_hash": config.sha256,
        "preflight": preflight,
        "gates": gates,
        "failed_gates": failed,
        "external_blockers": external_blockers,
    }
    _write_json_exclusive(Path(output_path), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cg-phase0-acceptance")
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--decision30-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_phase0_acceptance(
            repository=args.repository,
            config_path=args.config,
            data_root=args.data_root,
            release_path=args.release,
            decision30_report_path=args.decision30_report,
            output_path=args.output,
        )
    except (AcceptanceError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "failed_gates": report["failed_gates"]}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
