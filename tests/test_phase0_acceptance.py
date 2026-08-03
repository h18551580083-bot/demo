from __future__ import annotations

from pathlib import Path

import pytest

from cg_pipeline.acceptance import (
    AcceptanceError,
    _documentation_audit,
    audit_decision30_report,
    audit_isolation_claim_text,
    audit_tracked_files,
)
from cg_pipeline.pipeline import _write_json_exclusive

REPOSITORY = Path(__file__).resolve().parents[1]


def _write_required_documentation(root: Path) -> None:
    required = [
        "AGENTS.md",
        "CONTEXT.md",
        "README.md",
        "docs/DEVELOPMENT_SPEC.md",
        "docs/DECISIONS.md",
        "docs/PHASE0_ACCEPTANCE_MATRIX.md",
        "docs/PHASE0_GAP_REGISTER.md",
        "docs/TRAINING_PROTOCOL.md",
        "docs/EVALUATION_PROTOCOL.md",
        "docs/adr/0010-phase1-preregistered-baseline.md",
    ]
    for relative in required:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("safe\n", encoding="utf-8")


def test_decision30_formal_report_hash_and_gate_are_audited() -> None:
    report = audit_decision30_report(
        REPOSITORY / "artifacts" / "decision30_formal_acceptance_rtx4090_20260802.json"
    )

    assert report["status"] == "PASS"
    assert report["mode"] == "formal_acceptance"
    assert report["formal_gate_pass"] is True
    assert report["overall_pass"] is True
    assert report["sha256"] == (
        "sha256:fe0c0a3d704a2ae458e26e894ef82be0fddcdf13ad430ac5f483bf72b1836117"
    )


def test_decision30_tamper_and_forbidden_tracked_files_fail_closed(tmp_path: Path) -> None:
    tampered = tmp_path / "report.json"
    tampered.write_text("{}", encoding="utf-8")
    with pytest.raises(AcceptanceError, match="SHA-256"):
        audit_decision30_report(tampered)

    assert audit_tracked_files(REPOSITORY)["status"] == "PASS"


@pytest.mark.parametrize(
    ("text", "canonical_claim"),
    [
        (
            "patient-level isolation verified",
            "affirmative patient-level isolation claim",
        ),
        (
            "patient isolation has been verified",
            "affirmative patient-level isolation claim",
        ),
        (
            "patients are isolated across all splits",
            "affirmative patient-level isolation claim",
        ),
        (
            '"patient_level_isolation": "verified"',
            "patient_level_isolation must be not_evaluated",
        ),
        (
            "patient_level_isolation: PASS",
            "patient_level_isolation must be not_evaluated",
        ),
        (
            "patient_level_claim_allowed = true",
            "patient_level_claim_allowed must be false",
        ),
        (
            "patient-level isolation: verified",
            "affirmative patient-level isolation claim",
        ),
        (
            "patient-level isolation is safe",
            "affirmative patient-level isolation claim",
        ),
        (
            "patient-level leakage is prevented",
            "affirmative patient-level isolation claim",
        ),
        (
            "no patient crosses splits",
            "affirmative patient-level isolation claim",
        ),
        (
            "patient-level protection achieved",
            "unqualified patient-level safety statement",
        ),
    ],
)
def test_patient_level_safety_claim_fails_documentation_audit(
    text: str, canonical_claim: str
) -> None:
    allowed = audit_isolation_claim_text(
        "group_id/slide_id split isolation verified; "
        "patient_level_isolation = not_evaluated; "
        "patient_level_claim_allowed = false"
    )
    forbidden = audit_isolation_claim_text(text)

    assert allowed == {"status": "PASS", "forbidden_claims": []}
    assert forbidden["status"] == "FAIL"
    assert forbidden["forbidden_claims"] == [canonical_claim]


def test_every_markdown_protocol_and_generated_report_are_claim_audited(
    tmp_path: Path,
) -> None:
    _write_required_documentation(tmp_path)
    draft = tmp_path / "docs" / "extra-protocol-draft.md"
    draft.write_text("patients are isolated across all splits\n", encoding="utf-8")

    audit = _documentation_audit(tmp_path)

    assert audit["status"] == "FAIL"
    assert audit["forbidden_patient_level_claims"] == [
        "docs/extra-protocol-draft.md"
    ]
    output = tmp_path / "unsafe-report.json"
    with pytest.raises(ValueError, match="unsafe patient-level claim"):
        _write_json_exclusive(
            output,
            {
                "patient_level_isolation": "PASS",
                "patient_level_claim_allowed": False,
            },
        )
    assert not output.exists()
    unsafe_payloads = [
        {
            "Patient_Level_Isolation": "verified",
            "Patient_Level_Claim_Allowed": True,
        },
        {"patient-level-isolation": "verified"},
        {"patient level isolation": "verified"},
        {"patient_level_isolation ": "verified"},
        {"patientLevelIsolation": "verified"},
        {"patient_isolation": "verified"},
        {"patientIsolation": "verified"},
        {"patientSplitIsolation": "verified"},
        {"patientSafetyVerified": True},
    ]
    for payload in unsafe_payloads:
        with pytest.raises(ValueError, match="unsafe patient-level claim"):
            _write_json_exclusive(output, payload)
        assert not output.exists()


def test_context_is_included_in_fail_closed_documentation_claim_audit(
    tmp_path: Path,
) -> None:
    _write_required_documentation(tmp_path)
    (tmp_path / "CONTEXT.md").write_text(
        "patient isolation has been verified\n", encoding="utf-8"
    )

    audit = _documentation_audit(tmp_path)

    assert audit["status"] == "FAIL"
    assert audit["forbidden_patient_level_claims"] == ["CONTEXT.md"]
