from __future__ import annotations

from pathlib import Path

import pytest

from cg_pipeline.acceptance import (
    AcceptanceError,
    audit_decision30_report,
    audit_tracked_files,
)

REPOSITORY = Path(__file__).resolve().parents[1]


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
