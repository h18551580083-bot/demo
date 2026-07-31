"""Decision 30 cross-device numerical-equivalence acceptance interfaces."""

from .calibration import CalibrationReport, NegativeControlResult, run_calibration_gate
from .comparator import (
    AuditContext,
    ComparisonAuditRecord,
    ComparisonReport,
    FailureDiagnostic,
    FormalObjectKind,
    MaximumErrorDiagnostic,
    Verdict,
    compare_object,
)
from .fixture import CalibrationFixture, CalibrationMode, build_deterministic_fixture

__all__ = [
    "AuditContext",
    "CalibrationFixture",
    "CalibrationMode",
    "CalibrationReport",
    "ComparisonAuditRecord",
    "ComparisonReport",
    "FailureDiagnostic",
    "FormalObjectKind",
    "MaximumErrorDiagnostic",
    "NegativeControlResult",
    "Verdict",
    "build_deterministic_fixture",
    "compare_object",
    "run_calibration_gate",
]
