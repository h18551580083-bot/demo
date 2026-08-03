"""Frozen CAM16 split-isolation claim vocabulary and fail-closed audits."""

from __future__ import annotations

import re
from typing import Any

GROUP_SLIDE_ISOLATION_CLAIM = "group_id/slide_id split isolation verified"
PATIENT_LEVEL_ISOLATION = "not_evaluated"
PATIENT_LEVEL_CLAIM_ALLOWED = False

_AFFIRMATIVE_PATIENT_CLAIM = re.compile(
    r"\bpatient(?:-level)?\s+(?:split\s+)?isolation"
    r"\s*(?:(?:has\s+been|is|was)\s+|[:=]\s*)?"
    r"(?:verified|validated|passed|safe|assured|established|demonstrated|confirmed|guaranteed)\b"
    r"|\bpatients?\s+(?:are|is|remain|remains|were|was)\s+"
    r"(?:fully\s+)?isolated\b"
    r"|\bpatient(?:-level)?\s+leakage\s+"
    r"(?:(?:has\s+been|is|was)\s+)?"
    r"(?:prevented|excluded|eliminated|controlled|absent|impossible)\b"
    r"|\bno\s+patients?\s+(?:identity\s+)?"
    r"(?:crosses|crossed|occurs?|appears?)\s+(?:the\s+)?splits?\b",
    re.IGNORECASE,
)
_PATIENT_SAFETY_TOPIC = re.compile(
    r"\bpatient(?:-level)?\b[^\n]{0,80}\b"
    r"(?:isolation|isolated|leakage|safety|protection|claim)\b"
    r"|\bno\s+patients?\b[^\n]{0,80}\bsplits?\b",
    re.IGNORECASE,
)
_SAFE_PATIENT_CONTEXT = re.compile(
    r"not_evaluated|not evaluated|not applicable|\bfalse\b|\bmust not\b|"
    r"\bdoes not\b|\bdo not\b|\bcannot\b|\bnot allowed\b|\bwithout\b|"
    r"\bonly when\b|\brequires?\b|\bforbidden\b|\bdisallow(?:ed)?\b|"
    r"\breject(?:ed|s)?\b|\bfails?\b|\bunmet\b|\boutside\b[^\n]{0,30}\bscope\b|"
    r"avoid|\bunless\b|\binsufficient\b|\bsupported by\b[^\n]{0,30}\breliable\b|"
    r"\bno patient-level\b[^\n]{0,30}\bclaim\b|\bnot\b",
    re.IGNORECASE,
)
_STATE_ASSIGNMENT = re.compile(
    r'["\']?patient_level_isolation["\']?\s*(?:=|:)\s*["\']?'
    r"(?P<value>[A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
_CLAIM_ASSIGNMENT = re.compile(
    r'["\']?patient_level_claim_allowed["\']?\s*(?:=|:)\s*["\']?'
    r"(?P<value>[A-Za-z0-9_-]+)",
    re.IGNORECASE,
)


def isolation_claim_fields() -> dict[str, str | bool]:
    """Return a fresh canonical claim payload for every generated report."""

    return {
        "isolation_claim": GROUP_SLIDE_ISOLATION_CLAIM,
        "patient_level_isolation": PATIENT_LEVEL_ISOLATION,
        "patient_level_claim_allowed": PATIENT_LEVEL_CLAIM_ALLOWED,
    }


def audit_isolation_claim_text(text: str) -> dict[str, Any]:
    """Reject affirmative prose and noncanonical machine-readable patient claims."""

    forbidden: list[str] = []
    affirmative_claim = _AFFIRMATIVE_PATIENT_CLAIM.search(text) is not None
    if affirmative_claim:
        forbidden.append("affirmative patient-level isolation claim")
    if not affirmative_claim:
        for paragraph in re.split(r"\r?\n\s*\r?\n", text):
            if _PATIENT_SAFETY_TOPIC.search(paragraph) and not _SAFE_PATIENT_CONTEXT.search(
                paragraph
            ):
                forbidden.append("unqualified patient-level safety statement")
                break
    for match in _STATE_ASSIGNMENT.finditer(text):
        if match.group("value").casefold() != PATIENT_LEVEL_ISOLATION:
            forbidden.append("patient_level_isolation must be not_evaluated")
            break
    for match in _CLAIM_ASSIGNMENT.finditer(text):
        if match.group("value").casefold() != "false":
            forbidden.append("patient_level_claim_allowed must be false")
            break
    return {"status": "PASS" if not forbidden else "FAIL", "forbidden_claims": forbidden}


def audit_isolation_claim_payload(value: Any) -> dict[str, Any]:
    """Recursively reject unsafe claim fields or prose in a report/log payload."""

    forbidden: list[str] = []
    reserved_fields = {
        "isolationclaim": "isolation_claim",
        "patientlevelisolation": "patient_level_isolation",
        "patientlevelclaimallowed": "patient_level_claim_allowed",
    }

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                semantic_key = (
                    re.sub(r"[^a-z0-9]", "", key.casefold())
                    if isinstance(key, str)
                    else key
                )
                normalized_key = reserved_fields.get(semantic_key, key)
                key_phrase = (
                    re.sub(
                        r"[^a-z0-9]+",
                        " ",
                        re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", key).casefold(),
                    ).strip()
                    if isinstance(key, str)
                    else ""
                )
                if (
                    semantic_key not in reserved_fields
                    and _PATIENT_SAFETY_TOPIC.search(key_phrase)
                ):
                    forbidden.append("unsupported patient-level safety field")
                if semantic_key in reserved_fields and key != normalized_key:
                    forbidden.append("isolation claim field names must use canonical lowercase")
                if normalized_key == "isolation_claim" and nested != GROUP_SLIDE_ISOLATION_CLAIM:
                    forbidden.append("isolation_claim must use the frozen group/slide statement")
                elif normalized_key == "patient_level_isolation" and nested != PATIENT_LEVEL_ISOLATION:
                    forbidden.append("patient_level_isolation must be not_evaluated")
                elif normalized_key == "patient_level_claim_allowed" and nested is not False:
                    forbidden.append("patient_level_claim_allowed must be false")
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
        elif isinstance(item, str):
            text_audit = audit_isolation_claim_text(item)
            forbidden.extend(text_audit["forbidden_claims"])

    visit(value)
    unique = list(dict.fromkeys(forbidden))
    return {"status": "PASS" if not unique else "FAIL", "forbidden_claims": unique}
