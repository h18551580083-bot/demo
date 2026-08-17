"""Small JSON artifact helpers shared by training entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .claims import audit_isolation_claim_payload


class PipelineBlockedError(RuntimeError):
    """Training or preflight was blocked before the first optimizer batch."""


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    claim_audit = audit_isolation_claim_payload(value)
    if claim_audit["status"] != "PASS":
        raise ValueError(f"unsafe patient-level claim in report: {claim_audit['forbidden_claims']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def read_json_object(path: Path) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise PipelineBlockedError(f"cannot read JSON record {path}: {error}") from error
    if not isinstance(value, dict):
        raise PipelineBlockedError(f"JSON record must be an object: {path}")
    return value
