"""Restricted RFC 8785 JCS serialization and domain-separated identities.

The repository's scientific envelopes intentionally encode non-integral numeric
parameters as canonical strings.  Rejecting JSON floats keeps this implementation
inside the exactly implemented subset of RFC 8785 instead of delegating number
formatting to a platform JSON encoder.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from typing import Any

_MAX_SAFE_INTEGER = (1 << 53) - 1


def _validate_string(value: str) -> None:
    for character in value:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise ValueError("unpaired Unicode surrogate is not valid JCS")


def _utf16_sort_key(value: str) -> bytes:
    _validate_string(value)
    return value.encode("utf-16-be")


def _serialize(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_INTEGER:
            raise ValueError("integer exceeds the RFC 8785 safe integer range")
        return str(value)
    if isinstance(value, float):
        raise TypeError("floating JSON values are prohibited; use a canonical numeric string")
    if isinstance(value, str):
        _validate_string(value)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JCS object keys must be strings")
        ordered = sorted(value, key=_utf16_sort_key)
        return "{" + ",".join(f"{_serialize(key)}:{_serialize(value[key])}" for key in ordered) + "}"
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        return "[" + ",".join(_serialize(item) for item in value) + "]"
    raise ValueError(f"unsupported JCS value type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return UTF-8 RFC 8785 bytes for the repository's exact JSON subset."""

    return _serialize(value).encode("utf-8")


def domain_hash(domain: str, header: Mapping[str, Any], payload: bytes = b"") -> str:
    """Hash ``T || 0x00 || uint64be(len(H)) || H || P``."""

    _validate_string(domain)
    if not domain.isascii() or not domain:
        raise ValueError("identity domains must be nonempty ASCII")
    header_bytes = canonical_json_bytes(header)
    declared_length = header.get("payload_length")
    if declared_length is not None and declared_length != len(payload):
        raise ValueError("header payload_length does not match payload bytes")
    preimage = domain.encode("utf-8") + b"\x00" + struct.pack(">Q", len(header_bytes))
    digest = hashlib.sha256(preimage + header_bytes + payload).hexdigest()
    return f"sha256:{digest}"


def raw_sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()
