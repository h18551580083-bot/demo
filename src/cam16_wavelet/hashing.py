from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, default=_json_default, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()

