from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cam16_wavelet.hashing import stable_hash


def write_run_manifest(output: Path, config: dict[str, Any], extra: dict[str, Any]) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": stable_hash(config),
        "git_revision": revision,
        "python": sys.version,
        "platform": platform.platform(),
        **extra,
    }
    path = output / "source_snapshot.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

