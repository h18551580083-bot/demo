"""Command-line entry point for the Decision 30 calibration gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from .calibration import run_calibration_gate
from .fixture import CalibrationMode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="Real non-CPU device, for example cuda:0")
    parser.add_argument("--run-id", required=True, help="Stable audit run identifier")
    parser.add_argument("--output", required=True, type=Path, help="Strict JSON report path")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in CalibrationMode],
        default=CalibrationMode.LOCAL_SMOKE.value,
    )
    arguments = parser.parse_args(argv)

    report = run_calibration_gate(
        torch.device(arguments.device),
        run_id=arguments.run_id,
        output_path=arguments.output,
        mode=CalibrationMode(arguments.mode),
    )
    print(
        json.dumps(
            {
                "device": report.device_identity["gpu_name"],
                "mode": report.mode.value,
                "object_count": len(report.object_reports),
                "negative_control_count": len(report.negative_controls),
                "output": str(arguments.output.resolve()),
                "overall_pass": report.overall_pass,
            },
            sort_keys=True,
        )
    )
    return 0 if report.overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
