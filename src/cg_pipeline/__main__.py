"""Command-line entry point for Phase 0 acceptance and guarded training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import Phase0BlockedError, run_dry_run, run_formal_training, run_preflight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cg-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry = subparsers.add_parser("dry-run", help="run the non-formal synthetic end-to-end chain")
    dry.add_argument("--config", type=Path, required=True)
    dry.add_argument("--workspace-root", type=Path, required=True)

    preflight = subparsers.add_parser("preflight", help="validate all gates before training")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--data-root", type=Path, required=True)
    preflight.add_argument("--release", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--patient-mapping", type=Path)
    preflight.add_argument("--patient-mapping-approval", type=Path)

    train = subparsers.add_parser("train", help="run formal train/validation after preflight")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--data-root", type=Path, required=True)
    train.add_argument("--release", type=Path, required=True)
    train.add_argument("--patient-mapping", type=Path)
    train.add_argument("--patient-mapping-approval", type=Path)
    train.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "dry-run":
            report = run_dry_run(args.config, workspace_root=args.workspace_root)
            print(json.dumps({"status": report["status"], "report": report["config_hash"]}))
            return 0
        if args.command == "preflight":
            report = run_preflight(
                args.config,
                data_root=args.data_root,
                release_path=args.release,
                output_path=args.output,
                patient_mapping_path=args.patient_mapping,
                patient_mapping_approval_path=args.patient_mapping_approval,
            )
            print(json.dumps({"status": report["status"], "blocking_gates": report["blocking_gates"]}))
            return 0 if report["status"] == "PASS" else 1
        if args.command == "train":
            report = run_formal_training(
                args.config,
                data_root=args.data_root,
                release_path=args.release,
                patient_mapping_path=args.patient_mapping,
                patient_mapping_approval_path=args.patient_mapping_approval,
                resume=args.resume,
            )
            print(json.dumps({"status": "PASS", "runs": report["runs"]}))
            return 0
    except Phase0BlockedError as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 3
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
