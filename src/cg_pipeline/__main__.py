"""Command-line entry point for exploratory and formal CAM16 training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import (
    Phase0BlockedError,
    run_exploratory_training,
    run_formal_training,
    run_preflight,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cg-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    exploratory = subparsers.add_parser(
        "exploratory-train",
        help="run non-formal CAM16 train/validation with lightweight safety checks",
    )
    exploratory.add_argument("--config", type=Path, required=True)
    exploratory.add_argument("--data-root", type=Path, required=True)
    exploratory.add_argument("--device")
    exploratory.add_argument("--seed", type=int)
    exploratory.add_argument("--output", type=Path)
    exploratory.add_argument("--run-id")
    exploratory.add_argument("--batch-size", type=int)
    exploratory.add_argument("--num-workers", type=int)
    exploratory.add_argument("--max-epochs", type=int)
    exploratory.add_argument("--max-steps", type=int)

    preflight = subparsers.add_parser(
        "formal-preflight", help="validate all formal gates before formal training"
    )
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--data-root", type=Path, required=True)
    preflight.add_argument("--release", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser(
        "formal-train", help="run release-bound formal train/validation after preflight"
    )
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--data-root", type=Path, required=True)
    train.add_argument("--release", type=Path, required=True)
    train.add_argument("--preflight-report", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "exploratory-train":
            report = run_exploratory_training(
                args.config,
                data_root=args.data_root,
                device=args.device,
                seed=args.seed,
                output=args.output,
                run_id=args.run_id,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_epochs=args.max_epochs,
                max_steps=args.max_steps,
            )
            print(json.dumps({"status": "PASS", "run": report["run"]}))
            return 0
        if args.command == "formal-preflight":
            report = run_preflight(
                args.config,
                data_root=args.data_root,
                release_path=args.release,
                output_path=args.output,
            )
            print(json.dumps({"status": report["status"], "blocking_gates": report["blocking_gates"]}))
            return 0 if report["status"] == "PASS" else 1
        if args.command == "formal-train":
            report = run_formal_training(
                args.config,
                data_root=args.data_root,
                release_path=args.release,
                preflight_report_path=args.preflight_report,
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
