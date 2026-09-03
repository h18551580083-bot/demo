"""Read frozen Phase1 completion records and print comparison Markdown only."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "morlet": "phase1-cam16-baseline-b32-v2",
    "matched_control": "phase1-cam16-matched-control-b32-v1",
}
SEEDS = (1729, 3407)


def summarize(values: list[float]) -> tuple[float, float]:
    """Return the seed mean and sample SD (denominator n - 1)."""
    return mean(values), stdev(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        average, sample_sd = summarize([0.0, 1.0])
        assert average == 0.5
        assert math.isclose(sample_sd, math.sqrt(0.5), abs_tol=1e-15)
        assert summarize([0.8, 0.8]) == (0.8, 0.0)
        print("PASS: mean, n-1 sample SD, and equal-value case")
        return

    rows = []
    statistics = {}
    for variant, run_id in RUNS.items():
        values = []
        for seed in SEEDS:
            path = ROOT / "artifacts/formal_runs" / run_id / f"seed-{seed}" / "completion.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            epoch = record["best_epoch"]
            value = record["best_validation_slide_auroc"]
            if not (
                record["run_id"] == run_id
                and record["seed"] == seed
                and record["status"] == "complete"
                and record["test_split_accessed"] is False
                and type(epoch) is int
                and 0 <= epoch < record["epochs_completed"]
                and type(value) in (int, float)
                and math.isfinite(value)
                and 0 <= value <= 1
            ):
                raise ValueError(f"Invalid frozen completion record: {path}")
            rows.append(f"| {variant} | {seed} | {epoch} | {value!r} |")
            values.append(value)
        statistics[variant] = summarize(values)

    print("| frontend_variant | seed | best_epoch | best_validation_slide_auroc |")
    print("| --- | --- | --- | --- |")
    print("\n".join(rows))
    print("\n| Metric | Morlet | matched-control | Difference |")
    print("| --- | --- | --- | --- |")
    for index, label in enumerate(("Mean AUROC", "Sample SD")):
        morlet = statistics["morlet"][index]
        control = statistics["matched_control"][index]
        print(f"| {label} | {morlet:.16f} | {control:.16f} | {morlet - control:+.16f} |")


if __name__ == "__main__":
    main()
