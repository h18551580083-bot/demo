from __future__ import annotations

import argparse
import json
import time
from pathlib import Path, PurePosixPath

import torch
from torch.nn import functional as F

from cg_pipeline.config import load_experiment_config
from cg_pipeline.data import PatchDataset, build_dataloader, validate_manifest
from cg_pipeline.model import FixedHEClassifier
from cg_pipeline.training import (
    build_adamw,
    configure_determinism,
    train_one_step,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_experiment_config("configs/phase1_baseline.toml")
    device = torch.device("cuda:0")

    assert torch.cuda.is_available()
    print("GPU:", torch.cuda.get_device_name(0))

    manifest = args.data_root / "metadata" / "training_manifest.csv"

    bundle = validate_manifest(
        args.data_root,
        manifest,
        check_files=False,
        reconcile_disk=False,
        effective_hash_splits=("train", "val"),
    )

    dataset = PatchDataset(bundle, "train")

    seed = 1729
    configure_determinism(seed)

    model = FixedHEClassifier(
        frontend_backend=str(config.model["frontend_backend"])
    ).to(device)

    optimizer = build_adamw(
        model,
        learning_rate=str(config.training["learning_rate"]),
        beta1=str(config.training["beta1"]),
        beta2=str(config.training["beta2"]),
        epsilon=str(config.training["epsilon"]),
        weight_decay=str(config.training["weight_decay"]),
    )

    loader = build_dataloader(
        dataset,
        batch_size=32,
        seed=seed,
        epoch=0,
        num_workers=8,
    )

    total_steps = len(loader) if args.steps == 0 else min(args.steps, len(loader))

    model.train()
    losses = []

    torch.cuda.synchronize()
    start = time.perf_counter()

    for step_index, batch in enumerate(loader, start=1):
        rgb = batch["rgb"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(rgb).logits
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)

        if step_index % 25 == 0 or step_index == total_steps:
            elapsed = time.perf_counter() - start
            print(
                f"step={step_index}/{total_steps} "
                f"loss={loss_value:.6f} "
                f"elapsed={elapsed:.1f}s "
                f"step_time={elapsed / step_index:.3f}s",
                flush=True,
            )

        if step_index >= total_steps:
            break

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    estimated_epoch_minutes = elapsed / total_steps * len(loader) / 60

    result = {
        "formal_experiment": False,
        "num_workers": 8,
        "batch_size": 32,
        "completed_steps": total_steps,
        "epoch_batches": len(loader),
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / total_steps,
        "samples_per_second": total_steps * 32 / elapsed,
        "estimated_epoch_minutes": estimated_epoch_minutes,
        "mean_loss": sum(losses) / len(losses),
        "gpu": torch.cuda.get_device_name(0),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
