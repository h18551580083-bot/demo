from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from cg_pipeline.config import load_experiment_config
from cg_pipeline.data import PatchDataset, build_dataloader, validate_manifest
from cg_pipeline.model import FixedHEClassifier
from cg_pipeline.pooling import flatten_pooled
from cg_pipeline.training import build_adamw, configure_determinism

DATA_ROOT = Path(
    "/root/gpufree-data/cam16_extracted/"
    "cam16_patch_256_s128_512_class_quota"
)
WARMUP = 2
MEASURE = 5

config = load_experiment_config("configs/phase1_baseline.toml")
device = torch.device("cuda:0")

bundle = validate_manifest(
    DATA_ROOT,
    DATA_ROOT / "metadata" / "training_manifest.csv",
    check_files=False,
    reconcile_disk=False,
    effective_hash_splits=("train", "val"),
)

dataset = PatchDataset(bundle, "train")
configure_determinism(1729)

loader = build_dataloader(
    dataset,
    batch_size=32,
    seed=1729,
    epoch=0,
    num_workers=8,
)

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

iterator = iter(loader)
records = []


def gpu_time(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


for step in range(WARMUP + MEASURE):
    data_start = time.perf_counter()
    batch = next(iterator)
    data_seconds = time.perf_counter() - data_start

    (rgb, target), transfer_seconds = gpu_time(
        lambda: (
            batch["rgb"].to(device),
            batch["target"].to(device),
        )
    )

    optimizer.zero_grad(set_to_none=True)

    frontend, frontend_seconds = gpu_time(
        lambda: model.frontend(rgb)
    )

    interaction, interaction_seconds = gpu_time(
        lambda: model.interaction(
            frontend.feature_h,
            frontend.feature_e,
            frontend.valid_support_mask,
        )
    )

    pooling, pooling_seconds = gpu_time(
        lambda: model.pooling(
            interaction.features,
            interaction.valid_support_mask,
            interaction.neighborhood_valid_support_mask,
        )
    )

    def classify():
        classifier_input = flatten_pooled(pooling.pool_float32)
        return model.classifier(classifier_input).squeeze(-1)

    logits, classifier_seconds = gpu_time(classify)

    loss, loss_seconds = gpu_time(
        lambda: F.binary_cross_entropy_with_logits(logits, target)
    )

    _, backward_seconds = gpu_time(loss.backward)
    _, optimizer_seconds = gpu_time(optimizer.step)

    values = {
        "data": data_seconds,
        "transfer": transfer_seconds,
        "frontend": frontend_seconds,
        "interaction": interaction_seconds,
        "pooling": pooling_seconds,
        "classifier": classifier_seconds,
        "loss": loss_seconds,
        "backward": backward_seconds,
        "optimizer": optimizer_seconds,
    }

    total = sum(values.values())

    print(
        f"step={step + 1} total={total:.3f}s "
        f"frontend={frontend_seconds:.3f}s "
        f"interaction={interaction_seconds:.3f}s "
        f"pooling={pooling_seconds:.3f}s "
        f"backward={backward_seconds:.3f}s",
        flush=True,
    )

    if step >= WARMUP:
        records.append(values)

print("\n=== 5-step average ===")
averages = {
    key: statistics.mean(record[key] for record in records)
    for key in records[0]
}
total = sum(averages.values())

for key, value in sorted(
    averages.items(),
    key=lambda item: item[1],
    reverse=True,
):
    print(f"{key:12s}: {value:8.3f}s  {value / total:6.1%}")

print(f"{'total':12s}: {total:8.3f}s")
