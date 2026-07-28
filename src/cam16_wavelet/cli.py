from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cam16_wavelet.config import load_yaml
from cam16_wavelet.contracts import DetectorConfig, DatasetSpec, FrontendConfig, KernelSpec
from cam16_wavelet.data import DatasetRegistry
from cam16_wavelet.models import LightweightBackend
from cam16_wavelet.optics import OpticalBackbone


def _dataset_spec(config: dict) -> DatasetSpec:
    return DatasetSpec(
        dataset_id=config["dataset_id"],
        root=Path(config["root"]),
        manifest=Path(config["manifest"]),
        class_mapping=config["class_mapping"],
        patient_id_column=config.get("patient_id_column", "slide_id"),
    )


def _frontend_config(config: dict) -> FrontendConfig:
    kernel = KernelSpec(**config["kernel"])
    detector = DetectorConfig(**config["detector"])
    return FrontendConfig(mode=config["mode"], kernel=kernel, detector=detector)


def validate_dataset(path: str, check_files: bool) -> None:
    bundle = DatasetRegistry.load(_dataset_spec(load_yaml(path)), check_files=check_files)
    print(
        json.dumps(
            {
                "dataset_id": bundle.spec.dataset_id,
                "rows": len(bundle.rows),
                "split_counts": bundle.split_counts,
                "label_counts": bundle.label_counts,
                "manifest_hash": bundle.manifest_hash,
                "isolation": "passed",
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def smoke_test(path: str) -> None:
    config = load_yaml(path)
    frontend = OpticalBackbone(_frontend_config(load_yaml(config["frontend_config"])))
    backend = LightweightBackend(len(frontend.channel_names), **config["backend"])
    image = torch.rand(config["batch_size"], 3, config["image_size"], config["image_size"])
    with torch.no_grad():
        features = frontend(image)
        logits = backend(features)
    print(
        json.dumps(
            {
                "feature_shape": list(features.shape),
                "logit_shape": list(logits.shape),
                "kernel_bank_hash": frontend.kernel_bank_hash,
                "backend_parameters": backend.parameter_count,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="cam16-wavelet")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-dataset")
    validate.add_argument("--config", required=True)
    validate.add_argument("--check-files", action="store_true")
    smoke = subparsers.add_parser("smoke-test")
    smoke.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.command == "validate-dataset":
        validate_dataset(args.config, args.check_files)
    else:
        smoke_test(args.config)


if __name__ == "__main__":
    main()

