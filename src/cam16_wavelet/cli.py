from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cam16_wavelet.config import load_yaml
from cam16_wavelet.contracts import DetectorConfig, DatasetSpec, FrontendConfig, KernelSpec
from cam16_wavelet.data import DatasetRegistry
from cam16_wavelet.audit import write_run_manifest
from cam16_wavelet.models import LightweightBackend
from cam16_wavelet.optics import OpticalBackbone


def _dataset_spec(config: dict) -> DatasetSpec:
    return DatasetSpec(
        dataset_id=config["dataset_id"],
        root=Path(config["root"]),
        manifest=Path(config["manifest"]),
        class_mapping=config["class_mapping"],
        patient_id_column=config.get("patient_id_column", "slide_id"),
        sample_id_column=config.get("sample_id_column", "patch_id"),
        path_column=config.get("path_column", "patch_path"),
        split_column=config.get("split_column", "split"),
        label_column=config.get("label_column", "label"),
        label_name_column=config.get("label_name_column", "label_name"),
        license_note=config.get("license_note", "TBD"),
        preprocessing_note=config.get("preprocessing_note", "TBD"),
    )


def _frontend_config(config: dict) -> FrontendConfig:
    kernel = KernelSpec(**config["kernel"])
    detector = DetectorConfig(**config["detector"])
    return FrontendConfig(mode=config["mode"], kernel=kernel, detector=detector)


def validate_dataset(path: str, check_files: bool, output: str | None) -> None:
    config = load_yaml(path)
    bundle = DatasetRegistry.load(_dataset_spec(config), check_files=check_files)
    result = {
        "dataset_id": bundle.spec.dataset_id,
        "rows": len(bundle.rows),
        "split_counts": bundle.split_counts,
        "label_counts": bundle.label_counts,
        "manifest_hash": bundle.manifest_hash,
        "isolation": "passed",
    }
    if output:
        write_run_manifest(Path(output), config, {"dataset_validation": result})
    print(json.dumps(result, indent=2, ensure_ascii=False))


def smoke_test(path: str, output: str | None) -> None:
    config = load_yaml(path)
    frontend = OpticalBackbone(_frontend_config(load_yaml(config["frontend_config"])))
    backend = LightweightBackend(len(frontend.channel_names), **config["backend"])
    image = torch.rand(config["batch_size"], 3, config["image_size"], config["image_size"])
    with torch.no_grad():
        features = frontend(image)
        logits = backend(features)
    result = {
        "feature_shape": list(features.shape),
        "logit_shape": list(logits.shape),
        "kernel_bank_hash": frontend.kernel_bank_hash,
        "backend_parameters": backend.parameter_count,
    }
    if output:
        write_run_manifest(Path(output), config, {"smoke_test": result})
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="cam16-wavelet")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-dataset")
    validate.add_argument("--config", required=True)
    validate.add_argument("--check-files", action="store_true")
    validate.add_argument("--output")
    smoke = subparsers.add_parser("smoke-test")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--output")
    args = parser.parse_args()
    if args.command == "validate-dataset":
        validate_dataset(args.config, args.check_files, args.output)
    else:
        smoke_test(args.config, args.output)


if __name__ == "__main__":
    main()
