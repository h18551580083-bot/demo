"""Protected deterministic training-step and checkpoint contracts."""

from __future__ import annotations

import hashlib
import math
import os
import random
import struct
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .model import FixedHEClassifier


class TrainingContractError(ValueError):
    """Training ownership, precision, determinism, or identity contract failed."""


@dataclass(frozen=True)
class TrainingStepReport:
    loss: float
    changed_backend_parameters: tuple[str, ...]
    fixed_frontend_unchanged: bool
    optimizer_state_precision: str


def _hash_state_value(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        header = f"tensor|{tensor.dtype}|{tuple(tensor.shape)}".encode("ascii")
        payload = tensor.numpy().tobytes(order="C")
        digest.update(len(header).to_bytes(8, "big") + header)
        digest.update(len(payload).to_bytes(8, "big") + payload)
    elif isinstance(value, dict):
        digest.update(b"dict" + len(value).to_bytes(8, "big"))
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _hash_state_value(digest, key)
            _hash_state_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii") + len(value).to_bytes(8, "big"))
        for item in value:
            _hash_state_value(digest, item)
    elif isinstance(value, bool):
        digest.update(b"bool:1" if value else b"bool:0")
    elif isinstance(value, int):
        digest.update(b"int:" + str(value).encode("ascii") + b";")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise TrainingContractError("optimizer state contains a non-finite scalar")
        digest.update(b"float64:" + struct.pack(">d", value))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"str:" + len(encoded).to_bytes(8, "big") + encoded)
    elif value is None:
        digest.update(b"none")
    else:
        raise TrainingContractError(f"unsupported checkpoint state value: {type(value).__name__}")


def model_state_identity(model: FixedHEClassifier) -> str:
    digest = hashlib.sha256(b"cg/model-state/v1\x00")
    _hash_state_value(digest, model.state_dict())
    return "sha256:" + digest.hexdigest()


def optimizer_state_identity(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256(b"cg/optimizer-state/v1\x00")
    _hash_state_value(digest, optimizer.state_dict())
    return "sha256:" + digest.hexdigest()


def configure_determinism(seed: int) -> dict[str, Any]:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed >= 2**32:
        raise TrainingContractError("seed must be an unsigned 32-bit integer")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "python_seed": seed,
        "numpy_seed": seed,
        "torch_cpu_seed": seed,
        "torch_cuda_seed": seed,
        "cuda_seed_applied": bool(torch.cuda.is_available()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def worker_seed(base_seed: int, epoch: int, worker_id: int) -> int:
    if min(base_seed, epoch, worker_id) < 0:
        raise TrainingContractError("worker seed inputs must be nonnegative")
    material = f"cg/worker-seed/v1\x00{base_seed}\x00{epoch}\x00{worker_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def hash_epoch_order(identifiers: tuple[str, ...], *, seed: int, epoch: int) -> tuple[str, ...]:
    if len(identifiers) != len(set(identifiers)):
        raise TrainingContractError("epoch order requires unique patch identifiers")
    if epoch < 0 or epoch >= 2**64:
        raise TrainingContractError("epoch must be an unsigned 64-bit integer")
    seed_bytes = str(seed).encode("ascii")

    def key(identifier: str) -> tuple[bytes, bytes]:
        identifier_bytes = identifier.encode("utf-8")
        preimage = (
            b"cg/cam16-train-order/v1\x00"
            + struct.pack(">Q", len(seed_bytes))
            + seed_bytes
            + struct.pack(">Q", epoch)
            + struct.pack(">Q", len(identifier_bytes))
            + identifier_bytes
        )
        return hashlib.sha256(preimage).digest(), identifier_bytes

    return tuple(sorted(identifiers, key=key))


def _decimal_float(value: str, name: str) -> float:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise TrainingContractError(f"{name} is not a canonical decimal") from error
    if not parsed.is_finite():
        raise TrainingContractError(f"{name} must be finite")
    result = float(parsed)
    if not np.isfinite(result):
        raise TrainingContractError(f"{name} is outside binary64 range")
    return result


def build_adamw(
    model: FixedHEClassifier,
    *,
    learning_rate: str,
    beta1: str,
    beta2: str,
    epsilon: str,
    weight_decay: str,
) -> torch.optim.AdamW:
    lr = _decimal_float(learning_rate, "learning_rate")
    b1 = _decimal_float(beta1, "beta1")
    b2 = _decimal_float(beta2, "beta2")
    eps = _decimal_float(epsilon, "epsilon")
    decay = _decimal_float(weight_decay, "weight_decay")
    if lr <= 0.0 or not (0.0 <= b1 < 1.0) or not (0.0 <= b2 < 1.0):
        raise TrainingContractError("AdamW learning rate and beta values are illegal")
    if eps <= 0.0 or decay < 0.0:
        raise TrainingContractError("AdamW epsilon and weight decay are illegal")
    optimizer = torch.optim.AdamW(
        list(model.electronic_parameters()),
        lr=lr,
        betas=(b1, b2),
        eps=eps,
        weight_decay=decay,
        amsgrad=False,
        maximize=False,
        foreach=False,
        fused=False,
    )
    audit_optimizer_ownership(model, optimizer)
    return optimizer


def audit_optimizer_ownership(
    model: FixedHEClassifier, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    electronic = list(model.electronic_parameters())
    optical = list(model.frontend.parameters())
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    counts = Counter(id(parameter) for parameter in optimizer_parameters)
    if any(counts[id(parameter)] != 1 for parameter in electronic) or len(counts) != len(electronic):
        raise TrainingContractError("every electronic parameter must appear exactly once")
    optical_in_optimizer = any(id(parameter) in counts for parameter in optical)
    if optical_in_optimizer:
        raise TrainingContractError("an optical parameter appears in the optimizer")
    if any(parameter.dtype != torch.float32 for parameter in electronic):
        raise TrainingContractError("every electronic parameter must be float32")
    return {
        "electronic_parameter_count": sum(parameter.numel() for parameter in electronic),
        "electronic_tensor_count": len(electronic),
        "optimizer_unique_tensor_count": len(counts),
        "optical_parameter_count": sum(parameter.numel() for parameter in optical),
        "all_electronic_exactly_once": True,
        "optical_in_optimizer": False,
    }


def _optimizer_precision(optimizer: torch.optim.Optimizer) -> str:
    floating_states = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    ]
    if not floating_states or any(value.dtype != torch.float32 for value in floating_states):
        raise TrainingContractError("optimizer state precision is not uniformly float32")
    return "float32"


def train_one_step(
    model: FixedHEClassifier,
    optimizer: torch.optim.Optimizer,
    rgb: torch.Tensor,
    targets: torch.Tensor,
) -> TrainingStepReport:
    audit_optimizer_ownership(model, optimizer)
    if targets.dtype != torch.float32 or targets.ndim != 1:
        raise TrainingContractError("binary targets must be float32 [B]")
    fixed_before = model.frontend.fixed_state_identity()
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=rgb.device.type, enabled=False):
        logits = model(rgb).logits
        if logits.shape != targets.shape:
            raise TrainingContractError("logit and target shapes differ")
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    if loss.dtype != torch.float32 or not torch.isfinite(loss):
        raise TrainingContractError("loss must be finite float32")
    loss.backward()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            raise TrainingContractError(f"missing electronic gradient: {name}")
        if parameter.grad.dtype != torch.float32 or not torch.isfinite(parameter.grad).all():
            raise TrainingContractError(f"invalid electronic gradient: {name}")
    optimizer.step()
    fixed_unchanged = model.frontend.fixed_state_identity() == fixed_before
    if not fixed_unchanged:
        raise TrainingContractError("fixed frontend changed during optimizer step")
    changed = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not torch.equal(before[name], parameter.detach())
    )
    if not changed:
        raise TrainingContractError("optimizer step changed no electronic parameter")
    return TrainingStepReport(
        loss=float(loss.detach().cpu()),
        changed_backend_parameters=changed,
        fixed_frontend_unchanged=True,
        optimizer_state_precision=_optimizer_precision(optimizer),
    )


def save_checkpoint(
    path: Path | str,
    model: FixedHEClassifier,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
) -> None:
    if metadata.get("checkpoint_identity") != model_state_identity(model):
        raise TrainingContractError("checkpoint metadata does not bind the model state identity")
    if metadata.get("optimizer_state_identity") != optimizer_state_identity(optimizer):
        raise TrainingContractError("checkpoint metadata does not bind the optimizer state identity")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "cg-checkpoint-v1",
        "metadata": metadata,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    with target.open("xb") as handle:
        torch.save(payload, handle)


def load_checkpoint(
    path: Path | str,
    model: FixedHEClassifier,
    optimizer: torch.optim.Optimizer,
    *,
    expected_metadata: dict[str, Any],
) -> dict[str, Any]:
    expected_fixed_identity = model.frontend.fixed_state_identity()
    try:
        payload = torch.load(Path(path), map_location="cpu")
    except (OSError, RuntimeError) as error:
        raise TrainingContractError(f"cannot load checkpoint: {error}") from error
    if not isinstance(payload, dict) or payload.get("format") != "cg-checkpoint-v1":
        raise TrainingContractError("unknown checkpoint format")
    metadata = payload.get("metadata")
    if metadata != expected_metadata:
        raise TrainingContractError("checkpoint metadata identity mismatch")
    try:
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
    except (KeyError, RuntimeError, ValueError) as error:
        raise TrainingContractError(f"checkpoint state is invalid: {error}") from error
    if model.frontend.fixed_state_identity() != expected_fixed_identity:
        raise TrainingContractError("checkpoint changed the canonical fixed frontend identity")
    audit_optimizer_ownership(model, optimizer)
    if metadata.get("checkpoint_identity") != model_state_identity(model):
        raise TrainingContractError("checkpoint model state identity mismatch")
    if metadata.get("optimizer_state_identity") != optimizer_state_identity(optimizer):
        raise TrainingContractError("checkpoint optimizer state identity mismatch")
    return dict(metadata)


def aggregate_seed_results(results: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    valid = [item for item in results if item.get("status") == "complete"]
    failed = [item for item in results if item.get("status") != "complete"]
    if not valid:
        raise TrainingContractError("no valid seed run is available for aggregation")
    individual = [
        {"seed": int(item["seed"]), "value": float(item["best_validation_slide_auroc"])}
        for item in valid
    ]
    values = [item["value"] for item in individual]
    mean = math.fsum(values) / len(values)
    sample_standard_deviation = (
        None
        if len(values) == 1
        else math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))
    )
    return {
        "method": "mean-sd-and-individual",
        "valid_seed_count": len(valid),
        "failed_seed_count": len(failed),
        "mean": mean,
        "sample_standard_deviation": sample_standard_deviation,
        "individual": individual,
        "failed_seeds": [
            {"seed": int(item["seed"]), "failure_reason": str(item.get("failure_reason", "unknown"))}
            for item in failed
        ],
    }
