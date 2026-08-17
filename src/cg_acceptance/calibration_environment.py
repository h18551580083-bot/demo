"""Device-environment checks for the Decision 30 calibration gate."""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import torch

from .fixture import CalibrationFixture, CalibrationMode, fixture_input_hashes


@contextmanager
def protected_device_environment(device: torch.device) -> Iterator[None]:
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


def _nvidia_identity() -> dict[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=driver_version,uuid,name",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {"driver_version": "unavailable", "gpu_uuid": "unavailable"}
    fields = [field.strip() for field in result.stdout.splitlines()[0].split(",", maxsplit=2)]
    return {"driver_version": fields[0], "gpu_uuid": fields[1]}


def device_identity(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    nvidia = _nvidia_identity()
    zero32 = torch.tensor(0.0, dtype=torch.float32, device=device)
    one32 = torch.tensor(1.0, dtype=torch.float32, device=device)
    subnormal32 = torch.nextafter(zero32, one32)
    zero64 = torch.tensor(0.0, dtype=torch.float64, device=device)
    one64 = torch.tensor(1.0, dtype=torch.float64, device=device)
    subnormal64 = torch.nextafter(zero64, one64)
    uname = platform.uname()
    return {
        "device_type": device.type,
        "device_index": device.index,
        "gpu_name": properties.name,
        "device_name": properties.name,
        "gpu_uuid": nvidia["gpu_uuid"],
        "driver_version": nvidia["driver_version"],
        "compute_capability": f"{properties.major}.{properties.minor}",
        "gpu_total_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cpu_identity": platform.processor(),
        "cpu_logical_count": __import__("os").cpu_count(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "system": uname.system,
        "system_release": uname.release,
        "system_version": uname.version,
        "machine": uname.machine,
        "node": uname.node,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "autocast_enabled": torch.is_autocast_enabled(),
        "gradient_scaling_enabled": False,
        "default_dtype": str(torch.get_default_dtype()),
        "rounding_policy": "ieee754_round_to_nearest_ties_to_even",
        "float32_subnormal_preserved": bool((subnormal32 * one32 == subnormal32).item()),
        "float64_subnormal_preserved": bool((subnormal64 * one64 == subnormal64).item()),
    }


def environment_pass(identity: Mapping[str, object]) -> bool:
    return (
        identity["device_type"] != "cpu"
        and identity["deterministic_algorithms"] is True
        and identity["matmul_allow_tf32"] is False
        and identity["cudnn_allow_tf32"] is False
        and identity["autocast_enabled"] is False
        and identity["float32_subnormal_preserved"] is True
        and identity["float64_subnormal_preserved"] is True
    )


def _required_formal_environment_keys() -> set[str]:
    return {
        "gpu_name",
        "gpu_uuid",
        "driver_version",
        "torch_version",
        "cuda_runtime",
        "cudnn_version",
        "cpu_identity",
        "python_version",
        "system",
    }


def validate_scope(
    mode: CalibrationMode,
    fixture: CalibrationFixture,
    expected_environment: Mapping[str, object] | None,
) -> None:
    if mode is CalibrationMode.LOCAL_SMOKE:
        return
    if not fixture.is_preregistered:
        raise ValueError("formal_acceptance requires a pre-registered fixture")
    missing = _required_formal_environment_keys() - set(expected_environment or {})
    if missing:
        raise ValueError(
            "formal_acceptance expected_environment is missing: " + ", ".join(sorted(missing))
        )
    actual_hashes = fixture_input_hashes(fixture)
    if dict(fixture.registered_input_hashes or {}) != actual_hashes:
        raise ValueError("pre-registered fixture input hash mismatch")
    if fixture.registered_formal_shape != tuple(fixture.z.shape):
        raise ValueError("pre-registered fixture formal shape mismatch")


def validate_expected_environment(
    mode: CalibrationMode,
    expected: Mapping[str, object] | None,
    actual: Mapping[str, object],
) -> None:
    if mode is CalibrationMode.LOCAL_SMOKE:
        return
    mismatches = {
        key: (expected[key], actual.get(key))
        for key in _required_formal_environment_keys()
        if expected is not None and expected[key] != actual.get(key)
    }
    if mismatches:
        raise ValueError(f"formal_acceptance environment identity mismatch: {mismatches}")
