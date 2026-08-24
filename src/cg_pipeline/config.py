"""Strict configuration for exploratory and formal CAM16 training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on supported Python 3.10
    import tomli as tomllib


class ConfigError(ValueError):
    """Configuration is incomplete, unknown, illegal, or not canonical."""


_SECTION_KEYS: dict[str, set[str]] = {
    "execution": {"kind", "run_id", "device", "output_root", "max_steps", "allow_test"},
    "data": {
        "contract_id",
        "train_manifest_relpath",
        "validation_manifest_relpath",
        "identity_level",
        "identity_column",
        "sample_id_column",
        "path_column",
        "split_column",
        "label_column",
        "label_name_column",
        "allowed_splits",
        "class_names",
        "image_shape",
        "image_dtype",
        "channel_order",
        "patient_mapping_evidence",
    },
    "model": {
        "contract_id",
        "input_height",
        "input_width",
        "stain_basis",
        "morlet_scales",
        "morlet_orientations",
        "morlet_support",
        "sigma0",
        "xi0",
        "gamma",
        "frontend_backend",
        "pyramid_levels",
        "classifier",
        "precision_policy",
    },
    "training": {
        "loss",
        "loss_precision",
        "optimizer",
        "optimizer_state_precision",
        "learning_rate",
        "beta1",
        "beta2",
        "epsilon",
        "weight_decay",
        "scheduler",
        "batch_size",
        "max_epochs",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "checkpoint_metric",
        "checkpoint_tie_break",
        "checkpoint_save",
        "class_imbalance",
        "gradient_clip",
        "seeds",
        "sampler",
        "num_workers",
        "failed_run_policy",
        "multi_seed_aggregation",
        "output_layout",
        "resume_policy",
    },
    "evaluation": {
        "contract_id",
        "positive_class",
        "primary_metric",
        "secondary_metrics",
        "score_transform",
        "ranking_key",
        "slide_aggregation",
        "threshold_selection",
        "checkpoint_selection",
        "test_access",
        "confidence_interval",
        "invalid_run_policy",
        "result_schema",
    },
    "determinism": {
        "python_seed",
        "numpy_seed",
        "torch_cpu_seed",
        "torch_cuda_seed",
        "deterministic_algorithms",
        "warn_only",
        "cudnn_benchmark",
        "amp",
        "tf32",
        "float16",
        "bfloat16",
        "gradient_scaling",
    },
}

_EXACT_VALUES: dict[tuple[str, str], Any] = {
    ("data", "contract_id"): "cam16-existing-patch-v1",
    ("data", "identity_level"): "slide_id",
    ("data", "identity_column"): "slide_id",
    ("data", "sample_id_column"): "patch_id",
    ("data", "path_column"): "patch_path",
    ("data", "split_column"): "split",
    ("data", "label_column"): "label",
    ("data", "label_name_column"): "label_name",
    ("data", "allowed_splits"): ["train", "val", "test"],
    ("data", "class_names"): ["normal", "tumor"],
    ("data", "image_shape"): [256, 256, 3],
    ("data", "image_dtype"): "uint8",
    ("data", "channel_order"): "RGB",
    ("model", "contract_id"): "fixed-he-morlet-linear-v1",
    ("model", "input_height"): 256,
    ("model", "input_width"): 256,
    ("model", "morlet_scales"): 4,
    ("model", "morlet_orientations"): 8,
    ("model", "morlet_support"): 105,
    ("model", "sigma0"): "0.8",
    ("model", "xi0"): "3*pi/4",
    ("model", "gamma"): "0.5",
    ("model", "frontend_backend"): "fft",
    ("model", "pyramid_levels"): [1, 2, 4],
    ("model", "classifier"): "linear-logit-v1",
    ("model", "precision_policy"): "protected-float32-v1",
    ("training", "loss"): "bce-with-logits",
    ("training", "loss_precision"): "float32",
    ("training", "optimizer"): "adamw",
    ("training", "optimizer_state_precision"): "float32",
    ("training", "learning_rate"): "0.001",
    ("training", "beta1"): "0.9",
    ("training", "beta2"): "0.999",
    ("training", "epsilon"): "0.00000001",
    ("training", "weight_decay"): "0.0001",
    ("training", "scheduler"): "none",
    ("training", "batch_size"): 16,
    ("training", "max_epochs"): 20,
    ("training", "early_stopping_patience"): 5,
    ("training", "early_stopping_min_delta"): "0",
    ("training", "checkpoint_metric"): "val_slide_auroc",
    ("training", "checkpoint_tie_break"): "earliest_epoch",
    ("training", "class_imbalance"): "uniform-unweighted",
    ("training", "gradient_clip"): "none",
    ("training", "seeds"): [1729, 3407, 7919],
    ("training", "sampler"): "hash-order-once-per-epoch",
    ("training", "num_workers"): 8,
    ("training", "checkpoint_save"): "every-complete-epoch-immutable",
    ("training", "failed_run_policy"): "exclude-and-report-no-auto-retry",
    ("training", "multi_seed_aggregation"): "mean-sd-and-individual",
    ("training", "output_layout"): "run-seed-epoch-v1",
    ("training", "resume_policy"): "strict-latest-complete-epoch",
    ("evaluation", "contract_id"): "cam16-eval-v1",
    ("evaluation", "positive_class"): "tumor",
    ("evaluation", "primary_metric"): "slide_auroc",
    (
        "evaluation",
        "secondary_metrics",
    ): ["patch_auroc", "sensitivity", "specificity", "balanced_accuracy", "f1", "brier", "ece10"],
    ("evaluation", "score_transform"): "binary64-sigmoid-for-reporting",
    ("evaluation", "ranking_key"): "raw_float32_logit",
    ("evaluation", "slide_aggregation"): "manifest-bounded-max-logit",
    ("evaluation", "threshold_selection"): "validation-distinct-logits-youden-largest-tie",
    ("evaluation", "checkpoint_selection"): "validation-slide-auroc-earliest-tie",
    ("evaluation", "test_access"): "final-once-authorization-required",
    ("evaluation", "confidence_interval"): "stratified-slide-bootstrap-percentile-2000",
    ("evaluation", "invalid_run_policy"): "exclude-and-report",
    ("evaluation", "result_schema"): "cam16-result-v1",
}

_EXECUTION_PROFILES: dict[str, dict[str, Any]] = {
    "exploratory_train": {
        "run_id": "exploratory-default",
        "device": "cuda:0",
        "output_root": "artifacts/exploratory_runs/exploratory-default",
        "max_steps": 0,
        "allow_test": False,
    },
    "formal_train": {
        "device": "cuda:0",
        "max_steps": 0,
        "allow_test": False,
    },
}

_EXPLORATORY_UNLOCKED_EXACT_FIELDS = {
    ("training", "batch_size"),
    ("training", "max_epochs"),
    ("training", "seeds"),
    ("training", "num_workers"),
}

_EXPLORATORY_OVERRIDE_TARGETS = {
    "device": ("execution", "device"),
    "seed": ("training", "seeds"),
    "batch_size": ("training", "batch_size"),
    "num_workers": ("training", "num_workers"),
    "max_epochs": ("training", "max_epochs"),
    "max_steps": ("execution", "max_steps"),
    "output": ("execution", "output_root"),
    "run_id": ("execution", "run_id"),
}

_BOOL_FIELDS = set(_SECTION_KEYS["determinism"]) | {"allow_test"}
_INT_FIELDS = {
    "max_steps",
    "input_height",
    "input_width",
    "morlet_scales",
    "morlet_orientations",
    "morlet_support",
    "batch_size",
    "max_epochs",
    "early_stopping_patience",
    "num_workers",
}
_LIST_FIELDS = {
    "allowed_splits",
    "class_names",
    "image_shape",
    "stain_basis",
    "pyramid_levels",
    "seeds",
    "secondary_metrics",
}


def _reject_floats_and_tbd(value: Any, path: str = "config") -> None:
    if isinstance(value, float):
        raise ConfigError(f"floating TOML value at {path}; use the required canonical string")
    if isinstance(value, str) and value.strip().upper() == "TBD":
        raise ConfigError(f"unresolved TBD at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_floats_and_tbd(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_floats_and_tbd(item, f"{path}[{index}]")


def _validate_shape(document: dict[str, Any]) -> None:
    expected_top = {"schema_version", *_SECTION_KEYS}
    missing_top = expected_top.difference(document)
    unknown_top = set(document).difference(expected_top)
    if missing_top:
        raise ConfigError(f"missing top-level fields: {sorted(missing_top)}")
    if unknown_top:
        raise ConfigError(f"unknown top-level fields: {sorted(unknown_top)}")
    if document["schema_version"] != "phase0-experiment-config-v2":
        raise ConfigError("unsupported schema_version")
    for section, expected in _SECTION_KEYS.items():
        value = document[section]
        if not isinstance(value, dict):
            raise ConfigError(f"section {section} must be a table")
        missing = expected.difference(value)
        unknown = set(value).difference(expected)
        if missing:
            raise ConfigError(f"missing fields in {section}: {sorted(missing)}")
        if unknown:
            raise ConfigError(f"unknown fields in {section}: {sorted(unknown)}")
        for key, item in value.items():
            if key in _BOOL_FIELDS and not isinstance(item, bool):
                raise ConfigError(f"{section}.{key} must be Boolean")
            if key in _INT_FIELDS and (not isinstance(item, int) or isinstance(item, bool)):
                raise ConfigError(f"{section}.{key} must be an integer")
            if key in _LIST_FIELDS and not isinstance(item, list):
                raise ConfigError(f"{section}.{key} must be an array")


def _validate_semantics(document: dict[str, Any]) -> None:
    execution = document["execution"]
    if execution["kind"] not in {"exploratory_train", "formal_train"}:
        raise ConfigError("execution.kind must be exploratory_train or formal_train")
    if execution["allow_test"] is not False:
        raise ConfigError("test access must be false in every training configuration")
    profile = _EXECUTION_PROFILES[execution["kind"]]
    locked_execution_fields = ("allow_test",)
    if execution["kind"] == "formal_train":
        locked_execution_fields = ("device", "max_steps", "allow_test")
    for key in locked_execution_fields:
        if execution[key] != profile[key]:
            raise ConfigError(
                f"execution.{key} conflicts with the locked {execution['kind']} profile"
            )
    manifest_paths = tuple(
        PurePosixPath(str(document["data"][key]))
        for key in ("train_manifest_relpath", "validation_manifest_relpath")
    )
    if any(
        path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts)
        for path in manifest_paths
    ):
        raise ConfigError("data manifest paths must be normalized relative paths")
    if manifest_paths[0] == manifest_paths[1]:
        raise ConfigError("train and validation manifest paths must be distinct")
    mapping_evidence = document["data"]["patient_mapping_evidence"]
    if mapping_evidence != "not_available":
        raise ConfigError("patient mapping evidence must remain not_available for CAM16 Phase 1")
    for (section, key), expected in _EXACT_VALUES.items():
        if (
            execution["kind"] == "exploratory_train"
            and (
                section,
                key,
            )
            in _EXPLORATORY_UNLOCKED_EXACT_FIELDS
        ):
            continue
        if document[section][key] != expected:
            raise ConfigError(f"{section}.{key} conflicts with the locked contract")
    if execution["kind"] == "formal_train":
        run_id = execution["run_id"]
        if not isinstance(run_id, str) or not run_id or run_id != run_id.strip():
            raise ConfigError("formal execution.run_id must be nonempty without outer whitespace")
        if run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
            raise ConfigError("formal execution.run_id must be a single safe path component")
        expected_output_root = f"artifacts/formal_runs/{run_id}"
        if execution["output_root"] != expected_output_root:
            raise ConfigError(
                "formal execution.output_root must exactly match execution.run_id under "
                "artifacts/formal_runs"
            )
    else:
        if not execution["run_id"] or execution["run_id"] != execution["run_id"].strip():
            raise ConfigError(
                "exploratory execution.run_id must be nonempty without outer whitespace"
            )
        device = execution["device"]
        valid_device = isinstance(device, str) and (
            device == "cpu"
            or device == "cuda"
            or (device.startswith("cuda:") and device[5:].isdigit())
        )
        if not valid_device:
            raise ConfigError("exploratory execution.device must be cpu, cuda, or cuda:<index>")
        output_root = PurePosixPath(str(execution["output_root"]))
        if (
            output_root.is_absolute()
            or output_root.parts[:2] != ("artifacts", "exploratory_runs")
            or any(part in {"", ".", ".."} for part in output_root.parts)
        ):
            raise ConfigError(
                "exploratory execution.output_root must be under artifacts/exploratory_runs"
            )
        seeds = document["training"]["seeds"]
        if (
            len(seeds) != 1
            or not isinstance(seeds[0], int)
            or isinstance(seeds[0], bool)
            or seeds[0] < 0
        ):
            raise ConfigError("exploratory training requires exactly one nonnegative integer seed")
        if document["training"]["batch_size"] < 1:
            raise ConfigError("exploratory training.batch_size must be positive")
        if document["training"]["num_workers"] < 0:
            raise ConfigError("exploratory training.num_workers must be nonnegative")
        if document["training"]["max_epochs"] < 1:
            raise ConfigError("exploratory training.max_epochs must be positive")
        if execution["max_steps"] < 0:
            raise ConfigError("exploratory execution.max_steps must be nonnegative")
    expected_basis = [
        "0.644211",
        "0.716556",
        "0.266844",
        "0.092789",
        "0.954111",
        "0.283111",
    ]
    if document["model"]["stain_basis"] != expected_basis:
        raise ConfigError("model.stain_basis conflicts with the locked H/E basis")
    determinism = document["determinism"]
    required_true = {
        "python_seed",
        "numpy_seed",
        "torch_cpu_seed",
        "torch_cuda_seed",
        "deterministic_algorithms",
    }
    required_false = {
        "warn_only",
        "cudnn_benchmark",
        "amp",
        "tf32",
        "float16",
        "bfloat16",
        "gradient_scaling",
    }
    if any(determinism[key] is not True for key in required_true) or any(
        determinism[key] is not False for key in required_false
    ):
        raise ConfigError("determinism and precision guards must match protected-float32-v1")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable validated training configuration."""

    _document: Mapping[str, Any]
    source: Path

    @property
    def schema_version(self) -> str:
        return str(self._document["schema_version"])

    @property
    def execution_kind(self) -> str:
        return str(self._document["execution"]["kind"])

    @property
    def execution(self) -> Mapping[str, Any]:
        return self._document["execution"]

    @property
    def data(self) -> Mapping[str, Any]:
        return self._document["data"]

    @property
    def model(self) -> Mapping[str, Any]:
        return self._document["model"]

    @property
    def training(self) -> Mapping[str, Any]:
        return self._document["training"]

    @property
    def evaluation(self) -> Mapping[str, Any]:
        return self._document["evaluation"]

    @property
    def determinism(self) -> Mapping[str, Any]:
        return self._document["determinism"]

    def as_dict(self) -> dict[str, Any]:
        return _thaw(self._document)


def load_experiment_config(
    path: Path | str,
    *,
    exploratory_overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    source = Path(path)
    try:
        document = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot load configuration: {error}") from error
    overrides = dict(exploratory_overrides or {})
    unknown_overrides = set(overrides).difference(_EXPLORATORY_OVERRIDE_TARGETS)
    if unknown_overrides:
        raise ConfigError(f"unknown exploratory overrides: {sorted(unknown_overrides)}")
    if overrides:
        if document.get("execution", {}).get("kind") != "exploratory_train":
            raise ConfigError("exploratory overrides require execution.kind=exploratory_train")
        for name, value in overrides.items():
            if name in {"seed", "batch_size", "num_workers", "max_epochs", "max_steps"} and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise ConfigError(f"exploratory override {name} must be an integer")
            if name in {"device", "output", "run_id"} and not isinstance(value, str):
                raise ConfigError(f"exploratory override {name} must be a string")
            section, key = _EXPLORATORY_OVERRIDE_TARGETS[name]
            document[section][key] = [value] if name == "seed" else value
    _reject_floats_and_tbd(document)
    _validate_shape(document)
    _validate_semantics(document)
    return ExperimentConfig(_freeze(document), source.resolve())
