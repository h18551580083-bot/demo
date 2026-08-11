from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cg_pipeline.config import ConfigError, load_experiment_config
from cg_pipeline.identity import canonical_json_bytes, domain_hash


def _document() -> str:
    return """
schema_version = "phase0-experiment-config-v1"

[execution]
kind = "exploratory_train"
run_id = "exploratory-default"
device = "cuda:0"
output_root = "artifacts/exploratory_runs/exploratory-default"
max_steps = 0
allow_test = false

[data]
contract_id = "cam16-existing-patch-v1"
manifest_relpath = "cam16_class_quota/metadata/training_manifest.csv"
identity_level = "slide_id"
identity_column = "slide_id"
sample_id_column = "patch_id"
path_column = "patch_path"
split_column = "split"
label_column = "label"
label_name_column = "label_name"
allowed_splits = ["train", "val", "test"]
class_names = ["normal", "tumor"]
image_shape = [256, 256, 3]
image_dtype = "uint8"
channel_order = "RGB"
patient_mapping_evidence = "not_available"

[model]
contract_id = "fixed-he-morlet-linear-v1"
input_height = 256
input_width = 256
stain_basis = ["0.644211", "0.716556", "0.266844", "0.092789", "0.954111", "0.283111"]
morlet_scales = 4
morlet_orientations = 8
morlet_support = 105
sigma0 = "0.8"
xi0 = "3*pi/4"
gamma = "0.5"
frontend_backend = "fft"
pyramid_levels = [1, 2, 4]
classifier = "linear-logit-v1"
precision_policy = "protected-float32-v1"

[training]
loss = "bce-with-logits"
loss_precision = "float32"
optimizer = "adamw"
optimizer_state_precision = "float32"
learning_rate = "0.001"
beta1 = "0.9"
beta2 = "0.999"
epsilon = "0.00000001"
weight_decay = "0.0001"
scheduler = "none"
batch_size = 32
max_epochs = 1
early_stopping_patience = 5
early_stopping_min_delta = "0"
checkpoint_metric = "val_slide_auroc"
checkpoint_tie_break = "earliest_epoch"
checkpoint_save = "every-complete-epoch-immutable"
class_imbalance = "uniform-unweighted"
gradient_clip = "none"
seeds = [1729]
sampler = "hash-order-once-per-epoch"
num_workers = 8
failed_run_policy = "exclude-and-report-no-auto-retry"
multi_seed_aggregation = "mean-sd-and-individual"
output_layout = "run-seed-epoch-v1"
resume_policy = "strict-latest-complete-epoch"

[evaluation]
contract_id = "cam16-eval-v1"
positive_class = "tumor"
primary_metric = "slide_auroc"
secondary_metrics = ["patch_auroc", "sensitivity", "specificity", "balanced_accuracy", "f1", "brier", "ece10"]
score_transform = "binary64-sigmoid-for-reporting"
ranking_key = "raw_float32_logit"
slide_aggregation = "manifest-bounded-max-logit"
threshold_selection = "validation-distinct-logits-youden-largest-tie"
checkpoint_selection = "validation-slide-auroc-earliest-tie"
test_access = "final-once-authorization-required"
confidence_interval = "stratified-slide-bootstrap-percentile-2000"
invalid_run_policy = "exclude-and-report"
result_schema = "cam16-result-v1"

[determinism]
python_seed = true
numpy_seed = true
torch_cpu_seed = true
torch_cuda_seed = true
deterministic_algorithms = true
warn_only = false
cudnn_benchmark = false
amp = false
tf32 = false
float16 = false
bfloat16 = false
gradient_scaling = false
""".strip()


def test_config_is_strict_normalized_and_hashed(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(_document(), encoding="utf-8")

    config = load_experiment_config(path)

    assert config.schema_version == "phase0-experiment-config-v1"
    assert config.execution_kind == "exploratory_train"
    assert config.data["identity_level"] == "slide_id"
    assert config.training["batch_size"] == 32
    assert config.training["num_workers"] == 8
    assert config.training["seeds"] == (1729,)
    assert config.normalized_bytes == canonical_json_bytes(config.as_dict())
    assert config.sha256 == "sha256:" + hashlib.sha256(config.normalized_bytes).hexdigest()


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        (lambda text: text.replace("allow_test = false", "allow_test = false\nunknown = 1"), "unknown"),
        (lambda text: text.replace('run_id = "exploratory-default"\n', ""), "missing"),
        (lambda text: text.replace('gradient_clip = "none"', 'gradient_clip = "TBD"'), "TBD"),
        (lambda text: text.replace('learning_rate = "0.001"', "learning_rate = 0.001"), "floating"),
        (lambda text: text.replace("allow_test = false", "allow_test = true"), "test access"),
        (
            lambda text: text.replace(
                'patient_mapping_evidence = "not_available"',
                'patient_mapping_evidence = "sha256:' + "0" * 64 + '"',
            ),
            "must remain not_available",
        ),
    ],
)
def test_config_rejects_unknown_missing_tbd_float_and_test_access(
    tmp_path: Path,
    changed,
    message: str,
) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(changed(_document()), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_experiment_config(path)


def test_canonical_json_rejects_non_jcs_values_and_preserves_unicode() -> None:
    assert canonical_json_bytes({"z": "e\u0301", "a": [True, None, 2]}) == (
        b'{"a":[true,null,2],"z":"e\xcc\x81"}'
    )
    with pytest.raises(TypeError, match="floating"):
        canonical_json_bytes({"value": 0.5})
    with pytest.raises(ValueError, match="safe integer"):
        canonical_json_bytes({"value": 2**53})


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('learning_rate = "0.001"', 'learning_rate = "0.002"'),
        ('checkpoint_metric = "val_slide_auroc"', 'checkpoint_metric = "patch_auroc"'),
        (
            'confidence_interval = "stratified-slide-bootstrap-percentile-2000"',
            'confidence_interval = "none"',
        ),
    ],
)
def test_locked_scientific_fields_cannot_be_mutated(tmp_path: Path, old: str, new: str) -> None:
    path = tmp_path / "mutated.toml"
    path.write_text(_document().replace(old, new), encoding="utf-8")

    with pytest.raises(ConfigError, match="locked contract"):
        load_experiment_config(path)


def test_domain_hash_separates_domains_and_payloads() -> None:
    header = {"payload_length": 3, "shape": [3]}
    first = domain_hash("cg/test-a/v1", header, b"abc")
    second = domain_hash("cg/test-b/v1", header, b"abc")
    third = domain_hash("cg/test-a/v1", header, b"abd")

    assert first.startswith("sha256:") and len(first) == 71
    assert len({first, second, third}) == 3


def test_exploratory_engineering_overrides_are_typed_and_hashed(tmp_path: Path) -> None:
    path = tmp_path / "exploratory.toml"
    path.write_text(_document(), encoding="utf-8")

    config = load_experiment_config(
        path,
        exploratory_overrides={
            "device": "cpu",
            "seed": 41,
            "output": "artifacts/exploratory_runs/profile-41",
            "run_id": "profile-41",
            "batch_size": 8,
            "num_workers": 2,
            "max_epochs": 3,
            "max_steps": 11,
        },
    )

    assert config.training["seeds"] == (41,)
    assert config.execution["device"] == "cpu"
    assert config.training["batch_size"] == 8
    assert config.training["num_workers"] == 2
    assert config.training["max_epochs"] == 3
    assert config.execution["max_steps"] == 11
    assert config.execution["output_root"] == "artifacts/exploratory_runs/profile-41"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("batch_size = 32", "batch_size = 8"),
        ("num_workers = 8", "num_workers = 0"),
    ],
)
def test_formal_config_keeps_engineering_fields_exact_locked(
    tmp_path: Path, old: str, new: str
) -> None:
    source = (Path(__file__).resolve().parents[1] / "configs" / "phase1_baseline.toml").read_text(
        encoding="utf-8"
    )
    path = tmp_path / "formal.toml"
    path.write_text(source.replace(old, new), encoding="utf-8")

    with pytest.raises(ConfigError, match="locked contract"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed": "7"},
        {"device": "gpu:0"},
        {"batch_size": 0},
        {"num_workers": -1},
        {"max_epochs": 0},
        {"max_steps": -1},
        {"output": "artifacts/formal_runs/not-allowed"},
    ],
)
def test_exploratory_overrides_fail_closed(tmp_path: Path, overrides: dict[str, object]) -> None:
    path = tmp_path / "exploratory.toml"
    path.write_text(_document(), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_experiment_config(path, exploratory_overrides=overrides)
