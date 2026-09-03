from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cg_pipeline.config import ConfigError, load_experiment_config
from cg_pipeline.control_bank import (
    CONTROL_GENERATOR_VERSION,
    CONTROL_SEED,
    generate_matched_control_bundle,
)
from cg_pipeline.frontend import FixedHEMatchedControlFrontend, FixedHEMorletFrontend
from cg_pipeline.interaction import HEInteractionBlock
from cg_pipeline.model import FixedHEClassifier
from cg_pipeline.morlet import generate_morlet_bundle
from cg_pipeline.pooling import SupportAlignedPool
from cg_pipeline.training import (
    configure_determinism,
    load_checkpoint,
    model_state_identity,
    optimizer_state_identity,
    save_checkpoint,
)


def _with_variant(source: str, variant: str) -> str:
    contract_id = (
        "fixed-he-matched-control-linear-v1"
        if variant == "matched_control"
        else "fixed-he-morlet-linear-v1"
    )
    return source.replace(
        'contract_id = "fixed-he-morlet-linear-v1"',
        f'contract_id = "{contract_id}"\nfrontend_variant = "{variant}"',
        1,
    )


def test_frontend_variant_config_supports_both_modes_and_is_not_a_cli_override(
    tmp_path: Path,
) -> None:
    exploratory_source = Path("configs/exploratory_train.toml").read_text(encoding="utf-8")
    legacy_path = tmp_path / "legacy.toml"
    legacy_path.write_text(exploratory_source, encoding="utf-8")

    legacy = load_experiment_config(legacy_path)

    assert legacy.frontend_variant == "morlet"
    assert "frontend_variant" not in legacy.as_dict()["model"]

    formal_source = Path("configs/phase1_baseline.toml").read_text(encoding="utf-8")
    baseline_path = tmp_path / "formal-morlet.toml"
    baseline_path.write_text(formal_source, encoding="utf-8")
    baseline = load_experiment_config(baseline_path)
    assert baseline.frontend_variant == "morlet"
    assert "frontend_variant" not in baseline.as_dict()["model"]

    control_path = tmp_path / "control.toml"
    control_path.write_text(_with_variant(exploratory_source, "matched_control"), encoding="utf-8")
    control = load_experiment_config(control_path)

    assert control.frontend_variant == "matched_control"
    with pytest.raises(ConfigError, match="unknown exploratory overrides"):
        load_experiment_config(control_path, exploratory_overrides={"frontend_variant": "morlet"})

    missing_path = tmp_path / "missing-control-variant.toml"
    missing_path.write_text(
        exploratory_source.replace(
            'contract_id = "fixed-he-morlet-linear-v1"',
            'contract_id = "fixed-he-matched-control-linear-v1"',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="frontend_variant is required"):
        load_experiment_config(missing_path)

    invalid_path = tmp_path / "invalid.toml"
    invalid_path.write_text(_with_variant(exploratory_source, "unknown"), encoding="utf-8")
    with pytest.raises(ConfigError, match="frontend_variant"):
        load_experiment_config(invalid_path)

    formal_path = tmp_path / "formal-control.toml"
    formal_path.write_text(_with_variant(formal_source, "matched_control"), encoding="utf-8")
    formal_control = load_experiment_config(formal_path)
    assert formal_control.execution_kind == "formal_train"
    assert formal_control.frontend_variant == "matched_control"


@pytest.mark.parametrize(
    ("variant", "contract_id"),
    [
        ("morlet", "fixed-he-matched-control-linear-v1"),
        ("matched_control", "fixed-he-morlet-linear-v1"),
        ("matched_control", "unknown"),
        ("morlet", "unknown"),
        ("unknown", "fixed-he-morlet-linear-v1"),
    ],
)
def test_formal_frontend_contract_mismatches_fail_closed(
    tmp_path: Path, variant: str, contract_id: str
) -> None:
    source = Path("configs/phase1_baseline.toml").read_text(encoding="utf-8")
    path = tmp_path / "invalid-formal.toml"
    path.write_text(
        source.replace(
            'contract_id = "fixed-he-morlet-linear-v1"',
            f'contract_id = "{contract_id}"\nfrontend_variant = "{variant}"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_experiment_config(path)


def test_matched_control_bank_is_frozen_deterministic_and_ordered() -> None:
    first = generate_matched_control_bundle()
    second = generate_matched_control_bundle()

    assert CONTROL_GENERATOR_VERSION == "frozen-envelope-matched-random-phase-v1"
    assert CONTROL_SEED == 20260901
    assert first.kernels128.shape == (32, 105, 105)
    assert first.kernels128.dtype == np.complex128
    assert first.kernels64.dtype == np.complex64
    assert first.kernels64.flags.c_contiguous
    assert np.array_equal(first.kernels64, second.kernels64)
    assert first.channel_metadata == generate_morlet_bundle().channel_metadata
    assert first.specification_hash == second.specification_hash
    assert first.canonical_kernel_hash == second.canonical_kernel_hash
    assert first.spatial_execution_hash == second.spatial_execution_hash
    assert first.specification_hash == (
        "sha256:30c1d75b428e27efcdf9c6a0e71b3c7326f12aebf93b930e15c7b628fe84a308"
    )
    assert first.canonical_kernel_hash == (
        "sha256:dcf537ddb83b4699ebe6acbfb6d78d37b1e507fac54a0cf656919950c1b8848a"
    )
    assert first.spatial_execution_hash == (
        "sha256:e1002b5ab0eefd1cc00072935e8edc3ec2240667aab804831fa2977e65d37cde"
    )
    assert max(first.validation["complex128_zero_dc_error"]) <= 1e-12
    assert max(first.validation["complex128_unit_energy_error"]) <= 1e-12
    assert max(first.validation["complex64_zero_dc_error"]) <= 1e-6
    assert max(first.validation["complex64_unit_energy_error"]) <= 1e-6


def test_frontend_variants_preserve_shape_dtype_channel_order_support_and_capacity() -> None:
    morlet = FixedHEMorletFrontend(backend="fft")
    control = FixedHEMatchedControlFrontend(backend="fft")
    rgb = torch.arange(3 * 110 * 110, dtype=torch.int64).remainder(256)
    rgb = rgb.reshape(1, 3, 110, 110).to(torch.uint8)

    morlet_output = morlet(rgb)
    control_output = control(rgb)

    assert (
        morlet_output.feature_h.shape
        == control_output.feature_h.shape
        == (
            1,
            4,
            8,
            110,
            110,
        )
    )
    assert morlet_output.feature_e.shape == control_output.feature_e.shape
    assert morlet_output.feature_h.dtype == control_output.feature_h.dtype == torch.float32
    assert morlet.channel_metadata == control.channel_metadata
    assert torch.equal(morlet_output.valid_support_mask, control_output.valid_support_mask)
    assert list(morlet.parameters()) == list(control.parameters()) == []
    assert control.control_kernels.requires_grad is False

    models = {
        variant: FixedHEClassifier(frontend_backend="fft", frontend_variant=variant)
        for variant in ("morlet", "matched_control")
    }
    for model in models.values():
        assert (
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            == 9473
        )
        assert isinstance(model.interaction, HEInteractionBlock)
        assert isinstance(model.pooling, SupportAlignedPool)
        assert model.classifier.in_features == 9408
        assert model.classifier.out_features == 1

    assert type(models["morlet"].interaction) is type(models["matched_control"].interaction)
    assert type(models["morlet"].pooling) is type(models["matched_control"].pooling)
    assert type(models["morlet"].classifier) is type(models["matched_control"].classifier)
    assert isinstance(FixedHEClassifier(frontend_backend="fft").frontend, FixedHEMorletFrontend)


def test_frontend_variants_have_distinct_frozen_artifact_identities() -> None:
    morlet_identity = FixedHEMorletFrontend(backend="fft").fixed_state_identity()
    first_frontend = FixedHEMatchedControlFrontend(backend="fft")
    second_frontend = FixedHEMatchedControlFrontend(backend="fft")
    first_control = first_frontend.fixed_state_identity()
    second_control = second_frontend.fixed_state_identity()

    assert first_control == second_control
    assert first_control != morlet_identity
    assert morlet_identity["fixed_state_sha256"] == (
        "sha256:e2549d6305ae0cb40c0b3092fe60244f305ca64278a6e14e608ab686ea53f904"
    )
    assert first_control["filter_bank_specification_hash"].startswith("sha256:")
    assert first_control["canonical_kernel_hash"].startswith("sha256:")
    assert first_control["spatial_execution_hash"].startswith("sha256:")
    assert first_control["fixed_state_sha256"] == (
        "sha256:15645cbe124ec8a7036cd4d4e4b082fa005dbafe686c3d758cf6b59a3205c438"
    )
    assert all(value.startswith("sha256:") for value in first_control.values())

    artifact_identity = first_frontend.artifact_identity()
    assert artifact_identity["frontend_variant"] == "matched_control"
    assert artifact_identity["frontend_contract_id"] == "fixed-he-matched-control-linear-v1"
    assert artifact_identity["generator_version"] == CONTROL_GENERATOR_VERSION
    assert artifact_identity["rng"] == "PCG64DXSM"
    assert artifact_identity["control_seed"] == str(CONTROL_SEED)


def test_synthetic_forward_backward_and_checkpoint_identity_smoke(tmp_path: Path) -> None:
    configure_determinism(CONTROL_SEED)
    rgb = torch.arange(3 * 110 * 110, dtype=torch.int64).remainder(256)
    rgb = rgb.reshape(1, 3, 110, 110).to(torch.uint8)

    for variant in ("morlet", "matched_control"):
        model = FixedHEClassifier(frontend_backend="fft", frontend_variant=variant)
        optimizer = torch.optim.AdamW(model.electronic_parameters(), lr=0.001)
        fixed_before = model.frontend.fixed_state_identity()
        output = model(rgb)
        torch.nn.functional.binary_cross_entropy_with_logits(
            output.logits, torch.ones_like(output.logits)
        ).backward()
        optimizer.step()
        metadata = {
            "checkpoint_identity": model_state_identity(model),
            "optimizer_state_identity": optimizer_state_identity(optimizer),
            "fixed_frontend_identity": fixed_before,
            "frontend_artifact_identity": model.frontend.artifact_identity(),
        }
        checkpoint = tmp_path / f"{variant}.pt"

        save_checkpoint(checkpoint, model, optimizer, metadata)
        restored = FixedHEClassifier(frontend_backend="fft", frontend_variant=variant)
        restored_optimizer = torch.optim.AdamW(restored.electronic_parameters(), lr=0.001)
        load_checkpoint(
            checkpoint,
            restored,
            restored_optimizer,
            expected_metadata=metadata,
        )

        assert output.frontend.feature_h.shape == (1, 4, 8, 110, 110)
        assert output.logits.shape == (1,)
        assert model.frontend.fixed_state_identity() == fixed_before
        assert restored.frontend.fixed_state_identity() == fixed_before
        assert restored.frontend.artifact_identity() == model.frontend.artifact_identity()
