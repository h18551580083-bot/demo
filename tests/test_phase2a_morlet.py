"""Data-free Phase2-A parameter plumbing and Phase1 numerical regression."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cg_pipeline import preflight, training_runs
from cg_pipeline.artifacts import PipelineBlockedError
from cg_pipeline.config import ConfigError, load_experiment_config
from cg_pipeline.model import FixedHEClassifier
from cg_pipeline.morlet import LOCKED_MORLET_PARAMETER_HASH, generate_morlet_bundle
from cg_pipeline.morlet_validity import validate_phase2_morlet


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_default_matches_original_formula():
    axis = np.arange(-52, 53, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    original = []
    for j in range(4):
        sigma = np.float64(0.8 * (2**j))
        xi = np.float64((3.0 * np.pi / 4.0) * (2.0 ** (-j)))
        for ell in range(8):
            theta = np.float64(ell * np.pi / 8.0)
            parallel = xx * np.cos(theta) + yy * np.sin(theta)
            perpendicular = -xx * np.sin(theta) + yy * np.cos(theta)
            envelope = np.exp(
                -(parallel * parallel + np.float64(0.25) * perpendicular * perpendicular)
                / (2.0 * sigma * sigma)
            )
            carrier = np.exp(1j * xi * parallel)
            beta = np.sum(envelope * carrier, dtype=np.complex128) / np.sum(
                envelope, dtype=np.float64
            )
            kernel = envelope * (carrier - beta)
            original.append(kernel / np.sqrt(np.sum(np.abs(kernel) ** 2, dtype=np.float64)))
    actual = generate_morlet_bundle()
    assert actual.parameter_hash == LOCKED_MORLET_PARAMETER_HASH
    np.testing.assert_array_equal(actual.kernels128, np.stack(original))
    np.testing.assert_array_equal(actual.kernels64, np.stack(original).astype(np.complex64))
    explicit = generate_morlet_bundle(sigma0="0.8", xi0="3*pi/4", gamma="0.5")
    assert actual.parameter_hash == explicit.parameter_hash
    assert actual.canonical_kernel_hash == explicit.canonical_kernel_hash
    assert actual.spatial_execution_hash == explicit.spatial_execution_hash


@pytest.mark.parametrize(
    "suffix,change",
    [
        ("baseline", {}),
        ("sigma0_0p7", {"sigma0": "0.7"}),
        ("xi0_2pi3", {"xi0": "2*pi/3"}),
        ("gamma_0p625", {"gamma": "0.625"}),
    ],
)
def test_config_model_generator_and_formal_plumbing(suffix, change, monkeypatch, tmp_path):
    baseline = load_experiment_config("configs/phase1_baseline.toml")
    config = load_experiment_config(f"configs/phase2a_morlet_{suffix}.toml")
    expected = baseline.as_dict()
    expected["execution"].update(
        run_id=config.execution["run_id"], output_root=config.execution["output_root"]
    )
    expected["model"].update(
        contract_id="fixed-he-morlet-phase2a-linear-v1", frontend_variant="morlet", **change
    )
    assert config.as_dict() == expected
    parameters = {key: config.model[key] for key in ("sigma0", "xi0", "gamma")}
    bundle = generate_morlet_bundle(**parameters)
    model = FixedHEClassifier(frontend_backend="fft", **parameters)
    np.testing.assert_array_equal(model.frontend.morlet_kernels.numpy(), bundle.kernels64)
    assert list(model.frontend.parameters()) == []
    assert sum(p.numel() for p in model.electronic_parameters()) == 9473
    assert np.array_equal(bundle.kernels64, generate_morlet_bundle().kernels64) == (not change)

    # Stop at the real model constructor: no dataset, optimizer step, or checkpoint IO.
    class Constructed(Exception):
        pass

    def capture(**kwargs):
        constructed = FixedHEClassifier(**kwargs)
        np.testing.assert_array_equal(constructed.frontend.morlet_kernels.numpy(), bundle.kernels64)
        assert {key: kwargs[key] for key in parameters} == parameters
        raise Constructed

    monkeypatch.setattr(preflight, "FixedHEClassifier", capture)
    with pytest.raises(Constructed):
        preflight._model_audits(config, torch.device("cpu"))
    monkeypatch.setattr(training_runs, "FixedHEClassifier", capture)
    monkeypatch.setattr(training_runs, "configure_determinism", lambda seed: None)
    with pytest.raises(Constructed):
        training_runs.run_formal_seed(
            config,
            SimpleNamespace(),
            None,
            None,
            torch.device("cpu"),
            tmp_path,
            seed=1729,
            resume=False,
        )


def test_legacy_rejects_perturbation_and_phase2_rejects_invalid_value(tmp_path):
    source = Path("configs/phase1_baseline.toml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.toml"
    path.write_text(source.replace('sigma0 = "0.8"', 'sigma0 = "0.7"'), encoding="utf-8")
    with pytest.raises(ConfigError, match="locked contract"):
        load_experiment_config(path)
    source = Path("configs/phase2a_morlet_baseline.toml").read_text(encoding="utf-8")
    path.write_text(source.replace('gamma = "0.5"', 'gamma = "nan"'), encoding="utf-8")
    with pytest.raises(ConfigError, match="unsupported Morlet gamma"):
        load_experiment_config(path)


@pytest.mark.parametrize("suffix", ["baseline", "sigma0_0p7", "xi0_2pi3", "gamma_0p625"])
def test_phase2_preflight_uses_validity_and_preserves_diagnostics(suffix):
    config = load_experiment_config(f"configs/phase2a_morlet_{suffix}.toml")
    assert preflight._frontend_gate(config) == "phase2a_morlet_validity"
    report = preflight._model_audits(config, torch.device("cpu"))
    validity = report["phase2a_morlet_validity"]
    assert validity["status"] == "PASS"
    assert all(validity["interface_checks"].values())
    assert report["morlet_spectral_diagnostics"]["status"] == (
        "PASS" if suffix == "baseline" else "FAIL"
    )
    assert len(report["morlet_spectral_diagnostics"]["ring_uniformity"]) == 7
    assert len(report["morlet_numerical_diagnostics"]["beta_reference_error"]) == 32


def test_preflight_consumption_rejects_other_parameters(monkeypatch):
    config = load_experiment_config("configs/phase2a_morlet_sigma0_0p7.toml")
    report = {
        "status": "PASS",
        "blocking_gates": [],
        "training_started": False,
        "test_split_accessed": False,
        "passed_gates": list(preflight._REQUIRED_PASSED_GATES | {"morlet_spectral_coverage"}),
        "morlet_parameters": {"sigma0": "0.8", "xi0": "3*pi/4", "gamma": "0.5"},
    }
    monkeypatch.setattr(preflight, "read_json_object", lambda path: report)
    with pytest.raises(PipelineBlockedError, match="morlet_parameters"):
        preflight.consume_preflight_report(
            config,
            data_root=Path("unused"),
            authorization_path=Path("unused"),
            preflight_report_path=Path("unused"),
        )


def test_a0_preflight_records_actual_parameters():
    config = load_experiment_config("configs/phase2a_morlet_baseline.toml")
    report = preflight._model_audits(config, torch.device("cpu"))
    assert report["morlet_parameters"] == {"sigma0": "0.8", "xi0": "3*pi/4", "gamma": "0.5"}
    assert report["phase2a_morlet_validity"]["status"] == "PASS"
    assert (
        report["fixed_frontend_identity"]
        == FixedHEClassifier(frontend_backend="fft").frontend.fixed_state_identity()
    )


def test_phase1_still_uses_original_gate_and_rejects_failure(monkeypatch):
    config = load_experiment_config("configs/phase1_baseline.toml")
    assert preflight._frontend_gate(config) == "morlet_spectral_coverage"
    assert (
        preflight._model_audits(config, torch.device("cpu"))["morlet_spectral_coverage"]["status"]
        == "PASS"
    )
    monkeypatch.setattr(
        preflight, "validate_spectral_coverage", lambda *a, **kw: {"status": "FAIL"}
    )
    with pytest.raises(PipelineBlockedError):
        preflight._model_audits(config, torch.device("cpu"))


def test_theoretical_peak_depends_on_configuration():
    bundle = generate_morlet_bundle(sigma0="0.7")
    report = validate_phase2_morlet(bundle, sigma0="0.8", xi0="3*pi/4", gamma="0.5")
    assert report["status"] == "FAIL"
    assert not report["checks"]["configured_spectral_peaks"]


def test_nyquist_peak_and_runtime_only_corruption_fail():
    bundle = generate_morlet_bundle()
    axis = np.arange(-52, 53)
    yy, xx = np.meshgrid(axis, axis)
    edge = np.exp(-(xx**2 + yy**2) / 32) * np.cos(np.pi * xx)
    edge -= edge.mean()
    edge /= np.sqrt((edge**2).sum())
    kernels = bundle.kernels128.copy()
    kernels[0] = edge
    bad = replace(bundle, kernels128=kernels, kernels64=kernels.astype(np.complex64))
    report = validate_phase2_morlet(bad, sigma0="0.8", xi0="3*pi/4", gamma="0.5")
    assert report["status"] == "FAIL"
    assert not report["peaks"][0]["checks"]["nyquist_margin"]
    runtime_bad = replace(bundle, kernels64=bundle.kernels64.conj())
    report = validate_phase2_morlet(runtime_bad, sigma0="0.8", xi0="3*pi/4", gamma="0.5")
    assert report["status"] == "FAIL"
    assert report["peaks"][0]["checks"]["peak_matches_theory"]
    assert not report["peaks"][0]["checks"]["runtime_peak_matches_theory"]


@pytest.mark.parametrize(
    "defect", ["shape", "finite", "dc", "energy", "channels", "reflected_peak"]
)
def test_phase2_invalid_kernels_fail_closed(defect):
    bundle = generate_morlet_bundle()
    kernels = bundle.kernels128.copy()
    if defect == "shape":
        kernels = kernels[:, :-1]
    elif defect == "finite":
        kernels[0, 0, 0] = np.nan
    elif defect == "dc":
        kernels[0] += 0.01
    elif defect == "energy":
        kernels *= 2
    elif defect == "reflected_peak":
        kernels = kernels.conj()
    bundle = replace(bundle, kernels128=kernels, kernels64=kernels.astype(np.complex64))
    if defect == "channels":
        bundle = replace(bundle, channel_metadata=bundle.channel_metadata[:-1])
    report = validate_phase2_morlet(bundle, sigma0="0.8", xi0="3*pi/4", gamma="0.5")
    assert report["status"] == "FAIL"
