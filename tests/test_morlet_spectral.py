from __future__ import annotations

from cg_pipeline.morlet import generate_morlet_bundle, validate_spectral_coverage


def test_locked_morlet_identity_vectors_are_stable() -> None:
    bundle = generate_morlet_bundle()

    assert bundle.parameter_hash == "sha256:020c5bd67ba9ae5f234cc750ef4781de7c7ed6eb96991a5ce5e3868697598127"
    assert bundle.canonical_kernel_hash == "sha256:ec3a1c8dbec0a455e0b8bfdf159bc749cd926184403b83cc4e56f22e9884ba4c"
    assert bundle.spatial_execution_hash == "sha256:d89eee57ee11284646dd32ace899c3b7d31b2790c468cf59a2b5ed84cde96c19"


def test_all_spectral_coverage_gates_pass_per_kernel_pair_and_radius() -> None:
    report = validate_spectral_coverage(generate_morlet_bundle())

    assert report["status"] == "PASS"
    assert len(report["carrier_direction_error_degrees"]) == 32
    assert max(report["carrier_direction_error_degrees"]) <= 1.0
    assert len(report["radial_error"]) == 32
    assert all(report["radial_pass"])
    assert len(report["adjacent_orientation_overlap"]) == 32
    assert all(0.50 <= value <= 0.70 for value in report["adjacent_orientation_overlap"])
    assert len(report["adjacent_scale_overlap"]) == 24
    assert all(0.45 <= value <= 0.60 for value in report["adjacent_scale_overlap"])
    assert len(report["ring_uniformity"]) == 7
    assert all(item["min_over_median"] >= 0.85 for item in report["ring_uniformity"])
    assert all(item["min_over_max"] >= 0.75 for item in report["ring_uniformity"])
    assert all(item["coefficient_of_variation"] <= 0.10 for item in report["ring_uniformity"])
