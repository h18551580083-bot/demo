from __future__ import annotations

from dataclasses import replace

from cg_pipeline.morlet import (
    APPROVED_MORLET_BITWISE_IDENTITIES,
    LOCKED_MORLET_PARAMETER_HASH,
    audit_morlet_identity,
    generate_morlet_bundle,
    validate_spectral_coverage,
)


def test_locked_morlet_identity_vectors_are_stable() -> None:
    bundle = generate_morlet_bundle()
    audit = audit_morlet_identity(bundle)

    assert bundle.parameter_hash == LOCKED_MORLET_PARAMETER_HASH
    assert (
        bundle.canonical_kernel_hash,
        bundle.spatial_execution_hash,
    ) in APPROVED_MORLET_BITWISE_IDENTITIES.values()
    assert audit["status"] == "PASS"
    assert audit["parameter_identity_pass"] is True
    assert audit["bitwise_identity_pass"] is True
    assert audit["numerical_validation_pass"] is True
    assert audit["spectral_coverage_pass"] is True


def test_morlet_identity_audit_accepts_only_approved_complete_pairs() -> None:
    bundle = generate_morlet_bundle()
    spectral_pass = {"status": "PASS"}
    legacy = APPROVED_MORLET_BITWISE_IDENTITIES["legacy"]
    linux_verified = APPROVED_MORLET_BITWISE_IDENTITIES["linux_verified"]

    for variant, pair in APPROVED_MORLET_BITWISE_IDENTITIES.items():
        audit = audit_morlet_identity(
            replace(
                bundle,
                canonical_kernel_hash=pair[0],
                spatial_execution_hash=pair[1],
            ),
            spectral_coverage=spectral_pass,
        )
        assert audit["status"] == "PASS"
        assert audit["identity_variant"] == variant

    rejected = (
        replace(
            bundle,
            canonical_kernel_hash=legacy[0],
            spatial_execution_hash=linux_verified[1],
        ),
        replace(
            bundle,
            canonical_kernel_hash=linux_verified[0],
            spatial_execution_hash=legacy[1],
        ),
        replace(bundle, canonical_kernel_hash="sha256:" + "0" * 64),
        replace(bundle, spatial_execution_hash="sha256:" + "0" * 64),
        replace(bundle, parameter_hash="sha256:" + "0" * 64),
    )
    for candidate in rejected:
        assert (
            audit_morlet_identity(candidate, spectral_coverage=spectral_pass)["status"]
            == "FAIL"
        )


def test_morlet_identity_audit_fails_closed_on_numerical_or_spectral_failure() -> None:
    bundle = generate_morlet_bundle()
    invalid_validation = dict(bundle.validation)
    invalid_validation["complex64_zero_dc_error"] = (1.1e-6,)

    numerical_audit = audit_morlet_identity(
        replace(bundle, validation=invalid_validation),
        spectral_coverage={"status": "PASS"},
    )
    spectral_audit = audit_morlet_identity(
        bundle,
        spectral_coverage={"status": "FAIL"},
    )

    assert numerical_audit["status"] == "FAIL"
    assert numerical_audit["numerical_validation_pass"] is False
    assert spectral_audit["status"] == "FAIL"
    assert spectral_audit["spectral_coverage_pass"] is False


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
