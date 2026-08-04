"""Read-only audit of the active Phase 1 batch-32 training contract."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cg_pipeline.config import load_experiment_config
from cg_pipeline.data import build_dataloader, expected_batch_count

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY / "configs" / "phase1_baseline.toml"
RELEASE_PATH = REPOSITORY / "configs" / "phase1_training_release_b32_v3.json"
RELEASE_ID = "phase1-training-b32-v3"
SOURCE_MANIFEST_SHA256 = (
    "sha256:23c681a3a338e4df96c2e3443b39349c4758e08009eb47d46928d148f62045ab"
)
TRAIN_EFFECTIVE_SHA256 = (
    "sha256:8c54e7f8b1674e4e94c9a46e0d9abf01e4c0c8a88605e7831b2701c0ddbe58c5"
)
VAL_EFFECTIVE_SHA256 = (
    "sha256:1a6fd51cb6d7ae5da920f06974a871deef2f21147f0df9c4d2c902d30ed3decc"
)
RELEASE_COMMIT_ALLOWED_PATHS = [
    "configs/phase1_training_release_b32_v3.json",
    "docs/DECISIONS.md",
    "docs/PHASE1_TRAINING_RUNBOOK.md",
]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class _CountingDataset:
    def __init__(self, row_count: int, prefix: str) -> None:
        self.rows = tuple(
            SimpleNamespace(patch_id=f"{prefix}-{index:06d}")
            for index in range(row_count)
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return {"patch_id": self.rows[index].patch_id}


def _loader_integrity(row_count: int, prefix: str, batch_size: int) -> dict[str, Any]:
    dataset = _CountingDataset(row_count, prefix)
    loader = build_dataloader(
        dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        seed=1729,
        epoch=0,
        num_workers=0,
    )
    observed = [patch_id for batch in loader for patch_id in batch["patch_id"]]
    expected = {row.patch_id for row in dataset.rows}
    return {
        "row_count": len(observed),
        "unique_row_count": len(set(observed)),
        "batch_count": len(loader),
        "batch_size": loader.batch_size,
        "drop_last": loader.drop_last,
        "exact_once_coverage": len(observed) == row_count and set(observed) == expected,
    }


def audit() -> dict[str, Any]:
    config = load_experiment_config(CONFIG_PATH)
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    batch_size = int(config.training["batch_size"])
    train_integrity = _loader_integrity(79_570, "train", batch_size)
    validation_integrity = _loader_integrity(18_171, "validation", batch_size)
    checks = {
        "release_schema_v2": release["schema"] == "phase1-training-release-v2",
        "release_id": release["release_id"] == RELEASE_ID,
        "annotated_tag": release["annotated_tag"] == RELEASE_ID
        and _git("cat-file", "-t", RELEASE_ID) == "tag"
        and _git("rev-parse", f"{RELEASE_ID}^{{}}") == _git("rev-parse", "HEAD"),
        "single_parent_code_commit": len(
            _git("rev-list", "--parents", "-n", "1", "HEAD").split()
        )
        == 2
        and _git("rev-parse", "HEAD^") == release["formal_code_commit"],
        "release_commit_whitelist": release["release_commit_allowed_paths"]
        == RELEASE_COMMIT_ALLOWED_PATHS
        and _git(
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "--no-renames",
            "-r",
            "HEAD^",
            "HEAD",
        ).splitlines()
        == RELEASE_COMMIT_ALLOWED_PATHS,
        "formal_run_id": config.execution["run_id"] == "phase1-cam16-baseline-b32-v2",
        "run_release_roles_distinct": release["run_id_role"]
        == "unchanged-training-config-identity"
        and release["release_id_role"] == "release-governance-identity",
        "batch_size_32": batch_size == 32 and release["batch_size"] == 32,
        "drop_last_false": train_integrity["drop_last"] is False
        and validation_integrity["drop_last"] is False
        and release["drop_last"] is False,
        "config_hash_bound": release["config_hash"] == config.sha256,
        "normalized_hash_bound": release["normalized_config_sha256"] == config.sha256,
        "train_rows": release["expected_train_rows"]
        == train_integrity["row_count"]
        == 79_570,
        "train_exact_once_coverage": train_integrity["exact_once_coverage"],
        "train_batch_count": release["expected_train_batch_count"]
        == train_integrity["batch_count"]
        == expected_batch_count(79_570, batch_size, drop_last=False)
        == 2_487,
        "maximum_optimizer_updates": release["maximum_optimizer_updates"]
        == 2_487 * int(config.training["max_epochs"])
        == 49_740,
        "validation_rows": release["expected_validation_rows"]
        == validation_integrity["row_count"]
        == 18_171,
        "validation_exact_once_coverage": validation_integrity["exact_once_coverage"],
        "validation_batch_count": release["expected_validation_batch_count"]
        == validation_integrity["batch_count"]
        == expected_batch_count(18_171, batch_size, drop_last=False)
        == 568,
        "test_access_disabled": config.execution["allow_test"] is False
        and release["test_access_authorized"] is False,
        "source_manifest_identity": release["manifest_hash_algorithm"] == "sha256"
        and release["source_manifest_hash_rule"] == "raw-file-bytes-v1"
        and release["source_manifest_sha256"] == SOURCE_MANIFEST_SHA256,
        "effective_train_val_identities": release["effective_split_hash_rule"]
        == "cg/cam16-eval-manifest/v1"
        and release["effective_split_hashes"]
        == {"train": TRAIN_EFFECTIVE_SHA256, "val": VAL_EFFECTIVE_SHA256}
        and "test" not in release["effective_split_hashes"],
        "patient_claim_forbidden": release["patient_level_isolation"] == "not_evaluated"
        and release["patient_level_claim_allowed"] is False,
    }
    return {
        "schema": "phase1-training-contract-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "config_hash": config.sha256,
        "release_id": release["release_id"],
        "checks": checks,
        "train_integrity": train_integrity,
        "validation_integrity": validation_integrity,
        "execution_performed": False,
        "cam16_data_accessed": False,
    }


def main() -> int:
    report = audit()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
