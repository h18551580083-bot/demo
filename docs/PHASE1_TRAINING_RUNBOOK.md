# RTX 4090 formal CAM16 training runbook — release v3

## 1. Authorization boundary

This runbook applies only to the annotated release tag
`phase1-training-b32-v3`. It authorizes the already frozen CAM16 train/validation
entry on an RTX 4090; it does not itself start training.

- Release ID/tag: `phase1-training-b32-v3` (release-governance identity).
- Run ID: `phase1-cam16-baseline-b32-v2` (unchanged training-config identity).
- Formal code commit: `174e059915fabdba687f7c904f2b184f2627a674`.
- Config SHA-256:
  `sha256:e44768d80d7c1545138d7d5e1368de4ed53b7b07b71202e2c5bdee6efac7cf3b`.
- Test access remains false. Do not enumerate, hash, load, or evaluate the test
  effective split. Patient-level isolation remains `not_evaluated`.

Use a clean Linux cloud checkout with one RTX 4090. Do not run the formal command
on the local development GPU. Dataset download or split modification is outside
this runbook.

## 2. Set paths without changing the contract

From the repository root:

```bash
set -euo pipefail
export PYTHONPATH="$PWD/src"
export DATA_ROOT="/absolute/path/to/approved/cam16_patch"
export RELEASE_ID="phase1-training-b32-v3"
export RUN_ID="phase1-cam16-baseline-b32-v2"
export PREFLIGHT_REPORT="artifacts/preflight/${RELEASE_ID}/preflight.json"
export FORMAL_OUTPUT="artifacts/formal_runs/${RUN_ID}"
test -d "$DATA_ROOT"
```

`DATA_ROOT` must already contain the approved package. Do not create aliases for a
different manifest or data package.

## 3. Verify Git and release topology

```bash
git checkout --detach phase1-training-b32-v3
test "$(git cat-file -t phase1-training-b32-v3)" = "tag"
test "$(git rev-parse 'phase1-training-b32-v3^{}')" = "$(git rev-parse HEAD)"
test "$(git rev-list --parents -n 1 HEAD | wc -w)" -eq 2
test "$(git rev-parse HEAD^)" = "174e059915fabdba687f7c904f2b184f2627a674"
test -z "$(git status --porcelain --untracked-files=no)"

git diff-tree --no-commit-id --name-only --no-renames -r HEAD^ HEAD > /tmp/release-paths.txt
printf '%s\n' \
  configs/phase1_training_release_b32_v3.json \
  docs/DECISIONS.md \
  docs/PHASE1_TRAINING_RUNBOOK.md > /tmp/approved-release-paths.txt
diff -u /tmp/approved-release-paths.txt /tmp/release-paths.txt
```

Any failure stops the run. A lightweight tag, wrong parent, multiple parents,
extra changed path, switched commit, or dirty tracked checkout is not acceptable.

## 4. Verify environment and static contract

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 1
name = torch.cuda.get_device_name(0)
assert "4090" in name, name
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": name})
PY

python tools/audit_phase1_training_contract.py
python -m pytest tests -q
python -m compileall -q src tests tools
python -m ruff check .
git diff --check
```

The contract audit, tests, compile check, and Ruff must all pass. The audit is
read-only and does not access CAM16 data or start training.

## 5. Confirm clean artifact destinations

The unchanged Run ID may be used only when neither the v3 preflight report nor the
formal output exists:

```bash
test ! -e "$PREFLIGHT_REPORT"
test ! -e "$FORMAL_OUTPUT"
```

Do not delete, rename, merge, or overwrite an earlier formal artifact to make these
checks pass. Stop and preserve it. A different output directory requires a new,
separately approved config/release identity that includes the Release ID.

## 6. Run the standalone preflight exactly once

```bash
python -m cg_pipeline preflight \
  --config configs/phase1_baseline.toml \
  --data-root "$DATA_ROOT" \
  --release configs/phase1_training_release_b32_v3.json \
  --output "$PREFLIGHT_REPORT"
```

The command exclusively creates the release-bound path. It recomputes and compares:

- current HEAD, annotated tag, single parent, formal code commit, and whitelist;
- Release ID and normalized config identity;
- raw source-manifest SHA-256
  `sha256:23c681a3a338e4df96c2e3443b39349c4758e08009eb47d46928d148f62045ab`;
- train effective identity
  `sha256:8c54e7f8b1674e4e94c9a46e0d9abf01e4c0c8a88605e7831b2701c0ddbe58c5`;
- validation effective identity
  `sha256:1a6fd51cb6d7ae5da920f06974a871deef2f21147f0df9c4d2c902d30ed3decc`;
- fixed frontend, spectral, optimizer ownership, determinism, isolation, and disabled
  test access gates.

Do not compute a test effective hash. Confirm the command exits zero and the report
shows `status = PASS`, `blocking_gates = []`, `training_started = false`, and
`test_split_accessed = false`.

## 7. Start formal training by consuming that report

```bash
python -m cg_pipeline train \
  --config configs/phase1_baseline.toml \
  --data-root "$DATA_ROOT" \
  --release configs/phase1_training_release_b32_v3.json \
  --preflight-report "$PREFLIGHT_REPORT"
```

Training does not rerun the full standalone preflight. Before its first batch it
revalidates the report schema/hash and the current HEAD/tag, release, config, source
manifest, train/validation split, disk/isolation, and governance identities. Any
change or injected field fails closed. `created_at` is retained only for audit and
has no expiry rule.

## 8. Interruption and resume

Do not use `--resume` for a new run. After a genuine interruption, resume only in
the same exact checkout with the same report and identities:

```bash
python -m cg_pipeline train \
  --config configs/phase1_baseline.toml \
  --data-root "$DATA_ROOT" \
  --release configs/phase1_training_release_b32_v3.json \
  --preflight-report "$PREFLIGHT_REPORT" \
  --resume
```

Resume requires the continuous immutable checkpoint/report history and exact
config, code, release, preflight, source/effective manifest, fixed-frontend, model,
optimizer, seed, and epoch identities. It never repairs or overwrites artifacts.

## 9. Stop conditions and claims

Stop immediately on any nonzero command, identity mismatch, existing destination,
non-4090 device, non-annotated tag, changed manifest/split, report corruption, or
resume discontinuity. Preserve the terminal output and artifacts for audit; do not
retry by changing the contract.

Training and validation may report only the frozen slide-level metrics. The sole
isolation statement remains `group_id/slide_id split isolation verified`. Do not
claim patient-level leakage prevention, do not access test data, and do not advance
the project phase from this runbook alone.
