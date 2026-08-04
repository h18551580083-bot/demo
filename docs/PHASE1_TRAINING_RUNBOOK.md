# Phase 1 formal training runbook — batch-32 revision

## Active identity and authorization boundary

The active formal contract is `phase1-cam16-baseline-b32-v2`:

- config: `configs/phase1_baseline.toml`;
- training release: `configs/phase1_training_release_b32_v2.json`;
- normalized config SHA-256:
  `sha256:e44768d80d7c1545138d7d5e1368de4ed53b7b07b71202e2c5bdee6efac7cf3b`;
- historical superseded batch-4 SHA-256:
  `sha256:0653ae0003dac9062b73749e879a9a541a3f9dae18b034bdc1632f8410910e75`.

The historical `configs/phase0_release.json` and `phase0-closed-v1` tag are not
valid substitutes for the hash-bound training release. This runbook does not by
itself start execution. Test access remains false, and test rows or metrics
must not be loaded by training or validation.

## Contract check before any formal run

From `E:\cg` in PowerShell:

```powershell
$env:PYTHONPATH = 'E:\cg\src'
python tools\audit_phase1_training_contract.py
python -m pytest tests\test_pipeline_config.py tests\test_data_contract.py `
  tests\test_pipeline_entrypoints.py tests\test_training_protocol.py -q
python -m compileall -q src tests tools
python -m ruff check .
git diff --check
```

The audit must report batch size 32, `drop_last = false`, 79,570 train rows,
2,487 train batches, 18,171 validation rows, 568 validation batches, and matching
config/release hashes. It exercises exact-size synthetic identifiers to prove
once-only DataLoader coverage; the real manifest counts are enforced later by the
hash-bound preflight release. This audit does not access CAM16 patches or any test
metric.

## Preflight only

Preflight does not start training and must report `training_started = false` and
`test_split_accessed = false`:

```powershell
$env:PYTHONPATH = 'E:\cg\src'
python -m cg_pipeline preflight `
  --config configs\phase1_baseline.toml `
  --data-root cam16_patch `
  --release configs\phase1_training_release_b32_v2.json `
  --output artifacts\phase1_b32_preflight.json
```

Do not proceed if the config hash, release identity, manifest row counts,
batch-count derivation, fixed-frontend identity, optimizer ownership, determinism,
or test-access gate differs from the approved contract.

## Formal entry command

Use this command only after a separate operator decision to start the formal run:

```powershell
$env:PYTHONPATH = 'E:\cg\src'
python -m cg_pipeline train `
  --config configs\phase1_baseline.toml `
  --data-root cam16_patch `
  --release configs\phase1_training_release_b32_v2.json
```

The output root is
`artifacts/formal_runs/phase1-cam16-baseline-b32-v2`. Existing output is never
overwritten. Resume requires the exact normalized config and latest complete
immutable checkpoint/report pair.

## Frozen values and current uncertainty

AdamW, learning rate `0.001`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay
`1e-4`, no scheduler, no class weighting, no augmentation, no gradient clipping,
20 maximum epochs, patience 5/minimum delta 0, seeds `1729`, `3407`, `7919`,
per-complete-epoch immutable checkpoints, validation slide-AUROC selection, and
the final-once test gate remain unchanged.

Batch 32 reduces optimizer updates from 19,893 to 2,487 per complete epoch and
from 397,860 to 49,740 at the 20-epoch cap. Learning rate `0.001` and 20 epochs are
retained without linear scaling, but their suitability has not been validated by
formal training. Any later LR or epoch-budget change requires a separate human
decision based on validation convergence evidence.
