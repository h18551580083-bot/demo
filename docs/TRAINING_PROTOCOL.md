# Phase 1 preregistered training baseline

## Authority and claim

The executable source of formal parameters is
`configs/phase1_baseline.toml`, schema `phase0-experiment-config-v1`. The normalized
RFC 8785-subset JSON bytes and their SHA-256 are written before a formal run.
Unknown fields, missing fields, TOML floating values, illegal values, and `TBD`
fail before model construction.

The active batch-32 training configuration is `phase1-cam16-baseline-b32-v2`.
Once its exact three-file release commit and annotated tag are present, it is
released under governance identity `phase1-training-b32-v3` by
`configs/phase1_training_release_b32_v3.json`. Its normalized configuration hash
is `sha256:e44768d80d7c1545138d7d5e1368de4ed53b7b07b71202e2c5bdee6efac7cf3b`.
It supersedes the batch-4 configuration hash
`sha256:0653ae0003dac9062b73749e879a9a541a3f9dae18b034bdc1632f8410910e75`;
the historical Phase 0 release and `phase0-closed-v1` tag remain immutable.

This is the preregistered Phase 1 starting baseline. It is intentionally simple
and reproducible; it is not claimed to be optimal or validated by CAM16 training.
Changing a scientific field creates a new contract and requires a new decision.

## Frozen optimization contract

| Item | Frozen value |
| --- | --- |
| Loss | mean binary cross entropy with raw logits (`BCEWithLogits`), float32 |
| Optimizer | AdamW, all state float32 |
| Learning rate | `0.001` |
| Betas | `(0.9, 0.999)` |
| Epsilon | `0.00000001` |
| Weight decay | `0.0001` |
| Scheduler | none |
| Batch size | 32; last partial batch retained (`drop_last = false`) |
| Epoch limit | 20 |
| Early stopping | validation slide AUROC, patience 5 completed epochs, minimum improvement 0 |
| Checkpoint selection | highest validation slide AUROC; earliest epoch wins an exact tie |
| Checkpoint persistence | immutable checkpoint after every complete epoch |
| Sampling | every training-manifest row exactly once per epoch in the frozen SHA-256 order |
| Class imbalance | uniform unweighted rows; no resampling and no class/loss weight |
| Augmentation | none, including no stain, color, geometric, mix, or test-time augmentation |
| Gradient clipping | none |
| Seeds | `1729`, `3407`, `7919` |
| Failed run | record and exclude; no automatic retry or replacement seed |
| Multi-seed report | every valid seed, arithmetic mean, and sample standard deviation |
| Resume | explicit `--resume`; exact release/preflight/config/code/data/kernel/seed identity and latest complete epoch only |

The global epoch order is keyed by seed, zero-based epoch, and exact UTF-8
`patch_id` under domain `cg/cam16-train-order/v1`. DataLoader batching and worker
sharding cannot drop, pad, repeat, or reorder membership. Worker seeds are derived
and recorded. `num_workers = 0` is the preregistered baseline.

For the approved manifest, train has 79,570 rows and therefore 2,487 optimizer
updates per complete epoch: `ceil(79570 / 32)`. The maximum 20-epoch budget is
49,740 updates. Validation reuses the same batch-size configuration and has 18,171
rows, therefore 568 batches: `ceil(18171 / 32)`. The final partial train and
validation batches are retained, so every row remains covered. Batch 32 is not
optimizer-step-budget-equivalent to the historical batch-4 contract (19,893
updates per epoch; 397,860 at 20 epochs).

## Precision and ownership

Python, NumPy, PyTorch CPU, and all CUDA devices receive the same run seed. PyTorch
deterministic algorithms are required with `warn_only = false`; cuDNN benchmarking,
AMP, gradient scaling, float16, bfloat16, and TF32 are disabled. The loss,
parameters, gradients, AdamW state, and logits are float32; only the already locked
pooling statistic path uses float64 intermediates.

The optimizer must own each of the 9473 electronic scalars exactly once and no
optical value. Every training step checks finite loss/gradients, optimizer-state
precision, and the fixed-front-end byte identity before and after the step.

## Output and interruption contract

The formal layout is `output_root/seed-<seed>/epoch-<zero-padded>.{pt,json}` plus
the normalized config and final summary. Existing files are never overwritten.
Resume first verifies the normalized config and release/preflight identities, then requires a continuous
zero-based sequence of immutable `.pt`/`.json` pairs. It loads only the latest pair
after exact config, code revision/code, source/effective manifest, complete fixed-
frontend, model-state checkpoint, seed, and epoch identity validation. A partial,
gapped, or mismatched history fails; it is not repaired. The paired report binds
the checkpoint-file hash, and restore recomputes the electronic model, optimizer,
and canonical fixed-front-end identities. Test rows are not loaded by training or
validation.

The entry requires one standalone preflight report created from the published v3
release. `train --preflight-report` verifies the exact report schema and canonical
hash, then recomputes current
HEAD/tag/release/config/source-manifest and train/validation effective identities;
it also rechecks the frozen governance fields, while `created_at` remains audit-only
and has no expiry gate. It does not repeat the full model/optimizer/spectral
preflight. A patient mapping or mapping-approval file is not an input or
prerequisite. Every preflight and training report records
`group_id/slide_id split isolation verified`,
`patient_level_isolation = not_evaluated`, and
`patient_level_claim_allowed = false`:

```bash
export PYTHONPATH="$PWD/src"
python -m cg_pipeline preflight \
  --config configs/phase1_baseline.toml \
  --data-root "$DATA_ROOT" \
  --release configs/phase1_training_release_b32_v3.json \
  --output artifacts/preflight/phase1-training-b32-v3/preflight.json

python -m cg_pipeline train \
  --config configs/phase1_baseline.toml \
  --data-root "$DATA_ROOT" \
  --release configs/phase1_training_release_b32_v3.json \
  --preflight-report artifacts/preflight/phase1-training-b32-v3/preflight.json
```
