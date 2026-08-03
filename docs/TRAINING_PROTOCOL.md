# Phase 1 preregistered training baseline

## Authority and claim

The executable source of formal parameters is
`configs/phase1_baseline.toml`, schema `phase0-experiment-config-v1`. The normalized
RFC 8785-subset JSON bytes and their SHA-256 are written before a formal run.
Unknown fields, missing fields, TOML floating values, illegal values, and `TBD`
fail before model construction.

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
| Batch size | 4; last partial batch retained |
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
| Resume | explicit `--resume`; exact config/code/data/kernel/seed identity and latest complete epoch only |

The global epoch order is keyed by seed, zero-based epoch, and exact UTF-8
`patch_id` under domain `cg/cam16-train-order/v1`. DataLoader batching and worker
sharding cannot drop, pad, repeat, or reorder membership. Worker seeds are derived
and recorded. `num_workers = 0` is the preregistered baseline.

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
Resume first verifies the normalized config bytes, then requires a continuous
zero-based sequence of immutable `.pt`/`.json` pairs. It loads only the latest pair
after exact config, code revision/code, source/effective manifest, complete fixed-
frontend, model-state checkpoint, seed, and epoch identity validation. A partial,
gapped, or mismatched history fails; it is not repaired. The paired report binds
the checkpoint-file hash, and restore recomputes the electronic model, optimizer,
and canonical fixed-front-end identities. Test rows are not loaded by training or
validation.

The entry remains forbidden until patient mapping structure/isolation passes, an
attributable provenance-reliability approval is bound to the mapping and source-
manifest hashes, and the explicit Phase 0 release record passes:

```powershell
$env:PYTHONPATH = 'E:\cg\src'
python -m cg_pipeline preflight --config configs\phase1_baseline.toml `
  --data-root cam16_patch `
  --patient-mapping <approved-local-patient-mapping.csv> `
  --patient-mapping-approval <approved-provenance-artifact.json> `
  --release configs\phase0_release.json `
  --output artifacts\phase0_preflight.json
```
