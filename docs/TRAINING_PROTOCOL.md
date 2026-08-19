# Phase 1 preregistered training baseline

## Authority and claim

The executable source of formal parameters is
`configs/phase1_baseline.toml`, schema `phase0-experiment-config-v2`. Version 2 replaces
the combined-manifest input with explicit train and validation manifest paths; split
membership and scientific training parameters are unchanged. Unknown fields,
missing fields, TOML floating values, illegal values, and `TBD` fail before model
construction. The config is not hashed or authenticated as a frozen identity.

The active batch-32 training configuration is `phase1-cam16-baseline-b32-v2`.
`configs/formal_training_authorization.json` is a lightweight
formal-training authorization record. Historical release and tag evidence remains
historical and is not a startup gate.

This is the preregistered Phase 1 starting baseline. It is intentionally simple
and reproducible; it is not claimed to be optimal or validated by CAM16 training.
Changing a scientific field creates a new contract and requires a new decision.

## Two-mode training workflow

`exploratory_train` uses real CAM16 train/validation data for profiling, engineering
performance work, single-epoch or `max_steps` checks, and other non-formal
experiments. It requires only explicit train/validation manifest readability, legal
splits, and `group_id`/`slide_id` cross-split isolation checks. It permits a dirty or
untracked source tree, one seed, and CLI overrides for run/output identity, batch
size, worker count, device, epoch count, and step count. Every exploratory artifact records
`formal_experiment = false` and `experiment_mode = exploratory_train`; it cannot be
promoted or renamed as formal evidence.

`formal_train` retains the experiment contract below. It requires lightweight human
authorization, a passing standalone preflight, legal train/validation data and
isolation, CUDA, fixed frontend and Morlet checks, optimizer ownership, fixed seeds,
complete epochs, validation checkpoint selection, immutable outputs, and provenance.
Git state, tags, commit paths, code identity, and config identity are not gates. Test
access remains prohibited for both modes pending separate final-once authorization.

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
| Resume | explicit `--resume`; continuous checkpoint/report pairs and matching data, fixed frontend, model, optimizer, seed, and epoch state |

The global epoch order is keyed by seed, zero-based epoch, and exact UTF-8
`patch_id` under domain `cg/cam16-train-order/v1`. DataLoader batching and worker
sharding cannot drop, pad, repeat, or reorder membership. Worker seeds are derived
and recorded. Formal and default exploratory execution fix `num_workers = 8` as an
explicit engineering configuration; it is not selected or tuned at runtime.

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
the final summary. Existing files are never overwritten. Resume requires a continuous
zero-based sequence of `.pt`/`.json` pairs. It loads only the latest pair after
source/effective manifest, complete fixed-frontend, model-state checkpoint, optimizer,
seed, and epoch validation. A partial, gapped, or mismatched history fails; it is not
repaired. The paired report binds
the checkpoint-file hash, and restore recomputes the electronic model, optimizer,
and canonical fixed-front-end identities. Test rows are not loaded by training or
validation.

The entry requires one standalone preflight report. `formal-train --preflight-report`
checks that it is readable, passed, has no blocking gates, did not start training,
and did not access test. It revalidates the current authorization, CUDA availability,
manifest, train/validation files, and split isolation without attaching a checksum,
signature, fixed path, Git state, code identity, or config identity. It does not
repeat the full model/optimizer/spectral preflight. A patient mapping or
mapping-approval file is not an input or
prerequisite. Every preflight and training report records
`group_id/slide_id split isolation verified`,
`patient_level_isolation = not_evaluated`, and
`patient_level_claim_allowed = false`:

```bash
export PYTHONPATH="$PWD/src"
python -m cg_pipeline formal-preflight \
  --config configs/phase1_baseline.toml \
  --data-root "$DATA_ROOT" \
  --authorization configs/formal_training_authorization.json \
  --output artifacts/preflight/phase1-training-b32-workers8-v1/preflight.json

python -m cg_pipeline formal-train \
  --config configs/phase1_baseline.toml \
  --data-root "$DATA_ROOT" \
  --authorization configs/formal_training_authorization.json \
  --preflight-report artifacts/preflight/phase1-training-b32-workers8-v1/preflight.json
```
