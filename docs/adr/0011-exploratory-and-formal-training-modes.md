# ADR 0011: Separate exploratory and formal CAM16 training modes

Status: accepted on 2026-08-10 by explicit human instruction.

The project has exactly two real CAM16 training modes: `exploratory_train` and
`formal_train`. The earlier synthetic dry-run workflow is retired.

`exploratory_train` uses real train/validation data for profiling, performance
optimization, bounded-step validation, and other non-formal engineering work. It
requires the shared data-path, manifest-readability, legal train/validation split,
and supplied `group_id`/`slide_id` isolation checks. It does not require a release,
tag, clean source tree, formal configuration hash, or formal preflight report. It
uses one seed and may override run/output identity, batch size, worker count, epoch
count, step count, and runtime device. Its output is physically isolated under
`artifacts/exploratory_runs/`, and all reports and checkpoint metadata state
`formal_experiment=false` and `experiment_mode=exploratory_train`.

`formal_train` remains unchanged in authority and scientific meaning. It consumes
the release-bound standalone preflight report and retains the annotated-tag,
single-parent release, clean-tree, release/config/source/manifest identity, fixed
seeds, complete-epoch, validation checkpoint-selection, immutable checkpoint,
provenance, and non-overwrite requirements. Formal output remains under
`artifacts/formal_runs/`.

Neither mode constructs a test dataset or exposes a test-access CLI option. Test
evaluation remains prohibited until a separate identity-bound final-once
authorization. This decision does not change the fixed optical frontend, split
membership, isolation claim, formal metrics, checkpoint winner rule, seeds, or
project phase.
