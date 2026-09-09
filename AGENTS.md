# Project instructions

## Research and data boundaries

- The optical frontend is fixed and must not be trainable; the project is simulation-only.
- CAM16 is the primary development dataset; other pathology datasets are transfer-only.
- Do not add physical deployment, fabrication, SLM control, or clinical deployment code.
- Do not access CAM16 test: no enumeration, hashing, loading, or evaluation without
  the separate approved final-once gate. Training uses train/validation only.
- Prevent cross-split leakage at the declared `group_id`/`slide_id` level.
  Record `patient_level_isolation = not_evaluated` and `patient_level_claim_allowed = false`.
  No reliable patient-to-slide mapping exists for this study; do not infer patient
  identity from filenames/identifiers. This is not a Phase 0 or formal-training blocker.
- Never silently change splits, download datasets automatically, or commit images,
  checkpoints, credentials, or patient metadata.

## Authority and execution

- `docs/DEVELOPMENT_SPEC.md` is the highest-level contract. Only explicit human
  approval may fill a TBD or change a scientific contract; record approved decisions
  in `docs/DECISIONS.md`. Never advance Phase automatically. Report conflicts.
- The only real CAM16 training modes are `exploratory_train` and `formal_train`.
  Exploratory work may use a dirty/untracked tree and controlled engineering overrides;
  every artifact must retain `formal_experiment=false` and `experiment_mode=exploratory_train`.
  Never promote or rename exploratory results as formal evidence.
- Formal training follows `docs/TRAINING_PROTOCOL.md`: lightweight authorization,
  standalone preflight, frozen seeds, full epoch/checkpoint/provenance contract and
  non-overwriting outputs. Git/release/tag/hash identity is not a startup gate.
- Keep configuration explicit and optical frontend, electronic backend, evaluation,
  and dataset adapters separate. Prefer small testable modules and typed interfaces.
- Run unit tests for changed modules and the non-training project smoke; obey narrower
  task-specific verification boundaries. Report changed files, test commands/results/
  failures/skips, unresolved issues, assumptions, and locked-specification impact.

## Context loading policy

Always read:
- `AGENTS.md` (the sole default project entry).

Read only when relevant:
- Optical/frontend -> relevant section of `docs/specs/FIXED_OPTICAL_FRONTEND_SPEC.md`
  and relevant ADR; Phase2-A work also needs its scoped approved decisions.
- Interaction/backend -> relevant section of `docs/specs/ELECTRONIC_BACKEND_SPEC.md`
  and relevant ADR.
- Training -> `docs/TRAINING_PROTOCOL.md` and task-relevant approved amendments.
- Evaluation -> `docs/EVALUATION_PROTOCOL.md`.
- Terminology/paper writing -> relevant `CONTEXT.md` section.
- Architecture history -> relevant ADR only; use `docs/agents/domain.md` for domain work.
- Historical decision verification -> search `docs/DECISIONS.md` by topic/date.
- Highest-level scope/change -> `docs/DEVELOPMENT_SPEC.md`.
- GitHub issue/triage work only -> `docs/agents/issue-tracker.md` and `docs/agents/triage-labels.md`.
- Maintenance only -> `docs/MAINTENANCE.md`; refactor audit only -> `docs/SPEC_REFACTOR_MAP.md`.

Do not read all docs, all ADRs, or all DECISIONS by default.
Historical reports, `docs/archive/`, and historical ADR bodies are never default context.
Reports explain evidence; they cannot authorize scientific changes.
