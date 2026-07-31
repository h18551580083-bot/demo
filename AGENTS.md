# Project instructions

## Research boundary

- The optical frontend is fixed and must not be trainable.
- CAM16 is the primary development dataset.
- Other pathology datasets are reserved for transfer evaluation.
- The project is simulation-only.
- Do not add physical deployment, fabrication, SLM control, or clinical deployment code.

## Specification rules

- docs/DEVELOPMENT_SPEC.md is the highest-level specification.
- Never fill a TBD without explicit human approval.
- Record approved decisions in docs/DECISIONS.md.
- Do not advance to the next phase automatically.

## Data rules

- Prevent patient-level leakage.
- Never change dataset splits silently.
- Never download datasets automatically.
- Do not commit images, checkpoints, credentials, or patient metadata.

## Coding rules

- Use typed interfaces where practical.
- Configuration must not be hidden in source code.
- Keep optical frontend, electronic backend, evaluation, and dataset adapters separate.
- Prefer small, testable modules.

## Testing

- Run unit tests for all changed modules.
- Run the project smoke test before finishing.
- Report commands, results, failures, and skipped tests.

## Completion response

Report:
1. changed files;
2. tests executed;
3. unresolved issues;
4. assumptions;
5. whether any locked specification was affected.

## Agent skills

### Issue tracker

PRDs and work items use GitHub Issues; external pull requests are not a request or triage
surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the canonical labels `needs-triage`, `needs-info`, `ready-for-agent`,
`ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Use the single-context layout with root `CONTEXT.md` and repository-wide ADRs under
`docs/adr/`. See `docs/agents/domain.md`.
