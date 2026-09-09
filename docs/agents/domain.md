# Domain docs

This repository uses a single-context domain documentation layout.

## Before exploring

- Read only the task-relevant sections of root `CONTEXT.md` when it exists.
- Read only ADRs under `docs/adr/` that touch the area being changed.
- Do not load all ADRs or the full glossary by default.
- If these files do not exist, proceed silently. Do not invent domain decisions or
  create placeholder decisions.

## Vocabulary

Use terms as defined in `CONTEXT.md`. If a required concept is absent, treat that
as either a vocabulary mismatch or a domain-modeling gap rather than silently
introducing a synonym.

## ADR conflicts

Surface conflicts with an existing ADR explicitly. Do not silently override an
accepted architectural decision.

ADRs explain design choices; they do not override DEVELOPMENT_SPEC, approved
DECISIONS, or current normative specs/protocols. Report an unresolved conflict
with those authorities; do not silently override either document.
