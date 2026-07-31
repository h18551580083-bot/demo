# Domain docs

This repository uses a single-context domain documentation layout.

## Before exploring

- Read `CONTEXT.md` at the repository root when it exists.
- Read ADRs under `docs/adr/` that touch the area being changed.
- If these files do not exist, proceed silently. Do not invent domain decisions or
  create placeholder decisions.

## Vocabulary

Use terms as defined in `CONTEXT.md`. If a required concept is absent, treat that
as either a vocabulary mismatch or a domain-modeling gap rather than silently
introducing a synonym.

## ADR conflicts

Surface conflicts with an existing ADR explicitly. Do not silently override an
accepted architectural decision.
