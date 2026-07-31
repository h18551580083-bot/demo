# Issue tracker: GitHub

Issues and PRDs for this repository live in GitHub Issues at
`h18551580083-bot/demo`. Use the `gh` CLI for issue operations and infer the
repository from the configured Git remote.

## Conventions

- Create issues with `gh issue create`.
- Read issues and comments with `gh issue view <number> --comments`.
- List issues with `gh issue list`, selecting labels and state explicitly.
- Add or remove labels with `gh issue edit`.
- Comment with `gh issue comment`; close with `gh issue close`.

## Pull requests as a triage surface

External pull requests are not a request or triage surface. Do not pull them into
the issue triage state machine.

## Skill routing

- When a skill says to publish to the issue tracker, create a GitHub Issue.
- When a skill says to fetch a ticket, read the corresponding GitHub Issue and its
  comments.
