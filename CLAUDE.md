# Repository Commit Convention

This repository should present commits as authored by Bowen Wang only.

## Commit Message Format

Use Conventional Commits:

```text
type(scope): short imperative summary
```

Examples:

```text
docs(v2): draft default-overlap rebaseline protocol
feat(v2): add default-overlap rebaseline runner
fix(report): correct case C summary wording
chore(repo): update commit convention
```

Allowed common types: `docs`, `feat`, `fix`, `chore`, `refactor`, `test`, `perf`, `ci`.

## Attribution Rules

- Do **not** add `Co-Authored-By` trailers for Claude, Anthropic, or any AI assistant.
- Do **not** mention Claude, Anthropic, or AI assistants in commit subjects, bodies, scopes, or trailers.
- Keep `git config user.name` and `git config user.email` set to Bowen's identity before committing.
- Before pushing, inspect recent commits:

```bash
git log --max-count=5 --format='%h %s%n%B' | rg -i 'claude|anthropic|co-authored-by'
```

The command should return no matches for new commits.

## Staging Rules

- Do not stage `.claude/settings.local.json`.
- Do not stage raw JSON, traces, logs, or generated experiment outputs unless the user explicitly asks.
- Keep commits focused: planning/docs changes, runner changes, and experiment results should be separate commits.
