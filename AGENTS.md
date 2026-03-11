## Python, Typing & Docs

* Support Python >=3.8 at runtime, but deferred typing can support 3.14+.
* Avoid typing code that adds runtime overhead as a guideline, but not a hard rule.
* Prefer putting typing code in if `typing.TYPE_CHECKING` blocks.

## Linting & Style

* Follow PEP 8; mark exceptions with `# NOQA`.
* Use Google-style docstrings with runnable examples.
* Use xdoctest style doctests.
* Use comments to make the code intent more readable for humans and machines.

## Developer journal
Keep a running journal at `dev/journals/<agent_name>.md` (e.g.
`dev/journals/codex.md`) to capture the story of the work (decisions, progress,
challenges). This is not a changelog.  Write at a high level for future
maintainers: enough context for someone to pick up where you left off.

- Format: Each entry starts with `## YYYY-MM-DD HH:MM:SS -ZZZZ` (local time).
- Must include: what you were working on, a substantive entry about your state of mind / reflections, uncertainties/risks, tradeoffs, what might break, what you're confident about.
- Include what the user requested, if you think the user might be wrong or have misconceptions, where you agree or where you might challenge or change what the request is based on your understanding of what the code base is supposed to be. 
- Include what happened, rationale, testing notes, next steps, open questions.
- Rules: Prefer append-only. You may edit only the most recent entry *during the same session* (use timestamp + context to judge); never modify the timestamp line; once a new session starts, create a new entry. Never modify older entries. Avoid large diffs; reference files/modules/issues instead.
