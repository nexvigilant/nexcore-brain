# AI Guidance — nexcore-brain

Working memory and artifact management for persistent AI context.

## Use When
- Saving critical session state (task lists, implementation plans).
- Recovering context after an agent restart or session handoff.
- Versioning large text files that require manual or automated "resolution" (snapshots).
- Accessing implicit learning patterns derived from previous interactions.

## Grounding Patterns
- **Persistence (π)**: Use `BrainSession` as the root of all durable state.
- **Locking**: Respect the "Bathroom Lock" pattern; always acquire a lock before multi-file operations.
- **T1 Primitives**:
  - `π + σ`: Use for any time-series or versioned artifact.
  - `ς + μ`: Use for stateful coordination of project contexts.

## Maintenance SOPs
- **Artifact Names**: Prefer standard names: `task.md`, `plan.md`, `walkthrough.md`.
- **Resolution**: Call `resolve_artifact` only after a major sub-task is verified green.
- **Dual-Write**: Note that sessions and artifacts are dual-written to disk (JSON/Files) and SQLite (`brain.db`).

## Key Entry Points
- `src/session.rs`: Root lifecycle management.
- `src/artifact.rs`: Versioning and metadata logic.
- `src/implicit.rs`: The implicit learning engine.
