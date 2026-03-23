# nexcore-brain

Working memory engine for Claude Code. Manages the persistence, versioning, and recovery of session context, artifacts, and implicit learning patterns.

## Intent
To provide AI agents with a durable, versioned "working memory" that persists across sessions. It enables context recovery, structured task tracking, and automatic pattern assimilation through implicit learning.

## T1 Grounding (Lex Primitiva)
Dominant Primitives:
- **π (Persistence)**: The primary primitive for durable storage of sessions and artifacts.
- **σ (Sequence)**: Versioning of artifacts (`.resolved.N`) and the time-series nature of session history.
- **μ (Mapping)**: Implicit knowledge mapping and project-to-session association.
- **ς (State)**: Management of active session state and "Bathroom Lock" coordination.

## Core Components
1. **Session Management**: Lifecycle control for AI agent working contexts.
2. **Artifact Versioning**: Secure, immutable snapshots of tasks, plans, and walkthroughs.
3. **Implicit Learning**: Background tracking of user/agent interaction patterns.
4. **Bathroom Lock**: Cryptographic coordination mechanism to prevent concurrent write collisions on artifacts.

## SOPs for Use
### Creating a new Session
```rust
use nexcore_brain::session::BrainSession;
let session = BrainSession::create_with_options(Some("my-project".into()), None, Some("Task description".into()))?;
```

### Saving and Resolving an Artifact
1. Save the current state: `session.save_artifact(&artifact)?;`
2. Create an immutable snapshot: `session.resolve_artifact("task.md")?;`
3. This creates `task.md.resolved` and `task.md.resolved.N` (where N is the new version).

## Directory Structure
Stored in `~/.claude/brain/`:
- `sessions/`: Individual session directories.
- `index.json`: Global session registry.
- `implicit/`: Persisted implicit learning patterns.

## License
Proprietary. Copyright (c) 2026 NexVigilant LLC. All Rights Reserved.
