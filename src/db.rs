//! SQLite dual-write layer for Brain working memory.
//!
//! Every Brain mutation (session create, artifact save/resolve, implicit save,
//! code track) writes to both the filesystem (JSON) and SQLite.
//!
//! **Read paths use DB-first with index.json fallback.** Session resolution
//! (`BrainSession::load`, `list_all`, `load_latest`) queries brain.db first
//! via direct `DbPool::open_default()`, falling back to index.json only if
//! the DB is unavailable. Changed in Phase 1 Foundation Repair (2026-03-08)
//! to fix dual-store divergence where 196/218 sessions were invisible to CLI.
//!
//! # Design
//!
//! - **Lazy global `DbPool`**: Used by write-mirror functions (sync_session,
//!   sync_artifact, etc.). Uses `tracing::warn` on failure — silent in CLI
//!   binaries without a tracing subscriber.
//! - **Direct `DbPool::open_default()`**: Used by read functions in
//!   `session.rs`. Bypasses the LazyLock so errors propagate via `eprintln!`.
//! - **Non-fatal writes**: If writes fail, we warn but don't fail the op.
//! - **Loud read failures**: Read failures emit to stderr before fallback.
//! - **INSERT OR IGNORE** for sessions (may already exist from migration).
//! - **Upsert** for artifacts and implicit knowledge (idempotent updates).

use nexcore_chrono::DateTime;
use nexcore_db::pool::DbPool;
use std::sync::{LazyLock, Mutex};

/// Global database pool, lazily initialized.
///
/// `None` if the database failed to open (logs warning, degrades gracefully).
static DB_POOL: LazyLock<Mutex<Option<DbPool>>> = LazyLock::new(|| match DbPool::open_default() {
    Ok(pool) => {
        tracing::info!(
            path = %pool.path().display(),
            "Brain SQLite backend initialized"
        );
        Mutex::new(Some(pool))
    }
    Err(e) => {
        tracing::warn!("Brain SQLite backend unavailable: {e}");
        Mutex::new(None)
    }
});

/// Execute a closure against the global database pool.
///
/// If the pool is unavailable or the lock is poisoned, returns `None`.
/// If the closure fails, logs a warning and returns `None`.
fn with_db<F, T>(op_name: &str, f: F) -> Option<T>
where
    F: FnOnce(&DbPool) -> nexcore_db::error::Result<T>,
{
    let guard = DB_POOL.lock().ok()?;
    let pool = guard.as_ref()?;

    match f(pool) {
        Ok(val) => Some(val),
        Err(e) => {
            tracing::warn!(op = op_name, error = %e, "Brain SQLite sync failed");
            None
        }
    }
}

// ========== Session Sync ==========

/// Mirror a session creation to SQLite.
///
/// Uses `INSERT OR IGNORE` — safe to call even if the session already exists
/// (e.g., from a prior migration).
pub fn sync_session(
    id: &str,
    project: Option<&str>,
    git_commit: Option<&str>,
    description: Option<&str>,
    created_at: DateTime,
) {
    with_db("sync_session", |pool| {
        pool.with_conn(|conn| {
            nexcore_db::sessions::insert_or_ignore(
                conn,
                &nexcore_db::sessions::SessionRow {
                    id: id.to_string(),
                    project: project.unwrap_or("").to_string(),
                    git_commit: git_commit.map(String::from),
                    description: description.unwrap_or("").to_string(),
                    created_at,
                },
            )
        })
    });
}

// ========== Artifact Sync ==========

/// Mirror an artifact save (upsert) to SQLite.
pub fn sync_artifact(
    session_id: &str,
    name: &str,
    artifact_type: &str,
    content: &str,
    summary: &str,
    current_version: u32,
    tags: &str,
    custom_meta: &str,
    created_at: DateTime,
    updated_at: DateTime,
) {
    with_db("sync_artifact", |pool| {
        pool.with_conn(|conn| {
            nexcore_db::artifacts::upsert(
                conn,
                &nexcore_db::artifacts::ArtifactRow {
                    id: None,
                    session_id: session_id.to_string(),
                    name: name.to_string(),
                    artifact_type: artifact_type.to_string(),
                    content: content.to_string(),
                    summary: summary.to_string(),
                    current_version,
                    tags: tags.to_string(),
                    custom_meta: custom_meta.to_string(),
                    created_at,
                    updated_at,
                },
            )
        })
    });
}

/// Mirror an artifact version (resolved snapshot) to SQLite.
pub fn sync_artifact_version(session_id: &str, artifact_name: &str, version: u32, content: &str) {
    with_db("sync_artifact_version", |pool| {
        pool.with_conn(|conn| {
            nexcore_db::artifacts::insert_version(
                conn,
                &nexcore_db::artifacts::ArtifactVersionRow {
                    id: None,
                    session_id: session_id.to_string(),
                    artifact_name: artifact_name.to_string(),
                    version,
                    content: content.to_string(),
                    resolved_at: DateTime::now(),
                },
            )
        })
    });
}

// ========== Implicit Knowledge Sync ==========

/// Mirror a preference to SQLite.
pub fn sync_preference(
    key: &str,
    value: &serde_json::Value,
    description: Option<&str>,
    confidence: f64,
    reinforcement_count: u32,
    updated_at: DateTime,
) {
    with_db("sync_preference", |pool| {
        let value_str = serde_json::to_string(value).unwrap_or_else(|_| "null".to_string());
        pool.with_conn(|conn| {
            nexcore_db::implicit::upsert_preference(
                conn,
                &nexcore_db::implicit::PreferenceRow {
                    key: key.to_string(),
                    value: value_str,
                    description: description.map(String::from),
                    confidence,
                    reinforcement_count,
                    updated_at,
                },
            )
        })
    });
}

/// Mirror a pattern to SQLite.
pub fn sync_pattern(
    id: &str,
    pattern_type: &str,
    description: &str,
    examples: &[String],
    detected_at: DateTime,
    updated_at: DateTime,
    confidence: f64,
    occurrence_count: u32,
    t1_grounding: Option<&str>,
) {
    with_db("sync_pattern", |pool| {
        let examples_json = serde_json::to_string(examples).unwrap_or_else(|_| "[]".to_string());
        pool.with_conn(|conn| {
            nexcore_db::implicit::upsert_pattern(
                conn,
                &nexcore_db::implicit::PatternRow {
                    id: id.to_string(),
                    pattern_type: pattern_type.to_string(),
                    description: description.to_string(),
                    examples: examples_json,
                    detected_at,
                    updated_at,
                    confidence,
                    occurrence_count,
                    t1_grounding: t1_grounding.map(String::from),
                },
            )
        })
    });
}

/// Mirror a correction to SQLite.
pub fn sync_correction(
    mistake: &str,
    correction: &str,
    context: Option<&str>,
    learned_at: DateTime,
    application_count: u32,
) {
    with_db("sync_correction", |pool| {
        pool.with_conn(|conn| {
            nexcore_db::implicit::insert_correction(
                conn,
                &nexcore_db::implicit::CorrectionRow {
                    id: None,
                    mistake: mistake.to_string(),
                    correction: correction.to_string(),
                    context: context.map(String::from),
                    learned_at,
                    application_count,
                },
            )
        })
    });
}

/// Mirror a belief to SQLite.
pub fn sync_belief(
    id: &str,
    proposition: &str,
    category: &str,
    confidence: f64,
    evidence_json: &str,
    t1_grounding: Option<&str>,
    formed_at: DateTime,
    updated_at: DateTime,
    validation_count: u32,
    user_confirmed: bool,
) {
    with_db("sync_belief", |pool| {
        pool.with_conn(|conn| {
            nexcore_db::implicit::upsert_belief(
                conn,
                &nexcore_db::implicit::BeliefRow {
                    id: id.to_string(),
                    proposition: proposition.to_string(),
                    category: category.to_string(),
                    confidence,
                    evidence: evidence_json.to_string(),
                    t1_grounding: t1_grounding.map(String::from),
                    formed_at,
                    updated_at,
                    validation_count,
                    user_confirmed,
                },
            )
        })
    });
}

/// Mirror a trust accumulator to SQLite.
pub fn sync_trust(
    domain: &str,
    demonstrations: u32,
    failures: u32,
    created_at: DateTime,
    updated_at: DateTime,
    t1_grounding: Option<&str>,
) {
    with_db("sync_trust", |pool| {
        pool.with_conn(|conn| {
            nexcore_db::implicit::upsert_trust(
                conn,
                &nexcore_db::implicit::TrustRow {
                    domain: domain.to_string(),
                    demonstrations,
                    failures,
                    created_at,
                    updated_at,
                    t1_grounding: t1_grounding.map(String::from),
                },
            )
        })
    });
}

/// Mirror a belief implication to SQLite.
pub fn sync_implication(
    from_belief: &str,
    to_belief: &str,
    strength: &str,
    established_at: DateTime,
) {
    with_db("sync_implication", |pool| {
        pool.with_conn(|conn| {
            nexcore_db::implicit::insert_implication(
                conn,
                &nexcore_db::implicit::ImplicationRow {
                    from_belief: from_belief.to_string(),
                    to_belief: to_belief.to_string(),
                    strength: strength.to_string(),
                    established_at,
                },
            )
        })
    });
}

// ========== Code Tracker Sync ==========

/// Mirror a tracked file to SQLite.
pub fn sync_tracked_file(
    project: &str,
    file_path: &str,
    content_hash: &str,
    file_size: u64,
    tracked_at: DateTime,
    mtime: DateTime,
) {
    with_db("sync_tracked_file", |pool| {
        pool.with_conn(|conn| {
            nexcore_db::tracker::upsert(
                conn,
                &nexcore_db::tracker::TrackedFileRow {
                    id: None,
                    project: project.to_string(),
                    file_path: file_path.to_string(),
                    content_hash: content_hash.to_string(),
                    file_size,
                    tracked_at,
                    mtime,
                },
            )
        })
    });
}

// ========== Decision Sync ==========

/// Mirror a decision audit entry to SQLite.
///
/// Called from the decision-journal hook's insert binary.
pub fn sync_decision(
    timestamp: DateTime,
    session_id: &str,
    tool: &str,
    action: &str,
    target: &str,
    risk_level: &str,
    reversible: bool,
) {
    with_db("sync_decision", |pool| {
        pool.with_conn(|conn| {
            nexcore_db::decisions::insert(
                conn,
                &nexcore_db::decisions::DecisionRow {
                    id: None,
                    timestamp,
                    session_id: session_id.to_string(),
                    tool: tool.to_string(),
                    action: action.to_string(),
                    target: target.to_string(),
                    risk_level: risk_level.to_string(),
                    reversible,
                },
            )
        })
    });
}

// ========== Read-from-DB Methods ==========

/// Get the global DbPool (if available).
///
/// Useful for direct queries from MCP tools or other consumers.
pub fn get_pool() -> Option<DbPool> {
    let guard = DB_POOL.lock().ok()?;
    guard.clone()
}
