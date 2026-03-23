//! # NexVigilant Core — Brain (Working Memory System)
//!
//! A unified brain system for Claude Code providing:
//!
//! - **Session Management**: Per-session working memory directories
//! - **Artifact Versioning**: Immutable `.resolved.N` snapshots with metadata
//! - **Code Tracking**: Content-addressable file tracking with SHA-256 hashes
//! - **Implicit Knowledge**: Learned preferences, patterns, and corrections
//! - **Health Metrics**: Comprehensive monitoring and telemetry snapshots
//!
//! ## Architecture
//!
//! ```text
//! ~/.claude/brain/                    # Working memory
//! ├── sessions/{session-id}/          # Per-session brain
//! │   ├── task.md                     # Current mutable state
//! │   ├── task.md.metadata.json       # Artifact metadata
//! │   ├── task.md.resolved            # Latest resolved state
//! │   └── task.md.resolved.N          # Immutable version N
//! ├── annotations/                    # View timestamps per session
//! ├── telemetry/                      # Health metrics snapshots
//! │   └── brain_snapshots.jsonl       # JSONL time-series data
//! └── index.json                      # Session registry
//!
//! ~/.claude/code_tracker/             # Content-addressable tracking
//! ├── active/{project}_{commit}/      # Active project snapshots
//! │   └── {hash}_{filename}           # Hash-prefixed copies
//! ├── history/                        # Archived snapshots
//! └── index.json                      # File registry
//!
//! ~/.claude/implicit/                 # Learned knowledge
//! ├── preferences.json                # User working style
//! ├── patterns.json                   # Detected patterns
//! ├── corrections.json                # Learned corrections
//! └── beliefs.json                    # PROJECT GROUNDED beliefs
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use nexcore_brain::{BrainSession, Artifact, ArtifactType};
//!
//! // Create a new session
//! let session = BrainSession::create()?;
//!
//! // Save an artifact
//! let artifact = Artifact::new("task.md", ArtifactType::Task, "# My Task\n...");
//! session.save_artifact(&artifact)?;
//!
//! // Resolve (create immutable snapshot)
//! let version = session.resolve_artifact("task.md")?;
//! println!("Created version {}", version);
//!
//! // Health monitoring
//! use nexcore_brain::metrics::{BrainHealth, BrainSnapshot};
//! let health = BrainHealth::collect()?;
//! let snapshot = BrainSnapshot::collect()?;
//! ```

#![forbid(unsafe_code)]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
#![warn(missing_docs)]

use nexcore_fs::dirs;

pub mod artifact;
pub mod config;
pub mod coordination;
pub mod db;
pub mod error;
pub mod grounding;
pub mod implicit;
pub mod metrics;
pub mod nmd_bridge;
pub mod pipeline;
pub mod recovery;
pub mod session;
pub mod synapse;
pub mod tombstone;
pub mod tracker;

/// Insight trait adapter — makes Brain a first-class InsightEngine implementor.
#[cfg(feature = "insight")]
pub mod insight_adapter;

pub use artifact::{Artifact, ArtifactMetadata, ArtifactType};
pub use coordination::{AgentId, CoordinationRegistry, LockDuration, LockStatus};
pub use error::{BrainError, Result};
pub use implicit::{
    Belief, BeliefGraph, BeliefImplication, Correction, EvidenceRef, EvidenceType,
    ImplicationStrength, ImplicitKnowledge, ImplicitStats, Pattern, Preference, T1Primitive,
    TrustAccumulator,
};
pub use metrics::{
    ArtifactMetrics, ArtifactSizeInfo, BrainHealth, BrainSnapshot, GrowthRate, SessionMetrics,
    ensure_telemetry_dir, largest_artifacts, load_snapshots, save_snapshot, telemetry_dir,
};
pub use pipeline::{Checkpoint, PipelineRun, PipelineState, RunStatus};
pub use recovery::{
    RecoveryResult, attempt_recovery, check_brain_availability, check_index_health,
    detect_partial_writes, rebuild_index_from_sessions, repair_partial_writes,
};
pub use session::{BrainSession, SessionEvent, SessionEventKind};
pub use synapse::{
    PersistentSynapseBank, SynapseBankStats, SynapseInfo, pattern_amplitude, reinforce_pattern,
};
pub use tracker::{CodeTracker, TrackedFile};

/// Default brain directory path
pub const BRAIN_DIR: &str = ".claude/brain";

/// Antigravity brain directory path
pub const ANTIGRAVITY_BRAIN_DIR: &str = ".gemini/antigravity/brain";

/// Default code tracker directory path
pub const TRACKER_DIR: &str = ".claude/code_tracker";

/// Default implicit knowledge directory path
pub const IMPLICIT_DIR: &str = ".claude/implicit";

/// Get the brain directory path
#[must_use]
pub fn brain_dir() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(BRAIN_DIR)
}

/// Get the Antigravity brain directory path
#[must_use]
pub fn brain_dir_antigravity() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(ANTIGRAVITY_BRAIN_DIR)
}

/// Get the code tracker directory path
#[must_use]
pub fn tracker_dir() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(TRACKER_DIR)
}

/// Get the implicit knowledge directory path
#[must_use]
pub fn implicit_dir() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(IMPLICIT_DIR)
}

/// Initialize all brain directories
///
/// Creates the required directory structure if it doesn't exist.
///
/// # Errors
///
/// Returns an error if directories cannot be created.
pub fn initialize_directories() -> Result<()> {
    let brain = brain_dir();
    let antigravity_brain = brain_dir_antigravity();
    let tracker = tracker_dir();
    let implicit = implicit_dir();

    // Brain directories
    std::fs::create_dir_all(brain.join("sessions"))?;
    std::fs::create_dir_all(brain.join("annotations"))?;
    std::fs::create_dir_all(brain.join("telemetry"))?;

    // Ensure antigravity brain dir exists (integration point)
    if !antigravity_brain.exists() {
        std::fs::create_dir_all(&antigravity_brain)?;
    }

    // Tracker directories
    std::fs::create_dir_all(tracker.join("active"))?;
    std::fs::create_dir_all(tracker.join("history"))?;

    // Implicit directory (single level)
    std::fs::create_dir_all(&implicit)?;

    // Initialize index files if they don't exist
    let brain_index = brain.join("index.json");
    if !brain_index.exists() {
        std::fs::write(&brain_index, "[]")?;
    }

    let tracker_index = tracker.join("index.json");
    if !tracker_index.exists() {
        std::fs::write(&tracker_index, "{}")?;
    }

    // Initialize implicit knowledge files
    let preferences = implicit.join("preferences.json");
    if !preferences.exists() {
        std::fs::write(&preferences, "{}")?;
    }

    let patterns = implicit.join("patterns.json");
    if !patterns.exists() {
        std::fs::write(&patterns, "[]")?;
    }

    let corrections = implicit.join("corrections.json");
    if !corrections.exists() {
        std::fs::write(&corrections, "[]")?;
    }

    // Initialize beliefs (PROJECT GROUNDED)
    let beliefs = implicit.join("beliefs.json");
    if !beliefs.exists() {
        std::fs::write(&beliefs, "[]")?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brain_dir_exists() {
        // Just verify the function doesn't panic
        let _ = brain_dir();
    }

    #[test]
    fn test_tracker_dir_exists() {
        let _ = tracker_dir();
    }

    #[test]
    fn test_implicit_dir_exists() {
        let _ = implicit_dir();
    }

    #[test]
    fn test_telemetry_dir_exists() {
        let _ = telemetry_dir();
    }
}
