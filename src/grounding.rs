//! # Lex Primitiva Grounding for nexcore-brain
//!
//! GroundsTo implementations for all nexcore-brain public types.
//! The Brain crate handles working memory: sessions, artifacts, code tracking,
//! implicit learning (preferences, patterns, corrections, beliefs), pipeline
//! state, coordination, recovery, synapse integration, and configuration.
//!
//! ## Type Grounding Table
//!
//! | Type | Primitives | Dominant | Tier | Rationale |
//! |------|-----------|----------|------|-----------|
//! | ArtifactType | Σ | Σ | T1 | Sum enum: exactly one variant |
//! | ArtifactMetadata | ς π N | ς | T2-P | Encapsulated state with version counter and timestamps |
//! | Artifact | ς π σ | ς | T2-P | Stateful document with versioned content |
//! | SessionEntry | ς π λ | ς | T2-P | Session identity with timestamps and location |
//! | BrainSession | ς π λ ∂ | ς | T2-C | Session with directory boundary and persistence |
//! | TrackerIndex | μ ς | μ | T2-P | Maps project keys to snapshots |
//! | ProjectSnapshot | π ς μ | π | T2-P | Persisted snapshot of project state |
//! | TrackedFile | π λ κ | π | T2-P | Content-addressed file with hash comparison |
//! | CodeTracker | ς π μ λ | ς | T2-C | Stateful tracker managing file mappings |
//! | T1Primitive | Σ | Σ | T1 | Sum enum of primitive variants |
//! | Preference | ς ν N | ς | T2-P | Stateful preference with reinforcement count |
//! | Pattern | ς ν ∃ | ς | T2-P | Detected pattern with frequency and decay |
//! | Correction | μ → | μ | T2-P | Maps mistake to correction (causality) |
//! | EvidenceType | Σ | Σ | T1 | Sum enum of evidence categories |
//! | EvidenceRef | ∃ N → | ∃ | T2-P | Evidence existence with weight and causality |
//! | Belief | ς ∃ N → ν | ς | T2-C | Stateful proposition with evidence and decay |
//! | TrustAccumulator | ς N ν | ς | T2-P | Accumulated trust with success/failure counts |
//! | ImplicationStrength | Σ | Σ | T1 | Sum enum: Strong/Moderate/Weak |
//! | BeliefImplication | → ς | → | T2-P | Causal edge between beliefs |
//! | BeliefGraph | ρ → ς | ρ | T2-P | Recursive graph of belief implications |
//! | ImplicitKnowledge | ς π μ σ ∃ ρ | ς | T3 | Full domain knowledge store |
//! | ImplicitStats | N Σ | N | T2-P | Numeric summary statistics |
//! | ArtifactMetrics | N μ | N | T2-P | Numeric counts with type mapping |
//! | SessionMetrics | N ∂ | N | T2-P | Numeric counts with active/total boundary |
//! | BrainSnapshot | π N ς | π | T2-P | Persisted point-in-time metrics |
//! | ArtifactSizeInfo | N λ π | N | T2-P | Size measurement with location |
//! | BrainHealth | ς N ∂ λ σ | ς | T2-C | Health state with warnings sequence |
//! | GrowthRate | N ν ∂ | N | T2-P | Numeric rate with frequency over period |
//! | BrainError | Σ ∂ | Σ | T2-P | Sum error type with boundary semantics |
//! | LockStatus | Σ | Σ | T1 | Sum enum: Vacant/Occupied |
//! | AgentId | ς | ς | T1 | Newtype identity wrapper |
//! | LockDuration | N | N | T1 | Newtype quantity wrapper |
//! | FileLock | ς ∂ λ N | ς | T2-C | Lock state with path and TTL |
//! | CoordinationRegistry | ς μ ∂ π | ς | T2-C | Coordination state managing lock map |
//! | PipelineRun | ς σ π | ς | T2-P | Run with sequential checkpoints |
//! | RunStatus | Σ | Σ | T1 | Sum enum of run states |
//! | Checkpoint | ς π | ς | T2-P | Named state snapshot at a point in time |
//! | PipelineState | ς σ π | ς | T2-P | Pipeline with ordered runs |
//! | RecoveryResult | ∃ N ∂ | ∃ | T2-P | Recovery outcome with counts |
//! | PersistentSynapseBank | ς π ν Σ | ς | T2-C | Persistent synapse store with amplitude |
//! | SynapseBankStats | N Σ | N | T2-P | Numeric summary of synapse bank |
//! | SynapseInfo | ς N ∃ | ς | T2-P | Synapse state with amplitude and status |
//! | BrainConfig | ς ∂ N | ς | T2-P | Configuration state with boundary limits |

use nexcore_lex_primitiva::grounding::GroundsTo;
use nexcore_lex_primitiva::primitiva::{LexPrimitiva, PrimitiveComposition};
use nexcore_lex_primitiva::state_mode::StateMode;

use crate::artifact::{Artifact, ArtifactMetadata, ArtifactType};
use crate::config::BrainConfig;
use crate::coordination::{AgentId, CoordinationRegistry, FileLock, LockDuration, LockStatus};
use crate::error::BrainError;
use crate::implicit::{
    Belief, BeliefGraph, BeliefImplication, Correction, EvidenceRef, EvidenceType,
    ImplicationStrength, ImplicitKnowledge, ImplicitStats, Pattern, Preference, T1Primitive,
    TrustAccumulator,
};
use crate::metrics::{
    ArtifactMetrics, ArtifactSizeInfo, BrainHealth, BrainSnapshot, GrowthRate, SessionMetrics,
};
use crate::pipeline::{Checkpoint, PipelineRun, PipelineState, RunStatus};
use crate::recovery::RecoveryResult;
use crate::session::{BrainSession, SessionEntry};
use crate::synapse::{PersistentSynapseBank, SynapseBankStats, SynapseInfo};
use crate::tracker::{CodeTracker, ProjectSnapshot, TrackedFile, TrackerIndex};

// ============================================================================
// T1 Universal (1 unique primitive)
// ============================================================================

/// ArtifactType: Sum enum with exactly one variant selected from 7 options.
/// Tier: T1Universal. Dominant: Σ Sum.
/// WHY: A pure sum type -- each artifact is classified as exactly one of N
/// categories (Task, Plan, Walkthrough, etc). No other primitives involved.
impl GroundsTo for ArtifactType {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Sum, // Σ -- exclusive disjunction of 7 variants
        ])
        .with_dominant(LexPrimitiva::Sum, 1.0)
    }
}

/// T1Primitive: Sum enum representing 15 of the universal primitives.
/// Tier: T1Universal. Dominant: Σ Sum.
/// WHY: Each value IS exactly one of the fundamental primitives. Meta-recursive
/// (a sum of sums) but structurally it is a pure sum type (15 variants).
impl GroundsTo for T1Primitive {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Sum, // Σ -- one-of-15 variant selection
        ])
        .with_dominant(LexPrimitiva::Sum, 1.0)
    }
}

/// EvidenceType: Sum enum classifying evidence sources.
/// Tier: T1Universal. Dominant: Σ Sum.
/// WHY: Exactly one of 6 variants (Observation, TestResult, UserFeedback,
/// Inference, Authority, Prior). Pure classification.
impl GroundsTo for EvidenceType {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Sum, // Σ -- one-of-6 classification
        ])
        .with_dominant(LexPrimitiva::Sum, 1.0)
    }
}

/// ImplicationStrength: Sum enum for edge weights (Strong/Moderate/Weak).
/// Tier: T1Universal. Dominant: Σ Sum.
/// WHY: Three-variant sum type used as edge labels in belief graph.
impl GroundsTo for ImplicationStrength {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Sum, // Σ -- one-of-3 strength levels
        ])
        .with_dominant(LexPrimitiva::Sum, 1.0)
    }
}

/// LockStatus: Sum enum (Vacant/Occupied).
/// Tier: T1Universal. Dominant: Σ Sum.
/// WHY: Binary sum type representing lock availability.
impl GroundsTo for LockStatus {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Sum, // Σ -- binary Vacant/Occupied
        ])
        .with_dominant(LexPrimitiva::Sum, 1.0)
    }
}

/// RunStatus: Sum enum (Running/Completed/Failed/Cancelled).
/// Tier: T1Universal. Dominant: Σ Sum.
/// WHY: Four-variant sum type for pipeline run lifecycle.
impl GroundsTo for RunStatus {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Sum, // Σ -- one-of-4 run states
        ])
        .with_dominant(LexPrimitiva::Sum, 1.0)
    }
}

/// AgentId: Newtype wrapper over String for agent identity.
/// Tier: T1Universal. Dominant: ς State.
/// WHY: Pure identity wrapper -- encapsulates agent name as state.
impl GroundsTo for AgentId {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State, // ς -- encapsulated identity
        ])
        .with_dominant(LexPrimitiva::State, 1.0)
        .with_state_mode(StateMode::Mutable)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Mutable)
    }
}

/// LockDuration: Newtype wrapper over u64 for lock TTL in seconds.
/// Tier: T1Universal. Dominant: N Quantity.
/// WHY: Pure numeric quantity -- duration measured in seconds.
impl GroundsTo for LockDuration {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Quantity, // N -- numeric seconds
        ])
        .with_dominant(LexPrimitiva::Quantity, 1.0)
    }
}

// ============================================================================
// T2-P (2-3 unique primitives)
// ============================================================================

/// ArtifactMetadata: Encapsulated metadata state with version counter and timestamps.
/// Tier: T2Primitive. Dominant: ς State.
/// WHY: Tracks mutable version counter (N), created/updated timestamps (π),
/// and artifact type classification -- but its primary role is encapsulating
/// the current state of an artifact's metadata.
impl GroundsTo for ArtifactMetadata {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,       // ς -- mutable metadata state
            LexPrimitiva::Persistence, // π -- timestamps survive across sessions
            LexPrimitiva::Quantity,    // N -- version counter
        ])
        .with_dominant(LexPrimitiva::State, 0.85)
        .with_state_mode(StateMode::Mutable)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Mutable)
    }
}

/// Artifact: A versioned document with mutable content.
/// Tier: T2Primitive. Dominant: ς State.
/// WHY: Content is mutable state, versions form a sequence, timestamps
/// provide persistence. The artifact IS state -- you update_content on it.
impl GroundsTo for Artifact {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,       // ς -- mutable content
            LexPrimitiva::Persistence, // π -- survives across sessions via disk
            LexPrimitiva::Sequence,    // σ -- resolved versions form ordered sequence
        ])
        .with_dominant(LexPrimitiva::State, 0.85)
        .with_state_mode(StateMode::Mutable)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Mutable)
    }
}

/// SessionEntry: Registry entry recording a session's identity and metadata.
/// Tier: T2Primitive. Dominant: ς State.
/// WHY: Encapsulates session identity with optional project/commit fields.
/// Timestamps provide persistence, project path provides location context.
impl GroundsTo for SessionEntry {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,       // ς -- encapsulated session identity
            LexPrimitiva::Persistence, // π -- created_at timestamp
            LexPrimitiva::Location,    // λ -- project path context
        ])
        .with_dominant(LexPrimitiva::State, 0.85)
        .with_state_mode(StateMode::Accumulated)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Accumulated)
    }
}

/// TrackerIndex: Maps project snapshot keys to their tracked file sets.
/// Tier: T2Primitive. Dominant: μ Mapping.
/// WHY: Core structure is HashMap<String, ProjectSnapshot> -- a mapping
/// from project keys to snapshot state.
impl GroundsTo for TrackerIndex {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Mapping, // μ -- key-value project map
            LexPrimitiva::State,   // ς -- mutable registry state
        ])
        .with_dominant(LexPrimitiva::Mapping, 0.90)
        .with_state_mode(StateMode::Mutable)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Mutable)
    }
}

/// ProjectSnapshot: A persisted snapshot of tracked files for a project.
/// Tier: T2Primitive. Dominant: π Persistence.
/// WHY: The snapshot captures project state at a point in time and persists
/// it to disk. Contains file mapping (μ) and mutable state (ς).
impl GroundsTo for ProjectSnapshot {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Persistence, // π -- snapshot survives across sessions
            LexPrimitiva::State,       // ς -- captured project state
            LexPrimitiva::Mapping,     // μ -- file path to TrackedFile map
        ])
        .with_dominant(LexPrimitiva::Persistence, 0.85)
        .with_state_mode(StateMode::Accumulated)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Accumulated)
    }
}

/// TrackedFile: A content-addressed file with SHA-256 hash for change detection.
/// Tier: T2Primitive. Dominant: π Persistence.
/// WHY: The file is persisted with its content hash. Comparison (κ) arises
/// from content-addressed hashing (has_changed). Location (λ) from the path.
impl GroundsTo for TrackedFile {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Persistence, // π -- file persisted to disk
            LexPrimitiva::Location,    // λ -- file path
            LexPrimitiva::Comparison,  // κ -- content hash comparison
        ])
        .with_dominant(LexPrimitiva::Persistence, 0.85)
    }
}

/// Preference: A learned user preference with reinforcement tracking.
/// Tier: T2Primitive. Dominant: ς State.
/// WHY: Preference is mutable state (reinforce/weaken mutate confidence).
/// Frequency (ν) from reinforcement_count, Quantity (N) from confidence.
impl GroundsTo for Preference {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,     // ς -- mutable confidence state
            LexPrimitiva::Frequency, // ν -- reinforcement count
            LexPrimitiva::Quantity,  // N -- confidence value 0.0-1.0
        ])
        .with_dominant(LexPrimitiva::State, 0.85)
        .with_state_mode(StateMode::Mutable)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Mutable)
    }
}

/// Pattern: A detected behavioral pattern with confidence decay.
/// Tier: T2Primitive. Dominant: ς State.
/// WHY: Patterns are mutable state (add_example, set_grounding). Frequency (ν)
/// from occurrence_count. Existence (∃) because detection validates pattern presence.
impl GroundsTo for Pattern {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,     // ς -- mutable pattern state
            LexPrimitiva::Frequency, // ν -- occurrence count
            LexPrimitiva::Existence, // ∃ -- pattern detection validates existence
        ])
        .with_dominant(LexPrimitiva::State, 0.85)
        .with_state_mode(StateMode::Mutable)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Mutable)
    }
}

/// Correction: Maps a mistake to its correction (causal transform).
/// Tier: T2Primitive. Dominant: μ Mapping.
/// WHY: Fundamentally a mapping from mistake -> correction. Causality (→)
/// because the mistake causes the need for correction.
impl GroundsTo for Correction {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Mapping,   // μ -- mistake -> correction transform
            LexPrimitiva::Causality, // → -- mistake causes correction need
        ])
        .with_dominant(LexPrimitiva::Mapping, 0.85)
    }
}

/// EvidenceRef: A reference to evidence with weight and source.
/// Tier: T2Primitive. Dominant: ∃ Existence.
/// WHY: Evidence EXISTS as a record supporting/contradicting a belief.
/// Weight is a quantity (N), and evidence implies causality (→).
impl GroundsTo for EvidenceRef {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Existence, // ∃ -- evidence record exists
            LexPrimitiva::Quantity,  // N -- weight -1.0 to 1.0
            LexPrimitiva::Causality, // → -- evidence supports/contradicts belief
        ])
        .with_dominant(LexPrimitiva::Existence, 0.85)
    }
}

/// TrustAccumulator: Accumulated trust score from demonstrated competence.
/// Tier: T2Primitive. Dominant: ς State.
/// WHY: Mutable state tracking demonstrations and failures. Quantity (N)
/// for the score, Frequency (ν) for interaction counts.
impl GroundsTo for TrustAccumulator {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,     // ς -- mutable trust state
            LexPrimitiva::Quantity,  // N -- score = demos / (demos + failures + 1)
            LexPrimitiva::Frequency, // ν -- interaction count
        ])
        .with_dominant(LexPrimitiva::State, 0.85)
        .with_state_mode(StateMode::Accumulated)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Accumulated)
    }
}

/// BeliefImplication: A causal edge from one belief to another.
/// Tier: T2Primitive. Dominant: → Causality.
/// WHY: An implication IS causality -- belief A implies belief B.
/// State (ς) from the strength classification.
impl GroundsTo for BeliefImplication {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Causality, // → -- antecedent implies consequent
            LexPrimitiva::State,     // ς -- strength as encapsulated state
        ])
        .with_dominant(LexPrimitiva::Causality, 0.90)
        .with_state_mode(StateMode::Modal)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Modal)
    }
}

/// BeliefGraph: Directed graph of belief implications with cycle detection.
/// Tier: T2Primitive. Dominant: ρ Recursion.
/// WHY: Graph traversal (DFS reachability) is recursive. Causality (→)
/// from implication edges. State (ς) from the mutable edge collection.
impl GroundsTo for BeliefGraph {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Recursion, // ρ -- DFS graph traversal
            LexPrimitiva::Causality, // → -- directed implication edges
            LexPrimitiva::State,     // ς -- mutable graph state
        ])
        .with_dominant(LexPrimitiva::Recursion, 0.85)
        .with_state_mode(StateMode::Mutable)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Mutable)
    }
}

/// ImplicitStats: Numeric summary statistics of implicit knowledge.
/// Tier: T2Primitive. Dominant: N Quantity.
/// WHY: Pure numeric counts (total_preferences, total_patterns, etc).
/// Sum (Σ) because it aggregates across categories.
impl GroundsTo for ImplicitStats {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Quantity, // N -- count values
            LexPrimitiva::Sum,      // Σ -- aggregated from multiple categories
        ])
        .with_dominant(LexPrimitiva::Quantity, 0.90)
    }
}

/// ArtifactMetrics: Numeric artifact counts with type breakdown.
/// Tier: T2Primitive. Dominant: N Quantity.
/// WHY: total count is a quantity, by_type is a mapping from type to count.
impl GroundsTo for ArtifactMetrics {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Quantity, // N -- total count
            LexPrimitiva::Mapping,  // μ -- by_type HashMap<String, usize>
        ])
        .with_dominant(LexPrimitiva::Quantity, 0.85)
    }
}

/// SessionMetrics: Active vs total session counts.
/// Tier: T2Primitive. Dominant: N Quantity.
/// WHY: Numeric counts with boundary between active and inactive.
impl GroundsTo for SessionMetrics {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Quantity, // N -- active and total counts
            LexPrimitiva::Boundary, // ∂ -- 24-hour active/inactive threshold
        ])
        .with_dominant(LexPrimitiva::Quantity, 0.88)
    }
}

/// BrainSnapshot: Point-in-time telemetry snapshot persisted to JSONL.
/// Tier: T2Primitive. Dominant: π Persistence.
/// WHY: A snapshot IS persistence -- capturing state at a moment in time
/// for later analysis. Contains quantities (N) and state (ς).
impl GroundsTo for BrainSnapshot {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Persistence, // π -- persisted to brain_snapshots.jsonl
            LexPrimitiva::Quantity,    // N -- metrics values
            LexPrimitiva::State,       // ς -- captured system state
        ])
        .with_dominant(LexPrimitiva::Persistence, 0.85)
        .with_state_mode(StateMode::Accumulated)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Accumulated)
    }
}

/// ArtifactSizeInfo: Size measurement with location context.
/// Tier: T2Primitive. Dominant: N Quantity.
/// WHY: Primary data is size_bytes (quantity). Location (λ) from session_id
/// and artifact name. Persistence (π) from modified_at timestamp.
impl GroundsTo for ArtifactSizeInfo {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Quantity,    // N -- size_bytes
            LexPrimitiva::Location,    // λ -- session_id + name path
            LexPrimitiva::Persistence, // π -- modified_at timestamp
        ])
        .with_dominant(LexPrimitiva::Quantity, 0.85)
    }
}

/// GrowthRate: Rate of change over a time period.
/// Tier: T2Primitive. Dominant: N Quantity.
/// WHY: All fields are numeric rates and counts. Frequency (ν) from the
/// per-day rates. Boundary (∂) from the period start/end.
impl GroundsTo for GrowthRate {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Quantity,  // N -- rate values
            LexPrimitiva::Frequency, // ν -- per-day calculation
            LexPrimitiva::Boundary,  // ∂ -- period_start/period_end
        ])
        .with_dominant(LexPrimitiva::Quantity, 0.85)
    }
}

/// BrainError: Sum error type with boundary semantics for error classification.
/// Tier: T2Primitive. Dominant: Σ Sum.
/// WHY: Error enum with 16 variants -- a sum type. Boundary (∂) because errors
/// represent transitions across the valid/invalid boundary.
impl GroundsTo for BrainError {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Sum,      // Σ -- 16-variant error enum
            LexPrimitiva::Boundary, // ∂ -- error boundary crossing
        ])
        .with_dominant(LexPrimitiva::Sum, 0.90)
    }
}

/// PipelineRun: A run with sequential checkpoints and lifecycle state.
/// Tier: T2Primitive. Dominant: ς State.
/// WHY: Run has mutable status (Running -> Completed/Failed). Checkpoints
/// form a sequence (σ). Completed timestamp provides persistence (π).
impl GroundsTo for PipelineRun {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,       // ς -- mutable run status
            LexPrimitiva::Sequence,    // σ -- ordered checkpoints
            LexPrimitiva::Persistence, // π -- start/completion timestamps
        ])
        .with_dominant(LexPrimitiva::State, 0.85)
        .with_state_mode(StateMode::Modal)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Modal)
    }
}

/// Checkpoint: A named state snapshot at a point in time within a run.
/// Tier: T2Primitive. Dominant: ς State.
/// WHY: A checkpoint captures state (arbitrary JSON data) at a moment.
/// Persistence (π) from the timestamp.
impl GroundsTo for Checkpoint {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,       // ς -- captured checkpoint state
            LexPrimitiva::Persistence, // π -- timestamp
        ])
        .with_dominant(LexPrimitiva::State, 0.88)
        .with_state_mode(StateMode::Accumulated)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Accumulated)
    }
}

/// PipelineState: Named pipeline with ordered runs and last checkpoint.
/// Tier: T2Primitive. Dominant: ς State.
/// WHY: Mutable pipeline state managing a sequence of runs. Sequence (σ)
/// from ordered runs. Persistence (π) from file-backed save/load.
impl GroundsTo for PipelineState {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,       // ς -- mutable pipeline state
            LexPrimitiva::Sequence,    // σ -- ordered runs
            LexPrimitiva::Persistence, // π -- save/load to file
        ])
        .with_dominant(LexPrimitiva::State, 0.85)
        .with_state_mode(StateMode::Mutable)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Mutable)
    }
}

/// RecoveryResult: Outcome of a recovery operation with success/failure counts.
/// Tier: T2Primitive. Dominant: ∃ Existence.
/// WHY: Recovery validates existence of recovered items. Quantity (N) for
/// recovered/failed counts. Boundary (∂) for the success/failure threshold.
impl GroundsTo for RecoveryResult {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Existence, // ∃ -- recovery validates item existence
            LexPrimitiva::Quantity,  // N -- recovered_count, failed_count
            LexPrimitiva::Boundary,  // ∂ -- success/failure threshold
        ])
        .with_dominant(LexPrimitiva::Existence, 0.85)
    }
}

/// SynapseBankStats: Numeric summary of the synapse bank.
/// Tier: T2Primitive. Dominant: N Quantity.
/// WHY: All fields are numeric counts and averages. Sum (Σ) because
/// it aggregates pattern/preference/belief synapse counts.
impl GroundsTo for SynapseBankStats {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::Quantity, // N -- counts and amplitude values
            LexPrimitiva::Sum,      // Σ -- aggregated from categories
        ])
        .with_dominant(LexPrimitiva::Quantity, 0.90)
    }
}

/// SynapseInfo: Information about a single synapse's state.
/// Tier: T2Primitive. Dominant: ς State.
/// WHY: Captures synapse state (amplitude, status, persistence flag).
/// Quantity (N) from amplitude and observation_count. Existence (∃)
/// from is_persistent flag.
impl GroundsTo for SynapseInfo {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,     // ς -- synapse status and persistence flag
            LexPrimitiva::Quantity,  // N -- amplitude and observation_count
            LexPrimitiva::Existence, // ∃ -- is_persistent validates survival
        ])
        .with_dominant(LexPrimitiva::State, 0.85)
        .with_state_mode(StateMode::Mutable)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Mutable)
    }
}

/// BrainConfig: Configuration state with boundary limits and timeouts.
/// Tier: T2Primitive. Dominant: ς State.
/// WHY: Configuration IS encapsulated state. Boundary (∂) from threshold
/// limits (max_artifact_size, max_sessions). Quantity (N) from timeout values.
impl GroundsTo for BrainConfig {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,    // ς -- encapsulated configuration
            LexPrimitiva::Boundary, // ∂ -- limits and thresholds
            LexPrimitiva::Quantity, // N -- timeout durations and size limits
        ])
        .with_dominant(LexPrimitiva::State, 0.85)
        .with_state_mode(StateMode::Mutable)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Mutable)
    }
}

// ============================================================================
// T2-C (4-5 unique primitives)
// ============================================================================

/// BrainSession: Working memory session with directory management and persistence.
/// Tier: T2Composite. Dominant: ς State.
/// WHY: Session manages mutable state (artifacts), persists to directory (π),
/// operates within a filesystem boundary (∂), and has path location (λ).
impl GroundsTo for BrainSession {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,       // ς -- mutable session context
            LexPrimitiva::Persistence, // π -- file-backed session directory
            LexPrimitiva::Location,    // λ -- session_dir path
            LexPrimitiva::Boundary,    // ∂ -- BathroomLock write protection
        ])
        .with_dominant(LexPrimitiva::State, 0.85)
        .with_state_mode(StateMode::Mutable)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Mutable)
    }
}

/// CodeTracker: Stateful tracker managing content-addressed file snapshots.
/// Tier: T2Composite. Dominant: ς State.
/// WHY: Manages mutable file tracking state, persists snapshots to disk (π),
/// maps paths to tracked files (μ), and operates on filesystem paths (λ).
impl GroundsTo for CodeTracker {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,       // ς -- mutable tracker state
            LexPrimitiva::Persistence, // π -- file-backed tracker index
            LexPrimitiva::Mapping,     // μ -- path -> TrackedFile mapping
            LexPrimitiva::Location,    // λ -- tracker_path and file paths
        ])
        .with_dominant(LexPrimitiva::State, 0.85)
        .with_state_mode(StateMode::Mutable)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Mutable)
    }
}

/// Belief: A propositional attitude with evidence-based confidence and decay.
/// Tier: T2Composite. Dominant: ς State.
/// WHY: Belief has mutable state (add_evidence, confirm, reject). Existence (∃)
/// because beliefs validate/test propositions. Quantity (N) for confidence.
/// Causality (→) from evidence-to-belief relationship. Frequency (ν) from decay.
impl GroundsTo for Belief {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,     // ς -- mutable belief state
            LexPrimitiva::Existence, // ∃ -- proposition existence validation
            LexPrimitiva::Quantity,  // N -- confidence 0.0-1.0
            LexPrimitiva::Causality, // → -- evidence supports/contradicts
            LexPrimitiva::Frequency, // ν -- time-based decay rate
        ])
        .with_dominant(LexPrimitiva::State, 0.82)
        .with_state_mode(StateMode::Mutable)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Mutable)
    }
}

/// FileLock: Lock state tracking with path, agent, TTL and expiry.
/// Tier: T2Composite. Dominant: ς State.
/// WHY: Lock IS state (acquired/expired). Boundary (∂) from TTL expiry check.
/// Location (λ) from file path. Quantity (N) from TTL duration.
impl GroundsTo for FileLock {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,    // ς -- lock occupancy state
            LexPrimitiva::Boundary, // ∂ -- TTL expiry boundary
            LexPrimitiva::Location, // λ -- locked file path
            LexPrimitiva::Quantity, // N -- TTL in seconds
        ])
        .with_dominant(LexPrimitiva::State, 0.85)
        .with_state_mode(StateMode::Modal)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Modal)
    }
}

/// CoordinationRegistry: Multi-agent lock coordination with persistence.
/// Tier: T2Composite. Dominant: ς State.
/// WHY: Registry manages mutable lock state across agents. Mapping (μ) from
/// path to FileLock. Boundary (∂) from access control. Persistence (π) from
/// file-backed atomic saves.
impl GroundsTo for CoordinationRegistry {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,       // ς -- mutable lock registry
            LexPrimitiva::Mapping,     // μ -- path -> FileLock map
            LexPrimitiva::Boundary,    // ∂ -- lock access control
            LexPrimitiva::Persistence, // π -- file-backed registry
        ])
        .with_dominant(LexPrimitiva::State, 0.85)
        .with_state_mode(StateMode::Mutable)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Mutable)
    }
}

/// BrainHealth: Comprehensive health report with status determination.
/// Tier: T2Composite. Dominant: ς State.
/// WHY: Health IS the system's current state. Quantity (N) for metrics counts.
/// Boundary (∂) for healthy/warning/degraded thresholds. Location (λ) for
/// directory paths. Sequence (σ) for warnings list.
impl GroundsTo for BrainHealth {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,    // ς -- health status state
            LexPrimitiva::Quantity, // N -- metric values
            LexPrimitiva::Boundary, // ∂ -- healthy/warning/degraded thresholds
            LexPrimitiva::Location, // λ -- brain_dir and telemetry_dir paths
            LexPrimitiva::Sequence, // σ -- warnings list
        ])
        .with_dominant(LexPrimitiva::State, 0.82)
        .with_state_mode(StateMode::Mutable)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Mutable)
    }
}

/// PersistentSynapseBank: Persistent synapse store with amplitude growth model.
/// Tier: T2Composite. Dominant: ς State.
/// WHY: Manages mutable synapse state. Persistence (π) from file-backed
/// save/load. Frequency (ν) from observation counts. Sum (Σ) from amplitude
/// accumulation across observations.
impl GroundsTo for PersistentSynapseBank {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,       // ς -- mutable synapse bank state
            LexPrimitiva::Persistence, // π -- file-backed synapses.json
            LexPrimitiva::Frequency,   // ν -- observation counts
            LexPrimitiva::Sum,         // Σ -- amplitude accumulation
        ])
        .with_dominant(LexPrimitiva::State, 0.82)
        .with_state_mode(StateMode::Mutable)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Mutable)
    }
}

// ============================================================================
// T3 Domain-Specific (6+ unique primitives)
// ============================================================================

/// ImplicitKnowledge: The full implicit knowledge store combining preferences,
/// patterns, corrections, beliefs, trust accumulators, and belief graph.
/// Tier: T3DomainSpecific. Dominant: ς State.
/// WHY: This is the domain-specific aggregate of all brain learning subsystems.
/// State (ς) for mutable knowledge store. Persistence (π) for file-backed
/// save/load. Mapping (μ) for preference and trust maps. Sequence (σ) for
/// patterns and corrections lists. Existence (∃) for belief validation.
/// Recursion (ρ) for belief graph traversal.
impl GroundsTo for ImplicitKnowledge {
    fn primitive_composition() -> PrimitiveComposition {
        PrimitiveComposition::new(vec![
            LexPrimitiva::State,       // ς -- mutable knowledge state
            LexPrimitiva::Persistence, // π -- file-backed storage
            LexPrimitiva::Mapping,     // μ -- preferences and trust maps
            LexPrimitiva::Sequence,    // σ -- patterns, corrections, beliefs lists
            LexPrimitiva::Existence,   // ∃ -- belief existence validation
            LexPrimitiva::Recursion,   // ρ -- belief graph traversal
        ])
        .with_dominant(LexPrimitiva::State, 0.80)
        .with_state_mode(StateMode::Mutable)
    }

    fn state_mode() -> Option<StateMode> {
        Some(StateMode::Mutable)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nexcore_lex_primitiva::tier::Tier;

    // ==========================================================================
    // T1 Universal types
    // ==========================================================================

    #[test]
    fn test_artifact_type_grounding() {
        assert_eq!(ArtifactType::dominant_primitive(), Some(LexPrimitiva::Sum));
        assert!(ArtifactType::is_pure_primitive());
        assert_eq!(ArtifactType::tier(), Tier::T1Universal);
    }

    #[test]
    fn test_t1_primitive_grounding() {
        assert_eq!(T1Primitive::dominant_primitive(), Some(LexPrimitiva::Sum));
        assert!(T1Primitive::is_pure_primitive());
        assert_eq!(T1Primitive::tier(), Tier::T1Universal);
    }

    #[test]
    fn test_evidence_type_grounding() {
        assert_eq!(EvidenceType::dominant_primitive(), Some(LexPrimitiva::Sum));
        assert!(EvidenceType::is_pure_primitive());
        assert_eq!(EvidenceType::tier(), Tier::T1Universal);
    }

    #[test]
    fn test_implication_strength_grounding() {
        assert_eq!(
            ImplicationStrength::dominant_primitive(),
            Some(LexPrimitiva::Sum)
        );
        assert_eq!(ImplicationStrength::tier(), Tier::T1Universal);
    }

    #[test]
    fn test_lock_status_grounding() {
        assert_eq!(LockStatus::dominant_primitive(), Some(LexPrimitiva::Sum));
        assert_eq!(LockStatus::tier(), Tier::T1Universal);
    }

    #[test]
    fn test_run_status_grounding() {
        assert_eq!(RunStatus::dominant_primitive(), Some(LexPrimitiva::Sum));
        assert_eq!(RunStatus::tier(), Tier::T1Universal);
    }

    #[test]
    fn test_agent_id_grounding() {
        assert_eq!(AgentId::dominant_primitive(), Some(LexPrimitiva::State));
        assert_eq!(AgentId::tier(), Tier::T1Universal);
    }

    #[test]
    fn test_lock_duration_grounding() {
        assert_eq!(
            LockDuration::dominant_primitive(),
            Some(LexPrimitiva::Quantity)
        );
        assert_eq!(LockDuration::tier(), Tier::T1Universal);
    }

    // ==========================================================================
    // T2-P types
    // ==========================================================================

    #[test]
    fn test_artifact_metadata_grounding() {
        assert_eq!(
            ArtifactMetadata::dominant_primitive(),
            Some(LexPrimitiva::State)
        );
        assert_eq!(ArtifactMetadata::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_artifact_grounding() {
        assert_eq!(Artifact::dominant_primitive(), Some(LexPrimitiva::State));
        assert_eq!(Artifact::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_session_entry_grounding() {
        assert_eq!(
            SessionEntry::dominant_primitive(),
            Some(LexPrimitiva::State)
        );
        assert_eq!(SessionEntry::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_tracker_index_grounding() {
        assert_eq!(
            TrackerIndex::dominant_primitive(),
            Some(LexPrimitiva::Mapping)
        );
        assert_eq!(TrackerIndex::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_project_snapshot_grounding() {
        assert_eq!(
            ProjectSnapshot::dominant_primitive(),
            Some(LexPrimitiva::Persistence)
        );
        assert_eq!(ProjectSnapshot::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_tracked_file_grounding() {
        assert_eq!(
            TrackedFile::dominant_primitive(),
            Some(LexPrimitiva::Persistence)
        );
        assert_eq!(TrackedFile::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_preference_grounding() {
        assert_eq!(Preference::dominant_primitive(), Some(LexPrimitiva::State));
        assert_eq!(Preference::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_pattern_grounding() {
        assert_eq!(Pattern::dominant_primitive(), Some(LexPrimitiva::State));
        assert_eq!(Pattern::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_correction_grounding() {
        assert_eq!(
            Correction::dominant_primitive(),
            Some(LexPrimitiva::Mapping)
        );
        assert_eq!(Correction::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_evidence_ref_grounding() {
        assert_eq!(
            EvidenceRef::dominant_primitive(),
            Some(LexPrimitiva::Existence)
        );
        assert_eq!(EvidenceRef::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_trust_accumulator_grounding() {
        assert_eq!(
            TrustAccumulator::dominant_primitive(),
            Some(LexPrimitiva::State)
        );
        assert_eq!(TrustAccumulator::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_belief_implication_grounding() {
        assert_eq!(
            BeliefImplication::dominant_primitive(),
            Some(LexPrimitiva::Causality)
        );
        assert_eq!(BeliefImplication::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_belief_graph_grounding() {
        assert_eq!(
            BeliefGraph::dominant_primitive(),
            Some(LexPrimitiva::Recursion)
        );
        assert_eq!(BeliefGraph::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_implicit_stats_grounding() {
        assert_eq!(
            ImplicitStats::dominant_primitive(),
            Some(LexPrimitiva::Quantity)
        );
        assert_eq!(ImplicitStats::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_artifact_metrics_grounding() {
        assert_eq!(
            ArtifactMetrics::dominant_primitive(),
            Some(LexPrimitiva::Quantity)
        );
        assert_eq!(ArtifactMetrics::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_session_metrics_grounding() {
        assert_eq!(
            SessionMetrics::dominant_primitive(),
            Some(LexPrimitiva::Quantity)
        );
        assert_eq!(SessionMetrics::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_brain_snapshot_grounding() {
        assert_eq!(
            BrainSnapshot::dominant_primitive(),
            Some(LexPrimitiva::Persistence)
        );
        assert_eq!(BrainSnapshot::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_artifact_size_info_grounding() {
        assert_eq!(
            ArtifactSizeInfo::dominant_primitive(),
            Some(LexPrimitiva::Quantity)
        );
        assert_eq!(ArtifactSizeInfo::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_growth_rate_grounding() {
        assert_eq!(
            GrowthRate::dominant_primitive(),
            Some(LexPrimitiva::Quantity)
        );
        assert_eq!(GrowthRate::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_brain_error_grounding() {
        assert_eq!(BrainError::dominant_primitive(), Some(LexPrimitiva::Sum));
        assert_eq!(BrainError::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_pipeline_run_grounding() {
        assert_eq!(PipelineRun::dominant_primitive(), Some(LexPrimitiva::State));
        assert_eq!(PipelineRun::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_checkpoint_grounding() {
        assert_eq!(Checkpoint::dominant_primitive(), Some(LexPrimitiva::State));
        assert_eq!(Checkpoint::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_pipeline_state_grounding() {
        assert_eq!(
            PipelineState::dominant_primitive(),
            Some(LexPrimitiva::State)
        );
        assert_eq!(PipelineState::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_recovery_result_grounding() {
        assert_eq!(
            RecoveryResult::dominant_primitive(),
            Some(LexPrimitiva::Existence)
        );
        assert_eq!(RecoveryResult::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_synapse_bank_stats_grounding() {
        assert_eq!(
            SynapseBankStats::dominant_primitive(),
            Some(LexPrimitiva::Quantity)
        );
        assert_eq!(SynapseBankStats::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_synapse_info_grounding() {
        assert_eq!(SynapseInfo::dominant_primitive(), Some(LexPrimitiva::State));
        assert_eq!(SynapseInfo::tier(), Tier::T2Primitive);
    }

    #[test]
    fn test_brain_config_grounding() {
        assert_eq!(BrainConfig::dominant_primitive(), Some(LexPrimitiva::State));
        assert_eq!(BrainConfig::tier(), Tier::T2Primitive);
    }

    // ==========================================================================
    // T2-C types
    // ==========================================================================

    #[test]
    fn test_brain_session_grounding() {
        assert_eq!(
            BrainSession::dominant_primitive(),
            Some(LexPrimitiva::State)
        );
        assert_eq!(BrainSession::tier(), Tier::T2Composite);
    }

    #[test]
    fn test_code_tracker_grounding() {
        assert_eq!(CodeTracker::dominant_primitive(), Some(LexPrimitiva::State));
        assert_eq!(CodeTracker::tier(), Tier::T2Composite);
    }

    #[test]
    fn test_belief_grounding() {
        assert_eq!(Belief::dominant_primitive(), Some(LexPrimitiva::State));
        assert_eq!(Belief::tier(), Tier::T2Composite);
    }

    #[test]
    fn test_file_lock_grounding() {
        assert_eq!(FileLock::dominant_primitive(), Some(LexPrimitiva::State));
        assert_eq!(FileLock::tier(), Tier::T2Composite);
    }

    #[test]
    fn test_coordination_registry_grounding() {
        assert_eq!(
            CoordinationRegistry::dominant_primitive(),
            Some(LexPrimitiva::State)
        );
        assert_eq!(CoordinationRegistry::tier(), Tier::T2Composite);
    }

    #[test]
    fn test_brain_health_grounding() {
        assert_eq!(BrainHealth::dominant_primitive(), Some(LexPrimitiva::State));
        assert_eq!(BrainHealth::tier(), Tier::T2Composite);
    }

    #[test]
    fn test_brain_synapse_bank_grounding() {
        assert_eq!(
            PersistentSynapseBank::dominant_primitive(),
            Some(LexPrimitiva::State)
        );
        assert_eq!(PersistentSynapseBank::tier(), Tier::T2Composite);
    }

    // ==========================================================================
    // T3 Domain-Specific types
    // ==========================================================================

    #[test]
    fn test_implicit_knowledge_grounding() {
        assert_eq!(
            ImplicitKnowledge::dominant_primitive(),
            Some(LexPrimitiva::State)
        );
        assert_eq!(ImplicitKnowledge::tier(), Tier::T3DomainSpecific);
    }

    // ==========================================================================
    // Cross-cutting: dominant primitive distribution
    // ==========================================================================

    #[test]
    fn test_dominant_primitive_distribution() {
        // Verify that the Brain crate has a rich distribution of dominant primitives.
        // Brain is primarily about State (ς) but has diverse secondary groundings.

        let state_dominant = [
            ArtifactMetadata::dominant_primitive(),
            Artifact::dominant_primitive(),
            SessionEntry::dominant_primitive(),
            Preference::dominant_primitive(),
            Pattern::dominant_primitive(),
            TrustAccumulator::dominant_primitive(),
            BrainSession::dominant_primitive(),
            CodeTracker::dominant_primitive(),
            Belief::dominant_primitive(),
            PipelineRun::dominant_primitive(),
            Checkpoint::dominant_primitive(),
            PipelineState::dominant_primitive(),
            FileLock::dominant_primitive(),
            CoordinationRegistry::dominant_primitive(),
            BrainHealth::dominant_primitive(),
            PersistentSynapseBank::dominant_primitive(),
            SynapseInfo::dominant_primitive(),
            BrainConfig::dominant_primitive(),
            ImplicitKnowledge::dominant_primitive(),
            AgentId::dominant_primitive(),
        ];
        let state_count = state_dominant
            .iter()
            .filter(|d| **d == Some(LexPrimitiva::State))
            .count();
        // State should dominate -- Brain is a working memory system
        assert!(
            state_count >= 15,
            "Expected at least 15 State-dominant types, got {state_count}"
        );

        // But there should also be non-State dominants for diversity
        let non_state_types = [
            TrackerIndex::dominant_primitive(),      // μ Mapping
            ProjectSnapshot::dominant_primitive(),   // π Persistence
            TrackedFile::dominant_primitive(),       // π Persistence
            Correction::dominant_primitive(),        // μ Mapping
            EvidenceRef::dominant_primitive(),       // ∃ Existence
            BeliefImplication::dominant_primitive(), // → Causality
            BeliefGraph::dominant_primitive(),       // ρ Recursion
            RecoveryResult::dominant_primitive(),    // ∃ Existence
        ];
        assert!(
            non_state_types
                .iter()
                .all(|d| *d != Some(LexPrimitiva::State)),
            "Expected non-State dominants for diversity"
        );
    }

    #[test]
    fn test_all_sum_enums_are_t1() {
        // All pure sum enums should be T1Universal
        assert_eq!(ArtifactType::tier(), Tier::T1Universal);
        assert_eq!(T1Primitive::tier(), Tier::T1Universal);
        assert_eq!(EvidenceType::tier(), Tier::T1Universal);
        assert_eq!(ImplicationStrength::tier(), Tier::T1Universal);
        assert_eq!(LockStatus::tier(), Tier::T1Universal);
        assert_eq!(RunStatus::tier(), Tier::T1Universal);
    }

    #[test]
    fn test_newtypes_are_t1() {
        // Pure newtypes wrapping a single primitive should be T1Universal
        assert_eq!(AgentId::tier(), Tier::T1Universal);
        assert_eq!(LockDuration::tier(), Tier::T1Universal);
    }
}
