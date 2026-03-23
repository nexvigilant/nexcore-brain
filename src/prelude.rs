//! # Prelude — Convenient re-exports
//!
//! Import `use nexcore_brain::prelude::*;` for the most commonly used types.
//!
//! ## What's included
//!
//! - **Core types**: `BrainSession`, `Artifact`, `ArtifactType`, `ArtifactMetadata`
//! - **Implicit knowledge**: `Belief`, `Pattern`, `Preference`, `Correction`, `T1Primitive`
//! - **Evidence**: `EvidenceRef`, `EvidenceType`
//! - **Trust**: `TrustAccumulator`
//! - **Metrics**: `BrainHealth`, `BrainSnapshot`
//! - **Error**: `BrainError`, `Result`
//! - **Pipeline**: `PipelineState`, `PipelineRun`
//! - **Coordination**: `CoordinationRegistry`, `AgentId`
//! - **Transfer**: `TransferMapping`

pub use crate::artifact::{Artifact, ArtifactMetadata, ArtifactType};
pub use crate::coordination::{AgentId, CoordinationRegistry, LockDuration, LockStatus};
pub use crate::error::{BrainError, Result};
pub use crate::implicit::{
    Belief, BeliefGraph, Correction, EvidenceRef, EvidenceType, ImplicitKnowledge, ImplicitStats,
    Pattern, Preference, T1Primitive, TrustAccumulator,
};
pub use crate::metrics::{BrainHealth, BrainSnapshot};
pub use crate::pipeline::{PipelineRun, PipelineState};
pub use crate::session::BrainSession;
pub use crate::synapse::{PersistentSynapseBank, SynapseInfo};
pub use crate::tracker::{CodeTracker, TrackedFile};
pub use crate::transfer::TransferMapping;
