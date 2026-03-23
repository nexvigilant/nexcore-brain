//! Implicit knowledge management
//!
//! Stores learned preferences, patterns, and corrections that Claude
//! learns from user interactions. This is "implicit" knowledge because
//! it's inferred rather than explicitly stated.
//!
//! ## Pattern Matching
//!
//! Corrections use token-based fuzzy matching (Jaccard similarity) rather
//! than naive substring search. Patterns support T1 primitive grounding
//! and time-based confidence decay.

use nexcore_chrono::DateTime;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::PathBuf;

use crate::error::Result;
use crate::{implicit_dir, initialize_directories};

// ========== T1 Primitive Grounding ==========

/// Universal T1 primitives that patterns can be grounded to.
///
/// Every pattern in the Brain maps to exactly one T1 primitive,
/// enabling cross-domain transfer and primitive-first filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum T1Primitive {
    /// Ordered operations (iterators, chains, pipelines)
    Sequence,
    /// Transformation A → B (From/Into, map, type conversions)
    Mapping,
    /// Self-reference (enum, Box, Arc, recursive fns)
    Recursion,
    /// Encapsulated context (struct+impl, typestate, invariants)
    State,
    /// Absence of value ((), Option::None, Default)
    Void,
    /// Delimiters, transition points, or scale limits.
    Boundary,
    /// Rate of occurrence; iterative counting.
    Frequency,
    /// Simple presence; instantiation of being.
    Existence,
    /// Continuity of state through temporal duration.
    Persistence,
    /// Producer (Cause) and consequence (Effect).
    Causality,
    /// Logical test; predicate matching.
    Comparison,
    /// Numerical magnitude; amount or ratio.
    Quantity,
    /// Positional context; address or coordinate.
    Location,
    /// One-way state transition; entropy floor.
    Irreversibility,
    /// Exclusive disjunction; exactly one of N variants (enum, match).
    Sum,
}

impl fmt::Display for T1Primitive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Sequence => "sequence",
            Self::Mapping => "mapping",
            Self::Recursion => "recursion",
            Self::State => "state",
            Self::Void => "void",
            Self::Boundary => "boundary",
            Self::Frequency => "frequency",
            Self::Existence => "existence",
            Self::Persistence => "persistence",
            Self::Causality => "causality",
            Self::Comparison => "comparison",
            Self::Quantity => "quantity",
            Self::Location => "location",
            Self::Irreversibility => "irreversibility",
            Self::Sum => "sum",
        };
        write!(f, "{s}")
    }
}

// ========== Fuzzy Matching ==========

/// Compute token-based Jaccard similarity between two strings.
///
/// Tokenizes on whitespace and punctuation, lowercases, then computes
/// |intersection| / |union|. Returns 0.0–1.0.
fn token_similarity(a: &str, b: &str) -> f64 {
    let tokenize = |s: &str| -> std::collections::HashSet<String> {
        s.to_lowercase()
            .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|t| !t.is_empty())
            .map(String::from)
            .collect()
    };

    let set_a = tokenize(a);
    let set_b = tokenize(b);

    if set_a.is_empty() && set_b.is_empty() {
        return 1.0;
    }
    if set_a.is_empty() || set_b.is_empty() {
        return 0.0;
    }

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Default similarity threshold for fuzzy correction matching
const DEFAULT_SIMILARITY_THRESHOLD: f64 = 0.3;

/// Confidence decay half-life in days.
/// After this many days without reinforcement, effective confidence halves.
const DECAY_HALF_LIFE_DAYS: f64 = 30.0;

/// A learned user preference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preference {
    /// Preference key (e.g., "code_style", "commit_format")
    pub key: String,

    /// Preference value
    pub value: serde_json::Value,

    /// Optional description of what this preference means
    pub description: Option<String>,

    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,

    /// When this preference was last updated
    pub updated_at: DateTime,

    /// Number of times this preference was reinforced
    pub reinforcement_count: u32,
}

impl Preference {
    /// Create a new preference
    #[must_use]
    pub fn new(key: impl Into<String>, value: serde_json::Value) -> Self {
        Self {
            key: key.into(),
            value,
            description: None,
            confidence: 0.5, // Start neutral
            updated_at: DateTime::now(),
            reinforcement_count: 1,
        }
    }

    /// Reinforce this preference (increase confidence)
    pub fn reinforce(&mut self) {
        self.reinforcement_count += 1;
        // Asymptotically approach 1.0
        self.confidence = 1.0 - (1.0 / (self.reinforcement_count as f64 + 1.0));
        self.updated_at = DateTime::now();
    }

    /// Weaken this preference (decrease confidence)
    pub fn weaken(&mut self) {
        if self.reinforcement_count > 0 {
            self.reinforcement_count = self.reinforcement_count.saturating_sub(1);
        }
        self.confidence = if self.reinforcement_count == 0 {
            0.0
        } else {
            1.0 - (1.0 / (self.reinforcement_count as f64 + 1.0))
        };
        self.updated_at = DateTime::now();
    }

    /// Effective confidence after time-based decay.
    ///
    /// Uses exponential decay: `confidence * 0.5^(days_since_update / half_life)`.
    /// Reinforcing resets `updated_at`, restarting the decay clock.
    #[must_use]
    pub fn effective_confidence(&self) -> f64 {
        self.effective_confidence_at(DateTime::now())
    }

    /// Effective confidence at a specific point in time (for testing/analysis).
    #[must_use]
    pub fn effective_confidence_at(&self, now: DateTime) -> f64 {
        let days_elapsed = now.signed_duration_since(self.updated_at).num_hours() as f64 / 24.0;
        if days_elapsed <= 0.0 {
            return self.confidence;
        }
        let decay_factor = (0.5_f64).powf(days_elapsed / DECAY_HALF_LIFE_DAYS);
        self.confidence * decay_factor
    }

    /// Check if this preference is still active after time decay.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.effective_confidence() > 0.05
    }
}

/// A detected pattern in user behavior or codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Pattern identifier
    pub id: String,

    /// Pattern type (e.g., "naming", "structure", "workflow")
    pub pattern_type: String,

    /// Description of the pattern
    pub description: String,

    /// Example instances of this pattern
    pub examples: Vec<String>,

    /// When this pattern was detected
    pub detected_at: DateTime,

    /// When this pattern was last reinforced (for decay calculation)
    #[serde(default = "DateTime::now")]
    pub updated_at: DateTime,

    /// Confidence level (0.0 to 1.0) — raw, before decay
    pub confidence: f64,

    /// Number of occurrences observed
    pub occurrence_count: u32,

    /// T1 primitive this pattern is grounded to
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t1_grounding: Option<T1Primitive>,
}

impl Pattern {
    /// Create a new pattern
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        pattern_type: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            pattern_type: pattern_type.into(),
            description: description.into(),
            examples: Vec::new(),
            detected_at: DateTime::now(),
            updated_at: DateTime::now(),
            confidence: 0.5,
            occurrence_count: 1,
            t1_grounding: None,
        }
    }

    /// Add an example of this pattern (reinforces confidence and resets decay clock)
    pub fn add_example(&mut self, example: impl Into<String>) {
        self.examples.push(example.into());
        self.occurrence_count += 1;
        self.updated_at = DateTime::now();
        // Increase confidence with more examples
        self.confidence =
            (self.occurrence_count as f64 / (self.occurrence_count as f64 + 2.0)).min(0.95);
    }

    /// Set the T1 primitive this pattern is grounded to
    pub fn set_grounding(&mut self, primitive: T1Primitive) {
        self.t1_grounding = Some(primitive);
        self.updated_at = DateTime::now();
    }

    /// Effective confidence after time-based decay.
    ///
    /// Uses exponential decay: `confidence × 0.5^(days_since_update / half_life)`.
    /// Patterns reinforced recently keep full confidence; stale ones decay toward 0.
    #[must_use]
    pub fn effective_confidence(&self) -> f64 {
        let days_elapsed = (DateTime::now() - self.updated_at).num_hours() as f64 / 24.0;
        if days_elapsed <= 0.0 {
            return self.confidence;
        }
        let decay_factor = (0.5_f64).powf(days_elapsed / DECAY_HALF_LIFE_DAYS);
        self.confidence * decay_factor
    }
}

/// Correction confidence decay half-life in days.
/// Corrections decay slower (60 days) because explicit feedback is more durable.
const CORRECTION_HALF_LIFE_DAYS: f64 = 60.0;

/// A learned correction from user feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Correction {
    /// What was wrong (the mistake)
    pub mistake: String,

    /// What should have been done (the correction)
    pub correction: String,

    /// Context when this occurred
    pub context: Option<String>,

    /// When this correction was learned
    pub learned_at: DateTime,

    /// Number of times this correction was applied
    pub application_count: u32,

    /// Confidence level (0.0 to 1.0). Defaults to 0.8.
    #[serde(default = "default_correction_confidence")]
    pub confidence: f64,

    /// When this correction was last reinforced (applied or confirmed).
    #[serde(default = "DateTime::now")]
    pub updated_at: DateTime,
}

fn default_correction_confidence() -> f64 {
    0.8
}

impl Correction {
    /// Create a new correction
    #[must_use]
    pub fn new(mistake: impl Into<String>, correction: impl Into<String>) -> Self {
        let now = DateTime::now();
        Self {
            mistake: mistake.into(),
            correction: correction.into(),
            context: None,
            learned_at: now,
            application_count: 0,
            confidence: 0.8,
            updated_at: now,
        }
    }

    /// Record that this correction was applied (resets decay clock)
    pub fn mark_applied(&mut self) {
        self.application_count += 1;
        self.confidence = 1.0 - (0.2 / (self.application_count as f64 + 1.0));
        self.updated_at = DateTime::now();
    }

    /// Effective confidence after time-based decay.
    #[must_use]
    pub fn effective_confidence(&self) -> f64 {
        self.effective_confidence_at(DateTime::now())
    }

    /// Effective confidence at a specific point in time.
    #[must_use]
    pub fn effective_confidence_at(&self, now: DateTime) -> f64 {
        let days_elapsed = now.signed_duration_since(self.updated_at).num_hours() as f64 / 24.0;
        if days_elapsed <= 0.0 {
            return self.confidence;
        }
        let decay_factor = (0.5_f64).powf(days_elapsed / CORRECTION_HALF_LIFE_DAYS);
        self.confidence * decay_factor
    }

    /// Check if this correction is still active after time decay.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.effective_confidence() > 0.05
    }
}

// ========== PROJECT GROUNDED: Belief System ==========

/// Evidence type classification for belief support.
///
/// Grounded in T1:Sum (exactly one of N variants).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceType {
    /// Empirical observation from execution
    Observation,
    /// Result of hypothesis test
    TestResult,
    /// User-provided explicit feedback
    UserFeedback,
    /// Inferred from pattern matching
    Inference,
    /// External authoritative source
    Authority,
    /// Inherited from prior belief
    Prior,
}

impl fmt::Display for EvidenceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Observation => "observation",
            Self::TestResult => "test_result",
            Self::UserFeedback => "user_feedback",
            Self::Inference => "inference",
            Self::Authority => "authority",
            Self::Prior => "prior",
        };
        write!(f, "{s}")
    }
}

fn default_evidence_id() -> String {
    format!("ev-{}", DateTime::now().timestamp_millis())
}

fn default_evidence_type() -> EvidenceType {
    EvidenceType::Observation
}

fn default_recorded_at() -> DateTime {
    DateTime::now()
}

/// Reference to evidence supporting or contradicting a belief.
///
/// PROJECT GROUNDED integration: Evidence enables the
/// hypothesis → test → belief integration loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRef {
    /// Unique evidence identifier (auto-generated if missing from legacy data)
    #[serde(default = "default_evidence_id")]
    pub id: String,

    /// Type of evidence (defaults to Observation for legacy data)
    #[serde(default = "default_evidence_type")]
    pub evidence_type: EvidenceType,

    /// Description of what this evidence shows
    pub description: String,

    /// Weight of this evidence (positive = supports, negative = contradicts)
    /// Range: -1.0 to 1.0
    pub weight: f64,

    /// Source of the evidence (e.g., "session:abc123", "test:unit_foo", "user:explicit")
    pub source: String,

    /// When this evidence was recorded (accepts legacy "observed_at" alias)
    #[serde(default = "default_recorded_at", alias = "observed_at")]
    pub recorded_at: DateTime,

    /// Optional link to artifact or external resource
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_ref: Option<String>,

    /// Link to MCP tool execution that produced this evidence (GROUNDED extension)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_id: Option<String>,

    /// Link to originating hypothesis belief (GROUNDED extension)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hypothesis_id: Option<String>,
}

impl EvidenceRef {
    /// Create new supporting evidence
    #[must_use]
    pub fn supporting(
        id: impl Into<String>,
        evidence_type: EvidenceType,
        description: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            evidence_type,
            description: description.into(),
            weight: 1.0,
            source: source.into(),
            recorded_at: DateTime::now(),
            artifact_ref: None,
            execution_id: None,
            hypothesis_id: None,
        }
    }

    /// Create new contradicting evidence
    #[must_use]
    pub fn contradicting(
        id: impl Into<String>,
        evidence_type: EvidenceType,
        description: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            evidence_type,
            description: description.into(),
            weight: -1.0,
            source: source.into(),
            recorded_at: DateTime::now(),
            artifact_ref: None,
            execution_id: None,
            hypothesis_id: None,
        }
    }

    /// Create evidence with custom weight
    #[must_use]
    pub fn weighted(
        id: impl Into<String>,
        evidence_type: EvidenceType,
        description: impl Into<String>,
        weight: f64,
        source: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            evidence_type,
            description: description.into(),
            weight: weight.clamp(-1.0, 1.0),
            source: source.into(),
            recorded_at: DateTime::now(),
            artifact_ref: None,
            execution_id: None,
            hypothesis_id: None,
        }
    }

    /// Create evidence from tool execution (GROUNDED loop)
    #[must_use]
    pub fn from_execution(
        id: impl Into<String>,
        evidence_type: EvidenceType,
        description: impl Into<String>,
        weight: f64,
        execution_id: impl Into<String>,
        hypothesis_id: Option<String>,
    ) -> Self {
        Self {
            id: id.into(),
            evidence_type,
            description: description.into(),
            weight: weight.clamp(-1.0, 1.0),
            source: "execution".into(),
            recorded_at: DateTime::now(),
            artifact_ref: None,
            execution_id: Some(execution_id.into()),
            hypothesis_id,
        }
    }

    /// Link this evidence to an execution
    pub fn with_execution(mut self, execution_id: impl Into<String>) -> Self {
        self.execution_id = Some(execution_id.into());
        self
    }

    /// Link this evidence to a hypothesis
    pub fn with_hypothesis(mut self, hypothesis_id: impl Into<String>) -> Self {
        self.hypothesis_id = Some(hypothesis_id.into());
        self
    }
}

/// Belief state tracking for PROJECT GROUNDED feedback loops.
///
/// A belief represents a propositional attitude held by the system,
/// with confidence derived from accumulated evidence. Beliefs decay
/// over time without reinforcing evidence (same decay model as Pattern).
///
/// ## The GROUNDED Loop
///
/// ```text
/// hypothesis → test design → execution → belief integration
///                                              ↓
///                                      [this struct]
/// ```
///
/// ## Confidence Computation
///
/// Raw confidence = `Σ(evidence.weight) / |evidence|`, normalized to [0, 1].
/// Effective confidence applies time decay: `raw × 0.5^(days / half_life)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Belief {
    /// Unique belief identifier
    pub id: String,

    /// The proposition this belief represents (natural language)
    pub proposition: String,

    /// Category of belief (e.g., "capability", "behavior", "preference", "constraint")
    pub category: String,

    /// Raw confidence level before decay (0.0 to 1.0)
    pub confidence: f64,

    /// Evidence supporting or contradicting this belief
    pub evidence: Vec<EvidenceRef>,

    /// T1 primitive this belief is grounded to
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t1_grounding: Option<T1Primitive>,

    /// When this belief was first formed
    pub formed_at: DateTime,

    /// When this belief was last updated (for decay calculation)
    pub updated_at: DateTime,

    /// Number of times this belief was tested/validated
    pub validation_count: u32,

    /// Whether this belief has been explicitly confirmed by user
    #[serde(default)]
    pub user_confirmed: bool,
}

/// Belief decay half-life in days (same as Pattern for consistency)
const BELIEF_DECAY_HALF_LIFE_DAYS: f64 = 30.0;

impl Belief {
    /// Create a new belief with initial proposition
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        proposition: impl Into<String>,
        category: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            proposition: proposition.into(),
            category: category.into(),
            confidence: 0.5, // Start neutral (prior)
            evidence: Vec::new(),
            t1_grounding: None,
            formed_at: DateTime::now(),
            updated_at: DateTime::now(),
            validation_count: 0,
            user_confirmed: false,
        }
    }

    /// Create a belief from a hypothesis (lower initial confidence)
    #[must_use]
    pub fn from_hypothesis(
        id: impl Into<String>,
        proposition: impl Into<String>,
        category: impl Into<String>,
    ) -> Self {
        let mut belief = Self::new(id, proposition, category);
        belief.confidence = 0.3; // Hypotheses start lower
        belief
    }

    /// Add evidence and recompute confidence
    pub fn add_evidence(&mut self, evidence: EvidenceRef) {
        self.evidence.push(evidence);
        self.recompute_confidence();
        self.updated_at = DateTime::now();
    }

    /// Add multiple pieces of evidence at once
    pub fn add_evidence_batch(&mut self, evidence: Vec<EvidenceRef>) {
        self.evidence.extend(evidence);
        self.recompute_confidence();
        self.updated_at = DateTime::now();
    }

    /// Recompute confidence from evidence weights
    fn recompute_confidence(&mut self) {
        if self.evidence.is_empty() {
            return;
        }

        // Sum of weights normalized to [-1, 1], then shifted to [0, 1]
        let total_weight: f64 = self.evidence.iter().map(|e| e.weight).sum();
        let avg_weight = total_weight / self.evidence.len() as f64;

        // Map [-1, 1] to [0, 1]: (x + 1) / 2
        // Then apply diminishing returns curve
        let normalized = (avg_weight + 1.0) / 2.0;

        // Apply evidence count boost (more evidence = higher confidence ceiling)
        let count_factor =
            (self.evidence.len() as f64 / (self.evidence.len() as f64 + 3.0)).min(0.95);

        self.confidence = (normalized * count_factor).clamp(0.0, 0.95);
    }

    /// Effective confidence after time-based decay
    ///
    /// Uses same exponential decay as Pattern: `confidence × 0.5^(days / half_life)`
    #[must_use]
    pub fn effective_confidence(&self) -> f64 {
        let days_elapsed = (DateTime::now() - self.updated_at).num_hours() as f64 / 24.0;
        if days_elapsed <= 0.0 {
            return self.confidence;
        }
        let decay_factor = (0.5_f64).powf(days_elapsed / BELIEF_DECAY_HALF_LIFE_DAYS);
        self.confidence * decay_factor
    }

    /// Record a validation attempt (successful test of the belief)
    pub fn record_validation(&mut self, success: bool) {
        self.validation_count += 1;
        self.updated_at = DateTime::now();

        // Add implicit evidence from validation
        let evidence = if success {
            EvidenceRef::weighted(
                format!("validation_{}", self.validation_count),
                EvidenceType::TestResult,
                "Validation test passed",
                0.5, // Moderate positive weight
                "system:validation",
            )
        } else {
            EvidenceRef::weighted(
                format!("validation_{}", self.validation_count),
                EvidenceType::TestResult,
                "Validation test failed",
                -0.5, // Moderate negative weight
                "system:validation",
            )
        };

        self.evidence.push(evidence);
        self.recompute_confidence();
    }

    /// Mark belief as confirmed by user (prevents decay below threshold)
    pub fn confirm(&mut self) {
        self.user_confirmed = true;
        self.updated_at = DateTime::now();

        // Add user confirmation evidence
        let evidence = EvidenceRef::supporting(
            format!("user_confirm_{}", DateTime::now().timestamp()),
            EvidenceType::UserFeedback,
            "User explicitly confirmed this belief",
            "user:explicit",
        );
        self.evidence.push(evidence);
        self.recompute_confidence();
    }

    /// Mark belief as rejected by user
    pub fn reject(&mut self) {
        self.user_confirmed = false;
        self.updated_at = DateTime::now();

        // Add user rejection evidence
        let evidence = EvidenceRef::contradicting(
            format!("user_reject_{}", DateTime::now().timestamp()),
            EvidenceType::UserFeedback,
            "User explicitly rejected this belief",
            "user:explicit",
        );
        self.evidence.push(evidence);
        self.recompute_confidence();
    }

    /// Set T1 primitive grounding
    pub fn set_grounding(&mut self, primitive: T1Primitive) {
        self.t1_grounding = Some(primitive);
        self.updated_at = DateTime::now();
    }

    /// Check if belief is stale (effective confidence below threshold)
    ///
    /// User-confirmed beliefs use `max(threshold, 0.3)` as their staleness floor,
    /// meaning they're harder to mark as stale (require lower effective confidence).
    #[must_use]
    pub fn is_stale(&self, threshold: f64) -> bool {
        // User-confirmed beliefs have a staleness floor — they're protected
        // from being marked stale unless confidence drops very low
        if self.user_confirmed {
            return self.effective_confidence() < threshold.min(0.1);
        }
        self.effective_confidence() < threshold
    }

    /// Get net evidence sentiment (positive = supporting, negative = contradicting)
    #[must_use]
    pub fn evidence_sentiment(&self) -> f64 {
        if self.evidence.is_empty() {
            return 0.0;
        }
        self.evidence.iter().map(|e| e.weight).sum::<f64>() / self.evidence.len() as f64
    }
}

// ========== Trust Accumulator (PROJECT GROUNDED) ==========

/// Domain-scoped trust accumulator for competence tracking.
///
/// Trust is earned through demonstrated competence, not granted through compliance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustAccumulator {
    /// Domain this trust applies to (e.g., "rust_development", "pv_signal_detection")
    pub domain: String,
    /// Number of successful competence demonstrations
    pub demonstrations: u32,
    /// Number of failures or mistakes
    pub failures: u32,
    /// When this accumulator was created
    pub created_at: DateTime,
    /// When last updated
    pub updated_at: DateTime,
    /// Optional T1 primitive grounding for the domain
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t1_grounding: Option<T1Primitive>,
}

impl TrustAccumulator {
    /// Create a new trust accumulator for a domain
    #[must_use]
    pub fn new(domain: impl Into<String>) -> Self {
        Self {
            domain: domain.into(),
            demonstrations: 0,
            failures: 0,
            created_at: DateTime::now(),
            updated_at: DateTime::now(),
            t1_grounding: None,
        }
    }

    /// Record a successful demonstration
    pub fn record_success(&mut self) {
        self.demonstrations += 1;
        self.updated_at = DateTime::now();
    }

    /// Record a failure
    pub fn record_failure(&mut self) {
        self.failures += 1;
        self.updated_at = DateTime::now();
    }

    /// Compute earned trust score: demonstrations / (demonstrations + failures + 1)
    #[must_use]
    pub fn score(&self) -> f64 {
        let total = self.demonstrations + self.failures + 1;
        self.demonstrations as f64 / total as f64
    }

    /// Check if trust is above threshold
    #[must_use]
    pub fn is_trusted(&self, threshold: f64) -> bool {
        self.score() >= threshold
    }

    /// Total interactions (demonstrations + failures)
    #[must_use]
    pub fn total_interactions(&self) -> u32 {
        self.demonstrations + self.failures
    }
}

// ========== Belief Graph (PROJECT GROUNDED) ==========

/// Edge type for belief implications
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImplicationStrength {
    /// Strong implication (propagation: 0.9)
    Strong,
    /// Moderate implication (propagation: 0.6)
    Moderate,
    /// Weak implication (propagation: 0.3)
    Weak,
}

impl ImplicationStrength {
    /// Get propagation factor for confidence updates
    #[must_use]
    pub fn propagation_factor(&self) -> f64 {
        match self {
            Self::Strong => 0.9,
            Self::Moderate => 0.6,
            Self::Weak => 0.3,
        }
    }
}

/// An implication edge between beliefs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefImplication {
    /// Source belief ID (antecedent)
    pub from: String,
    /// Target belief ID (consequent)
    pub to: String,
    /// Strength of implication
    pub strength: ImplicationStrength,
    /// When this implication was established
    pub established_at: DateTime,
}

/// Belief dependency graph for inter-belief causality.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BeliefGraph {
    /// Implication edges (from → to with strength)
    pub implications: Vec<BeliefImplication>,
}

impl BeliefGraph {
    /// Create an empty belief graph
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an implication edge
    pub fn add_implication(
        &mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        strength: ImplicationStrength,
    ) {
        let imp = BeliefImplication {
            from: from.into(),
            to: to.into(),
            strength,
            established_at: DateTime::now(),
        };
        self.implications.push(imp);
    }

    /// Get all beliefs implied by a given belief
    #[must_use]
    pub fn implied_by(&self, belief_id: &str) -> Vec<(&str, ImplicationStrength)> {
        self.implications
            .iter()
            .filter(|imp| imp.from == belief_id)
            .map(|imp| (imp.to.as_str(), imp.strength))
            .collect()
    }

    /// Get all beliefs that imply a given belief
    #[must_use]
    pub fn implies(&self, belief_id: &str) -> Vec<(&str, ImplicationStrength)> {
        self.implications
            .iter()
            .filter(|imp| imp.to == belief_id)
            .map(|imp| (imp.from.as_str(), imp.strength))
            .collect()
    }

    /// Remove an implication edge
    pub fn remove_implication(&mut self, from: &str, to: &str) {
        self.implications
            .retain(|imp| imp.from != from || imp.to != to);
    }

    /// Check if adding an edge would create a cycle
    #[must_use]
    pub fn would_create_cycle(&self, from: &str, to: &str) -> bool {
        self.can_reach(to, from)
    }

    /// Check if source can reach target via implications (DFS)
    fn can_reach(&self, source: &str, target: &str) -> bool {
        let mut visited = std::collections::HashSet::new();
        self.dfs_reach(source, target, &mut visited)
    }

    fn dfs_reach(
        &self,
        current: &str,
        target: &str,
        visited: &mut std::collections::HashSet<String>,
    ) -> bool {
        if current == target {
            return true;
        }
        if !visited.insert(current.to_string()) {
            return false;
        }
        self.implied_by(current)
            .iter()
            .any(|(next, _)| self.dfs_reach(next, target, visited))
    }
}

/// Implicit knowledge store
#[derive(Debug)]
pub struct ImplicitKnowledge {
    /// User preferences
    preferences: HashMap<String, Preference>,

    /// Detected patterns
    patterns: Vec<Pattern>,

    /// Learned corrections
    corrections: Vec<Correction>,

    /// Beliefs from GROUNDED feedback loops
    beliefs: Vec<Belief>,

    /// Trust accumulators per domain (GROUNDED)
    trust_accumulators: HashMap<String, TrustAccumulator>,

    /// Belief dependency graph (GROUNDED)
    belief_graph: BeliefGraph,

    /// Path to implicit directory
    implicit_path: PathBuf,
}

impl ImplicitKnowledge {
    /// Create an empty in-memory store for unit testing (no disk I/O).
    #[cfg(test)]
    fn empty_for_test() -> Self {
        Self {
            preferences: HashMap::new(),
            patterns: Vec::new(),
            corrections: Vec::new(),
            beliefs: Vec::new(),
            trust_accumulators: HashMap::new(),
            belief_graph: BeliefGraph::new(),
            implicit_path: PathBuf::from("/tmp/implicit-test"),
        }
    }

    /// Create or load implicit knowledge store
    ///
    /// # Errors
    ///
    /// Returns an error if the store cannot be initialized.
    pub fn load() -> Result<Self> {
        initialize_directories()?;

        let implicit_path = implicit_dir();

        // Load preferences
        let preferences_path = implicit_path.join("preferences.json");
        let preferences: HashMap<String, Preference> = if preferences_path.exists() {
            let content = fs::read_to_string(&preferences_path)?;
            serde_json::from_str(&content)?
        } else {
            HashMap::new()
        };

        // Load patterns
        let patterns_path = implicit_path.join("patterns.json");
        let patterns: Vec<Pattern> = if patterns_path.exists() {
            let content = fs::read_to_string(&patterns_path)?;
            serde_json::from_str(&content)?
        } else {
            Vec::new()
        };

        // Load corrections
        let corrections_path = implicit_path.join("corrections.json");
        let corrections: Vec<Correction> = if corrections_path.exists() {
            let content = fs::read_to_string(&corrections_path)?;
            serde_json::from_str(&content)?
        } else {
            Vec::new()
        };

        // Load beliefs (PROJECT GROUNDED)
        let beliefs_path = implicit_path.join("beliefs.json");
        let beliefs: Vec<Belief> = if beliefs_path.exists() {
            let content = fs::read_to_string(&beliefs_path)?;
            serde_json::from_str(&content)?
        } else {
            Vec::new()
        };

        // Load trust accumulators (PROJECT GROUNDED)
        let trust_path = implicit_path.join("trust.json");
        let trust_accumulators: HashMap<String, TrustAccumulator> = if trust_path.exists() {
            let content = fs::read_to_string(&trust_path)?;
            serde_json::from_str(&content)?
        } else {
            HashMap::new()
        };

        // Load belief graph (PROJECT GROUNDED)
        let graph_path = implicit_path.join("belief_graph.json");
        let belief_graph: BeliefGraph = if graph_path.exists() {
            let content = fs::read_to_string(&graph_path)?;
            serde_json::from_str(&content)?
        } else {
            BeliefGraph::new()
        };

        Ok(Self {
            preferences,
            patterns,
            corrections,
            beliefs,
            trust_accumulators,
            belief_graph,
            implicit_path,
        })
    }

    /// Save all implicit knowledge to disk
    ///
    /// # Errors
    ///
    /// Returns an error if the data cannot be saved.
    pub fn save(&self) -> Result<()> {
        // Save preferences
        let preferences_path = self.implicit_path.join("preferences.json");
        let content = serde_json::to_string_pretty(&self.preferences)?;
        fs::write(&preferences_path, content)?;

        // Save patterns
        let patterns_path = self.implicit_path.join("patterns.json");
        let content = serde_json::to_string_pretty(&self.patterns)?;
        fs::write(&patterns_path, content)?;

        // Save corrections
        let corrections_path = self.implicit_path.join("corrections.json");
        let content = serde_json::to_string_pretty(&self.corrections)?;
        fs::write(&corrections_path, content)?;

        // Save beliefs (PROJECT GROUNDED)
        let beliefs_path = self.implicit_path.join("beliefs.json");
        let content = serde_json::to_string_pretty(&self.beliefs)?;
        fs::write(&beliefs_path, content)?;

        // Save trust accumulators (PROJECT GROUNDED)
        let trust_path = self.implicit_path.join("trust.json");
        let content = serde_json::to_string_pretty(&self.trust_accumulators)?;
        fs::write(&trust_path, content)?;

        // Save belief graph (PROJECT GROUNDED)
        let graph_path = self.implicit_path.join("belief_graph.json");
        let content = serde_json::to_string_pretty(&self.belief_graph)?;
        fs::write(&graph_path, content)?;

        // Dual-write: mirror all implicit knowledge to SQLite
        self.sync_to_db();

        Ok(())
    }

    /// Mirror current implicit knowledge state to SQLite (non-fatal).
    fn sync_to_db(&self) {
        // Preferences
        for pref in self.preferences.values() {
            crate::db::sync_preference(
                &pref.key,
                &pref.value,
                pref.description.as_deref(),
                pref.confidence,
                pref.reinforcement_count,
                pref.updated_at,
            );
        }

        // Patterns
        for pat in &self.patterns {
            let grounding_str = pat.t1_grounding.map(|g| g.to_string());
            crate::db::sync_pattern(
                &pat.id,
                &pat.pattern_type,
                &pat.description,
                &pat.examples,
                pat.detected_at,
                pat.updated_at,
                pat.confidence,
                pat.occurrence_count,
                grounding_str.as_deref(),
            );
        }

        // Corrections
        for corr in &self.corrections {
            crate::db::sync_correction(
                &corr.mistake,
                &corr.correction,
                corr.context.as_deref(),
                corr.learned_at,
                corr.application_count,
            );
        }

        // Beliefs
        for belief in &self.beliefs {
            let evidence_json =
                serde_json::to_string(&belief.evidence).unwrap_or_else(|_| "[]".to_string());
            let grounding_str = belief.t1_grounding.map(|g| g.to_string());
            crate::db::sync_belief(
                &belief.id,
                &belief.proposition,
                &belief.category,
                belief.confidence,
                &evidence_json,
                grounding_str.as_deref(),
                belief.formed_at,
                belief.updated_at,
                belief.validation_count,
                belief.user_confirmed,
            );
        }

        // Trust accumulators
        for ta in self.trust_accumulators.values() {
            let grounding_str = ta.t1_grounding.map(|g| g.to_string());
            crate::db::sync_trust(
                &ta.domain,
                ta.demonstrations,
                ta.failures,
                ta.created_at,
                ta.updated_at,
                grounding_str.as_deref(),
            );
        }

        // Belief implications
        for impl_edge in &self.belief_graph.implications {
            let strength_str = format!("{:?}", impl_edge.strength).to_lowercase();
            crate::db::sync_implication(
                &impl_edge.from,
                &impl_edge.to,
                &strength_str,
                impl_edge.established_at,
            );
        }
    }

    // ========== Preferences ==========

    /// Get a preference by key
    #[must_use]
    pub fn get_preference(&self, key: &str) -> Option<&Preference> {
        self.preferences.get(key)
    }

    /// Set a preference
    pub fn set_preference(&mut self, preference: Preference) {
        self.preferences.insert(preference.key.clone(), preference);
    }

    /// Set a simple key-value preference
    pub fn set_preference_value(&mut self, key: impl Into<String>, value: serde_json::Value) {
        let key = key.into();
        if let Some(existing) = self.preferences.get_mut(&key) {
            existing.value = value;
            existing.reinforce();
        } else {
            self.preferences
                .insert(key.clone(), Preference::new(key, value));
        }
    }

    /// Reinforce a preference (increase confidence)
    pub fn reinforce_preference(&mut self, key: &str) {
        if let Some(pref) = self.preferences.get_mut(key) {
            pref.reinforce();
        }
    }

    /// List all preferences
    #[must_use]
    pub fn list_preferences(&self) -> Vec<&Preference> {
        self.preferences.values().collect()
    }

    /// Delete a preference
    pub fn delete_preference(&mut self, key: &str) -> Option<Preference> {
        self.preferences.remove(key)
    }

    // ========== Patterns ==========

    /// Add a detected pattern
    pub fn add_pattern(&mut self, pattern: Pattern) {
        // Check if similar pattern exists
        let existing = self.patterns.iter_mut().find(|p| p.id == pattern.id);

        if let Some(existing) = existing {
            // Merge examples
            for example in pattern.examples {
                existing.add_example(example);
            }
        } else {
            self.patterns.push(pattern);
        }
    }

    /// Get a pattern by ID
    #[must_use]
    pub fn get_pattern(&self, id: &str) -> Option<&Pattern> {
        self.patterns.iter().find(|p| p.id == id)
    }

    /// List patterns by type
    #[must_use]
    pub fn list_patterns_by_type(&self, pattern_type: &str) -> Vec<&Pattern> {
        self.patterns
            .iter()
            .filter(|p| p.pattern_type == pattern_type)
            .collect()
    }

    /// List all patterns
    #[must_use]
    pub fn list_patterns(&self) -> &[Pattern] {
        &self.patterns
    }

    /// List patterns grounded to a specific T1 primitive
    #[must_use]
    pub fn list_patterns_by_grounding(&self, primitive: T1Primitive) -> Vec<&Pattern> {
        self.patterns
            .iter()
            .filter(|p| p.t1_grounding == Some(primitive))
            .collect()
    }

    /// List patterns with no T1 grounding (need classification)
    #[must_use]
    pub fn list_ungrounded_patterns(&self) -> Vec<&Pattern> {
        self.patterns
            .iter()
            .filter(|p| p.t1_grounding.is_none())
            .collect()
    }

    /// List patterns sorted by effective confidence (decay-adjusted), descending
    #[must_use]
    pub fn list_patterns_by_relevance(&self) -> Vec<(&Pattern, f64)> {
        let mut scored: Vec<(&Pattern, f64)> = self
            .patterns
            .iter()
            .map(|p| (p, p.effective_confidence()))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    // ========== Corrections ==========

    /// Add a learned correction
    pub fn add_correction(&mut self, correction: Correction) {
        self.corrections.push(correction);
    }

    /// Find corrections relevant to a mistake using fuzzy token matching.
    ///
    /// Uses Jaccard similarity on tokenized strings. Returns corrections
    /// with similarity ≥ threshold (default 0.3), sorted by relevance.
    #[must_use]
    pub fn find_corrections(&self, mistake_query: &str) -> Vec<&Correction> {
        self.find_corrections_fuzzy(mistake_query, DEFAULT_SIMILARITY_THRESHOLD)
    }

    /// Find corrections with a custom similarity threshold (0.0–1.0).
    #[must_use]
    pub fn find_corrections_fuzzy(&self, mistake_query: &str, threshold: f64) -> Vec<&Correction> {
        let mut scored: Vec<(&Correction, f64)> = self
            .corrections
            .iter()
            .map(|c| {
                let sim = token_similarity(mistake_query, &c.mistake);
                (c, sim)
            })
            .filter(|(_, sim)| *sim >= threshold)
            .collect();

        // Sort descending by similarity
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(c, _)| c).collect()
    }

    /// List all corrections
    #[must_use]
    pub fn list_corrections(&self) -> &[Correction] {
        &self.corrections
    }

    /// Mark a correction as applied
    pub fn mark_correction_applied(&mut self, index: usize) {
        if let Some(correction) = self.corrections.get_mut(index) {
            correction.mark_applied();
        }
    }

    // ========== Beliefs (PROJECT GROUNDED) ==========

    /// Add a new belief
    pub fn add_belief(&mut self, belief: Belief) {
        // Check if belief with same ID exists
        let existing = self.beliefs.iter_mut().find(|b| b.id == belief.id);

        if let Some(existing) = existing {
            // Merge evidence from new belief into existing
            existing.add_evidence_batch(belief.evidence);
        } else {
            self.beliefs.push(belief);
        }
    }

    /// Get a belief by ID
    #[must_use]
    pub fn get_belief(&self, id: &str) -> Option<&Belief> {
        self.beliefs.iter().find(|b| b.id == id)
    }

    /// Get a mutable belief by ID
    pub fn get_belief_mut(&mut self, id: &str) -> Option<&mut Belief> {
        self.beliefs.iter_mut().find(|b| b.id == id)
    }

    /// List all beliefs
    #[must_use]
    pub fn list_beliefs(&self) -> &[Belief] {
        &self.beliefs
    }

    /// List beliefs by category
    #[must_use]
    pub fn list_beliefs_by_category(&self, category: &str) -> Vec<&Belief> {
        self.beliefs
            .iter()
            .filter(|b| b.category == category)
            .collect()
    }

    /// List beliefs grounded to a specific T1 primitive
    #[must_use]
    pub fn list_beliefs_by_grounding(&self, primitive: T1Primitive) -> Vec<&Belief> {
        self.beliefs
            .iter()
            .filter(|b| b.t1_grounding == Some(primitive))
            .collect()
    }

    /// List beliefs sorted by effective confidence (decay-adjusted), descending
    #[must_use]
    pub fn list_beliefs_by_confidence(&self) -> Vec<(&Belief, f64)> {
        let mut scored: Vec<(&Belief, f64)> = self
            .beliefs
            .iter()
            .map(|b| (b, b.effective_confidence()))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// List stale beliefs (effective confidence below threshold)
    #[must_use]
    pub fn list_stale_beliefs(&self, threshold: f64) -> Vec<&Belief> {
        self.beliefs
            .iter()
            .filter(|b| b.is_stale(threshold))
            .collect()
    }

    /// List ungrounded beliefs (need T1 classification)
    #[must_use]
    pub fn list_ungrounded_beliefs(&self) -> Vec<&Belief> {
        self.beliefs
            .iter()
            .filter(|b| b.t1_grounding.is_none())
            .collect()
    }

    /// Add evidence to an existing belief
    ///
    /// Returns `true` if belief was found and updated, `false` otherwise.
    pub fn add_evidence_to_belief(&mut self, belief_id: &str, evidence: EvidenceRef) -> bool {
        if let Some(belief) = self.beliefs.iter_mut().find(|b| b.id == belief_id) {
            belief.add_evidence(evidence);
            true
        } else {
            false
        }
    }

    /// Record a validation attempt on a belief
    ///
    /// Returns `true` if belief was found and updated, `false` otherwise.
    pub fn validate_belief(&mut self, belief_id: &str, success: bool) -> bool {
        if let Some(belief) = self.beliefs.iter_mut().find(|b| b.id == belief_id) {
            belief.record_validation(success);
            true
        } else {
            false
        }
    }

    /// Create a belief from hypothesis (lower initial confidence)
    pub fn hypothesize(
        &mut self,
        id: impl Into<String>,
        proposition: impl Into<String>,
        category: impl Into<String>,
    ) -> &Belief {
        let belief = Belief::from_hypothesis(id, proposition, category);
        self.beliefs.push(belief);
        // SAFETY: just pushed — index is guaranteed valid
        &self.beliefs[self.beliefs.len() - 1]
    }

    /// Delete a belief by ID
    pub fn delete_belief(&mut self, id: &str) -> Option<Belief> {
        let pos = self.beliefs.iter().position(|b| b.id == id)?;
        Some(self.beliefs.remove(pos))
    }

    /// Prune stale beliefs below threshold
    ///
    /// Returns number of beliefs pruned.
    pub fn prune_stale_beliefs(&mut self, threshold: f64) -> usize {
        let before = self.beliefs.len();
        self.beliefs
            .retain(|b| !b.is_stale(threshold) || b.user_confirmed);
        before - self.beliefs.len()
    }

    // ========== Trust Accumulators (PROJECT GROUNDED) ==========

    /// Get or create trust accumulator for a domain
    pub fn get_or_create_trust(&mut self, domain: &str) -> &mut TrustAccumulator {
        self.trust_accumulators
            .entry(domain.to_string())
            .or_insert_with(|| TrustAccumulator::new(domain))
    }

    /// Get trust accumulator for a domain (immutable)
    #[must_use]
    pub fn get_trust(&self, domain: &str) -> Option<&TrustAccumulator> {
        self.trust_accumulators.get(domain)
    }

    /// Record a success in a domain
    pub fn record_trust_success(&mut self, domain: &str) {
        self.get_or_create_trust(domain).record_success();
    }

    /// Record a failure in a domain
    pub fn record_trust_failure(&mut self, domain: &str) {
        self.get_or_create_trust(domain).record_failure();
    }

    /// List all trust accumulators
    #[must_use]
    pub fn list_trust(&self) -> Vec<&TrustAccumulator> {
        self.trust_accumulators.values().collect()
    }

    /// Get global trust score (average across all domains)
    #[must_use]
    pub fn global_trust_score(&self) -> f64 {
        if self.trust_accumulators.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.trust_accumulators.values().map(|t| t.score()).sum();
        sum / self.trust_accumulators.len() as f64
    }

    // ========== Belief Graph (PROJECT GROUNDED) ==========

    /// Get the belief graph
    #[must_use]
    pub fn belief_graph(&self) -> &BeliefGraph {
        &self.belief_graph
    }

    /// Get mutable belief graph
    pub fn belief_graph_mut(&mut self) -> &mut BeliefGraph {
        &mut self.belief_graph
    }

    /// Add an implication between beliefs (with cycle check)
    pub fn add_belief_implication(
        &mut self,
        from: &str,
        to: &str,
        strength: ImplicationStrength,
    ) -> bool {
        if self.belief_graph.would_create_cycle(from, to) {
            return false;
        }
        self.belief_graph.add_implication(from, to, strength);
        true
    }

    // ========== Bulk Operations ==========

    /// Clear all implicit knowledge
    pub fn clear_all(&mut self) {
        self.preferences.clear();
        self.patterns.clear();
        self.corrections.clear();
        self.beliefs.clear();
        self.trust_accumulators.clear();
        self.belief_graph = BeliefGraph::new();
    }

    /// Promote a tracked pattern into the formal pattern store.
    ///
    /// Call this when a pattern in `pattern_tracker.json` exceeds the
    /// promotion threshold. Returns `true` if promoted (new), `false`
    /// if the pattern already existed (merged instead).
    ///
    // VIOLATION CONFESSION
    // Commandment Violated: 4 — WRAP
    // Code in Question: `occurrence_count: u32` parameter
    // Nature of Violation: Naked u32 for occurrence count rather than newtype
    // Root Cause: Pre-existing Pattern struct uses raw u32; changing it here
    //   would break serialization compat with existing patterns.json files
    // Corrected Code: Future PR should introduce `OccurrenceCount(u32)` newtype
    // Prevention: Track as tech debt in patterns.json migration plan
    // I have confessed. The record stands.
    pub fn promote_pattern(
        &mut self,
        id: impl Into<String>,
        pattern_type: impl Into<String>,
        description: impl Into<String>,
        occurrence_count: u32,
        t1_grounding: Option<T1Primitive>,
    ) -> bool {
        let id = id.into();

        // Check if pattern already exists — merge if so
        if let Some(existing) = self.patterns.iter_mut().find(|p| p.id == id) {
            existing.occurrence_count = existing.occurrence_count.saturating_add(occurrence_count);
            existing.updated_at = DateTime::now();
            if existing.t1_grounding.is_none() {
                existing.t1_grounding = t1_grounding;
            }
            // Recalculate confidence from merged count
            existing.confidence = (existing.occurrence_count as f64
                / (existing.occurrence_count as f64 + 2.0))
                .min(0.95);
            return false;
        }

        // Create new promoted pattern
        let mut pattern = Pattern::new(id, pattern_type, description);
        pattern.occurrence_count = occurrence_count;
        pattern.t1_grounding = t1_grounding;
        pattern.confidence = (occurrence_count as f64 / (occurrence_count as f64 + 2.0)).min(0.95);
        self.patterns.push(pattern);
        true
    }

    // ========== Crystallization (Correction → Belief promotion) ==========

    /// Crystallize high-confidence, frequently-applied corrections into beliefs.
    ///
    /// A correction is eligible for crystallization when:
    /// - `effective_confidence() >= min_confidence` (default 0.6)
    /// - `application_count >= min_applications` (default 3)
    ///
    /// Crystallized beliefs inherit:
    /// - The correction text as the belief proposition
    /// - An `Observation` evidence ref citing the correction source
    /// - Category `"correction"` for traceability
    ///
    /// Corrections that crystallize are **not** removed — they remain in the
    /// correction store as source-of-truth. The belief is additive.
    ///
    /// Returns the list of newly created belief IDs.
    pub fn crystallize_corrections(
        &mut self,
        min_confidence: f64,
        min_applications: u32,
    ) -> Vec<String> {
        let mut new_ids = Vec::new();

        // Collect candidates first to avoid borrow conflict
        let candidates: Vec<(String, String, String, u32, f64)> = self
            .corrections
            .iter()
            .filter(|c| {
                c.effective_confidence() >= min_confidence
                    && c.application_count >= min_applications
            })
            .map(|c| {
                let id = format!(
                    "crystallized:{}",
                    c.mistake
                        .chars()
                        .take(40)
                        .collect::<String>()
                        .replace(' ', "_")
                        .to_lowercase()
                );
                (
                    id,
                    c.correction.clone(),
                    c.mistake.clone(),
                    c.application_count,
                    c.effective_confidence(),
                )
            })
            .collect();

        for (id, correction_text, mistake_text, app_count, eff_conf) in candidates {
            // Skip if belief with this ID already exists
            if self.beliefs.iter().any(|b| b.id == id) {
                // Reinforce existing belief instead
                if let Some(existing) = self.beliefs.iter_mut().find(|b| b.id == id) {
                    let evidence = EvidenceRef {
                        id: default_evidence_id(),
                        evidence_type: EvidenceType::Observation,
                        description: format!(
                            "Re-crystallized: {app_count} applications, confidence {eff_conf:.2}"
                        ),
                        weight: 0.3,
                        source: format!(
                            "crystallization:correction:{}",
                            mistake_text.chars().take(30).collect::<String>()
                        ),
                        recorded_at: DateTime::now(),
                        artifact_ref: None,
                        execution_id: None,
                        hypothesis_id: None,
                    };
                    existing.add_evidence(evidence);
                }
                continue;
            }

            let evidence = EvidenceRef {
                id: default_evidence_id(),
                evidence_type: EvidenceType::Observation,
                description: format!(
                    "Crystallized from correction: \"{}\". Applied {app_count} times, confidence {eff_conf:.2}.",
                    mistake_text.chars().take(60).collect::<String>()
                ),
                weight: eff_conf.min(1.0),
                source: format!(
                    "crystallization:correction:{}",
                    mistake_text.chars().take(30).collect::<String>()
                ),
                recorded_at: DateTime::now(),
                artifact_ref: None,
                execution_id: None,
                hypothesis_id: None,
            };

            let mut belief = Belief::new(&id, correction_text, "correction");
            belief.add_evidence(evidence);

            new_ids.push(id);
            self.beliefs.push(belief);
        }

        new_ids
    }

    /// Compute belief-specific statistics
    fn belief_stats(&self) -> (usize, usize, usize, usize) {
        let high_conf = self
            .beliefs
            .iter()
            .filter(|b| b.effective_confidence() >= 0.8)
            .count();
        let stale = self.beliefs.iter().filter(|b| b.is_stale(0.3)).count();
        let confirmed = self.beliefs.iter().filter(|b| b.user_confirmed).count();
        let ungrounded = self
            .beliefs
            .iter()
            .filter(|b| b.t1_grounding.is_none())
            .count();
        (high_conf, stale, confirmed, ungrounded)
    }

    /// Get summary statistics
    #[must_use]
    pub fn stats(&self) -> ImplicitStats {
        let high_confidence_prefs = self
            .preferences
            .values()
            .filter(|p| p.confidence >= 0.8)
            .count();
        let high_confidence_patterns = self.patterns.iter().filter(|p| p.confidence >= 0.8).count();
        let decayed_patterns = self
            .patterns
            .iter()
            .filter(|p| p.effective_confidence() < 0.5)
            .count();
        let ungrounded_patterns = self
            .patterns
            .iter()
            .filter(|p| p.t1_grounding.is_none())
            .count();
        let (high_confidence_beliefs, stale_beliefs, user_confirmed_beliefs, ungrounded_beliefs) =
            self.belief_stats();

        ImplicitStats {
            total_preferences: self.preferences.len(),
            total_patterns: self.patterns.len(),
            total_corrections: self.corrections.len(),
            total_beliefs: self.beliefs.len(),
            high_confidence_preferences: high_confidence_prefs,
            high_confidence_patterns,
            high_confidence_beliefs,
            decayed_patterns,
            stale_beliefs,
            user_confirmed_beliefs,
            ungrounded_patterns,
            ungrounded_beliefs,
        }
    }
}

/// Statistics about implicit knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplicitStats {
    /// Total number of preferences
    pub total_preferences: usize,

    /// Total number of patterns
    pub total_patterns: usize,

    /// Total number of corrections
    pub total_corrections: usize,

    /// Total number of beliefs (PROJECT GROUNDED)
    pub total_beliefs: usize,

    /// Preferences with confidence >= 0.8
    pub high_confidence_preferences: usize,

    /// Patterns with confidence >= 0.8
    pub high_confidence_patterns: usize,

    /// Beliefs with effective confidence >= 0.8
    pub high_confidence_beliefs: usize,

    /// Patterns whose effective confidence has decayed below 0.5
    pub decayed_patterns: usize,

    /// Beliefs whose effective confidence has decayed below 0.3
    pub stale_beliefs: usize,

    /// Beliefs explicitly confirmed by user
    pub user_confirmed_beliefs: usize,

    /// Patterns with no T1 primitive grounding
    pub ungrounded_patterns: usize,

    /// Beliefs with no T1 primitive grounding
    pub ungrounded_beliefs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preference_reinforcement() {
        let mut pref = Preference::new("test", serde_json::json!("value"));
        assert_eq!(pref.confidence, 0.5);
        assert_eq!(pref.reinforcement_count, 1);

        pref.reinforce();
        assert!(pref.confidence > 0.5);
        assert_eq!(pref.reinforcement_count, 2);

        // Reinforce many times
        for _ in 0..10 {
            pref.reinforce();
        }
        assert!(pref.confidence > 0.9);
        assert!(pref.confidence < 1.0); // Never quite reaches 1.0
    }

    #[test]
    fn test_preference_weakening() {
        let mut pref = Preference::new("test", serde_json::json!("value"));
        pref.reinforce();
        pref.reinforce();
        assert_eq!(pref.reinforcement_count, 3);

        pref.weaken();
        assert_eq!(pref.reinforcement_count, 2);

        pref.weaken();
        pref.weaken();
        pref.weaken(); // Can't go below 0
        assert_eq!(pref.reinforcement_count, 0);
        assert_eq!(pref.confidence, 0.0);
    }

    #[test]
    fn test_pattern_examples() {
        let mut pattern = Pattern::new("test_pattern", "naming", "Test pattern");
        assert_eq!(pattern.occurrence_count, 1);

        pattern.add_example("example1");
        pattern.add_example("example2");
        assert_eq!(pattern.occurrence_count, 3);
        assert_eq!(pattern.examples.len(), 2);
        assert!(pattern.confidence > 0.5);
    }

    #[test]
    fn test_correction() {
        let mut correction = Correction::new("used unwrap", "use expect with message");
        assert_eq!(correction.application_count, 0);

        correction.mark_applied();
        correction.mark_applied();
        assert_eq!(correction.application_count, 2);
    }

    // ========== CTVP Phase 0: Edge Case Tests ==========

    #[test]
    fn test_preference_unicode_key_and_value() {
        let pref = Preference::new("编程风格", serde_json::json!("函数式 🦀"));
        assert_eq!(pref.key, "编程风格");
        assert_eq!(pref.value, serde_json::json!("函数式 🦀"));
    }

    #[test]
    fn test_preference_complex_json_value() {
        let complex_value = serde_json::json!({
            "style": "functional",
            "prefer_async": true,
            "max_line_length": 100,
            "languages": ["rust", "typescript"]
        });

        let pref = Preference::new("code_style", complex_value.clone());
        assert_eq!(pref.value, complex_value);
    }

    #[test]
    fn test_preference_extreme_reinforcement() {
        let mut pref = Preference::new("test", serde_json::json!(true));

        // Reinforce 10,000 times
        for _ in 0..10_000 {
            pref.reinforce();
        }

        // Confidence should be very close to 1.0 but never reach it
        assert!(pref.confidence > 0.9999);
        assert!(pref.confidence < 1.0);
        assert_eq!(pref.reinforcement_count, 10_001);
    }

    #[test]
    fn test_preference_alternating_reinforce_weaken() {
        let mut pref = Preference::new("test", serde_json::json!("value"));

        // Alternate reinforcement and weakening
        for _ in 0..100 {
            pref.reinforce();
            pref.weaken();
        }

        // Should end up at initial count since we alternate
        assert_eq!(pref.reinforcement_count, 1);
        assert_eq!(pref.confidence, 0.5);
    }

    #[test]
    fn test_preference_description() {
        let mut pref = Preference::new("indent_style", serde_json::json!("tabs"));
        pref.description = Some("User prefers tabs for indentation".into());

        assert_eq!(
            pref.description,
            Some("User prefers tabs for indentation".into())
        );
    }

    #[test]
    fn test_pattern_empty_examples() {
        let pattern = Pattern::new("test", "naming", "Test pattern with no examples");
        assert!(pattern.examples.is_empty());
        assert_eq!(pattern.occurrence_count, 1);
        assert_eq!(pattern.confidence, 0.5);
    }

    #[test]
    fn test_pattern_many_examples() {
        let mut pattern = Pattern::new("naming_convention", "naming", "snake_case naming");

        // Add 100 examples
        for i in 0..100 {
            pattern.add_example(format!("example_{}", i));
        }

        assert_eq!(pattern.examples.len(), 100);
        assert_eq!(pattern.occurrence_count, 101); // 1 initial + 100 added

        // Confidence should be capped at 0.95
        assert!(pattern.confidence <= 0.95);
        assert!(pattern.confidence > 0.9);
    }

    #[test]
    fn test_pattern_confidence_formula() {
        let mut pattern = Pattern::new("test", "test", "Test");

        // Verify the confidence formula: count / (count + 2), capped at 0.95
        pattern.add_example("ex1"); // count = 2, confidence = 2/(2+2) = 0.5
        assert!((pattern.confidence - 0.5).abs() < 0.001);

        pattern.add_example("ex2"); // count = 3, confidence = 3/(3+2) = 0.6
        assert!((pattern.confidence - 0.6).abs() < 0.001);

        pattern.add_example("ex3"); // count = 4, confidence = 4/(4+2) = 0.667
        assert!((pattern.confidence - (4.0 / 6.0)).abs() < 0.001);
    }

    #[test]
    fn test_correction_with_context() {
        let mut correction = Correction::new("unwrap()", "expect() with message");
        correction.context = Some("In error handling code".into());

        assert_eq!(correction.context, Some("In error handling code".into()));
    }

    #[test]
    fn test_correction_unicode() {
        let correction = Correction::new("使用了 unwrap", "应该使用 expect 并添加错误消息");

        assert_eq!(correction.mistake, "使用了 unwrap");
        assert_eq!(correction.correction, "应该使用 expect 并添加错误消息");
    }

    #[test]
    fn test_implicit_stats_empty() {
        let stats = ImplicitStats {
            total_preferences: 0,
            total_patterns: 0,
            total_corrections: 0,
            total_beliefs: 0,
            high_confidence_preferences: 0,
            high_confidence_patterns: 0,
            high_confidence_beliefs: 0,
            decayed_patterns: 0,
            stale_beliefs: 0,
            user_confirmed_beliefs: 0,
            ungrounded_patterns: 0,
            ungrounded_beliefs: 0,
        };

        assert_eq!(stats.total_preferences, 0);
        assert_eq!(stats.total_patterns, 0);
        assert_eq!(stats.total_corrections, 0);
        assert_eq!(stats.total_beliefs, 0);
        assert_eq!(stats.decayed_patterns, 0);
        assert_eq!(stats.stale_beliefs, 0);
        assert_eq!(stats.ungrounded_patterns, 0);
        assert_eq!(stats.ungrounded_beliefs, 0);
    }

    // ========== Fuzzy Matching Tests ==========

    #[test]
    fn test_token_similarity_identical() {
        assert!((token_similarity("use unwrap", "use unwrap") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_token_similarity_disjoint() {
        assert!(token_similarity("apple banana", "cherry date").abs() < f64::EPSILON);
    }

    #[test]
    fn test_token_similarity_partial_overlap() {
        let sim = token_similarity("use unwrap in code", "unwrap code");
        // "unwrap" and "code" overlap — 2 of 4 union tokens
        assert!(sim >= 0.4);
        assert!(sim <= 0.6);
    }

    #[test]
    fn test_token_similarity_case_insensitive() {
        let sim = token_similarity("Use UNWRAP", "use unwrap");
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_token_similarity_empty_strings() {
        assert!((token_similarity("", "") - 1.0).abs() < f64::EPSILON);
        assert!(token_similarity("", "hello").abs() < f64::EPSILON);
        assert!(token_similarity("hello", "").abs() < f64::EPSILON);
    }

    // ========== T1 Grounding Tests ==========

    #[test]
    fn test_pattern_grounding() {
        let mut pattern = Pattern::new("test", "workflow", "Test pattern");
        assert!(pattern.t1_grounding.is_none());

        pattern.set_grounding(T1Primitive::Mapping);
        assert_eq!(pattern.t1_grounding, Some(T1Primitive::Mapping));
    }

    #[test]
    fn test_t1_primitive_display() {
        assert_eq!(T1Primitive::Sequence.to_string(), "sequence");
        assert_eq!(T1Primitive::Mapping.to_string(), "mapping");
        assert_eq!(T1Primitive::Recursion.to_string(), "recursion");
        assert_eq!(T1Primitive::State.to_string(), "state");
        assert_eq!(T1Primitive::Void.to_string(), "void");
        assert_eq!(T1Primitive::Boundary.to_string(), "boundary");
        assert_eq!(T1Primitive::Causality.to_string(), "causality");
    }

    #[test]
    fn test_t1_primitive_serde_roundtrip() {
        let original = T1Primitive::Mapping;
        let json = serde_json::to_string(&original).expect("serialize");
        let deserialized: T1Primitive = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, deserialized);
    }

    // ========== Confidence Decay Tests ==========

    #[test]
    fn test_effective_confidence_fresh_pattern() {
        let pattern = Pattern::new("test", "workflow", "Just created");
        // Just created — no decay
        let eff = pattern.effective_confidence();
        assert!((eff - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_effective_confidence_stale_pattern() {
        let mut pattern = Pattern::new("test", "workflow", "Old pattern");
        pattern.confidence = 0.9;
        // Simulate 30 days ago (one half-life)
        pattern.updated_at = DateTime::now() - nexcore_chrono::Duration::days(30);

        let eff = pattern.effective_confidence();
        // Should be ~0.45 (0.9 * 0.5)
        assert!(eff > 0.40);
        assert!(eff < 0.50);
    }

    #[test]
    fn test_effective_confidence_very_stale() {
        let mut pattern = Pattern::new("test", "workflow", "Ancient");
        pattern.confidence = 0.95;
        // 90 days = 3 half-lives → 0.95 * 0.125 ≈ 0.119
        pattern.updated_at = DateTime::now() - nexcore_chrono::Duration::days(90);

        let eff = pattern.effective_confidence();
        assert!(eff < 0.15);
        assert!(eff > 0.10);
    }

    #[test]
    fn test_preference_weaken_from_zero() {
        let mut pref = Preference::new("test", serde_json::json!("value"));

        // Weaken until count is 0
        pref.weaken();

        // Weaken again from 0 - should not go negative
        pref.weaken();
        pref.weaken();
        pref.weaken();

        assert_eq!(pref.reinforcement_count, 0);
        assert_eq!(pref.confidence, 0.0);
    }

    #[test]
    fn test_preference_null_json_value() {
        let pref = Preference::new("optional_setting", serde_json::Value::Null);
        assert!(pref.value.is_null());
    }

    #[test]
    fn test_preference_array_json_value() {
        let pref = Preference::new(
            "favorite_crates",
            serde_json::json!(["serde", "tokio", "axum", "thiserror"]),
        );

        assert!(pref.value.is_array());
        assert_eq!(pref.value.as_array().unwrap().len(), 4);
    }

    // ========== Belief Tests (PROJECT GROUNDED) ==========

    #[test]
    fn test_belief_creation() {
        let belief = Belief::new("test_belief", "Rust is memory safe", "capability");
        assert_eq!(belief.id, "test_belief");
        assert_eq!(belief.proposition, "Rust is memory safe");
        assert_eq!(belief.category, "capability");
        assert_eq!(belief.confidence, 0.5);
        assert!(belief.evidence.is_empty());
        assert!(!belief.user_confirmed);
    }

    #[test]
    fn test_belief_from_hypothesis() {
        let belief = Belief::from_hypothesis("hyp_1", "This function is pure", "behavior");
        assert_eq!(belief.confidence, 0.3); // Hypotheses start lower
    }

    #[test]
    fn test_evidence_creation() {
        let supporting = EvidenceRef::supporting(
            "ev_1",
            EvidenceType::Observation,
            "Observed correct behavior",
            "session:abc123",
        );
        assert_eq!(supporting.weight, 1.0);

        let contradicting = EvidenceRef::contradicting(
            "ev_2",
            EvidenceType::TestResult,
            "Test failed",
            "test:unit_foo",
        );
        assert_eq!(contradicting.weight, -1.0);

        let weighted = EvidenceRef::weighted(
            "ev_3",
            EvidenceType::Inference,
            "Weak signal",
            0.3,
            "system:infer",
        );
        assert!((weighted.weight - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_belief_add_evidence() {
        let mut belief = Belief::new("test", "Test proposition", "test");
        assert_eq!(belief.confidence, 0.5);

        // Add supporting evidence
        belief.add_evidence(EvidenceRef::supporting(
            "ev_1",
            EvidenceType::Observation,
            "Observed",
            "test",
        ));
        assert!(belief.confidence > 0.0);
        assert_eq!(belief.evidence.len(), 1);
    }

    #[test]
    fn test_belief_mixed_evidence() {
        let mut belief = Belief::new("test", "Test proposition", "test");

        // Add supporting evidence
        belief.add_evidence(EvidenceRef::supporting(
            "ev_1",
            EvidenceType::Observation,
            "Good",
            "test",
        ));
        belief.add_evidence(EvidenceRef::supporting(
            "ev_2",
            EvidenceType::TestResult,
            "Passed",
            "test",
        ));

        // Add contradicting evidence
        belief.add_evidence(EvidenceRef::contradicting(
            "ev_3",
            EvidenceType::TestResult,
            "Failed",
            "test",
        ));

        // Net sentiment should still be positive (2 supporting, 1 contradicting)
        assert!(belief.evidence_sentiment() > 0.0);
        assert_eq!(belief.evidence.len(), 3);
    }

    #[test]
    fn test_belief_validation() {
        let mut belief = Belief::new("test", "Test proposition", "test");
        assert_eq!(belief.validation_count, 0);

        belief.record_validation(true);
        assert_eq!(belief.validation_count, 1);
        assert!(belief.evidence.len() == 1);

        belief.record_validation(false);
        assert_eq!(belief.validation_count, 2);
        assert!(belief.evidence.len() == 2);
    }

    #[test]
    fn test_belief_confirmation() {
        let mut belief = Belief::new("test", "Test proposition", "test");
        assert!(!belief.user_confirmed);

        belief.confirm();
        assert!(belief.user_confirmed);
        assert!(
            belief
                .evidence
                .iter()
                .any(|e| e.evidence_type == EvidenceType::UserFeedback)
        );
    }

    #[test]
    fn test_belief_rejection() {
        let mut belief = Belief::new("test", "Test proposition", "test");
        belief.confirm();
        assert!(belief.user_confirmed);

        belief.reject();
        assert!(!belief.user_confirmed);
    }

    #[test]
    fn test_belief_grounding() {
        let mut belief = Belief::new("test", "Sequence operations", "behavior");
        assert!(belief.t1_grounding.is_none());

        belief.set_grounding(T1Primitive::Sequence);
        assert_eq!(belief.t1_grounding, Some(T1Primitive::Sequence));
    }

    #[test]
    fn test_belief_effective_confidence_fresh() {
        let belief = Belief::new("test", "Fresh belief", "test");
        let eff = belief.effective_confidence();
        assert!((eff - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_belief_effective_confidence_stale() {
        let mut belief = Belief::new("test", "Stale belief", "test");
        belief.confidence = 0.8;
        belief.updated_at = DateTime::now() - nexcore_chrono::Duration::days(30);

        let eff = belief.effective_confidence();
        // After one half-life, should be ~0.4
        assert!(eff > 0.35);
        assert!(eff < 0.45);
    }

    #[test]
    fn test_belief_is_stale() {
        let mut belief = Belief::new("test", "Test", "test");
        belief.confidence = 0.5;
        belief.updated_at = DateTime::now() - nexcore_chrono::Duration::days(60);

        // After 2 half-lives: 0.5 * 0.25 = 0.125
        assert!(belief.is_stale(0.2));
        assert!(!belief.is_stale(0.1));
    }

    #[test]
    fn test_belief_user_confirmed_stale_floor() {
        let mut belief = Belief::new("test", "Confirmed belief", "test");
        belief.confirm();
        belief.confidence = 0.8;
        belief.updated_at = DateTime::now() - nexcore_chrono::Duration::days(30); // 1 half-life

        // Effective confidence ≈ 0.4 (0.8 * 0.5)
        // Confirmed beliefs use min(threshold, 0.1) as floor
        // So at threshold 0.3: min(0.3, 0.1) = 0.1, eff 0.4 > 0.1 → not stale
        assert!(!belief.is_stale(0.3));

        // Unconfirmed belief with same decay WOULD be stale at threshold 0.5
        let mut unconfirmed = Belief::new("test2", "Not confirmed", "test");
        unconfirmed.confidence = 0.8;
        unconfirmed.updated_at = DateTime::now() - nexcore_chrono::Duration::days(30);
        assert!(unconfirmed.is_stale(0.5)); // 0.4 < 0.5 → stale
    }

    #[test]
    fn test_evidence_type_display() {
        assert_eq!(EvidenceType::Observation.to_string(), "observation");
        assert_eq!(EvidenceType::TestResult.to_string(), "test_result");
        assert_eq!(EvidenceType::UserFeedback.to_string(), "user_feedback");
        assert_eq!(EvidenceType::Inference.to_string(), "inference");
        assert_eq!(EvidenceType::Authority.to_string(), "authority");
        assert_eq!(EvidenceType::Prior.to_string(), "prior");
    }

    #[test]
    fn test_evidence_weight_clamping() {
        let evidence = EvidenceRef::weighted("test", EvidenceType::Observation, "desc", 5.0, "src");
        assert!((evidence.weight - 1.0).abs() < f64::EPSILON); // Clamped to 1.0

        let evidence =
            EvidenceRef::weighted("test", EvidenceType::Observation, "desc", -5.0, "src");
        assert!((evidence.weight - (-1.0)).abs() < f64::EPSILON); // Clamped to -1.0
    }

    // ========== Crystallization Tests ==========

    fn make_applied_correction(mistake: &str, correction: &str, applications: u32) -> Correction {
        let mut c = Correction::new(mistake, correction);
        c.confidence = 0.8;
        c.application_count = applications;
        c
    }

    #[test]
    fn test_crystallize_creates_belief_from_high_confidence_correction() {
        let mut store = ImplicitKnowledge::empty_for_test();

        let c = make_applied_correction("used unwrap", "use proper error handling", 5);
        store.add_correction(c);

        let ids = store.crystallize_corrections(0.6, 3);
        assert_eq!(ids.len(), 1);
        assert!(ids[0].starts_with("crystallized:"));

        let belief = store.get_belief(&ids[0]);
        assert!(belief.is_some());
        let belief = belief.unwrap();
        assert_eq!(belief.category, "correction");
        assert_eq!(belief.proposition, "use proper error handling");
        assert!(!belief.evidence.is_empty());
    }

    #[test]
    fn test_crystallize_skips_low_confidence() {
        let mut store = ImplicitKnowledge::empty_for_test();

        let mut c = Correction::new("some mistake", "some fix");
        c.confidence = 0.3; // Below threshold
        c.application_count = 10;
        store.add_correction(c);

        let ids = store.crystallize_corrections(0.6, 3);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_crystallize_skips_low_application_count() {
        let mut store = ImplicitKnowledge::empty_for_test();

        let mut c = Correction::new("some mistake", "some fix");
        c.confidence = 0.9;
        c.application_count = 1; // Below threshold
        store.add_correction(c);

        let ids = store.crystallize_corrections(0.6, 3);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_crystallize_reinforces_existing_belief() {
        let mut store = ImplicitKnowledge::empty_for_test();

        let c = make_applied_correction("used unwrap", "use proper error handling", 5);
        store.add_correction(c);

        // First crystallization
        let ids1 = store.crystallize_corrections(0.6, 3);
        assert_eq!(ids1.len(), 1);

        let ev_count_before = store.get_belief(&ids1[0]).unwrap().evidence.len();

        // Second crystallization — should reinforce, not duplicate
        let ids2 = store.crystallize_corrections(0.6, 3);
        assert!(ids2.is_empty()); // No NEW beliefs

        let ev_count_after = store.get_belief(&ids1[0]).unwrap().evidence.len();
        assert_eq!(ev_count_after, ev_count_before + 1); // Evidence added
    }

    #[test]
    fn test_crystallize_multiple_corrections() {
        let mut store = ImplicitKnowledge::empty_for_test();

        store.add_correction(make_applied_correction("mistake A", "fix A", 4));
        store.add_correction(make_applied_correction("mistake B", "fix B", 3));
        store.add_correction(make_applied_correction("mistake C", "fix C", 1)); // Below threshold

        let ids = store.crystallize_corrections(0.6, 3);
        assert_eq!(ids.len(), 2); // A and B, not C
    }
}
