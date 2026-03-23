//! Belief state tracking for PROJECT GROUNDED feedback loops
//!
//! A belief represents a propositional attitude held by the system,
//! with confidence derived from accumulated evidence. Beliefs decay
//! over time without reinforcing evidence (same decay model as Pattern).

use nexcore_chrono::DateTime;
use serde::{Deserialize, Serialize};

use super::evidence::{EvidenceRef, EvidenceType};
use super::t1::T1Primitive;

/// Belief decay half-life in days (same as Pattern for consistency)
const BELIEF_DECAY_HALF_LIFE_DAYS: f64 = 30.0;

/// Belief state tracking for PROJECT GROUNDED feedback loops.
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
        let normalized = f64::midpoint(avg_weight, 1.0);

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
