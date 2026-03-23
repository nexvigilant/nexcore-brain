//! Domain-scoped trust accumulators for competence tracking
//!
//! Trust is earned through demonstrated competence, not granted through compliance.

use nexcore_chrono::DateTime;
use serde::{Deserialize, Serialize};

use super::t1::T1Primitive;

/// Domain-scoped trust accumulator for competence tracking.
///
/// Trust is earned through demonstrated competence, not granted through compliance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustAccumulator {
    /// Domain this trust applies to (e.g., "`rust_development`", "`pv_signal_detection`")
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
        f64::from(self.demonstrations) / f64::from(total)
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
