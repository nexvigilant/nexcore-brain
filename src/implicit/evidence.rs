//! Evidence types for belief support
//!
//! Evidence enables the GROUNDED hypothesis → test → belief integration loop.

use nexcore_chrono::DateTime;
use serde::{Deserialize, Serialize};
use std::fmt;

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

/// Reference to evidence supporting or contradicting a belief.
///
/// PROJECT GROUNDED integration: Evidence enables the
/// hypothesis → test → belief integration loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRef {
    /// Unique evidence identifier
    pub id: String,

    /// Type of evidence
    pub evidence_type: EvidenceType,

    /// Description of what this evidence shows
    pub description: String,

    /// Weight of this evidence (positive = supports, negative = contradicts)
    /// Range: -1.0 to 1.0
    pub weight: f64,

    /// Source of the evidence (e.g., "session:abc123", "`test:unit_foo`", "user:explicit")
    pub source: String,

    /// When this evidence was recorded
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
