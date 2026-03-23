//! # Cross-Domain Transfer Confidence
//!
//! Maps nexcore-brain concepts (working memory, beliefs, patterns, artifacts)
//! to analogous concepts in other domains (PV, Biology, Economics, Psychology).
//!
//! ## Transfer Mapping Table
//!
//! | Brain Type | PV Analog | Biology Analog | Confidence |
//! |-----------|-----------|---------------|------------|
//! | BrainSession | Case narrative | Episodic memory | 0.88 |
//! | Artifact | ICSR document | Long-term memory trace | 0.85 |
//! | Pattern | Signal detection | Learned reflex | 0.90 |
//! | Belief | Causality assessment | Bayesian prior | 0.85 |
//! | TrustAccumulator | Reporter reliability | Immune memory | 0.82 |
//! | Correction | Regulatory amendment | Error correction (DNA repair) | 0.80 |
//! | Preference | Standard operating procedure | Conditioned response | 0.78 |
//! | EvidenceRef | Supporting documentation | Experimental observation | 0.92 |
//! | BeliefGraph | Causal network | Neural network | 0.80 |
//! | CodeTracker | Change audit trail | Cell lineage tracking | 0.85 |
//! | PipelineState | Workflow status | Metabolic pathway | 0.78 |
//! | ImplicitKnowledge | Safety database | Acquired immunity | 0.75 |

use serde::{Deserialize, Serialize};

/// A cross-domain transfer mapping from brain concepts to other domains.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransferMapping {
    /// Brain type name
    pub brain_type: &'static str,
    /// Target domain
    pub domain: &'static str,
    /// Analogous concept in target domain
    pub analog: &'static str,
    /// Transfer confidence (0.0-1.0)
    pub confidence: f64,
}

/// All cross-domain transfer mappings for brain types.
///
/// Each brain type maps to 3 domains (PV, Biology, Psychology/Economics)
/// with calibrated confidence based on structural + semantic similarity.
pub fn transfer_mappings() -> Vec<TransferMapping> {
    vec![
        // ========== BrainSession (ς+π+λ+∂) ==========
        TransferMapping {
            brain_type: "BrainSession",
            domain: "PV",
            analog: "Case narrative (ICSR lifecycle)",
            confidence: 0.88,
        },
        TransferMapping {
            brain_type: "BrainSession",
            domain: "Biology",
            analog: "Episodic memory (hippocampal encoding)",
            confidence: 0.85,
        },
        TransferMapping {
            brain_type: "BrainSession",
            domain: "Psychology",
            analog: "Working memory session (Baddeley model)",
            confidence: 0.92,
        },
        // ========== Artifact (ς+π+σ) ==========
        TransferMapping {
            brain_type: "Artifact",
            domain: "PV",
            analog: "ICSR document with version history",
            confidence: 0.85,
        },
        TransferMapping {
            brain_type: "Artifact",
            domain: "Biology",
            analog: "Long-term memory trace (engram)",
            confidence: 0.82,
        },
        TransferMapping {
            brain_type: "Artifact",
            domain: "Economics",
            analog: "Audit trail / ledger entry",
            confidence: 0.88,
        },
        // ========== Pattern (ς+ν+∃) ==========
        TransferMapping {
            brain_type: "Pattern",
            domain: "PV",
            analog: "Disproportionality signal (PRR/ROR)",
            confidence: 0.90,
        },
        TransferMapping {
            brain_type: "Pattern",
            domain: "Biology",
            analog: "Conditioned reflex (Pavlovian)",
            confidence: 0.85,
        },
        TransferMapping {
            brain_type: "Pattern",
            domain: "Psychology",
            analog: "Schema (Piaget cognitive structure)",
            confidence: 0.88,
        },
        // ========== Belief (ς+∃+N+→+ν) ==========
        TransferMapping {
            brain_type: "Belief",
            domain: "PV",
            analog: "Causality assessment (Naranjo score)",
            confidence: 0.85,
        },
        TransferMapping {
            brain_type: "Belief",
            domain: "Biology",
            analog: "Bayesian prior (predictive coding)",
            confidence: 0.82,
        },
        TransferMapping {
            brain_type: "Belief",
            domain: "Psychology",
            analog: "Attitude (Fishbein-Ajzen model)",
            confidence: 0.80,
        },
        // ========== TrustAccumulator (ς+N+ν) ==========
        TransferMapping {
            brain_type: "TrustAccumulator",
            domain: "PV",
            analog: "Reporter reliability score",
            confidence: 0.82,
        },
        TransferMapping {
            brain_type: "TrustAccumulator",
            domain: "Biology",
            analog: "Immune memory (T-cell repertoire)",
            confidence: 0.78,
        },
        TransferMapping {
            brain_type: "TrustAccumulator",
            domain: "Economics",
            analog: "Credit score (accumulated trust)",
            confidence: 0.90,
        },
        // ========== Correction (μ+→) ==========
        TransferMapping {
            brain_type: "Correction",
            domain: "PV",
            analog: "Regulatory amendment / CAPA",
            confidence: 0.80,
        },
        TransferMapping {
            brain_type: "Correction",
            domain: "Biology",
            analog: "DNA mismatch repair (MMR pathway)",
            confidence: 0.78,
        },
        TransferMapping {
            brain_type: "Correction",
            domain: "Psychology",
            analog: "Error-driven learning (Rescorla-Wagner)",
            confidence: 0.85,
        },
        // ========== EvidenceRef (∃+N+→) ==========
        TransferMapping {
            brain_type: "EvidenceRef",
            domain: "PV",
            analog: "Supporting documentation (E2B field)",
            confidence: 0.92,
        },
        TransferMapping {
            brain_type: "EvidenceRef",
            domain: "Biology",
            analog: "Experimental observation (lab result)",
            confidence: 0.88,
        },
        TransferMapping {
            brain_type: "EvidenceRef",
            domain: "Economics",
            analog: "Market indicator / economic data point",
            confidence: 0.80,
        },
        // ========== BeliefGraph (ρ+→+ς) ==========
        TransferMapping {
            brain_type: "BeliefGraph",
            domain: "PV",
            analog: "Causal network (Bradford Hill criteria web)",
            confidence: 0.80,
        },
        TransferMapping {
            brain_type: "BeliefGraph",
            domain: "Biology",
            analog: "Neural network (connectome)",
            confidence: 0.85,
        },
        TransferMapping {
            brain_type: "BeliefGraph",
            domain: "Psychology",
            analog: "Semantic network (Collins-Quillian)",
            confidence: 0.88,
        },
        // ========== CodeTracker (ς+π+μ+λ) ==========
        TransferMapping {
            brain_type: "CodeTracker",
            domain: "PV",
            analog: "Change audit trail (21 CFR Part 11)",
            confidence: 0.85,
        },
        TransferMapping {
            brain_type: "CodeTracker",
            domain: "Biology",
            analog: "Cell lineage tracking (fate mapping)",
            confidence: 0.78,
        },
        TransferMapping {
            brain_type: "CodeTracker",
            domain: "Economics",
            analog: "Transaction ledger (blockchain)",
            confidence: 0.82,
        },
        // ========== PipelineState (ς+σ+π) ==========
        TransferMapping {
            brain_type: "PipelineState",
            domain: "PV",
            analog: "ICSR processing workflow status",
            confidence: 0.82,
        },
        TransferMapping {
            brain_type: "PipelineState",
            domain: "Biology",
            analog: "Metabolic pathway (glycolysis stages)",
            confidence: 0.78,
        },
        TransferMapping {
            brain_type: "PipelineState",
            domain: "Economics",
            analog: "Supply chain state tracking",
            confidence: 0.80,
        },
        // ========== Preference (ς+ν+N) ==========
        TransferMapping {
            brain_type: "Preference",
            domain: "PV",
            analog: "Standard operating procedure (SOP)",
            confidence: 0.78,
        },
        TransferMapping {
            brain_type: "Preference",
            domain: "Biology",
            analog: "Tropism (growth direction preference)",
            confidence: 0.72,
        },
        TransferMapping {
            brain_type: "Preference",
            domain: "Psychology",
            analog: "Habitual response (operant conditioning)",
            confidence: 0.85,
        },
        // ========== ImplicitKnowledge (ς+π+μ+σ+∃+ρ) ==========
        TransferMapping {
            brain_type: "ImplicitKnowledge",
            domain: "PV",
            analog: "Safety database (EVDAS/EudraVigilance)",
            confidence: 0.75,
        },
        TransferMapping {
            brain_type: "ImplicitKnowledge",
            domain: "Biology",
            analog: "Acquired immunity (adaptive immune system)",
            confidence: 0.80,
        },
        TransferMapping {
            brain_type: "ImplicitKnowledge",
            domain: "Psychology",
            analog: "Implicit memory (procedural knowledge)",
            confidence: 0.90,
        },
    ]
}

/// Get transfer mappings for a specific brain type.
pub fn transfers_for(brain_type: &str) -> Vec<&'static TransferMapping> {
    // Use leaked static for O(1) per-call after first invocation
    use std::sync::LazyLock;
    static MAPPINGS: LazyLock<Vec<TransferMapping>> = LazyLock::new(transfer_mappings);

    MAPPINGS
        .iter()
        .filter(|m| m.brain_type == brain_type)
        .collect()
}

/// Get the average transfer confidence for a brain type across all domains.
pub fn avg_confidence(brain_type: &str) -> f64 {
    let mappings = transfers_for(brain_type);
    if mappings.is_empty() {
        return 0.0;
    }
    let sum: f64 = mappings.iter().map(|m| m.confidence).sum();
    sum / mappings.len() as f64
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_mappings_count() {
        let mappings = transfer_mappings();
        // 12 brain types × 3 domains each = 36 mappings
        assert_eq!(mappings.len(), 36);
    }

    #[test]
    fn test_all_confidences_in_range() {
        for m in transfer_mappings() {
            assert!(
                m.confidence >= 0.0 && m.confidence <= 1.0,
                "Out of range: {} → {} = {}",
                m.brain_type,
                m.domain,
                m.confidence
            );
        }
    }

    #[test]
    fn test_transfers_for_pattern() {
        let transfers = transfers_for("Pattern");
        assert_eq!(transfers.len(), 3);
        assert!(transfers.iter().any(|t| t.domain == "PV"));
        assert!(transfers.iter().any(|t| t.domain == "Biology"));
        assert!(transfers.iter().any(|t| t.domain == "Psychology"));
    }

    #[test]
    fn test_transfers_for_nonexistent() {
        let transfers = transfers_for("Nonexistent");
        assert!(transfers.is_empty());
    }

    #[test]
    fn test_avg_confidence_belief() {
        let avg = avg_confidence("Belief");
        // PV: 0.85, Biology: 0.82, Psychology: 0.80 → avg: 0.823
        assert!(avg > 0.80);
        assert!(avg < 0.86);
    }

    #[test]
    fn test_avg_confidence_nonexistent() {
        assert!((avg_confidence("Nonexistent") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_highest_confidence_mappings() {
        let mappings = transfer_mappings();
        let max = mappings
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .unwrap();
        // BrainSession→Psychology should be 0.92 (highest)
        assert!((max.confidence - 0.92).abs() < f64::EPSILON);
    }

    #[test]
    fn test_domain_coverage() {
        let mappings = transfer_mappings();
        let domains: std::collections::HashSet<&str> = mappings.iter().map(|m| m.domain).collect();
        // 4 target domains
        assert!(domains.contains("PV"));
        assert!(domains.contains("Biology"));
        assert!(domains.contains("Psychology"));
        assert!(domains.contains("Economics"));
    }

    #[test]
    fn test_serialize() {
        let mapping = TransferMapping {
            brain_type: "Belief",
            domain: "PV",
            analog: "Causality assessment",
            confidence: 0.85,
        };
        let json = serde_json::to_string(&mapping).unwrap();
        assert!(json.contains("Belief"));
        assert!(json.contains("PV"));
        assert!(json.contains("0.85"));
    }
}
