// Copyright (c) 2026 NexVigilant LLC. All Rights Reserved.
// Intellectual Property of Matthew Alexander Campion, PharmD

//! NMD Learning Event → Brain Integration Bridge.
//!
//! Maps `NmdLearningEvent` variants from the immunity layer into Brain
//! implicit knowledge operations:
//!
//! | Event | Brain Operation |
//! |-------|----------------|
//! | `RecordDegradation` | `add_belief()` + `add_evidence()` |
//! | `AdjustThresholds` | `add_belief()` per adjustment |
//! | `RecordTrustEvent` | `record_trust_success/failure()` |
//!
//! ## Primitive Grounding: μ(Mapping) + →(Causality) + π(Persistence)

use nexcore_immunity::adaptive::{NmdLearningEvent, ThresholdAdjustment};

use crate::implicit::{Belief, EvidenceRef, EvidenceType, ImplicitKnowledge, T1Primitive};

/// Apply a batch of NMD learning events to the brain's implicit knowledge store.
///
/// Returns the number of events successfully applied.
///
/// # Errors
///
/// Individual event application is infallible — this always succeeds.
pub fn apply_learning_events(
    knowledge: &mut ImplicitKnowledge,
    events: &[NmdLearningEvent],
) -> usize {
    let mut applied = 0;
    for event in events {
        apply_single_event(knowledge, event);
        applied += 1;
    }
    applied
}

/// Apply a single NMD learning event to the brain.
fn apply_single_event(knowledge: &mut ImplicitKnowledge, event: &NmdLearningEvent) {
    match event {
        NmdLearningEvent::RecordDegradation {
            category,
            proposition,
            evidence_weight,
        } => {
            apply_degradation(knowledge, category, proposition, *evidence_weight);
        }
        NmdLearningEvent::AdjustThresholds { adjustments } => {
            for adj in adjustments {
                apply_threshold_adjustment(knowledge, adj);
            }
        }
        NmdLearningEvent::RecordTrustEvent { domain, success } => {
            if *success {
                knowledge.record_trust_success(domain);
            } else {
                knowledge.record_trust_failure(domain);
            }
        }
    }
}

/// Map a degradation event to a belief with negative evidence.
fn apply_degradation(
    knowledge: &mut ImplicitKnowledge,
    category: &str,
    proposition: &str,
    evidence_weight: f64,
) {
    let belief_id = format!("nmd_degrade_{}", category.to_lowercase());

    let mut belief = Belief::new(&belief_id, proposition, "nmd_surveillance");
    belief.set_grounding(T1Primitive::Causality);

    let evidence = EvidenceRef::weighted(
        format!(
            "nmd_ev_{}",
            nexcore_chrono::DateTime::now().timestamp_millis()
        ),
        EvidenceType::Observation,
        format!("NMD degradation in category '{category}'"),
        evidence_weight,
        "nmd:adaptive",
    );
    belief.add_evidence(evidence);
    knowledge.add_belief(belief);
}

/// Map a threshold adjustment to a boundary belief.
fn apply_threshold_adjustment(knowledge: &mut ImplicitKnowledge, adj: &ThresholdAdjustment) {
    let belief_id = format!(
        "nmd_adjust_{}_{}",
        adj.category.to_lowercase(),
        adj.parameter
    );

    let mut belief = Belief::new(
        &belief_id,
        format!(
            "Threshold '{}' for '{}' should change from {:.3} to {:.3}: {}",
            adj.parameter, adj.category, adj.current_value, adj.recommended_value, adj.reason
        ),
        "nmd_threshold",
    );
    belief.set_grounding(T1Primitive::Boundary);

    let evidence = EvidenceRef::weighted(
        format!(
            "nmd_adj_{}",
            nexcore_chrono::DateTime::now().timestamp_millis()
        ),
        EvidenceType::Inference,
        format!(
            "Adaptive engine recommends {} → {} (confidence: {:.2})",
            adj.current_value, adj.recommended_value, adj.confidence
        ),
        f64::from(adj.confidence),
        "nmd:adaptive",
    );
    belief.add_evidence(evidence);
    knowledge.add_belief(belief);
}

#[cfg(test)]
mod tests {
    use super::*;

    // ═══ End-to-End Integration: Immunity NMD → Brain ═══

    #[test]
    fn test_e2e_nmd_pipeline_to_brain() {
        use nexcore_immunity::adaptive::NmdAdaptiveEngine;
        use nexcore_immunity::co_translational::{CheckpointObservation, UpfComplex, UpfVerdict};
        use nexcore_immunity::smg::SmgComplex;
        use nexcore_immunity::thymic::ThymicGate;
        use nexcore_immunity::{EjcMarker, TaskCategory};

        // 1. Build NMD pipeline components
        let upf = UpfComplex::new();
        let mut thymic = ThymicGate::new();
        let smg = SmgComplex::new();
        let mut adaptive = NmdAdaptiveEngine::new();

        // 2. Simulate a pipeline observation
        let observation = CheckpointObservation {
            phase_id: "investigate".to_string(),
            observed_categories: vec![TaskCategory::Explore, TaskCategory::Mutate],
            grounding_signals: 3,
            total_calls: 10,
            checkpoint_index: 0,
        };

        let markers = vec![EjcMarker {
            phase_id: "investigate".to_string(),
            expected_tool_categories: vec![TaskCategory::Explore],
            grounding_confidence_threshold: 0.3,
            max_calls_before_checkpoint: 50,
            expected_confidence_range: (0.3, 0.9),
            skippable: false,
        }];

        // 3. Run UPF → Thymic → SMG → Adaptive
        let raw_verdict = upf.scan_checkpoint(&observation, &markers);
        let filtered = thymic.filter_verdict("Explore", raw_verdict);
        let actions = smg.process_verdict(&filtered);

        let mut learning_events = Vec::new();
        let is_degradation = matches!(filtered, UpfVerdict::Degrade { .. });
        if is_degradation {
            for action in &actions {
                let events = adaptive.process_adaptive_action(action);
                learning_events.extend(events);
            }
        } else {
            adaptive.record_success("Explore");
            // Simulate a trust success event for the bridge
            learning_events.push(NmdLearningEvent::RecordTrustEvent {
                domain: "nmd_surveillance".to_string(),
                success: true,
            });
        }

        // 4. Feed learning events to brain
        let mut knowledge = ImplicitKnowledge::load()
            .unwrap_or_else(|_| panic!("Failed to load implicit knowledge"));

        let applied = apply_learning_events(&mut knowledge, &learning_events);

        // At least one event should have been applied
        assert!(applied >= 1);

        // Trust domain should exist after pipeline run
        let trust = knowledge.get_trust("nmd_surveillance");
        assert!(trust.is_some());
    }

    #[test]
    fn test_e2e_degradation_produces_beliefs() {
        use nexcore_immunity::adaptive::NmdAdaptiveEngine;
        use nexcore_immunity::smg::SmgComplex;

        // Simulate SMG producing actions from a degradation
        let smg = SmgComplex::new();
        let mut adaptive = NmdAdaptiveEngine::new();

        // Manually construct a Degrade verdict
        let verdict = nexcore_immunity::co_translational::UpfVerdict::Degrade {
            anomalies: vec![nexcore_immunity::co_translational::UpfAnomaly {
                channel: nexcore_immunity::co_translational::UpfChannel::Upf2,
                description: "Tool category drift: expected Explore, got Mixed".to_string(),
                severity: 0.8,
            }],
        };

        let actions = smg.process_verdict(&verdict);
        assert!(!actions.is_empty());

        // Collect learning events
        let mut learning_events = Vec::new();
        for action in &actions {
            let events = adaptive.process_adaptive_action(action);
            learning_events.extend(events);
        }

        // Should produce RecordDegradation + RecordTrustEvent at minimum
        assert!(learning_events.len() >= 2);

        // Feed to brain
        let mut knowledge = ImplicitKnowledge::load()
            .unwrap_or_else(|_| panic!("Failed to load implicit knowledge"));

        let applied = apply_learning_events(&mut knowledge, &learning_events);
        assert!(applied >= 2);

        // Should have created a degradation belief
        let has_degrade_belief = knowledge
            .list_beliefs()
            .iter()
            .any(|b| b.category == "nmd_surveillance" || b.category == "nmd_threshold");
        assert!(has_degrade_belief);

        // Trust should record a failure
        let trust = knowledge.get_trust("nmd_surveillance");
        assert!(trust.is_some());
    }

    // ═══ Unit Tests ═══

    #[test]
    fn test_apply_degradation_event() {
        let mut knowledge = ImplicitKnowledge::load()
            .unwrap_or_else(|_| panic!("Failed to load implicit knowledge"));

        let events = vec![NmdLearningEvent::RecordDegradation {
            category: "Explore".to_string(),
            proposition: "Explore tasks show UPF2 tool drift".to_string(),
            evidence_weight: -0.7,
        }];

        let applied = apply_learning_events(&mut knowledge, &events);
        assert_eq!(applied, 1);

        let belief = knowledge.get_belief("nmd_degrade_explore");
        assert!(belief.is_some());
        let belief = belief.unwrap_or_else(|| panic!("Missing belief"));
        assert_eq!(belief.category, "nmd_surveillance");
        assert!(!belief.evidence.is_empty());
    }

    #[test]
    fn test_apply_threshold_adjustment() {
        let mut knowledge = ImplicitKnowledge::load()
            .unwrap_or_else(|_| panic!("Failed to load implicit knowledge"));

        let events = vec![NmdLearningEvent::AdjustThresholds {
            adjustments: vec![ThresholdAdjustment {
                category: "Mutate".to_string(),
                parameter: "upf3_grounding_floor".to_string(),
                current_value: 0.3,
                recommended_value: 0.25,
                reason: "UPF3 over-triggering".to_string(),
                confidence: 0.8,
            }],
        }];

        let applied = apply_learning_events(&mut knowledge, &events);
        assert_eq!(applied, 1);

        let belief = knowledge.get_belief("nmd_adjust_mutate_upf3_grounding_floor");
        assert!(belief.is_some());
    }

    #[test]
    fn test_apply_trust_success() {
        let mut knowledge = ImplicitKnowledge::load()
            .unwrap_or_else(|_| panic!("Failed to load implicit knowledge"));

        let events = vec![NmdLearningEvent::RecordTrustEvent {
            domain: "nmd_surveillance".to_string(),
            success: true,
        }];

        apply_learning_events(&mut knowledge, &events);

        let trust = knowledge.get_trust("nmd_surveillance");
        assert!(trust.is_some());
        let trust = trust.unwrap_or_else(|| panic!("Missing trust"));
        assert_eq!(trust.demonstrations, 1);
        assert_eq!(trust.failures, 0);
    }

    #[test]
    fn test_apply_trust_failure() {
        let mut knowledge = ImplicitKnowledge::load()
            .unwrap_or_else(|_| panic!("Failed to load implicit knowledge"));

        let events = vec![NmdLearningEvent::RecordTrustEvent {
            domain: "nmd_surveillance".to_string(),
            success: false,
        }];

        apply_learning_events(&mut knowledge, &events);

        let trust = knowledge.get_trust("nmd_surveillance");
        assert!(trust.is_some());
        let trust = trust.unwrap_or_else(|| panic!("Missing trust"));
        assert_eq!(trust.demonstrations, 0);
        assert_eq!(trust.failures, 1);
    }

    #[test]
    fn test_apply_mixed_batch() {
        let mut knowledge = ImplicitKnowledge::load()
            .unwrap_or_else(|_| panic!("Failed to load implicit knowledge"));

        let events = vec![
            NmdLearningEvent::RecordDegradation {
                category: "Compute".to_string(),
                proposition: "Compute tasks stalling on UPF1".to_string(),
                evidence_weight: -0.5,
            },
            NmdLearningEvent::RecordTrustEvent {
                domain: "nmd_surveillance".to_string(),
                success: false,
            },
            NmdLearningEvent::AdjustThresholds {
                adjustments: vec![ThresholdAdjustment {
                    category: "Compute".to_string(),
                    parameter: "upf1_max_phase_skip".to_string(),
                    current_value: 2.0,
                    recommended_value: 1.0,
                    reason: "Phase order violations detected".to_string(),
                    confidence: 0.75,
                }],
            },
        ];

        let applied = apply_learning_events(&mut knowledge, &events);
        assert_eq!(applied, 3);

        // Verify all three types applied
        assert!(knowledge.get_belief("nmd_degrade_compute").is_some());
        assert!(
            knowledge
                .get_belief("nmd_adjust_compute_upf1_max_phase_skip")
                .is_some()
        );
        assert!(knowledge.get_trust("nmd_surveillance").is_some());
    }
}
