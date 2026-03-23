//! # Brain → Insight Trait Adapter
//!
//! Makes Brain's implicit learning system a first-class Insight implementor,
//! proving architecturally that NexCore IS an InsightEngine.
//!
//! ## Composite Mapping
//!
//! | Insight Composite | Brain Component |
//! |-------------------|-----------------|
//! | Pattern (σ+κ+μ) | Implicit Pattern detection |
//! | Recognition (κ+∃+σ) | Correction matching |
//! | Novelty (∅+∃+σ) | New preference/pattern discovery |
//! | Connection (μ+κ+ς) | Belief graph edges |
//! | Compression (N+μ+κ) | Preference consolidation (many obs → one pref) |
//! | Suddenness (σ+∂+N+κ) | Confidence spike detection |
//!
//! ## T1 Grounding
//!
//! BrainInsightAdapter ≡ ⟨σ, κ, μ, ∃, ς, ∅, N, ∂⟩ (T3)
//! - ς mode: Accumulated (ς-acc) — append-only learning history

#![allow(dead_code)]

use nexcore_insight::composites::{Compression, Connection, Pattern as InsightPattern};
use nexcore_insight::engine::{InsightConfig, InsightEngine, InsightEvent, Observation};
use nexcore_insight::traits::Insight;

/// A learning observation fed into the Brain insight adapter.
///
/// Represents a single piece of knowledge the Brain has encountered,
/// whether a preference, pattern, correction, or belief.
///
/// ## Tier: T2-P (∃ + σ — existence of knowledge at a point in time)
#[derive(Debug, Clone)]
pub struct LearningObservation {
    /// Unique key identifying the learned concept.
    ///
    /// Format: `{knowledge_type}:{identifier}`
    /// e.g., "preference:code_style", "pattern:naming_convention", "correction:unwrap_usage"
    pub key: String,

    /// The observed value or content.
    pub value: String,

    /// Confidence in this learning (0.0 - 1.0).
    pub confidence: f64,

    /// Category of knowledge: "preference", "pattern", "correction", "belief"
    pub knowledge_type: KnowledgeType,

    /// Optional T1 primitive this knowledge is grounded to.
    pub grounding: Option<String>,
}

/// Category of implicit knowledge.
///
/// ## Tier: T2-P (Σ — sum/coproduct of knowledge types)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KnowledgeType {
    /// Learned user preference (code style, commit format, etc.)
    Preference,
    /// Detected behavioral pattern
    Pattern,
    /// Error correction (wrong → right mapping)
    Correction,
    /// Grounded belief (evidence-backed assertion)
    Belief,
}

impl KnowledgeType {
    /// Get the tag string for this knowledge type.
    fn as_tag(&self) -> &'static str {
        match self {
            Self::Preference => "preference",
            Self::Pattern => "pattern",
            Self::Correction => "correction",
            Self::Belief => "belief",
        }
    }
}

impl LearningObservation {
    /// Create a new learning observation.
    #[must_use]
    pub fn new(
        key: impl Into<String>,
        value: impl Into<String>,
        confidence: f64,
        knowledge_type: KnowledgeType,
    ) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
            confidence: confidence.clamp(0.0, 1.0),
            knowledge_type,
            grounding: None,
        }
    }

    /// Create a preference observation.
    #[must_use]
    pub fn preference(key: impl Into<String>, value: impl Into<String>, confidence: f64) -> Self {
        let k: String = key.into();
        Self::new(
            format!("preference:{k}"),
            value,
            confidence,
            KnowledgeType::Preference,
        )
    }

    /// Create a pattern observation.
    #[must_use]
    pub fn pattern(id: impl Into<String>, description: impl Into<String>, confidence: f64) -> Self {
        let i: String = id.into();
        Self::new(
            format!("pattern:{i}"),
            description,
            confidence,
            KnowledgeType::Pattern,
        )
    }

    /// Create a correction observation.
    #[must_use]
    pub fn correction(
        trigger: impl Into<String>,
        replacement: impl Into<String>,
        confidence: f64,
    ) -> Self {
        let t: String = trigger.into();
        Self::new(
            format!("correction:{t}"),
            replacement,
            confidence,
            KnowledgeType::Correction,
        )
    }

    /// Create a belief observation.
    #[must_use]
    pub fn belief(
        subject: impl Into<String>,
        assertion: impl Into<String>,
        confidence: f64,
    ) -> Self {
        let s: String = subject.into();
        Self::new(
            format!("belief:{s}"),
            assertion,
            confidence,
            KnowledgeType::Belief,
        )
    }

    /// Set the T1 primitive grounding.
    #[must_use]
    pub fn with_grounding(mut self, primitive: impl Into<String>) -> Self {
        self.grounding = Some(primitive.into());
        self
    }
}

/// Brain adapter for the Insight trait.
///
/// Wraps Brain's implicit learning pipeline as an InsightEngine implementation.
/// Converts `LearningObservation` → `Observation` → InsightEngine pipeline.
///
/// Brain-tuned config: lower thresholds for exploratory learning.
///
/// ## Tier: T3 (system-level adapter)
///
/// ## T1 Grounding
/// BRAIN_INSIGHT ≡ ⟨σ, κ, μ, ∃, ς, ∅, N, ∂⟩
/// - σ: Temporal ordering of learning events
/// - κ: Pattern matching (correction triggers, preference keys)
/// - μ: Knowledge mapping (trigger → replacement, key → value)
/// - ∃: New knowledge detection
/// - ∅: Knowledge gaps (novel concepts)
/// - N: Confidence scores, reinforcement counts
/// - ∂: Confidence thresholds for pattern confirmation
/// - ς: Accumulated learning history (ς-acc)
pub struct BrainInsightAdapter {
    /// Internal engine performing the 6-stage pipeline.
    engine: InsightEngine,
}

impl BrainInsightAdapter {
    /// Create a new Brain insight adapter with learning-tuned configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(Self::learning_config())
    }

    /// Create with custom InsightConfig.
    #[must_use]
    pub fn with_config(config: InsightConfig) -> Self {
        Self {
            engine: InsightEngine::with_config(config),
        }
    }

    /// Learning-tuned configuration.
    ///
    /// Lower thresholds than Guardian to encourage exploration:
    /// - `pattern_min_occurrences: 2` (learn from fewer examples)
    /// - `pattern_confidence_threshold: 0.4` (accept developing patterns)
    /// - `enable_suddenness: false` (learning doesn't care about timing)
    #[must_use]
    pub fn learning_config() -> InsightConfig {
        InsightConfig {
            pattern_min_occurrences: 2,
            pattern_confidence_threshold: 0.4,
            connection_strength_threshold: 0.3,
            compression_min_ratio: 2.0,
            enable_suddenness: false,
            suddenness_threshold: 2.0,
            enable_recursive_learning: true,
        }
    }

    /// Convert a `LearningObservation` into a generic `Observation`.
    ///
    /// Mapping:
    /// - key: `"{knowledge_type}:{id}"` (already formatted in constructor)
    /// - value: the learned content
    /// - numeric_value: confidence score
    /// - tags: knowledge type + optional primitive grounding
    ///
    /// Tier: T2-P (μ — mapping between domains)
    #[must_use]
    pub fn learning_to_observation(obs: &LearningObservation) -> Observation {
        let mut observation = Observation::with_numeric(&obs.key, obs.confidence);

        // Tag with knowledge type for clustering
        observation = observation.with_tag(obs.knowledge_type.as_tag());

        // Tag with T1 grounding if present
        if let Some(ref grounding) = obs.grounding {
            observation = observation.with_tag(format!("grounded:{grounding}"));
        }

        observation
    }

    /// Ingest a learning observation through the 6-stage pipeline.
    pub fn ingest_learning(&mut self, obs: &LearningObservation) -> Vec<InsightEvent> {
        let observation = Self::learning_to_observation(obs);
        self.engine.ingest(observation)
    }

    /// Get the underlying engine (read-only access for inspection).
    #[must_use]
    pub fn engine(&self) -> &InsightEngine {
        &self.engine
    }
}

impl Default for BrainInsightAdapter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Insight Trait Implementation ──────────────────────────────────────────────

impl Insight for BrainInsightAdapter {
    type Obs = LearningObservation;

    fn ingest(&mut self, observation: LearningObservation) -> Vec<InsightEvent> {
        self.ingest_learning(&observation)
    }

    fn ingest_batch(&mut self, observations: Vec<LearningObservation>) -> Vec<InsightEvent> {
        let mut all_events = Vec::new();
        for obs in &observations {
            all_events.extend(self.ingest_learning(obs));
        }
        all_events
    }

    fn connect(&mut self, from: &str, to: &str, relation: &str, strength: f64) -> Connection {
        self.engine.connect(from, to, relation, strength)
    }

    fn compress(&mut self, keys: Vec<String>, principle: &str) -> Compression {
        self.engine.compress(keys, principle)
    }

    fn events(&self) -> &[InsightEvent] {
        self.engine.events()
    }

    fn observation_count(&self) -> usize {
        self.engine.observation_count()
    }

    fn pattern_count(&self) -> usize {
        self.engine.pattern_count()
    }

    fn patterns(&self) -> Vec<&InsightPattern> {
        self.engine.patterns()
    }

    fn connections(&self) -> &[Connection] {
        self.engine.connections()
    }

    fn unique_key_count(&self) -> usize {
        self.engine.unique_key_count()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = BrainInsightAdapter::new();
        assert_eq!(adapter.observation_count(), 0);
        assert_eq!(adapter.pattern_count(), 0);
        assert_eq!(adapter.unique_key_count(), 0);
    }

    #[test]
    fn test_default_impl() {
        let adapter = BrainInsightAdapter::default();
        assert_eq!(adapter.observation_count(), 0);
    }

    #[test]
    fn test_preference_observation() {
        let obs = LearningObservation::preference("code_style", "functional", 0.8);
        assert_eq!(obs.key, "preference:code_style");
        assert_eq!(obs.knowledge_type, KnowledgeType::Preference);
        assert!((obs.confidence - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pattern_observation() {
        let obs = LearningObservation::pattern("naming_convention", "snake_case for modules", 0.7);
        assert_eq!(obs.key, "pattern:naming_convention");
        assert_eq!(obs.knowledge_type, KnowledgeType::Pattern);
    }

    #[test]
    fn test_correction_observation() {
        let obs = LearningObservation::correction("unwrap_usage", "use ? operator instead", 0.9);
        assert_eq!(obs.key, "correction:unwrap_usage");
        assert_eq!(obs.knowledge_type, KnowledgeType::Correction);
    }

    #[test]
    fn test_belief_observation() {
        let obs =
            LearningObservation::belief("rust_safety", "forbid unsafe_code in all crates", 0.95);
        assert_eq!(obs.key, "belief:rust_safety");
        assert_eq!(obs.knowledge_type, KnowledgeType::Belief);
    }

    #[test]
    fn test_grounding_tag() {
        let obs =
            LearningObservation::pattern("naming", "snake_case", 0.7).with_grounding("sequence");
        let converted = BrainInsightAdapter::learning_to_observation(&obs);
        assert!(converted.tags.contains(&"pattern".to_string()));
        assert!(converted.tags.contains(&"grounded:sequence".to_string()));
    }

    #[test]
    fn test_observation_conversion() {
        let obs = LearningObservation::preference("indent", "4 spaces", 0.85);
        let converted = BrainInsightAdapter::learning_to_observation(&obs);
        assert_eq!(converted.key, "preference:indent");
        assert!(converted.numeric_value.is_some());
        let conf = converted.numeric_value.unwrap_or(0.0);
        assert!((conf - 0.85).abs() < f64::EPSILON);
        assert!(converted.tags.contains(&"preference".to_string()));
    }

    #[test]
    fn test_ingest_produces_novelty() {
        let mut adapter = BrainInsightAdapter::new();
        let obs = LearningObservation::preference("editor", "vim", 0.9);
        let events = adapter.ingest(obs);

        assert_eq!(adapter.observation_count(), 1);
        // First observation always novel
        assert!(
            events
                .iter()
                .any(|e| matches!(e, InsightEvent::NoveltyDetected(_))),
            "expected novelty event for first observation"
        );
    }

    #[test]
    fn test_ingest_batch() {
        let mut adapter = BrainInsightAdapter::new();
        let observations = vec![
            LearningObservation::preference("style", "functional", 0.8),
            LearningObservation::pattern("naming", "snake_case", 0.7),
            LearningObservation::correction("unwrap", "use ?", 0.9),
            LearningObservation::belief("safety", "forbid unsafe", 0.95),
        ];
        let events = adapter.ingest_batch(observations);

        assert_eq!(adapter.observation_count(), 4);
        assert_eq!(adapter.unique_key_count(), 4);
        // All 4 should be novel
        let novelty_count = events
            .iter()
            .filter(|e| matches!(e, InsightEvent::NoveltyDetected(_)))
            .count();
        assert_eq!(novelty_count, 4);
    }

    #[test]
    fn test_pattern_detection_from_repeated_learning() {
        let mut adapter = BrainInsightAdapter::with_config(InsightConfig {
            pattern_min_occurrences: 2,
            pattern_confidence_threshold: 0.3,
            ..InsightConfig::default()
        });

        // Learn same pattern type multiple times
        let obs_a = LearningObservation::correction("unwrap_used", "use ? operator", 0.8);
        let obs_b = LearningObservation::correction("expect_used", "use ? operator", 0.8);

        adapter.ingest(obs_a.clone());
        adapter.ingest(obs_b.clone());
        adapter.ingest(obs_a);
        let events = adapter.ingest(obs_b);

        // After repeated co-occurrence, pattern should emerge or events produced
        assert!(adapter.pattern_count() > 0 || !events.is_empty());
    }

    #[test]
    fn test_connect_beliefs() {
        let mut adapter = BrainInsightAdapter::new();
        let conn = adapter.connect(
            "belief:rust_safety",
            "correction:unwrap_usage",
            "implies",
            0.9,
        );

        assert_eq!(conn.from, "belief:rust_safety");
        assert_eq!(conn.to, "correction:unwrap_usage");
        assert_eq!(conn.relation, "implies");
        assert!((conn.strength - 0.9).abs() < f64::EPSILON);
        assert_eq!(adapter.connections().len(), 1);
    }

    #[test]
    fn test_compress_corrections() {
        let mut adapter = BrainInsightAdapter::new();

        adapter.ingest(LearningObservation::correction("unwrap", "use ?", 0.9));
        adapter.ingest(LearningObservation::correction("expect", "use ?", 0.85));
        adapter.ingest(LearningObservation::correction("panic", "return Err", 0.8));

        let compression = adapter.compress(
            vec![
                "correction:unwrap".to_string(),
                "correction:expect".to_string(),
                "correction:panic".to_string(),
            ],
            "error_handling_discipline",
        );

        assert_eq!(compression.principle, "error_handling_discipline");
        assert_eq!(compression.input_count, 3);
        assert!(compression.is_meaningful());
    }

    #[test]
    fn test_connections_for_key() {
        let mut adapter = BrainInsightAdapter::new();
        adapter.connect("belief:safety", "correction:unwrap", "implies", 0.9);
        adapter.connect("belief:safety", "correction:panic", "implies", 0.85);
        adapter.connect("preference:style", "pattern:naming", "influences", 0.5);

        let safety_conns = adapter.connections_for("belief:safety");
        assert_eq!(safety_conns.len(), 2);

        let style_conns = adapter.connections_for("preference:style");
        assert_eq!(style_conns.len(), 1);
    }

    #[test]
    fn test_trait_polymorphism() {
        fn count_events<E: Insight>(engine: &mut E, obs: E::Obs) -> usize {
            engine.ingest(obs).len()
        }

        let mut adapter = BrainInsightAdapter::new();
        let obs = LearningObservation::belief("test", "assertion", 0.5);
        let n = count_events(&mut adapter, obs);
        assert!(n > 0, "expected at least one event (novelty)");
    }

    #[test]
    fn test_knowledge_type_tags() {
        assert_eq!(KnowledgeType::Preference.as_tag(), "preference");
        assert_eq!(KnowledgeType::Pattern.as_tag(), "pattern");
        assert_eq!(KnowledgeType::Correction.as_tag(), "correction");
        assert_eq!(KnowledgeType::Belief.as_tag(), "belief");
    }
}
