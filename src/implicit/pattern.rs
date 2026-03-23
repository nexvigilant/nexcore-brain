//! Detected patterns in user behavior or codebase
//!
//! Patterns support T1 primitive grounding and time-based confidence decay.
//! Confidence decays exponentially with a 30-day half-life unless reinforced.

use nexcore_chrono::DateTime;
use serde::{Deserialize, Serialize};

use super::t1::T1Primitive;

/// Confidence decay half-life in days.
/// After this many days without reinforcement, effective confidence halves.
pub(crate) const DECAY_HALF_LIFE_DAYS: f64 = 30.0;

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
            (f64::from(self.occurrence_count) / (f64::from(self.occurrence_count) + 2.0)).min(0.95);
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_pattern_grounding() {
        let mut pattern = Pattern::new("test", "workflow", "Test pattern");
        assert!(pattern.t1_grounding.is_none());

        pattern.set_grounding(T1Primitive::Mapping);
        assert_eq!(pattern.t1_grounding, Some(T1Primitive::Mapping));
    }

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
}
