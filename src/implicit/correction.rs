//! Learned corrections from user feedback
//!
//! Corrections record mistakes and their fixes, with application tracking.

use nexcore_chrono::DateTime;
use serde::{Deserialize, Serialize};

/// A learned correction from user feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Correction {
    /// What was wrong (the mistake)
    #[serde(alias = "original")]
    pub mistake: String,

    /// What should have been done (the correction)
    #[serde(alias = "corrected")]
    pub correction: String,

    /// Context when this occurred
    #[serde(alias = "reason")]
    pub context: Option<String>,

    /// When this correction was learned
    #[serde(default = "DateTime::now")]
    pub learned_at: DateTime,

    /// Number of times this correction was applied
    #[serde(default)]
    pub application_count: u32,

    /// Confidence level (0.0 to 1.0). Defaults to 0.8 for corrections
    /// (higher than preferences since corrections are explicit user feedback).
    #[serde(default = "default_correction_confidence")]
    pub confidence: f64,

    /// When this correction was last reinforced (applied or explicitly confirmed).
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
        // Each application increases confidence asymptotically
        self.confidence =
            1.0 - (0.2 / (f64::from(self.application_count) + 1.0));
        self.updated_at = DateTime::now();
    }

    /// Half-life in days for correction confidence decay.
    ///
    /// Corrections decay slower than preferences (60 days vs 30)
    /// because explicit feedback is more durable than inferred preferences.
    const HALF_LIFE_DAYS: f64 = 60.0;

    /// Compute effective confidence with time-based decay.
    #[must_use]
    pub fn effective_confidence(&self) -> f64 {
        self.effective_confidence_at(DateTime::now())
    }

    /// Compute effective confidence at a specific point in time.
    #[must_use]
    pub fn effective_confidence_at(&self, now: DateTime) -> f64 {
        let elapsed = now.signed_duration_since(self.updated_at);
        let days = elapsed.num_seconds() as f64 / 86_400.0;
        if days <= 0.0 {
            return self.confidence;
        }
        let decay = 0.5_f64.powf(days / Self::HALF_LIFE_DAYS);
        self.confidence * decay
    }

    /// Check if this correction is still above minimum confidence after decay.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.effective_confidence() > 0.05
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correction() {
        let mut correction = Correction::new("used unwrap", "use expect with message");
        assert_eq!(correction.application_count, 0);

        correction.mark_applied();
        correction.mark_applied();
        assert_eq!(correction.application_count, 2);
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

    // ========== Decay-on-read tests ==========

    #[test]
    fn test_correction_initial_confidence() {
        let correction = Correction::new("mistake", "fix");
        assert!((correction.confidence - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_correction_mark_applied_increases_confidence() {
        let mut correction = Correction::new("mistake", "fix");
        let before = correction.confidence;
        correction.mark_applied();
        assert!(correction.confidence > before);
    }

    #[test]
    fn test_correction_effective_confidence_no_decay() {
        let correction = Correction::new("mistake", "fix");
        let eff = correction.effective_confidence_at(correction.updated_at);
        assert!((eff - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_correction_half_life_60_days() {
        let mut correction = Correction::new("mistake", "fix");
        correction.confidence = 1.0;
        let sixty_days = correction.updated_at + nexcore_chrono::Duration::days(60);
        let eff = correction.effective_confidence_at(sixty_days);
        assert!((eff - 0.5).abs() < 0.01, "expected ~0.5, got {eff}");
    }

    #[test]
    fn test_correction_is_active_fresh() {
        let correction = Correction::new("mistake", "fix");
        assert!(correction.is_active());
    }

    #[test]
    fn test_correction_is_active_very_old() {
        let mut correction = Correction::new("mistake", "fix");
        correction.updated_at = DateTime::now() - nexcore_chrono::Duration::days(730);
        assert!(!correction.is_active());
    }

    #[test]
    fn test_correction_mark_applied_resets_decay() {
        let mut correction = Correction::new("mistake", "fix");
        correction.updated_at = DateTime::now() - nexcore_chrono::Duration::days(60);
        let decayed = correction.effective_confidence();
        correction.mark_applied();
        let fresh = correction.effective_confidence();
        assert!(fresh > decayed);
    }

    #[test]
    fn test_correction_backward_compat_no_confidence_field() {
        // Simulate deserializing old corrections without confidence/updated_at
        let json = r#"{"mistake":"old","correction":"new","learned_at":"2026-01-01T00:00:00Z","application_count":3}"#;
        let correction: Correction = serde_json::from_str(json).unwrap();
        assert!((correction.confidence - 0.8).abs() < 1e-10); // default
    }
}
