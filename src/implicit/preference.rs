//! Learned user preferences
//!
//! Preferences are key-value pairs with confidence levels that
//! increase through reinforcement and decrease through weakening.

use nexcore_chrono::DateTime;
use serde::{Deserialize, Serialize};

/// A learned user preference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preference {
    /// Preference key (e.g., "`code_style`", "`commit_format`")
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
        self.confidence = 1.0 - (1.0 / (f64::from(self.reinforcement_count) + 1.0));
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
            1.0 - (1.0 / (f64::from(self.reinforcement_count) + 1.0))
        };
        self.updated_at = DateTime::now();
    }

    /// Half-life in days for confidence decay.
    ///
    /// After 30 days without reinforcement, effective confidence drops to 50%.
    const HALF_LIFE_DAYS: f64 = 30.0;

    /// Compute effective confidence with time-based decay.
    ///
    /// Uses exponential decay: `confidence * 0.5^(days_since_update / half_life)`.
    /// Reinforcing resets `updated_at`, restarting the decay clock.
    #[must_use]
    pub fn effective_confidence(&self) -> f64 {
        self.effective_confidence_at(DateTime::now())
    }

    /// Compute effective confidence at a specific point in time.
    ///
    /// Useful for testing and historical analysis.
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

    /// Check if this preference is still above a minimum confidence threshold
    /// after accounting for time decay.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.effective_confidence() > 0.05
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
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

    // ========== Decay-on-read tests ==========

    #[test]
    fn test_effective_confidence_no_decay_at_creation() {
        let pref = Preference::new("test", serde_json::json!("value"));
        // At creation, effective == base confidence
        let eff = pref.effective_confidence_at(pref.updated_at);
        assert!((eff - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_effective_confidence_half_life_30_days() {
        let mut pref = Preference::new("test", serde_json::json!("value"));
        pref.confidence = 1.0;
        let thirty_days_later = pref.updated_at + nexcore_chrono::Duration::days(30);
        let eff = pref.effective_confidence_at(thirty_days_later);
        // After 30 days, should be ~0.5
        assert!((eff - 0.5).abs() < 0.01, "expected ~0.5, got {eff}");
    }

    #[test]
    fn test_effective_confidence_60_days() {
        let mut pref = Preference::new("test", serde_json::json!("value"));
        pref.confidence = 1.0;
        let sixty_days_later = pref.updated_at + nexcore_chrono::Duration::days(60);
        let eff = pref.effective_confidence_at(sixty_days_later);
        // After 60 days (2 half-lives), should be ~0.25
        assert!((eff - 0.25).abs() < 0.01, "expected ~0.25, got {eff}");
    }

    #[test]
    fn test_effective_confidence_reinforce_resets_decay() {
        let mut pref = Preference::new("test", serde_json::json!("value"));
        // Simulate 30 days passing
        pref.updated_at = DateTime::now() - nexcore_chrono::Duration::days(30);
        let before_reinforce = pref.effective_confidence();
        // Reinforce resets updated_at to now
        pref.reinforce();
        let after_reinforce = pref.effective_confidence();
        assert!(after_reinforce > before_reinforce);
    }

    #[test]
    fn test_is_active_fresh_preference() {
        let pref = Preference::new("test", serde_json::json!("value"));
        assert!(pref.is_active());
    }

    #[test]
    fn test_is_active_very_old_preference() {
        let mut pref = Preference::new("test", serde_json::json!("value"));
        pref.confidence = 0.5;
        // 365 days without reinforcement: 0.5 * 0.5^(365/30) ≈ 0.00024
        pref.updated_at = DateTime::now() - nexcore_chrono::Duration::days(365);
        assert!(!pref.is_active());
    }

    #[test]
    fn test_zero_confidence_stays_zero_after_decay() {
        let mut pref = Preference::new("test", serde_json::json!("value"));
        pref.confidence = 0.0;
        let later = pref.updated_at + nexcore_chrono::Duration::days(30);
        assert_eq!(pref.effective_confidence_at(later), 0.0);
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
}
