//! Synapse integration for amplitude-based implicit learning.
//!
//! Bridges `nexcore_synapse` with the Brain's implicit knowledge system,
//! enabling amplitude growth for patterns, preferences, and beliefs.
//!
//! ## Architecture
//!
//! ```text
//! Pattern/Preference/Belief
//!        ↓
//!    Synapse (amplitude model)
//!        ↓
//!    SynapseBank (persistence)
//!        ↓
//!    ~/.claude/brain/synapses.json
//! ```
//!
//! ## T1 Primitive Grounding
//!
//! - ν (Frequency) → observation_count
//! - Σ (Sum) → amplitude accumulation
//! - π (Persistence) → file-backed storage
//! - ∂ (Boundary) → consolidation threshold
//! - ∝ (Irreversibility) → temporal decay

use crate::BrainError;
use crate::error::Result;
use crate::implicit::Pattern;
use nexcore_chrono::DateTime;
use nexcore_fs::dirs;
use nexcore_synapse::{
    Amplitude, AmplitudeConfig, ConsolidationStatus, LearningSignal, Synapse, SynapseBank,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

// Re-export synapse types for convenience
pub use nexcore_synapse::{Amplitude as SynapseAmplitude, AmplitudeConfig as SynapseConfig};

/// Default synapse bank file path.
const SYNAPSE_FILE: &str = ".claude/brain/synapses.json";

/// Get the synapse bank file path.
#[must_use]
pub fn synapse_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(SYNAPSE_FILE)
}

/// Persistent synapse bank with file-backed storage.
///
/// Tier: T3 (domain-specific Brain integration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentSynapseBank {
    /// Inner synapse bank from nexcore-synapse.
    #[serde(flatten)]
    bank: SynapseBank,

    /// Last save timestamp.
    pub last_saved: Option<DateTime>,
}

impl Default for PersistentSynapseBank {
    fn default() -> Self {
        Self::new()
    }
}

impl PersistentSynapseBank {
    /// Create a new empty synapse bank.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bank: SynapseBank::new(),
            last_saved: None,
        }
    }

    /// Load synapse bank from disk, or create new if not exists.
    pub fn load() -> Result<Self> {
        let path = synapse_path();
        if path.exists() {
            let content = fs::read_to_string(&path)?;
            let bank: Self = serde_json::from_str(&content)
                .map_err(|e| BrainError::Serialization(e.to_string()))?;
            Ok(bank)
        } else {
            Ok(Self::new())
        }
    }

    /// Save synapse bank to disk.
    pub fn save(&mut self) -> Result<()> {
        let path = synapse_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        self.last_saved = Some(DateTime::now());
        let content = serde_json::to_string_pretty(&self)
            .map_err(|e| BrainError::Serialization(e.to_string()))?;
        fs::write(&path, content)?;
        Ok(())
    }

    /// Get or create a synapse for a pattern.
    pub fn get_or_create_for_pattern(&mut self, pattern_id: &str) -> &mut Synapse {
        self.bank
            .get_or_create(&format!("pattern:{pattern_id}"), AmplitudeConfig::DEFAULT)
    }

    /// Get or create a synapse for a preference.
    pub fn get_or_create_for_preference(&mut self, key: &str) -> &mut Synapse {
        self.bank
            .get_or_create(&format!("preference:{key}"), AmplitudeConfig::DEFAULT)
    }

    /// Get or create a synapse for a belief.
    pub fn get_or_create_for_belief(&mut self, belief_id: &str) -> &mut Synapse {
        // Beliefs use faster learning since they're evidence-based
        self.bank
            .get_or_create(&format!("belief:{belief_id}"), AmplitudeConfig::FAST)
    }

    /// Get a synapse by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&Synapse> {
        self.bank.get(id)
    }

    /// Get mutable synapse by ID.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut Synapse> {
        self.bank.get_mut(id)
    }

    /// Observe a learning signal for a synapse.
    pub fn observe(&mut self, id: &str, confidence: f64, relevance: f64) -> Option<Amplitude> {
        if let Some(synapse) = self.bank.get_mut(id) {
            synapse.observe(LearningSignal::new(confidence, relevance));
            Some(synapse.current_amplitude())
        } else {
            None
        }
    }

    /// Get all consolidated synapses.
    pub fn consolidated(&self) -> Vec<&Synapse> {
        self.bank.consolidated().collect()
    }

    /// Get all accumulating synapses.
    pub fn accumulating(&self) -> Vec<&Synapse> {
        self.bank.accumulating().collect()
    }

    /// Prune decayed synapses (amplitude near zero).
    pub fn prune_decayed(&mut self) -> usize {
        self.bank.prune_decayed()
    }

    /// Get synapse count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bank.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bank.is_empty()
    }

    /// Get summary statistics.
    #[must_use]
    pub fn stats(&self) -> SynapseBankStats {
        let consolidated_count = self.bank.consolidated().count();
        let accumulating_count = self.bank.accumulating().count();
        let total = self.bank.len();

        let mut total_amplitude = 0.0;
        let mut peak_amplitude = 0.0;
        let mut pattern_count = 0;
        let mut preference_count = 0;
        let mut belief_count = 0;

        for (id, synapse) in self.bank.iter() {
            let amp = synapse.current_amplitude().value();
            total_amplitude += amp;
            if amp > peak_amplitude {
                peak_amplitude = amp;
            }

            if id.starts_with("pattern:") {
                pattern_count += 1;
            } else if id.starts_with("preference:") {
                preference_count += 1;
            } else if id.starts_with("belief:") {
                belief_count += 1;
            }
        }

        SynapseBankStats {
            total_synapses: total,
            consolidated_count,
            accumulating_count,
            pattern_synapses: pattern_count,
            preference_synapses: preference_count,
            belief_synapses: belief_count,
            average_amplitude: if total > 0 {
                total_amplitude / total as f64
            } else {
                0.0
            },
            peak_amplitude,
        }
    }

    /// List all synapse IDs and their status.
    pub fn list(&self) -> Vec<SynapseInfo> {
        self.bank
            .iter()
            .map(|(id, synapse)| SynapseInfo {
                id: id.clone(),
                amplitude: synapse.current_amplitude().value(),
                observation_count: synapse.observation_count(),
                status: synapse.status(),
                is_persistent: synapse.is_persistent(),
            })
            .collect()
    }
}

/// Summary statistics for the synapse bank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseBankStats {
    /// Total number of synapses.
    pub total_synapses: usize,
    /// Number of consolidated synapses.
    pub consolidated_count: usize,
    /// Number of accumulating synapses.
    pub accumulating_count: usize,
    /// Number of pattern synapses.
    pub pattern_synapses: usize,
    /// Number of preference synapses.
    pub preference_synapses: usize,
    /// Number of belief synapses.
    pub belief_synapses: usize,
    /// Average amplitude across all synapses.
    pub average_amplitude: f64,
    /// Peak amplitude among all synapses.
    pub peak_amplitude: f64,
}

/// Information about a single synapse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseInfo {
    /// Synapse ID.
    pub id: String,
    /// Current amplitude (with decay).
    pub amplitude: f64,
    /// Total observation count.
    pub observation_count: u32,
    /// Consolidation status.
    pub status: ConsolidationStatus,
    /// Whether persistent across sessions.
    pub is_persistent: bool,
}

// ============================================================================
// Pattern Integration
// ============================================================================

/// Convert a Pattern's effective_confidence to use synapse amplitude.
///
/// This provides a migration path from the simple decay formula to
/// the full amplitude growth model.
pub fn pattern_amplitude(pattern: &Pattern, bank: &PersistentSynapseBank) -> f64 {
    let synapse_id = format!("pattern:{}", pattern.id);
    if let Some(synapse) = bank.get(&synapse_id) {
        synapse.current_amplitude().value()
    } else {
        // Fall back to pattern's built-in effective_confidence
        pattern.effective_confidence()
    }
}

/// Reinforce a pattern by observing a signal on its synapse.
pub fn reinforce_pattern(
    pattern: &Pattern,
    bank: &mut PersistentSynapseBank,
    confidence: f64,
    relevance: f64,
) -> f64 {
    let synapse = bank.get_or_create_for_pattern(&pattern.id);
    synapse.observe(LearningSignal::new(confidence, relevance));
    synapse.current_amplitude().value()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brain_synapse_bank_new() {
        let bank = PersistentSynapseBank::new();
        assert!(bank.is_empty());
        assert_eq!(bank.len(), 0);
    }

    #[test]
    fn test_synapse_creation() {
        let mut bank = PersistentSynapseBank::new();

        let synapse = bank.get_or_create_for_pattern("test_pattern");
        assert_eq!(synapse.id, "pattern:test_pattern");

        let synapse2 = bank.get_or_create_for_preference("code_style");
        assert_eq!(synapse2.id, "preference:code_style");

        let synapse3 = bank.get_or_create_for_belief("rust_is_safe");
        assert_eq!(synapse3.id, "belief:rust_is_safe");

        assert_eq!(bank.len(), 3);
    }

    #[test]
    fn test_observe_and_amplitude() {
        let mut bank = PersistentSynapseBank::new();

        // Create and observe
        let _ = bank.get_or_create_for_pattern("test");
        for _ in 0..5 {
            bank.observe("pattern:test", 0.9, 1.0);
        }

        let synapse = bank.get("pattern:test");
        assert!(synapse.is_some());
        assert!(
            synapse
                .map(|s| s.current_amplitude().value() > 0.0)
                .unwrap_or(false)
        );
    }

    #[test]
    fn test_stats() {
        let mut bank = PersistentSynapseBank::new();

        bank.get_or_create_for_pattern("p1");
        bank.get_or_create_for_pattern("p2");
        bank.get_or_create_for_preference("pref1");
        bank.get_or_create_for_belief("b1");

        let stats = bank.stats();
        assert_eq!(stats.total_synapses, 4);
        assert_eq!(stats.pattern_synapses, 2);
        assert_eq!(stats.preference_synapses, 1);
        assert_eq!(stats.belief_synapses, 1);
    }
}
