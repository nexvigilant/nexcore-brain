//! Belief dependency graph for inter-belief causality
//!
//! Supports implication edges with cycle detection and DFS reachability.

use nexcore_chrono::DateTime;
use serde::{Deserialize, Serialize};

/// Edge type for belief implications
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImplicationStrength {
    /// Strong implication (propagation: 0.9)
    Strong,
    /// Moderate implication (propagation: 0.6)
    Moderate,
    /// Weak implication (propagation: 0.3)
    Weak,
}

impl ImplicationStrength {
    /// Get propagation factor for confidence updates
    #[must_use]
    pub fn propagation_factor(&self) -> f64 {
        match self {
            Self::Strong => 0.9,
            Self::Moderate => 0.6,
            Self::Weak => 0.3,
        }
    }
}

/// An implication edge between beliefs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefImplication {
    /// Source belief ID (antecedent)
    pub from: String,
    /// Target belief ID (consequent)
    pub to: String,
    /// Strength of implication
    pub strength: ImplicationStrength,
    /// When this implication was established
    pub established_at: DateTime,
}

/// Belief dependency graph for inter-belief causality.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BeliefGraph {
    /// Implication edges (from → to with strength)
    pub implications: Vec<BeliefImplication>,
}

impl BeliefGraph {
    /// Create an empty belief graph
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an implication edge
    pub fn add_implication(
        &mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        strength: ImplicationStrength,
    ) {
        let imp = BeliefImplication {
            from: from.into(),
            to: to.into(),
            strength,
            established_at: DateTime::now(),
        };
        self.implications.push(imp);
    }

    /// Get all beliefs implied by a given belief
    #[must_use]
    pub fn implied_by(&self, belief_id: &str) -> Vec<(&str, ImplicationStrength)> {
        self.implications
            .iter()
            .filter(|imp| imp.from == belief_id)
            .map(|imp| (imp.to.as_str(), imp.strength))
            .collect()
    }

    /// Get all beliefs that imply a given belief
    #[must_use]
    pub fn implies(&self, belief_id: &str) -> Vec<(&str, ImplicationStrength)> {
        self.implications
            .iter()
            .filter(|imp| imp.to == belief_id)
            .map(|imp| (imp.from.as_str(), imp.strength))
            .collect()
    }

    /// Remove an implication edge
    pub fn remove_implication(&mut self, from: &str, to: &str) {
        self.implications
            .retain(|imp| imp.from != from || imp.to != to);
    }

    /// Check if adding an edge would create a cycle
    #[must_use]
    pub fn would_create_cycle(&self, from: &str, to: &str) -> bool {
        self.can_reach(to, from)
    }

    /// Check if source can reach target via implications (DFS)
    fn can_reach(&self, source: &str, target: &str) -> bool {
        let mut visited = std::collections::HashSet::new();
        self.dfs_reach(source, target, &mut visited)
    }

    fn dfs_reach(
        &self,
        current: &str,
        target: &str,
        visited: &mut std::collections::HashSet<String>,
    ) -> bool {
        if current == target {
            return true;
        }
        if !visited.insert(current.to_string()) {
            return false;
        }
        self.implied_by(current)
            .iter()
            .any(|(next, _)| self.dfs_reach(next, target, visited))
    }
}
