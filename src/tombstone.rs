// Copyright © 2026 NexVigilant LLC. All Rights Reserved.

//! # Tombstone — Void Persistence Pattern
//!
//! Implements the tombstone pattern: deletion markers that persist through
//! garbage collection cycles, preventing accidental resurrection of deleted entities.
//!
//! ## Primitive Pairs
//!
//! - **Primary**: ∅(Void) × π(Persistence), score 0.78 — the void of deletion persists
//! - **∅ × ς**: State machine — `Pending → Entombed → Expired → Purged`
//! - **∅ × ρ**: Recursive cascade chains — parent entombment propagates to children
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use nexcore_brain::tombstone::{TombstoneRegistry, TombstonePolicy};
//!
//! let mut registry = TombstoneRegistry::new();
//! registry.entomb("session-123", "user deleted", TombstonePolicy::Permanent).ok();
//! assert!(registry.is_entombed("session-123"));
//! ```

use nexcore_chrono::{DateTime, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{BrainError, Result};

/// Serialization helpers for [`Duration`].
///
/// Stores as total milliseconds (i64) for compact, portable representation.
mod duration_serde {
    use nexcore_chrono::Duration;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(duration: &Duration, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.num_milliseconds().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = i64::deserialize(deserializer)?;
        Duration::try_milliseconds(millis)
            .ok_or_else(|| serde::de::Error::custom("duration milliseconds out of range"))
    }
}

/// Policy governing how long a tombstone persists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TombstonePolicy {
    /// Tombstone is permanent — it never expires and cannot be exhumed.
    Permanent,
    /// Tombstone expires after the specified duration from entombment time.
    Ttl(#[serde(with = "duration_serde")] Duration),
    /// Tombstone persists until an explicit acknowledgement removes it.
    UntilAcknowledged,
}

/// Lifecycle states for a tombstone entry.
///
/// Captures the ∅ × ς bonus pair: void crossed with state transitions.
///
/// ```text
/// Pending → Entombed → (TTL elapsed) → Expired → (sweep) → [removed]
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TombstoneState {
    /// Tombstone has been requested but not yet finalised.
    Pending,
    /// Entity has been fully entombed — deletion marker is active.
    Entombed,
    /// TTL has elapsed; eligible for purge on the next sweep.
    Expired,
    /// Tombstone has been removed by a GC sweep pass.
    Purged,
}

/// A deletion marker that survives garbage collection cycles.
///
/// Represents the ∅(Void) × π(Persistence) primitive pair: the fact of a
/// deletion persists beyond the deleted entity itself, blocking re-insertion
/// of stale data.
///
/// # Cascade Chains (∅ × ρ)
///
/// When `parent_id` is `Some`, this tombstone was created as part of a cascade
/// from its parent, enabling atomic entombment of whole entity subtrees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tombstone {
    /// Unique identifier of the deleted entity.
    pub id: String,
    /// Human-readable reason for the deletion.
    pub reason: String,
    /// Policy governing this tombstone's lifetime.
    pub policy: TombstonePolicy,
    /// Current lifecycle state.
    pub state: TombstoneState,
    /// Timestamp when the entity was entombed.
    pub entombed_at: DateTime,
    /// Parent tombstone ID for cascade chains (∅ × ρ recursive pair).
    pub parent_id: Option<String>,
}

impl Tombstone {
    /// Create a new tombstone in `Pending` state.
    fn new(id: String, reason: String, policy: TombstonePolicy, parent_id: Option<String>) -> Self {
        Self {
            id,
            reason,
            policy,
            state: TombstoneState::Pending,
            entombed_at: DateTime::now(),
            parent_id,
        }
    }

    /// Check whether this tombstone's TTL has elapsed.
    ///
    /// Returns `false` for [`TombstonePolicy::Permanent`] and
    /// [`TombstonePolicy::UntilAcknowledged`] policies.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        match &self.policy {
            TombstonePolicy::Ttl(duration) => {
                let elapsed = DateTime::now().signed_duration_since(self.entombed_at);
                elapsed >= *duration
            }
            TombstonePolicy::Permanent | TombstonePolicy::UntilAcknowledged => false,
        }
    }
}

/// Registry managing all tombstones for the brain system.
///
/// In-memory store keyed by entity ID. Serialise to/from JSON via `serde_json`
/// for persistence. Thread-safety is the caller's responsibility.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TombstoneRegistry {
    stones: HashMap<String, Tombstone>,
}

impl TombstoneRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Entomb an entity — creates an active deletion marker.
    ///
    /// The tombstone transitions immediately to [`TombstoneState::Entombed`].
    ///
    /// # Errors
    ///
    /// Returns [`BrainError::AlreadyExists`] if a tombstone already exists for `id`.
    pub fn entomb(
        &mut self,
        id: impl Into<String>,
        reason: impl Into<String>,
        policy: TombstonePolicy,
    ) -> Result<()> {
        let id = id.into();
        if self.stones.contains_key(&id) {
            return Err(BrainError::AlreadyExists(format!("tombstone:{id}")));
        }
        let mut stone = Tombstone::new(id.clone(), reason.into(), policy, None);
        stone.state = TombstoneState::Entombed;
        self.stones.insert(id, stone);
        Ok(())
    }

    /// Check whether an entity has an active (non-purged) tombstone.
    #[must_use]
    pub fn is_entombed(&self, id: &str) -> bool {
        self.stones.get(id).map_or(false, |s| {
            matches!(
                s.state,
                TombstoneState::Entombed | TombstoneState::Pending | TombstoneState::Expired
            )
        })
    }

    /// Exhume (resurrect) a tombstone — removes the deletion marker.
    ///
    /// Returns the removed [`Tombstone`] if found, or `None` if no tombstone
    /// exists for the given ID.
    ///
    /// # Errors
    ///
    /// Returns [`BrainError::Other`] if the tombstone has a
    /// [`TombstonePolicy::Permanent`] policy, since permanent tombstones cannot
    /// be resurrected.
    pub fn exhume(&mut self, id: &str) -> Result<Option<Tombstone>> {
        let Some(stone) = self.stones.get(id) else {
            return Ok(None);
        };
        if stone.policy == TombstonePolicy::Permanent {
            return Err(BrainError::Other(format!(
                "cannot exhume permanent tombstone: {id}"
            )));
        }
        Ok(self.stones.remove(id))
    }

    /// Sweep expired tombstones — GC pass removing TTL-elapsed entries.
    ///
    /// Transitions eligible `Entombed` tombstones to `Expired`, removes them
    /// from the registry, and returns the list of purged entity IDs.
    pub fn sweep(&mut self) -> Vec<String> {
        // Phase 1: mark TTL-elapsed entries as Expired
        for stone in self.stones.values_mut() {
            if stone.state == TombstoneState::Entombed && stone.is_expired() {
                stone.state = TombstoneState::Expired;
            }
        }
        // Phase 2: collect IDs eligible for purge
        let to_purge: Vec<String> = self
            .stones
            .values()
            .filter(|s| s.state == TombstoneState::Expired)
            .map(|s| s.id.clone())
            .collect();
        // Phase 3: remove from registry
        for id in &to_purge {
            self.stones.remove(id);
        }
        to_purge
    }

    /// Cascade entomb a parent entity and all its children.
    ///
    /// Implements the ∅ × ρ recursive pair: cascading void propagation through
    /// entity hierarchies. All children receive the parent's ID as `parent_id`.
    ///
    /// The operation is **not** atomic — entries successfully entombed before
    /// the first error are not rolled back.
    ///
    /// # Errors
    ///
    /// Returns [`BrainError::AlreadyExists`] if any entity (parent or child)
    /// already has a tombstone.
    pub fn cascade_entomb(
        &mut self,
        parent_id: impl Into<String>,
        children_ids: Vec<String>,
        reason: impl Into<String>,
        policy: TombstonePolicy,
    ) -> Result<()> {
        let parent_id = parent_id.into();
        let reason = reason.into();

        // Entomb parent first
        self.entomb(parent_id.clone(), reason.clone(), policy.clone())?;

        // Entomb each child, recording the parent link
        for child_id in children_ids {
            if self.stones.contains_key(&child_id) {
                return Err(BrainError::AlreadyExists(format!("tombstone:{child_id}")));
            }
            let mut stone = Tombstone::new(
                child_id.clone(),
                reason.clone(),
                policy.clone(),
                Some(parent_id.clone()),
            );
            stone.state = TombstoneState::Entombed;
            self.stones.insert(child_id, stone);
        }

        Ok(())
    }

    /// List all [`TombstoneState::Entombed`] tombstones.
    ///
    /// Excludes `Pending`, `Expired`, and `Purged` (purged stones are removed
    /// from the registry and never returned).
    #[must_use]
    pub fn list_entombed(&self) -> Vec<&Tombstone> {
        self.stones
            .values()
            .filter(|s| s.state == TombstoneState::Entombed)
            .collect()
    }

    /// List tombstones filtered by state.
    #[must_use]
    pub fn list_by_state(&self, state: &TombstoneState) -> Vec<&Tombstone> {
        self.stones.values().filter(|s| &s.state == state).collect()
    }

    /// Total number of tombstones in the registry (all states).
    #[must_use]
    pub fn len(&self) -> usize {
        self.stones.len()
    }

    /// Returns `true` if the registry contains no tombstones.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stones.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: returns a TTL policy already expired at creation time.
    fn expired_policy() -> TombstonePolicy {
        // -1 ms is within i64 range; unwrap_or fallback is unreachable in practice
        let dur = Duration::try_milliseconds(-1).unwrap_or(Duration::zero());
        TombstonePolicy::Ttl(dur)
    }

    /// Helper: returns a TTL policy expiring in 1 day (well in the future).
    fn long_ttl_policy() -> TombstonePolicy {
        // 86_400_000 ms == 1 day
        let dur = Duration::try_milliseconds(86_400_000_i64).unwrap_or(Duration::zero());
        TombstonePolicy::Ttl(dur)
    }

    // ── Basic entomb / is_entombed ────────────────────────────────────────────

    #[test]
    fn test_entomb_creates_active_tombstone() {
        let mut reg = TombstoneRegistry::new();
        let result = reg.entomb("session-001", "user deleted", TombstonePolicy::Permanent);
        assert!(result.is_ok(), "entomb should succeed on fresh id");
        assert!(reg.is_entombed("session-001"));
    }

    #[test]
    fn test_is_entombed_returns_false_for_unknown() {
        let reg = TombstoneRegistry::new();
        assert!(!reg.is_entombed("no-such-id"));
    }

    #[test]
    fn test_entomb_duplicate_returns_already_exists() {
        let mut reg = TombstoneRegistry::new();
        reg.entomb("dup-01", "first", TombstonePolicy::Permanent)
            .ok();
        let err = reg.entomb("dup-01", "second", TombstonePolicy::Permanent);
        assert!(
            matches!(err, Err(BrainError::AlreadyExists(_))),
            "duplicate entomb should yield AlreadyExists"
        );
    }

    #[test]
    fn test_until_acknowledged_policy_is_active() {
        let mut reg = TombstoneRegistry::new();
        reg.entomb("ack-01", "pending ack", TombstonePolicy::UntilAcknowledged)
            .ok();
        assert!(reg.is_entombed("ack-01"));
    }

    // ── Exhume ───────────────────────────────────────────────────────────────

    #[test]
    fn test_exhume_removes_non_permanent_tombstone() {
        let mut reg = TombstoneRegistry::new();
        reg.entomb("session-002", "test", TombstonePolicy::UntilAcknowledged)
            .ok();
        let result = reg.exhume("session-002");
        assert!(result.is_ok(), "exhume should succeed");
        let stone = result.ok().flatten();
        assert!(stone.is_some(), "returned tombstone should be Some");
        assert!(
            !reg.is_entombed("session-002"),
            "should be removed after exhume"
        );
    }

    #[test]
    fn test_exhume_returns_none_for_unknown() {
        let mut reg = TombstoneRegistry::new();
        let result = reg.exhume("does-not-exist");
        assert!(result.is_ok());
        assert!(
            result.ok().flatten().is_none(),
            "unknown id should return None"
        );
    }

    #[test]
    fn test_exhume_permanent_returns_error() {
        let mut reg = TombstoneRegistry::new();
        reg.entomb("perm-01", "never delete", TombstonePolicy::Permanent)
            .ok();
        let err = reg.exhume("perm-01");
        assert!(
            matches!(err, Err(BrainError::Other(_))),
            "exhuming permanent should yield Other error"
        );
        assert!(
            reg.is_entombed("perm-01"),
            "permanent tombstone survives failed exhume"
        );
    }

    #[test]
    fn test_exhume_ttl_policy_succeeds() {
        let mut reg = TombstoneRegistry::new();
        reg.entomb("ttl-exhume", "test", long_ttl_policy()).ok();
        let result = reg.exhume("ttl-exhume");
        assert!(result.is_ok(), "TTL tombstone should be exhumable");
        assert!(!reg.is_entombed("ttl-exhume"));
    }

    // ── Sweep / TTL ───────────────────────────────────────────────────────────

    #[test]
    fn test_sweep_removes_expired_tombstones() {
        let mut reg = TombstoneRegistry::new();
        reg.entomb("expired-01", "old", expired_policy()).ok();
        reg.entomb("permanent-01", "keep", TombstonePolicy::Permanent)
            .ok();

        let purged = reg.sweep();

        assert!(
            purged.contains(&"expired-01".to_string()),
            "expired-01 should be in purged list"
        );
        assert!(
            !purged.contains(&"permanent-01".to_string()),
            "permanent-01 must not be purged"
        );
        assert!(
            !reg.is_entombed("expired-01"),
            "expired-01 gone from registry"
        );
        assert!(reg.is_entombed("permanent-01"), "permanent-01 still alive");
    }

    #[test]
    fn test_sweep_returns_empty_when_nothing_expires() {
        let mut reg = TombstoneRegistry::new();
        reg.entomb("perm-sweep", "safe", TombstonePolicy::Permanent)
            .ok();
        let purged = reg.sweep();
        assert!(
            purged.is_empty(),
            "no expired entries — nothing should be purged"
        );
    }

    #[test]
    fn test_sweep_clears_multiple_expired() {
        let mut reg = TombstoneRegistry::new();
        for i in 0..5_u8 {
            reg.entomb(format!("expired-{i}"), "old", expired_policy())
                .ok();
        }
        let purged = reg.sweep();
        assert_eq!(purged.len(), 5, "all 5 expired tombstones should be purged");
        assert!(reg.is_empty(), "registry should be empty after sweep");
    }

    #[test]
    fn test_sweep_preserves_long_ttl() {
        let mut reg = TombstoneRegistry::new();
        reg.entomb("future", "r", long_ttl_policy()).ok();
        let purged = reg.sweep();
        assert!(purged.is_empty(), "1-day TTL should not expire immediately");
        assert!(
            reg.is_entombed("future"),
            "future tombstone should survive sweep"
        );
    }

    // ── Cascade entomb ────────────────────────────────────────────────────────

    #[test]
    fn test_cascade_entombs_parent_and_children() {
        let mut reg = TombstoneRegistry::new();
        let children = vec!["child-1".to_string(), "child-2".to_string()];
        let result = reg.cascade_entomb(
            "parent-01",
            children,
            "cascade delete",
            TombstonePolicy::Permanent,
        );
        assert!(result.is_ok(), "cascade_entomb should succeed");
        assert!(reg.is_entombed("parent-01"));
        assert!(reg.is_entombed("child-1"));
        assert!(reg.is_entombed("child-2"));
    }

    #[test]
    fn test_cascade_child_has_parent_id_set() {
        let mut reg = TombstoneRegistry::new();
        reg.cascade_entomb(
            "p-01",
            vec!["c-01".to_string()],
            "cascade",
            TombstonePolicy::Permanent,
        )
        .ok();

        let entombed = reg.list_entombed();
        let child = entombed.iter().find(|s| s.id == "c-01");
        assert!(child.is_some(), "child tombstone should exist");
        if let Some(c) = child {
            assert_eq!(
                c.parent_id.as_deref(),
                Some("p-01"),
                "child's parent_id must point to parent"
            );
        }
    }

    #[test]
    fn test_cascade_parent_has_no_parent_id() {
        let mut reg = TombstoneRegistry::new();
        reg.cascade_entomb(
            "root",
            vec!["leaf".to_string()],
            "r",
            TombstonePolicy::Permanent,
        )
        .ok();

        let entombed = reg.list_entombed();
        let root = entombed.iter().find(|s| s.id == "root");
        assert!(root.is_some());
        if let Some(r) = root {
            assert!(
                r.parent_id.is_none(),
                "root tombstone should have no parent"
            );
        }
    }

    #[test]
    fn test_cascade_fails_on_duplicate_child() {
        let mut reg = TombstoneRegistry::new();
        reg.entomb("existing-child", "pre-existing", TombstonePolicy::Permanent)
            .ok();
        let children = vec!["existing-child".to_string(), "new-child".to_string()];
        let result =
            reg.cascade_entomb("parent-02", children, "cascade", TombstonePolicy::Permanent);
        assert!(
            matches!(result, Err(BrainError::AlreadyExists(_))),
            "duplicate child should yield AlreadyExists"
        );
    }

    // ── List operations ───────────────────────────────────────────────────────

    #[test]
    fn test_list_entombed_returns_all_active() {
        let mut reg = TombstoneRegistry::new();
        reg.entomb("a", "r", TombstonePolicy::Permanent).ok();
        reg.entomb("b", "r", TombstonePolicy::Permanent).ok();
        assert_eq!(reg.list_entombed().len(), 2);
    }

    #[test]
    fn test_list_by_state_filters_entombed() {
        let mut reg = TombstoneRegistry::new();
        reg.entomb("e-01", "old", expired_policy()).ok();
        reg.entomb("perm-ls", "keep", TombstonePolicy::Permanent)
            .ok();

        // After sweep, expired is removed; only perm-ls remains
        let _purged = reg.sweep();

        let entombed = reg.list_by_state(&TombstoneState::Entombed);
        assert_eq!(entombed.len(), 1, "only perm-ls should remain entombed");
        assert_eq!(entombed[0].id, "perm-ls");
    }

    #[test]
    fn test_list_entombed_empty_on_fresh_registry() {
        let reg = TombstoneRegistry::new();
        assert!(reg.list_entombed().is_empty());
    }

    // ── len / is_empty ────────────────────────────────────────────────────────

    #[test]
    fn test_len_and_is_empty_transitions() {
        let mut reg = TombstoneRegistry::new();
        assert!(reg.is_empty(), "new registry should be empty");
        assert_eq!(reg.len(), 0);

        reg.entomb("x", "r", TombstonePolicy::Permanent).ok();
        assert!(!reg.is_empty());
        assert_eq!(reg.len(), 1);

        reg.entomb("y", "r", TombstonePolicy::Permanent).ok();
        assert_eq!(reg.len(), 2);
    }

    // ── State transition checks ───────────────────────────────────────────────

    #[test]
    fn test_fresh_entomb_is_in_entombed_state() {
        let mut reg = TombstoneRegistry::new();
        reg.entomb("state-test", "r", TombstonePolicy::Permanent)
            .ok();
        let by_state = reg.list_by_state(&TombstoneState::Entombed);
        assert_eq!(by_state.len(), 1);
        assert_eq!(by_state[0].state, TombstoneState::Entombed);
    }

    #[test]
    fn test_pending_state_not_present_after_entomb() {
        let mut reg = TombstoneRegistry::new();
        reg.entomb("no-pending", "r", TombstonePolicy::Permanent)
            .ok();
        // entomb() immediately sets Entombed, skipping Pending
        let pending = reg.list_by_state(&TombstoneState::Pending);
        assert!(
            pending.is_empty(),
            "entomb() should set Entombed directly, not Pending"
        );
    }
}
