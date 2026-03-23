//! Agent Coordination System (ACS)
//!
//! Provides a robust file-locking and notification framework to prevent
//! race conditions and ensure idempotent modifications in multi-agent environments.
//!
//! # Codex Compliance
//! - **Tier**: T2-C / T3
//! - **Commandments**: I (Quantify), II (Classify), IV (From), V (Wrap)

use nexcore_chrono::DateTime;
use nexcore_fs::dirs;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{BrainError, Result};
use crate::initialize_directories;
use nexcore_constants::bathroom_lock::BathroomLock;

/// Quantification of lock availability.
///
/// # Tier: T2-P
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LockStatus {
    /// File is available for locking.
    Vacant = 0,
    /// File is currently held by an agent.
    Occupied = 1,
}

/// Identifier for an agent session.
///
/// # Tier: T2-P
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);

impl From<String> for AgentId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for AgentId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Duration of a lock in seconds.
///
/// # Tier: T2-P
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockDuration(pub u64);

impl From<u64> for LockDuration {
    fn from(d: u64) -> Self {
        Self(d)
    }
}

/// A file lock entry in the registry.
///
/// # Tier: T2-C
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileLock {
    /// Absolute path to the locked file.
    pub path: PathBuf,
    /// ID of the agent holding the lock.
    pub agent_id: AgentId,
    /// Current status.
    pub status: LockStatus,
    /// When the lock was acquired.
    pub acquired_at: DateTime,
    /// How long the lock is valid.
    pub ttl: LockDuration,
}

impl FileLock {
    /// Check if the lock has expired.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        let now = DateTime::now();
        let expires_at = self.acquired_at + nexcore_chrono::Duration::seconds(self.ttl.0 as i64);
        now > expires_at
    }
}

/// Registry for managing inter-agent coordination.
///
/// # Tier: T3
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoordinationRegistry {
    /// Map of normalized paths to their active locks.
    pub locks: HashMap<String, FileLock>,
    /// Last global update timestamp.
    pub last_updated: DateTime,
}

impl CoordinationRegistry {
    /// Load the registry from the standard location.
    pub fn load() -> Result<Self> {
        initialize_directories()?;
        let path = Self::registry_path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = fs::read_to_string(&path)?;
        let registry: Self = serde_json::from_str(&content)?;
        Ok(registry)
    }

    /// Save the registry atomically.
    ///
    /// Protected by BathroomLock to prevent the meta-irony:
    /// two agents loading the registry simultaneously, both seeing
    /// Vacant, both saving — last write wins, first lock lost.
    pub fn save(&mut self) -> Result<()> {
        self.last_updated = DateTime::now();
        let path = Self::registry_path();

        // Bathroom lock the registry itself
        let lock = BathroomLock::for_coordination_registry();
        let _guard = lock
            .try_acquire("coordination-registry")
            .map_err(|e| BrainError::Other(format!("Registry lock failed: {e}")))?;

        let content = serde_json::to_string_pretty(self)?;

        // Atomic write via temp file
        let temp_path = path.with_extension("tmp");
        fs::write(&temp_path, content)?;
        fs::rename(temp_path, path)?;

        Ok(())
    }

    /// Attempt to acquire a lock for a file.
    pub fn acquire_lock(
        &mut self,
        file_path: &Path,
        agent_id: AgentId,
        ttl: LockDuration,
    ) -> Result<bool> {
        let path_str = Self::normalize_path(file_path)?;

        if self.is_lock_held_by_other(&path_str, &agent_id) {
            return Ok(false);
        }

        // Acquire or overwrite expired lock
        let lock = FileLock {
            path: file_path.to_path_buf(),
            agent_id,
            status: LockStatus::Occupied,
            acquired_at: DateTime::now(),
            ttl,
        };

        self.locks.insert(path_str, lock);
        self.save()?;
        Ok(true)
    }

    /// Check if a valid lock is held by a different agent.
    fn is_lock_held_by_other(&self, path_str: &str, agent_id: &AgentId) -> bool {
        if let Some(existing) = self.locks.get(path_str) {
            if existing.status == LockStatus::Occupied && !existing.is_expired() {
                return existing.agent_id != *agent_id;
            }
        }
        false
    }

    /// Release a lock held by an agent.
    pub fn release_lock(&mut self, file_path: &Path, agent_id: &AgentId) -> Result<bool> {
        let path_str = Self::normalize_path(file_path)?;

        if let Some(existing) = self.locks.get(&path_str) {
            if existing.agent_id == *agent_id {
                self.locks.remove(&path_str);
                self.save()?;
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Check the status of a file.
    pub fn check_status(&self, file_path: &Path) -> Result<LockStatus> {
        let path_str = Self::normalize_path(file_path)?;
        if let Some(lock) = self.locks.get(&path_str) {
            if !lock.is_expired() {
                return Ok(lock.status);
            }
        }
        Ok(LockStatus::Vacant)
    }

    fn registry_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".claude/file-locks/registry.json")
    }

    fn normalize_path(path: &Path) -> Result<String> {
        let absolute = fs::canonicalize(path).map_err(BrainError::Io)?;
        Ok(absolute.to_string_lossy().to_string())
    }
}

/// Helper for logging access attempts.
pub fn log_access(agent_id: &AgentId, path: &Path, action: &str) -> Result<()> {
    let log_path = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".claude/file-locks/access.log");

    let entry = format!(
        "{} | Agent: {} | File: {} | Action: {}\n",
        DateTime::now().to_rfc3339(),
        agent_id.0,
        path.display(),
        action
    );

    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?
        .write_all(entry.as_bytes())?;

    Ok(())
}

use std::io::Write;
