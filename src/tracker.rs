//! Content-addressable code tracking
//!
//! Tracks files using SHA-256 content hashes for change detection
//! and version management independent of git.

use nexcore_chrono::DateTime;
use nexcore_codec::hex;
use nexcore_hash::sha256::Sha256;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{BrainError, Result};
use crate::{initialize_directories, tracker_dir};

/// Index of all tracked files
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrackerIndex {
    /// Map of project snapshots to their tracked files
    pub projects: HashMap<String, ProjectSnapshot>,
}

/// A snapshot of tracked files for a project
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectSnapshot {
    /// Project identifier (name or path)
    pub project: String,

    /// Git commit hash when snapshot was created (if available)
    pub git_commit: Option<String>,

    /// When the snapshot was created
    pub created_at: DateTime,

    /// Tracked files in this snapshot
    pub files: HashMap<String, TrackedFile>,
}

/// A tracked file with content-addressable storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedFile {
    /// Original file path (relative or absolute)
    pub path: PathBuf,

    /// SHA-256 content hash (first 32 hex chars = 128 bits)
    pub content_hash: String,

    /// File size in bytes
    pub size: u64,

    /// When the file was tracked
    pub tracked_at: DateTime,

    /// File modification time when tracked
    pub mtime: DateTime,
}

impl TrackedFile {
    /// Create a tracked file from a path
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read.
    pub fn from_path(path: &Path) -> Result<Self> {
        let content = fs::read(path)?;
        let hash = compute_hash(&content);
        let metadata = fs::metadata(path)?;
        let mtime = metadata
            .modified()
            .map(DateTime::from)
            .unwrap_or_else(|_| DateTime::now());

        Ok(Self {
            path: path.to_path_buf(),
            content_hash: hash,
            size: metadata.len(),
            tracked_at: DateTime::now(),
            mtime,
        })
    }
}

/// Code tracker for content-addressable file management
#[derive(Debug)]
pub struct CodeTracker {
    /// Project identifier
    pub project: String,

    /// Git commit hash (optional)
    pub commit: Option<String>,

    /// Tracked files keyed by relative path
    files: HashMap<String, TrackedFile>,

    /// Path to tracker directory
    tracker_path: PathBuf,
}

impl CodeTracker {
    /// Create a new code tracker for a project
    ///
    /// # Errors
    ///
    /// Returns an error if directories cannot be created.
    pub fn new(project: impl Into<String>, commit: Option<String>) -> Result<Self> {
        initialize_directories()?;

        let project = project.into();
        let tracker_path = tracker_dir();

        Ok(Self {
            project,
            commit,
            files: HashMap::new(),
            tracker_path,
        })
    }

    /// Load an existing tracker for a project
    ///
    /// # Errors
    ///
    /// Returns an error if the tracker doesn't exist or cannot be loaded.
    pub fn load(project: &str) -> Result<Self> {
        let tracker_path = tracker_dir();
        let index = Self::load_index()?;

        let snapshot_key = Self::make_snapshot_key(project, None);
        let snapshot = index
            .projects
            .get(&snapshot_key)
            .ok_or_else(|| BrainError::FileNotTracked(format!("project: {project}")))?;

        Ok(Self {
            project: project.to_string(),
            commit: snapshot.git_commit.clone(),
            files: snapshot.files.clone(),
            tracker_path,
        })
    }

    /// Track a file
    ///
    /// Computes the content hash and stores a copy in the tracker.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or tracked.
    pub fn track_file(&mut self, path: &Path) -> Result<TrackedFile> {
        let tracked = TrackedFile::from_path(path)?;

        // Store a copy in the active directory
        let snapshot_dir = self.get_snapshot_dir();
        fs::create_dir_all(&snapshot_dir)?;

        // Hash-prefixed filename: {hash_prefix}_{original_name}
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        let hash_prefix = &tracked.content_hash[..16]; // First 16 chars = 64 bits
        let tracked_filename = format!("{hash_prefix}_{file_name}");
        let tracked_path = snapshot_dir.join(&tracked_filename);

        // Copy file content
        let content = fs::read(path)?;
        fs::write(&tracked_path, &content)?;

        // Store in memory
        let relative_path = path.to_string_lossy().to_string();
        self.files.insert(relative_path.clone(), tracked.clone());

        // Update index
        self.save_to_index()?;

        tracing::debug!("Tracked file {} with hash {}", path.display(), hash_prefix);

        // Dual-write: mirror to SQLite
        crate::db::sync_tracked_file(
            &self.project,
            &relative_path,
            &tracked.content_hash,
            tracked.size,
            tracked.tracked_at,
            tracked.mtime,
        );

        Ok(tracked)
    }

    /// Check if a file has changed since it was tracked
    ///
    /// # Errors
    ///
    /// Returns an error if the file is not tracked or cannot be read.
    pub fn has_changed(&self, path: &Path) -> Result<bool> {
        let relative_path = path.to_string_lossy().to_string();
        let tracked = self
            .files
            .get(&relative_path)
            .ok_or_else(|| BrainError::FileNotTracked(relative_path.clone()))?;

        let current_content = fs::read(path)?;
        let current_hash = compute_hash(&current_content);

        Ok(current_hash != tracked.content_hash)
    }

    /// Get the original content when the file was tracked
    ///
    /// # Errors
    ///
    /// Returns an error if the file is not tracked or the backup cannot be read.
    pub fn get_original(&self, path: &Path) -> Result<String> {
        let relative_path = path.to_string_lossy().to_string();
        let tracked = self
            .files
            .get(&relative_path)
            .ok_or_else(|| BrainError::FileNotTracked(relative_path.clone()))?;

        // Find the backup file
        let snapshot_dir = self.get_snapshot_dir();
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        let hash_prefix = &tracked.content_hash[..16];
        let tracked_filename = format!("{hash_prefix}_{file_name}");
        let tracked_path = snapshot_dir.join(tracked_filename);

        let content = fs::read_to_string(&tracked_path)?;
        Ok(content)
    }

    /// Snapshot multiple files from a project
    ///
    /// # Errors
    ///
    /// Returns an error if any file cannot be tracked.
    pub fn snapshot_project(&mut self, paths: &[PathBuf]) -> Result<usize> {
        let mut count = 0;
        for path in paths {
            if path.exists() && path.is_file() {
                self.track_file(path)?;
                count += 1;
            }
        }
        Ok(count)
    }

    /// List all tracked files
    #[must_use]
    pub fn list_files(&self) -> Vec<&TrackedFile> {
        self.files.values().collect()
    }

    /// Get a tracked file by path
    #[must_use]
    pub fn get_file(&self, path: &str) -> Option<&TrackedFile> {
        self.files.get(path)
    }

    /// Archive the current snapshot to history
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot cannot be archived.
    pub fn archive(&self) -> Result<PathBuf> {
        let snapshot_dir = self.get_snapshot_dir();
        if !snapshot_dir.exists() {
            return Err(BrainError::FileNotTracked("no active snapshot".into()));
        }

        // Move to history with timestamp
        let timestamp = DateTime::now().format("%Y%m%d_%H%M%S").unwrap_or_default();
        let history_name = format!("{}_{timestamp}", self.project.replace('/', "_"));
        let history_dir = self.tracker_path.join("history").join(&history_name);

        fs::rename(&snapshot_dir, &history_dir)?;

        tracing::info!("Archived snapshot to {}", history_dir.display());

        Ok(history_dir)
    }

    // Private helpers

    fn get_snapshot_dir(&self) -> PathBuf {
        let commit_suffix = self
            .commit
            .as_ref()
            .map(|c| format!("_{}", &c[..8.min(c.len())]))
            .unwrap_or_default();
        let snapshot_name = format!("{}{commit_suffix}", self.project.replace(['/', '\\'], "_"));
        self.tracker_path.join("active").join(snapshot_name)
    }

    fn make_snapshot_key(project: &str, commit: Option<&str>) -> String {
        match commit {
            Some(c) => format!("{}@{}", project, &c[..8.min(c.len())]),
            None => project.to_string(),
        }
    }

    fn load_index() -> Result<TrackerIndex> {
        let index_path = tracker_dir().join("index.json");

        if !index_path.exists() {
            return Ok(TrackerIndex::default());
        }

        let content = fs::read_to_string(&index_path)?;
        let index: TrackerIndex = serde_json::from_str(&content)?;
        Ok(index)
    }

    fn save_to_index(&self) -> Result<()> {
        let mut index = Self::load_index()?;

        let snapshot_key = Self::make_snapshot_key(&self.project, self.commit.as_deref());
        let snapshot = ProjectSnapshot {
            project: self.project.clone(),
            git_commit: self.commit.clone(),
            created_at: DateTime::now(),
            files: self.files.clone(),
        };

        index.projects.insert(snapshot_key, snapshot);

        let index_path = tracker_dir().join("index.json");
        let content = serde_json::to_string_pretty(&index)?;
        fs::write(&index_path, content)?;

        Ok(())
    }
}

/// Compute SHA-256 hash of content, returning first 32 hex chars
fn compute_hash(content: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content);
    let result = hasher.finalize();
    hex::encode(&result[..16]) // First 16 bytes = 32 hex chars
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_compute_hash() {
        let hash1 = compute_hash(b"hello world");
        let hash2 = compute_hash(b"hello world");
        let hash3 = compute_hash(b"different content");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.len(), 32); // 16 bytes = 32 hex chars
    }

    #[test]
    fn test_tracked_file_from_path() {
        let temp = TempDir::new().unwrap();
        let file_path = temp.path().join("test.txt");
        fs::write(&file_path, "test content").unwrap();

        let tracked = TrackedFile::from_path(&file_path).unwrap();
        assert_eq!(tracked.size, 12); // "test content" = 12 bytes
        assert_eq!(tracked.content_hash.len(), 32);
    }

    // ========== CTVP Phase 0: Edge Case Tests ==========

    #[test]
    fn test_compute_hash_empty_content() {
        let hash = compute_hash(b"");
        assert_eq!(hash.len(), 32);

        // Empty content should always produce the same hash
        let hash2 = compute_hash(b"");
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_compute_hash_binary_content() {
        // Binary content with null bytes and other non-printable characters
        let binary_content: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let hash = compute_hash(&binary_content);

        assert_eq!(hash.len(), 32);

        // Hash should be deterministic for binary content
        let hash2 = compute_hash(&binary_content);
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_compute_hash_unicode_content() {
        let unicode = "Hello 世界 🌍 Привет";
        let hash = compute_hash(unicode.as_bytes());

        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_compute_hash_large_content() {
        // 1MB of data
        let large_content: Vec<u8> = vec![0xAB; 1024 * 1024];
        let hash = compute_hash(&large_content);

        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_tracked_file_empty_file() {
        let temp = TempDir::new().unwrap();
        let file_path = temp.path().join("empty.txt");
        fs::write(&file_path, "").unwrap();

        let tracked = TrackedFile::from_path(&file_path).unwrap();
        assert_eq!(tracked.size, 0);
        assert_eq!(tracked.content_hash.len(), 32);
    }

    #[test]
    fn test_tracked_file_unicode_filename() {
        let temp = TempDir::new().unwrap();
        let file_path = temp.path().join("файл_日本語_🦀.txt");
        fs::write(&file_path, "unicode filename test").unwrap();

        let tracked = TrackedFile::from_path(&file_path).unwrap();
        assert!(tracked.path.to_string_lossy().contains("файл"));
        assert_eq!(tracked.size, 21);
    }

    #[test]
    fn test_tracked_file_nonexistent() {
        let temp = TempDir::new().unwrap();
        let file_path = temp.path().join("nonexistent.txt");

        let result = TrackedFile::from_path(&file_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_tracker_index_default() {
        let index = TrackerIndex::default();
        assert!(index.projects.is_empty());
    }

    #[test]
    fn test_project_snapshot_serialization() {
        let snapshot = ProjectSnapshot {
            project: "test-project".into(),
            git_commit: Some("abc123".into()),
            created_at: DateTime::now(),
            files: HashMap::new(),
        };

        let json = serde_json::to_string(&snapshot).unwrap();
        let parsed: ProjectSnapshot = serde_json::from_str(&json).unwrap();

        assert_eq!(snapshot.project, parsed.project);
        assert_eq!(snapshot.git_commit, parsed.git_commit);
    }

    #[test]
    fn test_tracked_file_serialization() {
        let temp = TempDir::new().unwrap();
        let file_path = temp.path().join("test.txt");
        fs::write(&file_path, "content").unwrap();

        let tracked = TrackedFile::from_path(&file_path).unwrap();

        let json = serde_json::to_string(&tracked).unwrap();
        let parsed: TrackedFile = serde_json::from_str(&json).unwrap();

        assert_eq!(tracked.content_hash, parsed.content_hash);
        assert_eq!(tracked.size, parsed.size);
    }

    #[test]
    fn test_hash_prefix_extraction() {
        let hash = compute_hash(b"test content");

        // The first 16 characters should be a valid hex prefix
        let prefix = &hash[..16];
        assert_eq!(prefix.len(), 16);
        assert!(prefix.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_make_snapshot_key_no_commit() {
        let key = CodeTracker::make_snapshot_key("my-project", None);
        assert_eq!(key, "my-project");
    }

    #[test]
    fn test_make_snapshot_key_with_commit() {
        let key = CodeTracker::make_snapshot_key("my-project", Some("abc123def456"));
        assert_eq!(key, "my-project@abc123de");
    }

    #[test]
    fn test_make_snapshot_key_short_commit() {
        // Commit shorter than 8 chars
        let key = CodeTracker::make_snapshot_key("my-project", Some("abc"));
        assert_eq!(key, "my-project@abc");
    }

    #[test]
    fn test_tracked_file_mtime_is_recent() {
        let temp = TempDir::new().unwrap();
        let file_path = temp.path().join("test.txt");
        fs::write(&file_path, "content").unwrap();

        let tracked = TrackedFile::from_path(&file_path).unwrap();

        // mtime should be within the last minute
        let now = DateTime::now();
        let diff = now.signed_duration_since(tracked.mtime);
        assert!(diff.num_seconds() < 60);
    }
}
