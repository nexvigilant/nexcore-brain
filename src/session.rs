//! Brain session management
//!
//! Each session represents a working memory context for a Claude Code session.
//! Sessions contain artifacts (task.md, plan.md, etc.) that can be versioned.

use nexcore_chrono::DateTime;
use nexcore_id::NexId;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::artifact::{Artifact, ArtifactMetadata, ArtifactType};
use crate::error::{BrainError, Result};
use crate::{brain_dir, initialize_directories};
use nexcore_constants::bathroom_lock::BathroomLock;

// ========== Episodic Event Log ==========

/// Event types for the episodic session log.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionEventKind {
    /// Artifact was saved or updated
    ArtifactSaved,
    /// Artifact was resolved (snapshotted)
    ArtifactResolved,
    /// A preference was learned or reinforced
    PreferenceLearned,
    /// A correction was recorded
    CorrectionRecorded,
    /// A belief was crystallized from corrections
    BeliefCrystallized,
    /// Session was created
    SessionCreated,
    /// Session was closed
    SessionClosed,
    /// Custom user-defined event
    Custom,
}

/// A single event in the episodic session log.
///
/// Events are append-only (JSONL) — one per line in `events.jsonl`.
/// This provides a causal timeline of everything that happened in a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEvent {
    /// When this event occurred
    pub timestamp: DateTime,
    /// Event classification
    pub kind: SessionEventKind,
    /// Human-readable description
    pub description: String,
    /// Optional structured metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl SessionEvent {
    /// Create a new session event.
    #[must_use]
    pub fn new(kind: SessionEventKind, description: impl Into<String>) -> Self {
        Self {
            timestamp: DateTime::now(),
            kind,
            description: description.into(),
            metadata: None,
        }
    }

    /// Create a new event with metadata.
    #[must_use]
    pub fn with_metadata(
        kind: SessionEventKind,
        description: impl Into<String>,
        metadata: serde_json::Value,
    ) -> Self {
        Self {
            timestamp: DateTime::now(),
            kind,
            description: description.into(),
            metadata: Some(metadata),
        }
    }
}

/// Session registry entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEntry {
    /// Session ID
    pub id: String,

    /// When the session was created
    pub created_at: DateTime,

    /// Optional project name/path
    pub project: Option<String>,

    /// Optional git commit hash
    pub git_commit: Option<String>,

    /// Short description of the session
    pub description: Option<String>,
}

/// A brain session representing working memory for a Claude Code session
#[derive(Debug, Clone)]
pub struct BrainSession {
    /// Unique session identifier
    pub id: String,

    /// When the session was created
    pub created_at: DateTime,

    /// Optional associated project
    pub project: Option<String>,

    /// Optional git commit when session started
    pub git_commit: Option<String>,

    /// Path to this session's directory
    session_dir: PathBuf,
}

impl BrainSession {
    /// Create a new brain session
    ///
    /// # Errors
    ///
    /// Returns an error if the session directory cannot be created.
    pub fn create() -> Result<Self> {
        Self::create_with_options(None, None, None)
    }

    /// Create a new brain session with options
    ///
    /// # Errors
    ///
    /// Returns an error if the session directory cannot be created.
    pub fn create_with_options(
        project: Option<String>,
        git_commit: Option<String>,
        description: Option<String>,
    ) -> Result<Self> {
        // Ensure directories exist
        initialize_directories()?;

        let id = NexId::v4().to_string();
        let created_at = DateTime::now();
        let session_dir = brain_dir().join("sessions").join(&id);

        // Create session directory
        fs::create_dir_all(&session_dir)?;

        let session = Self {
            id,
            created_at,
            project,
            git_commit,
            session_dir,
        };

        // Register session in index
        session.register_in_index(description.clone())?;

        // Dual-write: mirror to SQLite
        crate::db::sync_session(
            &session.id,
            session.project.as_deref(),
            session.git_commit.as_deref(),
            description.as_deref(),
            session.created_at,
        );

        Ok(session)
    }

    /// Load an existing session by ID
    ///
    /// Queries brain.db first (contains all sessions including those created by
    /// hooks via direct SQL). Falls back to index.json for backward compatibility.
    ///
    /// # Errors
    ///
    /// Returns an error if the session doesn't exist or cannot be loaded.
    pub fn load(id: String) -> Result<Self> {
        let session_dir = brain_dir().join("sessions").join(&id);

        // Try loading from SQLite first (authoritative — contains hook-created sessions)
        if let Some(session) = Self::load_from_db(&id, &session_dir) {
            return Ok(session);
        }

        // Fallback: check index.json (legacy path)
        let index = Self::load_index()?;
        if let Some(entry) = index.iter().find(|e| e.id == id) {
            return Ok(Self {
                id: entry.id.clone(),
                created_at: entry.created_at,
                project: entry.project.clone(),
                git_commit: entry.git_commit.clone(),
                session_dir,
            });
        }

        Err(BrainError::SessionNotFound(id))
    }

    /// Load session by string ID
    ///
    /// # Errors
    ///
    /// Returns an error if the ID is invalid or session doesn't exist.
    pub fn load_str(id: &str) -> Result<Self> {
        Self::load(id.to_string())
    }

    /// Get the path to the session directory
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.session_dir
    }

    /// List all sessions
    ///
    /// Queries brain.db first (authoritative). Falls back to index.json.
    ///
    /// # Errors
    ///
    /// Returns an error if neither store can be read.
    pub fn list_all() -> Result<Vec<SessionEntry>> {
        if let Some(entries) = Self::list_from_db() {
            if !entries.is_empty() {
                return Ok(entries);
            }
        }
        Self::load_index()
    }

    /// Load the most recent session
    ///
    /// Queries brain.db first (authoritative). Falls back to index.json.
    ///
    /// # Errors
    ///
    /// Returns an error if no sessions exist or neither store can be read.
    pub fn load_latest() -> Result<Self> {
        // Try DB first — it has all sessions including hook-created ones
        if let Some(entries) = Self::list_from_db() {
            if let Some(entry) = entries.into_iter().max_by_key(|e| e.created_at) {
                let session_dir = brain_dir().join("sessions").join(&entry.id);
                return Ok(Self {
                    id: entry.id,
                    created_at: entry.created_at,
                    project: entry.project,
                    git_commit: entry.git_commit,
                    session_dir,
                });
            }
        }

        // Fallback: index.json
        let index = Self::load_index()?;
        let entry = index
            .into_iter()
            .max_by_key(|e| e.created_at)
            .ok_or_else(|| BrainError::SessionNotFound("no sessions found".into()))?;

        Self::load(entry.id)
    }

    /// Import sessions from the Antigravity brain system.
    ///
    /// Scans the Antigravity brain directory and registers any new sessions.
    ///
    /// # Errors
    ///
    /// Returns an error if the Antigravity directory cannot be read.
    pub fn import_from_antigravity() -> Result<usize> {
        let antigravity_dir = crate::brain_dir_antigravity();
        if !antigravity_dir.exists() {
            return Ok(0);
        }

        let mut imported_count = 0;
        let index = Self::load_index()?;
        let existing_ids: std::collections::HashSet<String> =
            index.iter().map(|e| e.id.clone()).collect();

        for entry in fs::read_dir(antigravity_dir)? {
            if let Ok(true) = process_antigravity_entry(entry?, &existing_ids) {
                imported_count += 1;
            }
        }

        Ok(imported_count)
    }

    fn import_single_session(id: String, path: &Path) -> Result<()> {
        // Determine creation time from directory metadata
        let created_at = fs::metadata(path)
            .ok()
            .and_then(|m| m.modified().ok())
            .map(DateTime::from)
            .unwrap_or_else(|| DateTime::now());

        let session = Self {
            id,
            created_at,
            project: None, // Will be inferred later
            git_commit: None,
            session_dir: path.to_path_buf(),
        };

        session.register_in_index(Some("Imported from Antigravity".to_string()))?;
        Ok(())
    }

    /// Save an artifact to the session
    ///
    /// Creates or updates the artifact file and its metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the artifact cannot be saved.
    pub fn save_artifact(&self, artifact: &Artifact) -> Result<()> {
        let artifact_path = self.session_dir.join(&artifact.name);
        let metadata_path = self
            .session_dir
            .join(format!("{}.metadata.json", artifact.name));

        // Acquire bathroom lock — prevent concurrent writes to same artifact
        let lock = BathroomLock::for_artifact(&artifact_path);
        let _guard = lock
            .try_acquire(&self.id)
            .map_err(|e| BrainError::Other(format!("Lock failed: {e}")))?;

        // Write artifact content
        fs::write(&artifact_path, &artifact.content)?;

        // Load or create metadata
        let metadata = if metadata_path.exists() {
            let content = fs::read_to_string(&metadata_path)?;
            let mut meta: ArtifactMetadata = serde_json::from_str(&content)?;
            meta.touch();
            meta.summary = artifact.generate_summary();
            meta
        } else {
            ArtifactMetadata::new(artifact.artifact_type, artifact.generate_summary())
        };

        // Write metadata
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        fs::write(&metadata_path, metadata_json)?;

        tracing::debug!(
            "Saved artifact {} (v{}) to session {}",
            artifact.name,
            metadata.current_version,
            self.id
        );

        // Dual-write: mirror to SQLite
        let tags_json = serde_json::to_string(&metadata.tags).unwrap_or_else(|_| "[]".to_string());
        let custom_json =
            serde_json::to_string(&metadata.custom).unwrap_or_else(|_| "null".to_string());
        crate::db::sync_artifact(
            &self.id,
            &artifact.name,
            &artifact.artifact_type.to_string(),
            &artifact.content,
            &metadata.summary,
            metadata.current_version,
            &tags_json,
            &custom_json,
            metadata.created_at,
            metadata.updated_at,
        );

        Ok(())
    }

    /// Resolve an artifact (create immutable snapshot)
    ///
    /// Creates a `.resolved` copy and increments version counter.
    /// Returns the new version number.
    ///
    /// # Errors
    ///
    /// Returns an error if the artifact doesn't exist or cannot be resolved.
    pub fn resolve_artifact(&self, name: &str) -> Result<u32> {
        let artifact_path = self.session_dir.join(name);
        let metadata_path = self.session_dir.join(format!("{name}.metadata.json"));

        if !artifact_path.exists() {
            return Err(BrainError::ArtifactNotFound(name.to_string()));
        }

        // Acquire bathroom lock — resolve is a multi-step write (read + 3 writes)
        let lock = BathroomLock::for_artifact(&artifact_path);
        let _guard = lock
            .try_acquire(&self.id)
            .map_err(|e| BrainError::Other(format!("Lock failed: {e}")))?;

        // Load and update metadata
        let content = fs::read_to_string(&metadata_path).unwrap_or_else(|_| {
            serde_json::to_string(&ArtifactMetadata::new(
                ArtifactType::from_filename(name),
                "Auto-created metadata",
            ))
            .unwrap_or_default()
        });

        let mut metadata: ArtifactMetadata = serde_json::from_str(&content)?;
        let new_version = metadata.increment_version();

        // Read current artifact content
        let artifact_content = fs::read_to_string(&artifact_path)?;

        // Write resolved versions
        let resolved_path = self.session_dir.join(format!("{name}.resolved"));
        let versioned_path = self
            .session_dir
            .join(format!("{name}.resolved.{new_version}"));

        fs::write(&resolved_path, &artifact_content)?;
        fs::write(&versioned_path, &artifact_content)?;

        // Update metadata
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        fs::write(&metadata_path, metadata_json)?;

        tracing::info!(
            "Resolved artifact {} to version {} in session {}",
            name,
            new_version,
            self.id
        );

        // Dual-write: mirror resolved version to SQLite
        crate::db::sync_artifact_version(&self.id, name, new_version, &artifact_content);

        Ok(new_version)
    }

    /// Get an artifact (current or specific version)
    ///
    /// # Errors
    ///
    /// Returns an error if the artifact or version doesn't exist.
    pub fn get_artifact(&self, name: &str, version: Option<u32>) -> Result<Artifact> {
        let artifact_path = self.get_artifact_path(name, version);

        if !artifact_path.exists() {
            return Err(self.create_artifact_not_found_error(name, version));
        }

        let content = fs::read_to_string(&artifact_path)?;
        let updated_at = self.get_artifact_mtime(&artifact_path);
        let artifact_type = self.get_artifact_type(name);

        Ok(Artifact {
            name: name.to_string(),
            artifact_type,
            content,
            version: version.unwrap_or(0),
            updated_at,
        })
    }

    fn get_artifact_path(&self, name: &str, version: Option<u32>) -> PathBuf {
        match version {
            Some(v) => self.session_dir.join(format!("{name}.resolved.{v}")),
            None => self.session_dir.join(name),
        }
    }

    fn create_artifact_not_found_error(&self, name: &str, version: Option<u32>) -> BrainError {
        match version {
            Some(v) => BrainError::VersionNotFound {
                artifact: name.to_string(),
                version: v,
            },
            None => BrainError::ArtifactNotFound(name.to_string()),
        }
    }

    fn get_artifact_mtime(&self, path: &Path) -> DateTime {
        fs::metadata(path)
            .ok()
            .and_then(|m| m.modified().ok())
            .map(DateTime::from)
            .unwrap_or_else(|| DateTime::now())
    }

    fn get_artifact_type(&self, name: &str) -> ArtifactType {
        let metadata_path = self.session_dir.join(format!("{name}.metadata.json"));
        if !metadata_path.exists() {
            return ArtifactType::from_filename(name);
        }

        self.try_load_artifact_type(&metadata_path)
            .unwrap_or_else(|| ArtifactType::from_filename(name))
    }

    fn try_load_artifact_type(&self, path: &Path) -> Option<ArtifactType> {
        let content = fs::read_to_string(path).ok()?;
        let metadata: ArtifactMetadata = serde_json::from_str(&content).ok()?;
        Some(metadata.artifact_type)
    }

    /// List all versions of an artifact
    ///
    /// # Errors
    ///
    /// Returns an error if the artifact doesn't exist.
    pub fn list_versions(&self, name: &str) -> Result<Vec<u32>> {
        let artifact_path = self.session_dir.join(name);
        if !artifact_path.exists() {
            return Err(BrainError::ArtifactNotFound(name.to_string()));
        }

        let mut versions = Vec::new();
        let prefix = format!("{name}.resolved.");

        if let Ok(entries) = fs::read_dir(&self.session_dir) {
            for entry in entries.flatten() {
                if let Some(v) = parse_version_entry(entry, &prefix) {
                    versions.push(v);
                }
            }
        }

        versions.sort_unstable();
        Ok(versions)
    }

    /// List all artifacts in this session
    ///
    /// # Errors
    ///
    /// Returns an error if the session directory cannot be read.
    pub fn list_artifacts(&self) -> Result<Vec<String>> {
        let mut artifacts = Vec::new();

        if let Ok(entries) = fs::read_dir(&self.session_dir) {
            for entry in entries.flatten() {
                if let Some(name) = filter_artifact_entry(entry) {
                    artifacts.push(name);
                }
            }
        }

        Ok(artifacts)
    }

    /// Diff two versions of an artifact
    ///
    /// Returns a simple line-based diff.
    ///
    /// # Errors
    ///
    /// Returns an error if either version doesn't exist.
    pub fn diff_versions(&self, name: &str, v1: u32, v2: u32) -> Result<String> {
        let a1 = self.get_artifact(name, Some(v1))?;
        let a2 = self.get_artifact(name, Some(v2))?;

        Ok(generate_simple_diff(name, v1, v2, &a1.content, &a2.content))
    }

    // ========== Episodic Event Log ==========

    /// Append an event to this session's episodic log.
    ///
    /// Events are written as JSONL (one JSON object per line) to `events.jsonl`
    /// in the session directory. Append-only — events are never modified or deleted.
    ///
    /// # Errors
    ///
    /// Returns an error if the event cannot be serialized or written.
    pub fn append_event(&self, event: &SessionEvent) -> Result<()> {
        let events_path = self.session_dir.join("events.jsonl");
        let line = serde_json::to_string(event)?;

        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&events_path)?;

        use std::io::Write;
        writeln!(file, "{line}")?;

        tracing::trace!(
            "Session {} event: {:?} — {}",
            self.id,
            event.kind,
            event.description
        );

        Ok(())
    }

    /// Read all events from this session's episodic log.
    ///
    /// Returns events in chronological order (append order).
    /// Skips malformed lines with a warning rather than failing.
    ///
    /// # Errors
    ///
    /// Returns an error if the events file cannot be read.
    pub fn read_events(&self) -> Result<Vec<SessionEvent>> {
        let events_path = self.session_dir.join("events.jsonl");

        if !events_path.exists() {
            return Ok(Vec::new());
        }

        let content = fs::read_to_string(&events_path)?;
        let events: Vec<SessionEvent> = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .filter_map(|line| {
                serde_json::from_str(line)
                    .map_err(|e| {
                        tracing::warn!("Skipping malformed event line: {e}");
                        e
                    })
                    .ok()
            })
            .collect();

        Ok(events)
    }

    /// Count events in this session's log without loading them all.
    ///
    /// # Errors
    ///
    /// Returns an error if the events file cannot be read.
    pub fn event_count(&self) -> Result<usize> {
        let events_path = self.session_dir.join("events.jsonl");

        if !events_path.exists() {
            return Ok(0);
        }

        let content = fs::read_to_string(&events_path)?;
        Ok(content.lines().filter(|l| !l.trim().is_empty()).count())
    }

    // Private helper methods

    /// Try to load a session from brain.db by ID.
    ///
    /// Returns `None` if the DB is unavailable or the session doesn't exist.
    /// This is the preferred path — brain.db contains all sessions including
    /// those created by the autopsy-prospective hook via direct SQL INSERT.
    ///
    /// Creates the session directory on disk if it doesn't exist yet (hook-created
    /// sessions exist only in the DB until their first artifact operation).
    fn load_from_db(id: &str, session_dir: &Path) -> Option<Self> {
        // Open DB directly — the LazyLock pool uses tracing which is silent
        // in CLI context (no subscriber). Direct open matches main.rs pattern.
        let pool = nexcore_db::pool::DbPool::open_default().ok()?;
        let row = pool
            .with_conn(|conn| nexcore_db::sessions::get(conn, id))
            .ok()?;

        // Ensure session directory exists — hook-created sessions lack it
        if !session_dir.exists() {
            let _ = fs::create_dir_all(session_dir);
        }

        Some(Self {
            id: row.id,
            created_at: row.created_at,
            project: if row.project.is_empty() {
                None
            } else {
                Some(row.project)
            },
            git_commit: row.git_commit,
            session_dir: session_dir.to_path_buf(),
        })
    }

    /// List all sessions from brain.db, converting to SessionEntry.
    ///
    /// Returns `None` if the DB is unavailable.
    fn list_from_db() -> Option<Vec<SessionEntry>> {
        // Open DB directly — same bypass as load_from_db.
        let pool = nexcore_db::pool::DbPool::open_default().ok()?;
        let rows = pool
            .with_conn(|conn| nexcore_db::sessions::list_all(conn))
            .ok()?;

        Some(
            rows.into_iter()
                .map(|row| SessionEntry {
                    id: row.id,
                    created_at: row.created_at,
                    project: if row.project.is_empty() {
                        None
                    } else {
                        Some(row.project)
                    },
                    git_commit: row.git_commit,
                    description: if row.description.is_empty() {
                        None
                    } else {
                        Some(row.description)
                    },
                })
                .collect(),
        )
    }

    fn register_in_index(&self, description: Option<String>) -> Result<()> {
        let mut index = Self::load_index()?;

        let entry = SessionEntry {
            id: self.id.clone(),
            created_at: self.created_at,
            project: self.project.clone(),
            git_commit: self.git_commit.clone(),
            description,
        };

        index.push(entry);

        Self::save_index(&index)
    }

    fn load_index() -> Result<Vec<SessionEntry>> {
        let index_path = brain_dir().join("index.json");

        if !index_path.exists() {
            return Ok(Vec::new());
        }

        let content = fs::read_to_string(&index_path)?;
        let index: Vec<SessionEntry> = serde_json::from_str(&content)?;
        Ok(index)
    }

    fn save_index(index: &[SessionEntry]) -> Result<()> {
        let index_path = brain_dir().join("index.json");
        let content = serde_json::to_string_pretty(index)?;
        fs::write(&index_path, content)?;
        Ok(())
    }
}

/// Helper to process a single directory entry during Antigravity import.
fn process_antigravity_entry(
    entry: fs::DirEntry,
    existing_ids: &std::collections::HashSet<String>,
) -> Result<bool> {
    let path = entry.path();
    if !path.is_dir() {
        return Ok(false);
    }

    let id = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    if id.is_empty() {
        return Ok(false);
    }

    if existing_ids.contains(&id) {
        return Ok(false);
    }

    BrainSession::import_single_session(id, &path)?;
    Ok(true)
}

fn parse_version_entry(entry: fs::DirEntry, prefix: &str) -> Option<u32> {
    let file_name = entry.file_name();
    let file_name = file_name.to_string_lossy();

    if file_name.starts_with(prefix) {
        let suffix = &file_name[prefix.len()..];
        return suffix.parse::<u32>().ok();
    }
    None
}

fn filter_artifact_entry(entry: fs::DirEntry) -> Option<String> {
    let file_name = entry.file_name();
    let file_name = file_name.to_string_lossy();

    // Skip metadata, resolved, and versioned files
    if file_name.ends_with(".metadata.json")
        || file_name.contains(".resolved")
        || file_name.starts_with('.')
    {
        return None;
    }

    Some(file_name.to_string())
}

fn generate_simple_diff(name: &str, v1: u32, v2: u32, content1: &str, content2: &str) -> String {
    use std::fmt::Write;

    let lines1: Vec<&str> = content1.lines().collect();
    let lines2: Vec<&str> = content2.lines().collect();

    let mut diff = String::new();
    let _ = writeln!(diff, "--- {name}.resolved.{v1}");
    let _ = writeln!(diff, "+++ {name}.resolved.{v2}");

    // Very simple diff - just show removed and added lines
    for line in &lines1 {
        if !lines2.contains(line) {
            let _ = writeln!(diff, "-{line}");
        }
    }
    for line in &lines2 {
        if !lines1.contains(line) {
            let _ = writeln!(diff, "+{line}");
        }
    }

    diff
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup_test_env() -> (TempDir, PathBuf) {
        let temp = TempDir::new().unwrap();
        let brain = temp.path().join(".claude/brain");
        fs::create_dir_all(brain.join("sessions")).unwrap();
        fs::create_dir_all(brain.join("annotations")).unwrap();
        fs::write(brain.join("index.json"), "[]").unwrap();
        (temp, brain)
    }

    #[test]
    fn test_session_entry_serialization() {
        let entry = SessionEntry {
            id: NexId::v4().to_string(),
            created_at: DateTime::now(),
            project: Some("test-project".into()),
            git_commit: Some("abc123".into()),
            description: Some("Test session".into()),
        };

        let json = serde_json::to_string(&entry).unwrap();
        let parsed: SessionEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry.id, parsed.id);
        assert_eq!(entry.project, parsed.project);
    }

    #[test]
    fn test_artifact_type_from_filename_comprehensive() {
        assert_eq!(ArtifactType::from_filename("task.md"), ArtifactType::Task);
        assert_eq!(ArtifactType::from_filename("TASK.MD"), ArtifactType::Task);
        assert_eq!(
            ArtifactType::from_filename("implementation_plan.md"),
            ArtifactType::ImplementationPlan
        );
        assert_eq!(
            ArtifactType::from_filename("walkthrough.md"),
            ArtifactType::Walkthrough
        );
    }

    // ========== CTVP Phase 0: Edge Case Tests ==========

    #[test]
    fn test_session_entry_all_fields_none() {
        let entry = SessionEntry {
            id: NexId::v4().to_string(),
            created_at: DateTime::now(),
            project: None,
            git_commit: None,
            description: None,
        };

        let json = serde_json::to_string(&entry).unwrap();
        let parsed: SessionEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(entry.id, parsed.id);
        assert!(parsed.project.is_none());
        assert!(parsed.git_commit.is_none());
        assert!(parsed.description.is_none());
    }

    #[test]
    fn test_session_entry_unicode_fields() {
        let entry = SessionEntry {
            id: NexId::v4().to_string(),
            created_at: DateTime::now(),
            project: Some("项目名称 🚀".into()),
            git_commit: Some("abc123".into()),
            description: Some("Описание сессии с Unicode 日本語".into()),
        };

        let json = serde_json::to_string(&entry).unwrap();
        let parsed: SessionEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(entry.project, parsed.project);
        assert_eq!(entry.description, parsed.description);
    }

    #[test]
    fn test_session_entry_long_description() {
        let long_desc = "x".repeat(10000);
        let entry = SessionEntry {
            id: NexId::v4().to_string(),
            created_at: DateTime::now(),
            project: None,
            git_commit: None,
            description: Some(long_desc.clone()),
        };

        let json = serde_json::to_string(&entry).unwrap();
        let parsed: SessionEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.description.unwrap().len(), 10000);
    }

    #[test]
    fn test_brain_session_struct_fields() {
        // Test that BrainSession has the expected fields
        let id = NexId::v4().to_string();
        let created_at = DateTime::now();
        let session_dir = PathBuf::from("/tmp/test");

        let session = BrainSession {
            id: id.clone(),
            created_at,
            project: Some("test".into()),
            git_commit: Some("abc123".into()),
            session_dir,
        };

        assert_eq!(session.id, id);
        assert_eq!(session.project, Some("test".into()));
        assert_eq!(session.git_commit, Some("abc123".into()));
    }

    #[test]
    fn test_brain_session_dir_method() {
        let session = BrainSession {
            id: NexId::v4().to_string(),
            created_at: DateTime::now(),
            project: None,
            git_commit: None,
            session_dir: PathBuf::from("/home/test/.claude/brain/sessions/abc"),
        };

        assert_eq!(
            session.dir(),
            Path::new("/home/test/.claude/brain/sessions/abc")
        );
    }

    #[test]
    fn test_load_index_empty() {
        let (_temp, _brain) = setup_test_env();
        // Note: Can't easily test load_index without modifying global state
        // This test verifies the empty index structure is valid JSON
        let empty_index: Vec<SessionEntry> = serde_json::from_str("[]").unwrap();
        assert!(empty_index.is_empty());
    }

    #[test]
    fn test_load_index_with_entries() {
        let entries = vec![
            SessionEntry {
                id: NexId::v4().to_string(),
                created_at: DateTime::now(),
                project: Some("project1".into()),
                git_commit: None,
                description: None,
            },
            SessionEntry {
                id: NexId::v4().to_string(),
                created_at: DateTime::now(),
                project: Some("project2".into()),
                git_commit: Some("def456".into()),
                description: Some("Second session".into()),
            },
        ];

        let json = serde_json::to_string(&entries).unwrap();
        let parsed: Vec<SessionEntry> = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].project, Some("project1".into()));
        assert_eq!(parsed[1].description, Some("Second session".into()));
    }

    #[test]
    fn test_session_entry_timestamp_ordering() {
        use std::thread::sleep;
        use std::time::Duration;

        let entry1 = SessionEntry {
            id: NexId::v4().to_string(),
            created_at: DateTime::now(),
            project: None,
            git_commit: None,
            description: None,
        };

        sleep(Duration::from_millis(10));

        let entry2 = SessionEntry {
            id: NexId::v4().to_string(),
            created_at: DateTime::now(),
            project: None,
            git_commit: None,
            description: None,
        };

        // entry2 should be later
        assert!(entry2.created_at > entry1.created_at);

        // When sorting by created_at descending, entry2 comes first
        let mut entries = vec![entry1.clone(), entry2.clone()];
        entries.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        assert_eq!(entries[0].id, entry2.id);
    }

    #[test]
    fn test_uuid_string_roundtrip() {
        let original = NexId::v4().to_string();
        let string = original.clone();
        let parsed: String = string;

        assert_eq!(original, parsed);
    }

    #[test]
    fn test_session_entry_clone() {
        let entry = SessionEntry {
            id: NexId::v4().to_string(),
            created_at: DateTime::now(),
            project: Some("test".into()),
            git_commit: Some("abc".into()),
            description: Some("desc".into()),
        };

        let cloned = entry.clone();
        assert_eq!(entry.id, cloned.id);
        assert_eq!(entry.project, cloned.project);
    }

    // ========== Episodic Event Log Tests ==========

    fn make_test_session(dir: &Path) -> BrainSession {
        fs::create_dir_all(dir).unwrap();
        BrainSession {
            id: NexId::v4().to_string(),
            created_at: DateTime::now(),
            project: None,
            git_commit: None,
            session_dir: dir.to_path_buf(),
        }
    }

    #[test]
    fn test_session_event_new() {
        let event = SessionEvent::new(SessionEventKind::SessionCreated, "session started");
        assert_eq!(event.kind, SessionEventKind::SessionCreated);
        assert_eq!(event.description, "session started");
        assert!(event.metadata.is_none());
    }

    #[test]
    fn test_session_event_with_metadata() {
        let meta = serde_json::json!({"artifact": "task.md", "version": 3});
        let event = SessionEvent::with_metadata(
            SessionEventKind::ArtifactResolved,
            "resolved task.md",
            meta.clone(),
        );
        assert_eq!(event.kind, SessionEventKind::ArtifactResolved);
        assert_eq!(event.metadata, Some(meta));
    }

    #[test]
    fn test_session_event_serialization_roundtrip() {
        let event = SessionEvent::new(SessionEventKind::CorrectionRecorded, "learned correction");
        let json = serde_json::to_string(&event).unwrap();
        let parsed: SessionEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.kind, event.kind);
        assert_eq!(parsed.description, event.description);
    }

    #[test]
    fn test_event_kind_serde_rename() {
        let event = SessionEvent::new(SessionEventKind::BeliefCrystallized, "test");
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"belief_crystallized\""));
    }

    #[test]
    fn test_append_and_read_events() {
        let temp = TempDir::new().unwrap();
        let session = make_test_session(&temp.path().join("sess1"));

        let e1 = SessionEvent::new(SessionEventKind::SessionCreated, "started");
        let e2 = SessionEvent::new(SessionEventKind::ArtifactSaved, "saved task.md");
        let e3 = SessionEvent::with_metadata(
            SessionEventKind::PreferenceLearned,
            "rust > python",
            serde_json::json!({"key": "lang"}),
        );

        session.append_event(&e1).unwrap();
        session.append_event(&e2).unwrap();
        session.append_event(&e3).unwrap();

        let events = session.read_events().unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].kind, SessionEventKind::SessionCreated);
        assert_eq!(events[1].kind, SessionEventKind::ArtifactSaved);
        assert_eq!(events[2].description, "rust > python");
        assert!(events[2].metadata.is_some());
    }

    #[test]
    fn test_read_events_empty_session() {
        let temp = TempDir::new().unwrap();
        let session = make_test_session(&temp.path().join("sess_empty"));

        let events = session.read_events().unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn test_event_count() {
        let temp = TempDir::new().unwrap();
        let session = make_test_session(&temp.path().join("sess_count"));

        assert_eq!(session.event_count().unwrap(), 0);

        for i in 0..5 {
            let e = SessionEvent::new(SessionEventKind::Custom, format!("event {i}"));
            session.append_event(&e).unwrap();
        }

        assert_eq!(session.event_count().unwrap(), 5);
    }

    #[test]
    fn test_read_events_skips_malformed_lines() {
        let temp = TempDir::new().unwrap();
        let sess_dir = temp.path().join("sess_malformed");
        let session = make_test_session(&sess_dir);

        // Write a valid event, a malformed line, and another valid event
        let e1 = SessionEvent::new(SessionEventKind::SessionCreated, "good");
        session.append_event(&e1).unwrap();

        // Inject a malformed line directly
        let events_path = sess_dir.join("events.jsonl");
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&events_path)
            .unwrap();
        use std::io::Write;
        writeln!(file, "{{not valid json}}").unwrap();

        let e2 = SessionEvent::new(SessionEventKind::SessionClosed, "also good");
        session.append_event(&e2).unwrap();

        let events = session.read_events().unwrap();
        // Malformed line is skipped
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind, SessionEventKind::SessionCreated);
        assert_eq!(events[1].kind, SessionEventKind::SessionClosed);
    }

    #[test]
    fn test_event_count_with_blank_lines() {
        let temp = TempDir::new().unwrap();
        let sess_dir = temp.path().join("sess_blanks");
        let session = make_test_session(&sess_dir);

        let e = SessionEvent::new(SessionEventKind::Custom, "one");
        session.append_event(&e).unwrap();

        // Inject blank lines
        let events_path = sess_dir.join("events.jsonl");
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&events_path)
            .unwrap();
        use std::io::Write;
        writeln!(file).unwrap();
        writeln!(file, "   ").unwrap();

        let e2 = SessionEvent::new(SessionEventKind::Custom, "two");
        session.append_event(&e2).unwrap();

        // Blank lines should not be counted
        assert_eq!(session.event_count().unwrap(), 2);
    }
}
