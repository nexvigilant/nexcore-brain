//! Recovery and Graceful Degradation for the Brain System (CTVP Phase 1)
//!
//! Provides mechanisms for:
//! - Index recovery from session directories
//! - Partial write detection and repair
//! - Graceful degradation when brain is unavailable

use std::fs;
use std::path::Path;

use crate::artifact::{ArtifactMetadata, ArtifactType};
use crate::error::Result;
use crate::session::SessionEntry;
use crate::{brain_dir, config};
use nexcore_chrono::DateTime;
use nexcore_id::NexId;

/// Result of a recovery operation
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    /// Whether recovery was successful
    pub success: bool,

    /// Number of items recovered
    pub recovered_count: usize,

    /// Number of items that could not be recovered
    pub failed_count: usize,

    /// Details about what was recovered
    pub details: Vec<String>,

    /// Warnings encountered during recovery
    pub warnings: Vec<String>,
}

impl RecoveryResult {
    /// Create a successful result
    #[must_use]
    pub fn success(recovered_count: usize, details: Vec<String>) -> Self {
        Self {
            success: true,
            recovered_count,
            failed_count: 0,
            details,
            warnings: Vec::new(),
        }
    }

    /// Create a partial success result
    #[must_use]
    pub fn partial(
        recovered_count: usize,
        failed_count: usize,
        details: Vec<String>,
        warnings: Vec<String>,
    ) -> Self {
        Self {
            success: recovered_count > 0,
            recovered_count,
            failed_count,
            details,
            warnings,
        }
    }

    /// Create a failure result
    #[must_use]
    pub fn failure(reason: impl Into<String>) -> Self {
        Self {
            success: false,
            recovered_count: 0,
            failed_count: 0,
            details: Vec::new(),
            warnings: vec![reason.into()],
        }
    }
}

/// Check if the brain index is corrupted
///
/// Returns `Some(reason)` if corrupted, `None` if healthy.
#[must_use]
pub fn check_index_health() -> Option<String> {
    let index_path = brain_dir().join("index.json");

    if !index_path.exists() {
        return Some("Index file does not exist".to_string());
    }

    match fs::read_to_string(&index_path) {
        Ok(content) => validate_index_content(&content),
        Err(e) => Some(format!("Failed to read index: {}", e)),
    }
}

/// Helper to validate index file content.
fn validate_index_content(content: &str) -> Option<String> {
    if content.is_empty() {
        return Some("Index file is empty".to_string());
    }

    match serde_json::from_str::<Vec<SessionEntry>>(content) {
        Ok(entries) => validate_entries(&entries),
        Err(e) => Some(format!("Failed to parse index: {}", e)),
    }
}

/// Helper to validate individual session entries.
fn validate_entries(entries: &[SessionEntry]) -> Option<String> {
    for entry in entries {
        if entry.id.is_empty() {
            return Some(format!("Entry has empty ID: {:?}", entry));
        }
    }
    None
}

/// Rebuild the session index from session directories
///
/// This scans the sessions directory and reconstructs the index
/// from the session directories that exist.
pub fn rebuild_index_from_sessions() -> Result<RecoveryResult> {
    let sessions_dir = brain_dir().join("sessions");
    let index_path = brain_dir().join("index.json");

    if !sessions_dir.exists() {
        return Ok(RecoveryResult::failure("Sessions directory does not exist"));
    }

    let mut recovered_entries = Vec::new();
    let mut details = Vec::new();
    let mut warnings = Vec::new();
    let mut failed_count = 0;

    // Scan session directories
    if let Ok(entries) = fs::read_dir(&sessions_dir) {
        for entry in entries.flatten() {
            process_session_directory(
                entry,
                &mut recovered_entries,
                &mut details,
                &mut warnings,
                &mut failed_count,
            );
        }
    }

    // Sort by created_at descending
    recovered_entries.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    // Check max sessions limit
    truncate_to_max_sessions(&mut recovered_entries, &mut warnings);

    // Write new index
    let index_json = serde_json::to_string_pretty(&recovered_entries)?;
    fs::write(&index_path, index_json)?;

    Ok(RecoveryResult::partial(
        recovered_entries.len(),
        failed_count,
        details,
        warnings,
    ))
}

/// Helper to process a single session directory during index rebuild.
fn process_session_directory(
    entry: fs::DirEntry,
    recovered_entries: &mut Vec<SessionEntry>,
    details: &mut Vec<String>,
    warnings: &mut Vec<String>,
    failed_count: &mut usize,
) {
    let path = entry.path();
    if !path.is_dir() {
        return;
    }

    let dir_name = match path.file_name().and_then(|n| n.to_str()) {
        Some(name) => name,
        None => return,
    };

    // Try to parse directory name as UUID
    match dir_name.parse::<NexId>() {
        Ok(id) => {
            let session_entry = create_recovered_entry(id, &path);
            recovered_entries.push(session_entry);
            details.push(format!("Recovered session: {}", id));
        }
        Err(_) => {
            warnings.push(format!("Skipping non-UUID directory: {}", dir_name));
            *failed_count += 1;
        }
    }
}

/// Helper to create a SessionEntry for a recovered directory.
fn create_recovered_entry(id: NexId, path: &Path) -> SessionEntry {
    // Try to determine when session was created (use dir mtime)
    let created_at = fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .map(DateTime::from)
        .unwrap_or_else(|| DateTime::now());

    // Try to infer project name from artifacts
    let project = infer_project_from_artifacts(path);

    SessionEntry {
        id: id.to_string(),
        created_at,
        project,
        git_commit: None,
        description: Some("Recovered session".to_string()),
    }
}

/// Helper to truncate recovered entries to max sessions limit.
fn truncate_to_max_sessions(recovered_entries: &mut Vec<SessionEntry>, warnings: &mut Vec<String>) {
    let config = config::get_config();
    if recovered_entries.len() > config.max_sessions {
        let original_count = recovered_entries.len();
        recovered_entries.truncate(config.max_sessions);
        warnings.push(format!(
            "Truncated {} sessions to max limit of {}",
            original_count, config.max_sessions
        ));
    }
}

/// Infer project name from artifacts in a session directory
fn infer_project_from_artifacts(session_dir: &Path) -> Option<String> {
    // Look for common artifact files that might contain project info
    for artifact_name in ["task.md", "plan.md", "context.md"] {
        let artifact_path = session_dir.join(artifact_name);
        if let Some(project) = try_infer_from_file(&artifact_path) {
            return Some(project);
        }
    }
    None
}

/// Helper to try and infer project name from a single artifact file.
fn try_infer_from_file(path: &Path) -> Option<String> {
    if !path.exists() {
        return None;
    }

    let content = fs::read_to_string(path).ok()?;
    for line in content.lines().take(10) {
        if let Some(project) = extract_project_from_line(line) {
            return Some(project);
        }
    }
    None
}

/// Helper to extract project name from a single line of an artifact.
fn extract_project_from_line(line: &str) -> Option<String> {
    if line.starts_with("# ") {
        let title = line.trim_start_matches("# ").trim();
        if !title.is_empty() && title.len() < 100 {
            return Some(title.to_string());
        }
    }
    if line.to_lowercase().starts_with("project:") {
        let project = line.split(':').nth(1)?.trim();
        if !project.is_empty() {
            return Some(project.to_string());
        }
    }
    None
}

/// Detect partial writes (artifact without metadata or vice versa)
pub fn detect_partial_writes(session_dir: &Path) -> Vec<String> {
    let mut partial_writes = Vec::new();

    if let Ok(entries) = fs::read_dir(session_dir) {
        for entry in entries.flatten() {
            if let Some(name) = get_partial_write_name(session_dir, entry) {
                partial_writes.push(name);
            }
        }
    }

    partial_writes
}

/// Helper to check if a directory entry is a partial write.
fn get_partial_write_name(session_dir: &Path, entry: fs::DirEntry) -> Option<String> {
    let path = entry.path();
    let name = path.file_name()?.to_str()?.to_string();

    if is_ignored_file(&name) {
        return None;
    }

    let metadata_path = session_dir.join(format!("{}.metadata.json", name));
    if !metadata_path.exists() {
        return Some(name);
    }
    None
}

/// Helper to determine if a file should be ignored during partial write detection.
fn is_ignored_file(name: &str) -> bool {
    name.ends_with(".metadata.json") || name.contains(".resolved") || name.starts_with('.')
}

/// Repair partial writes by creating missing metadata
pub fn repair_partial_writes(session_dir: &Path) -> Result<RecoveryResult> {
    let partial_writes = detect_partial_writes(session_dir);

    if partial_writes.is_empty() {
        return Ok(RecoveryResult::success(
            0,
            vec!["No partial writes detected".to_string()],
        ));
    }

    let mut repaired = Vec::new();
    let mut failed = Vec::new();

    for artifact_name in partial_writes {
        match repair_single_artifact(session_dir, &artifact_name) {
            Ok(msg) => repaired.push(msg),
            Err(e) => failed.push(format!("{}: {}", artifact_name, e)),
        }
    }

    Ok(RecoveryResult::partial(
        repaired.len(),
        failed.len(),
        repaired,
        failed,
    ))
}

/// Helper to repair a single artifact by creating its metadata.
fn repair_single_artifact(session_dir: &Path, name: &str) -> Result<String> {
    let artifact_path = session_dir.join(name);
    let metadata_path = session_dir.join(format!("{}.metadata.json", name));

    let content = fs::read_to_string(&artifact_path)?;
    let artifact_type = ArtifactType::from_filename(name);
    let summary = extract_summary(&content, artifact_type);

    let metadata = ArtifactMetadata::new(artifact_type, summary);
    let metadata_json = serde_json::to_string_pretty(&metadata)?;

    fs::write(&metadata_path, metadata_json)?;
    Ok(format!("Created metadata for: {}", name))
}

/// Helper to extract a summary from artifact content.
fn extract_summary(content: &str, artifact_type: ArtifactType) -> String {
    content
        .lines()
        .find(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            if l.len() > 100 {
                format!("{}...", &l[..100])
            } else {
                l.to_string()
            }
        })
        .unwrap_or_else(|| format!("{} artifact", artifact_type))
}

/// Check if brain system is available and functional
///
/// This performs a quick health check and returns a reason if unavailable.
#[must_use]
pub fn check_brain_availability() -> Option<String> {
    let brain = brain_dir();

    // Check if brain directory exists
    if !brain.exists() {
        return Some("Brain directory does not exist".to_string());
    }

    // Check if sessions directory exists
    let sessions = brain.join("sessions");
    if !sessions.exists() {
        return Some("Sessions directory does not exist".to_string());
    }

    // Check if we can write to the brain directory
    let test_file = brain.join(".write_test");
    match fs::write(&test_file, "test") {
        Ok(()) => {
            let _ = fs::remove_file(&test_file);
        }
        Err(e) => {
            return Some(format!("Cannot write to brain directory: {}", e));
        }
    }

    // Check index health
    if let Some(reason) = check_index_health() {
        return Some(format!("Index unhealthy: {}", reason));
    }

    None // Brain is available
}

/// Attempt recovery if brain is in degraded state
///
/// Returns `Ok(())` if recovery succeeded or wasn't needed.
pub fn attempt_recovery() -> Result<RecoveryResult> {
    let config = config::get_config();

    if !config.auto_recovery {
        return Ok(RecoveryResult::failure("Auto-recovery is disabled"));
    }

    // Check if recovery is needed
    if check_brain_availability().is_none() {
        return Ok(RecoveryResult::success(
            0,
            vec!["Brain is healthy, no recovery needed".to_string()],
        ));
    }

    // Check if index needs rebuilding
    if check_index_health().is_some() {
        tracing::warn!("Index is corrupted, attempting to rebuild from sessions");
        return rebuild_index_from_sessions();
    }

    Ok(RecoveryResult::success(
        0,
        vec!["No recovery action taken".to_string()],
    ))
}

/// Wrapper for operations that should degrade gracefully
///
/// If the operation fails with a recoverable error and graceful degradation
/// is enabled, logs a warning and returns `Ok(default)` instead of `Err`.
pub fn with_graceful_degradation<T, F>(operation: F, default: T, context: &str) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    match operation() {
        Ok(value) => Ok(value),
        Err(e) if e.should_degrade_gracefully() => {
            let config = config::get_config();
            if config.graceful_degradation {
                tracing::warn!("{}: {}. Continuing with default.", context, e);
                Ok(default)
            } else {
                Err(e)
            }
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::BrainError;
    use std::fs;
    use tempfile::TempDir;

    #[allow(dead_code)]
    fn setup_test_brain() -> (TempDir, std::path::PathBuf) {
        let temp = TempDir::new().unwrap();
        let brain = temp.path().join(".claude/brain");
        fs::create_dir_all(brain.join("sessions")).unwrap();
        fs::create_dir_all(brain.join("annotations")).unwrap();
        fs::write(brain.join("index.json"), "[]").unwrap();
        (temp, brain)
    }

    #[test]
    fn test_recovery_result_success() {
        let result = RecoveryResult::success(5, vec!["item1".into(), "item2".into()]);
        assert!(result.success);
        assert_eq!(result.recovered_count, 5);
        assert_eq!(result.failed_count, 0);
    }

    #[test]
    fn test_recovery_result_partial() {
        let result =
            RecoveryResult::partial(3, 2, vec!["recovered".into()], vec!["warning".into()]);
        assert!(result.success);
        assert_eq!(result.recovered_count, 3);
        assert_eq!(result.failed_count, 2);
    }

    #[test]
    fn test_recovery_result_failure() {
        let result = RecoveryResult::failure("test failure");
        assert!(!result.success);
        assert_eq!(result.warnings[0], "test failure");
    }

    #[test]
    fn test_detect_partial_writes_empty_dir() {
        let temp = TempDir::new().unwrap();
        let session_dir = temp.path().join("session");
        fs::create_dir_all(&session_dir).unwrap();

        let partial = detect_partial_writes(&session_dir);
        assert!(partial.is_empty());
    }

    #[test]
    fn test_detect_partial_writes_with_missing_metadata() {
        let temp = TempDir::new().unwrap();
        let session_dir = temp.path().join("session");
        fs::create_dir_all(&session_dir).unwrap();

        // Create artifact without metadata
        fs::write(session_dir.join("task.md"), "# Task").unwrap();

        let partial = detect_partial_writes(&session_dir);
        assert_eq!(partial, vec!["task.md"]);
    }

    #[test]
    fn test_detect_partial_writes_with_metadata() {
        let temp = TempDir::new().unwrap();
        let session_dir = temp.path().join("session");
        fs::create_dir_all(&session_dir).unwrap();

        // Create artifact with metadata
        fs::write(session_dir.join("task.md"), "# Task").unwrap();
        fs::write(session_dir.join("task.md.metadata.json"), "{}").unwrap();

        let partial = detect_partial_writes(&session_dir);
        assert!(partial.is_empty());
    }

    #[test]
    fn test_repair_partial_writes() {
        let temp = TempDir::new().unwrap();
        let session_dir = temp.path().join("session");
        fs::create_dir_all(&session_dir).unwrap();

        // Create artifact without metadata
        fs::write(
            session_dir.join("task.md"),
            "# My Task\n\nThis is the content.",
        )
        .unwrap();

        let result = repair_partial_writes(&session_dir).unwrap();
        assert!(result.success);
        assert_eq!(result.recovered_count, 1);

        // Check metadata was created
        assert!(session_dir.join("task.md.metadata.json").exists());
    }

    #[test]
    fn test_infer_project_from_artifacts() {
        let temp = TempDir::new().unwrap();
        let session_dir = temp.path();

        // Create task.md with project header
        fs::write(session_dir.join("task.md"), "# MyProject\n\nContent here").unwrap();

        let project = infer_project_from_artifacts(session_dir);
        assert_eq!(project, Some("MyProject".to_string()));
    }

    #[test]
    fn test_infer_project_with_project_line() {
        let temp = TempDir::new().unwrap();
        let session_dir = temp.path();

        // Create task.md with Project: line
        fs::write(session_dir.join("task.md"), "Project: NexCore\n\nContent").unwrap();

        let project = infer_project_from_artifacts(session_dir);
        assert_eq!(project, Some("NexCore".to_string()));
    }

    #[test]
    fn test_with_graceful_degradation_success() {
        config::reset_config();

        let result: Result<i32> = with_graceful_degradation(|| Ok(42), 0, "test");
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_with_graceful_degradation_recoverable_error() {
        config::reset_config();
        config::set_config(config::BrainConfig::new().with_graceful_degradation(true));

        let result: Result<i32> =
            with_graceful_degradation(|| Err(BrainError::unavailable("test")), 42, "test");
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_with_graceful_degradation_disabled() {
        config::reset_config();
        config::set_config(config::BrainConfig::new().with_graceful_degradation(false));

        let result: Result<i32> =
            with_graceful_degradation(|| Err(BrainError::unavailable("test")), 42, "test");
        assert!(result.is_err());
    }
}
