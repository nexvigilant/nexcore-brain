//! Brain Health Metrics Collection
//!
//! Provides comprehensive health monitoring for the Brain system including:
//! - Artifact counts and sizes
//! - Session tracking (active, total)
//! - Implicit learning entry counts
//! - Code tracker file counts
//! - Growth rate analysis
//! - Telemetry snapshots for tracking growth over time

use nexcore_chrono::{DateTime, Duration};
use nexcore_constants::bathroom_lock::BathroomLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use crate::error::{BrainError, Result};
use crate::{ArtifactType, BrainSession, brain_dir, implicit_dir, tracker_dir};

// ============================================================================
// Telemetry Directory Constants
// ============================================================================

/// Telemetry directory name within brain
pub const TELEMETRY_DIR: &str = "telemetry";

/// Snapshots file name (JSONL format)
pub const SNAPSHOTS_FILE: &str = "brain_snapshots.jsonl";

// ============================================================================
// Metrics Types (T2-C Composites)
// ============================================================================

/// Artifact metrics breakdown by type
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArtifactMetrics {
    /// Total number of artifacts across all sessions
    pub total: usize,
    /// Breakdown by artifact type
    pub by_type: HashMap<String, usize>,
}

/// Session metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionMetrics {
    /// Number of active sessions (created in last 24 hours)
    pub active: usize,
    /// Total number of sessions
    pub total: usize,
}

/// Brain health snapshot for telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainSnapshot {
    /// ISO-8601 timestamp
    pub timestamp: DateTime,
    /// Artifact metrics
    pub artifacts: ArtifactMetrics,
    /// Total bytes across all artifacts
    pub total_bytes: u64,
    /// Session metrics
    pub sessions: SessionMetrics,
    /// Number of implicit knowledge entries (preferences + patterns + corrections + beliefs)
    pub implicit_entries: usize,
    /// Number of tracked files in code tracker
    pub tracked_files: usize,
}

impl BrainSnapshot {
    /// Create a new snapshot with current metrics
    ///
    /// # Errors
    ///
    /// Returns an error if metrics cannot be collected.
    pub fn collect() -> Result<Self> {
        let health = BrainHealth::collect()?;

        Ok(Self {
            timestamp: DateTime::now(),
            artifacts: health.artifacts,
            total_bytes: health.total_bytes,
            sessions: health.sessions,
            implicit_entries: health.implicit_entries,
            tracked_files: health.tracked_files,
        })
    }
}

/// Artifact size info for largest artifacts query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactSizeInfo {
    /// Session ID containing the artifact
    pub session_id: String,
    /// Artifact name
    pub name: String,
    /// Artifact type
    pub artifact_type: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// Last modified time
    pub modified_at: DateTime,
}

/// Comprehensive brain health report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainHealth {
    /// Collection timestamp
    pub collected_at: DateTime,
    /// Artifact metrics
    pub artifacts: ArtifactMetrics,
    /// Total bytes across all artifacts
    pub total_bytes: u64,
    /// Session metrics
    pub sessions: SessionMetrics,
    /// Number of implicit knowledge entries
    pub implicit_entries: usize,
    /// Number of tracked files
    pub tracked_files: usize,
    /// Brain directory path
    pub brain_dir: PathBuf,
    /// Telemetry directory path
    pub telemetry_dir: PathBuf,
    /// Health status ("healthy", "warning", "degraded")
    pub status: String,
    /// Any warnings or issues detected
    pub warnings: Vec<String>,
}

impl BrainHealth {
    /// Collect comprehensive brain health metrics
    ///
    /// # Errors
    ///
    /// Returns an error if directories cannot be read.
    pub fn collect() -> Result<Self> {
        let brain = brain_dir();
        let telemetry = brain.join(TELEMETRY_DIR);
        let mut warnings = Vec::new();

        // Ensure telemetry directory exists
        if !telemetry.exists() {
            fs::create_dir_all(&telemetry)?;
        }

        // Collect session metrics
        let sessions = collect_session_metrics(&mut warnings);

        // Collect artifact metrics across all sessions
        let (artifacts, total_bytes) = collect_artifact_metrics(&brain, &mut warnings);

        // Collect implicit knowledge metrics
        let implicit_entries = collect_implicit_metrics(&mut warnings);

        // Collect code tracker metrics
        let tracked_files = collect_tracker_metrics(&mut warnings);

        // Determine health status
        let status = determine_health_status(&sessions, &artifacts, &warnings);

        Ok(Self {
            collected_at: DateTime::now(),
            artifacts,
            total_bytes,
            sessions,
            implicit_entries,
            tracked_files,
            brain_dir: brain,
            telemetry_dir: telemetry,
            status,
            warnings,
        })
    }
}

/// Growth rate metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrowthRate {
    /// Analysis period start
    pub period_start: DateTime,
    /// Analysis period end
    pub period_end: DateTime,
    /// Number of days analyzed
    pub days_analyzed: u32,
    /// Artifacts created per day (average)
    pub artifacts_per_day: f64,
    /// Sessions created per day (average)
    pub sessions_per_day: f64,
    /// Total artifacts in period
    pub total_artifacts_created: usize,
    /// Total sessions in period
    pub total_sessions_created: usize,
    /// Bytes added per day (average)
    pub bytes_per_day: f64,
}

impl GrowthRate {
    /// Calculate growth rate from snapshots
    ///
    /// # Errors
    ///
    /// Returns an error if snapshots cannot be read.
    pub fn calculate(days: u32) -> Result<Self> {
        let snapshots = load_snapshots()?;
        let now = DateTime::now();
        let period_start = now - Duration::days(i64::from(days));

        // Filter snapshots within period
        let relevant: Vec<_> = snapshots
            .iter()
            .filter(|s| s.timestamp >= period_start)
            .collect();

        if relevant.len() < 2 {
            // Not enough data, calculate from current state
            let health = BrainHealth::collect()?;
            let actual_days = if days == 0 { 1 } else { days };

            return Ok(Self {
                period_start,
                period_end: now,
                days_analyzed: actual_days,
                artifacts_per_day: health.artifacts.total as f64 / f64::from(actual_days),
                sessions_per_day: health.sessions.total as f64 / f64::from(actual_days),
                total_artifacts_created: health.artifacts.total,
                total_sessions_created: health.sessions.total,
                bytes_per_day: health.total_bytes as f64 / f64::from(actual_days),
            });
        }

        // Calculate deltas between first and last snapshot in period
        let first = relevant
            .first()
            .ok_or_else(|| BrainError::Other("No snapshots in period".into()))?;
        let last = relevant
            .last()
            .ok_or_else(|| BrainError::Other("No snapshots in period".into()))?;

        let artifact_delta = last.artifacts.total.saturating_sub(first.artifacts.total);
        let session_delta = last.sessions.total.saturating_sub(first.sessions.total);
        let bytes_delta = last.total_bytes.saturating_sub(first.total_bytes);

        let actual_days = ((last.timestamp - first.timestamp).num_hours() as f64 / 24.0).max(1.0);

        Ok(Self {
            period_start: first.timestamp,
            period_end: last.timestamp,
            days_analyzed: actual_days as u32,
            artifacts_per_day: artifact_delta as f64 / actual_days,
            sessions_per_day: session_delta as f64 / actual_days,
            total_artifacts_created: artifact_delta,
            total_sessions_created: session_delta,
            bytes_per_day: bytes_delta as f64 / actual_days,
        })
    }
}

// ============================================================================
// Telemetry Functions
// ============================================================================

/// Get the telemetry directory path
#[must_use]
pub fn telemetry_dir() -> PathBuf {
    brain_dir().join(TELEMETRY_DIR)
}

/// Ensure telemetry directory exists
///
/// # Errors
///
/// Returns an error if the directory cannot be created.
pub fn ensure_telemetry_dir() -> Result<PathBuf> {
    let dir = telemetry_dir();
    if !dir.exists() {
        fs::create_dir_all(&dir)?;
    }
    Ok(dir)
}

/// Save a snapshot to the telemetry JSONL file
///
/// # Errors
///
/// Returns an error if the snapshot cannot be written.
pub fn save_snapshot(snapshot: &BrainSnapshot) -> Result<()> {
    let telemetry = ensure_telemetry_dir()?;
    let snapshots_path = telemetry.join(SNAPSHOTS_FILE);

    // Bathroom lock — multiple sessions can snapshot concurrently
    let lock = BathroomLock::for_brain_snapshots();
    let _guard = lock.try_acquire("brain-metrics").ok();

    let json_line = serde_json::to_string(snapshot)?;

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&snapshots_path)?;

    writeln!(file, "{json_line}")?;

    Ok(())
}

/// Load all snapshots from the telemetry file
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn load_snapshots() -> Result<Vec<BrainSnapshot>> {
    let telemetry = telemetry_dir();
    let snapshots_path = telemetry.join(SNAPSHOTS_FILE);

    if !snapshots_path.exists() {
        return Ok(Vec::new());
    }

    let content = fs::read_to_string(&snapshots_path)?;
    let mut snapshots = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(snapshot) = serde_json::from_str::<BrainSnapshot>(line) {
            snapshots.push(snapshot);
        }
    }

    Ok(snapshots)
}

/// Get the N largest artifacts across all sessions
///
/// # Errors
///
/// Returns an error if sessions cannot be enumerated.
pub fn largest_artifacts(n: usize) -> Result<Vec<ArtifactSizeInfo>> {
    let brain = brain_dir();
    let sessions_dir = brain.join("sessions");

    if !sessions_dir.exists() {
        return Ok(Vec::new());
    }

    let mut all_artifacts: Vec<ArtifactSizeInfo> = Vec::new();

    // Iterate through all session directories
    if let Ok(entries) = fs::read_dir(&sessions_dir) {
        for entry in entries.flatten() {
            let session_path = entry.path();
            if !session_path.is_dir() {
                continue;
            }

            let session_id = session_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            // Read artifacts in this session
            if let Ok(session_entries) = fs::read_dir(&session_path) {
                for artifact_entry in session_entries.flatten() {
                    let artifact_path = artifact_entry.path();

                    // Skip metadata, resolved, and hidden files
                    let name = artifact_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("");

                    if name.ends_with(".metadata.json")
                        || name.contains(".resolved")
                        || name.starts_with('.')
                    {
                        continue;
                    }

                    if let Ok(metadata) = fs::metadata(&artifact_path) {
                        if metadata.is_file() {
                            let modified_at = metadata
                                .modified()
                                .map(DateTime::from)
                                .unwrap_or_else(|_| DateTime::now());

                            let artifact_type = ArtifactType::from_filename(name);

                            all_artifacts.push(ArtifactSizeInfo {
                                session_id: session_id.clone(),
                                name: name.to_string(),
                                artifact_type: artifact_type.to_string(),
                                size_bytes: metadata.len(),
                                modified_at,
                            });
                        }
                    }
                }
            }
        }
    }

    // Sort by size descending
    all_artifacts.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));

    // Return top N
    all_artifacts.truncate(n);

    Ok(all_artifacts)
}

// ============================================================================
// Internal Helpers
// ============================================================================

fn collect_session_metrics(warnings: &mut Vec<String>) -> SessionMetrics {
    let sessions = match BrainSession::list_all() {
        Ok(s) => s,
        Err(e) => {
            warnings.push(format!("Failed to list sessions: {e}"));
            return SessionMetrics::default();
        }
    };

    let now = DateTime::now();
    let active_cutoff = now - Duration::hours(24);

    let active = sessions
        .iter()
        .filter(|s| s.created_at >= active_cutoff)
        .count();

    SessionMetrics {
        active,
        total: sessions.len(),
    }
}

fn collect_artifact_metrics(
    brain: &std::path::Path,
    warnings: &mut Vec<String>,
) -> (ArtifactMetrics, u64) {
    let sessions_dir = brain.join("sessions");
    let mut total = 0usize;
    let mut total_bytes = 0u64;
    let mut by_type: HashMap<String, usize> = HashMap::new();

    if !sessions_dir.exists() {
        return (ArtifactMetrics::default(), 0);
    }

    let entries = match fs::read_dir(&sessions_dir) {
        Ok(e) => e,
        Err(e) => {
            warnings.push(format!("Failed to read sessions directory: {e}"));
            return (ArtifactMetrics::default(), 0);
        }
    };

    for entry in entries.flatten() {
        let session_path = entry.path();
        if !session_path.is_dir() {
            continue;
        }

        if let Ok(session_entries) = fs::read_dir(&session_path) {
            for artifact_entry in session_entries.flatten() {
                let artifact_path = artifact_entry.path();
                let name = artifact_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");

                // Skip metadata, resolved, and hidden files
                if name.ends_with(".metadata.json")
                    || name.contains(".resolved")
                    || name.starts_with('.')
                {
                    continue;
                }

                if let Ok(metadata) = fs::metadata(&artifact_path) {
                    if metadata.is_file() {
                        total += 1;
                        total_bytes += metadata.len();

                        let artifact_type = ArtifactType::from_filename(name);
                        *by_type.entry(artifact_type.to_string()).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    (ArtifactMetrics { total, by_type }, total_bytes)
}

fn collect_implicit_metrics(warnings: &mut Vec<String>) -> usize {
    let implicit = implicit_dir();
    let mut count = 0usize;

    // Count preferences
    let prefs_path = implicit.join("preferences.json");
    if prefs_path.exists() {
        if let Ok(content) = fs::read_to_string(&prefs_path) {
            if let Ok(prefs) = serde_json::from_str::<HashMap<String, serde_json::Value>>(&content)
            {
                count += prefs.len();
            }
        }
    }

    // Count patterns
    let patterns_path = implicit.join("patterns.json");
    if patterns_path.exists() {
        if let Ok(content) = fs::read_to_string(&patterns_path) {
            if let Ok(patterns) = serde_json::from_str::<Vec<serde_json::Value>>(&content) {
                count += patterns.len();
            }
        }
    }

    // Count corrections
    let corrections_path = implicit.join("corrections.json");
    if corrections_path.exists() {
        if let Ok(content) = fs::read_to_string(&corrections_path) {
            if let Ok(corrections) = serde_json::from_str::<Vec<serde_json::Value>>(&content) {
                count += corrections.len();
            }
        }
    }

    // Count beliefs (PROJECT GROUNDED)
    let beliefs_path = implicit.join("beliefs.json");
    if beliefs_path.exists() {
        if let Ok(content) = fs::read_to_string(&beliefs_path) {
            if let Ok(beliefs) = serde_json::from_str::<Vec<serde_json::Value>>(&content) {
                count += beliefs.len();
            } else {
                warnings.push("Failed to parse beliefs.json".to_string());
            }
        }
    }

    count
}

fn collect_tracker_metrics(warnings: &mut Vec<String>) -> usize {
    let tracker = tracker_dir();
    let index_path = tracker.join("index.json");

    if !index_path.exists() {
        return 0;
    }

    let content = match fs::read_to_string(&index_path) {
        Ok(c) => c,
        Err(e) => {
            warnings.push(format!("Failed to read tracker index: {e}"));
            return 0;
        }
    };

    // Parse the tracker index to count files
    #[derive(Deserialize)]
    struct TrackerIndex {
        projects: HashMap<String, ProjectSnapshot>,
    }

    #[derive(Deserialize)]
    struct ProjectSnapshot {
        files: HashMap<String, serde_json::Value>,
    }

    match serde_json::from_str::<TrackerIndex>(&content) {
        Ok(index) => index.projects.values().map(|p| p.files.len()).sum(),
        Err(_) => {
            warnings.push("Failed to parse tracker index".to_string());
            0
        }
    }
}

fn determine_health_status(
    sessions: &SessionMetrics,
    artifacts: &ArtifactMetrics,
    warnings: &[String],
) -> String {
    if !warnings.is_empty() {
        return "warning".to_string();
    }

    if sessions.total == 0 && artifacts.total == 0 {
        return "empty".to_string();
    }

    "healthy".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_artifact_metrics_default() {
        let metrics = ArtifactMetrics::default();
        assert_eq!(metrics.total, 0);
        assert!(metrics.by_type.is_empty());
    }

    #[test]
    fn test_session_metrics_default() {
        let metrics = SessionMetrics::default();
        assert_eq!(metrics.active, 0);
        assert_eq!(metrics.total, 0);
    }

    #[test]
    fn test_brain_snapshot_serialization() {
        let snapshot = BrainSnapshot {
            timestamp: DateTime::now(),
            artifacts: ArtifactMetrics::default(),
            total_bytes: 1234,
            sessions: SessionMetrics {
                active: 1,
                total: 5,
            },
            implicit_entries: 10,
            tracked_files: 3,
        };

        let json = serde_json::to_string(&snapshot).unwrap();
        let parsed: BrainSnapshot = serde_json::from_str(&json).unwrap();

        assert_eq!(snapshot.total_bytes, parsed.total_bytes);
        assert_eq!(snapshot.sessions.total, parsed.sessions.total);
        assert_eq!(snapshot.implicit_entries, parsed.implicit_entries);
    }

    #[test]
    fn test_artifact_size_info_serialization() {
        let info = ArtifactSizeInfo {
            session_id: "test-session".to_string(),
            name: "task.md".to_string(),
            artifact_type: "task".to_string(),
            size_bytes: 2048,
            modified_at: DateTime::now(),
        };

        let json = serde_json::to_string(&info).unwrap();
        let parsed: ArtifactSizeInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(info.session_id, parsed.session_id);
        assert_eq!(info.size_bytes, parsed.size_bytes);
    }

    #[test]
    fn test_growth_rate_serialization() {
        let rate = GrowthRate {
            period_start: DateTime::now() - Duration::days(7),
            period_end: DateTime::now(),
            days_analyzed: 7,
            artifacts_per_day: 2.5,
            sessions_per_day: 1.0,
            total_artifacts_created: 17,
            total_sessions_created: 7,
            bytes_per_day: 1024.0,
        };

        let json = serde_json::to_string(&rate).unwrap();
        let parsed: GrowthRate = serde_json::from_str(&json).unwrap();

        assert_eq!(rate.days_analyzed, parsed.days_analyzed);
        assert!((rate.artifacts_per_day - parsed.artifacts_per_day).abs() < f64::EPSILON);
    }

    #[test]
    fn test_telemetry_dir_path() {
        let dir = telemetry_dir();
        assert!(dir.to_string_lossy().contains("telemetry"));
    }

    #[test]
    fn test_health_status_determination() {
        let sessions_empty = SessionMetrics {
            active: 0,
            total: 0,
        };
        let artifacts_empty = ArtifactMetrics::default();

        let status = determine_health_status(&sessions_empty, &artifacts_empty, &[]);
        assert_eq!(status, "empty");

        let sessions_active = SessionMetrics {
            active: 1,
            total: 5,
        };
        let artifacts_active = ArtifactMetrics {
            total: 10,
            by_type: HashMap::new(),
        };

        let status = determine_health_status(&sessions_active, &artifacts_active, &[]);
        assert_eq!(status, "healthy");

        let warnings = vec!["test warning".to_string()];
        let status = determine_health_status(&sessions_active, &artifacts_active, &warnings);
        assert_eq!(status, "warning");
    }
}
