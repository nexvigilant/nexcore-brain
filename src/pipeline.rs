//! Pipeline state tracking with checkpoints
//!
//! Provides run-level state management with checkpoint markers.
//! Integrates with BrainSession for persistence.

use nexcore_chrono::DateTime;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use crate::error::{BrainError, Result};

/// A pipeline run with checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineRun {
    /// Unique identifier for this run
    pub id: String,
    /// When the run started
    pub started: DateTime,
    /// Current status of the run
    pub status: RunStatus,
    /// Checkpoints recorded during the run
    pub checkpoints: Vec<Checkpoint>,
    /// When the run completed (if finished)
    pub completed: Option<DateTime>,
    /// EJC markers from spliceosome for NMD surveillance
    #[serde(default)]
    pub ejc_markers: Vec<serde_json::Value>,
}

/// Run status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    /// Run is currently executing
    Running,
    /// Run completed successfully
    Completed,
    /// Run failed with an error
    Failed,
    /// Run was cancelled by user
    Cancelled,
    /// Run was aborted by NMD surveillance
    Aborted,
}

/// A checkpoint within a run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Name of the checkpoint
    pub name: String,
    /// When the checkpoint was recorded
    pub timestamp: DateTime,
    /// Arbitrary data associated with the checkpoint
    pub data: serde_json::Value,
}

/// Pipeline state manager
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PipelineState {
    /// Name of the pipeline
    pub name: String,
    /// All runs for this pipeline
    pub runs: Vec<PipelineRun>,
    /// Most recent checkpoint across all runs
    pub last_checkpoint: Option<Checkpoint>,
}

impl PipelineState {
    /// Create new pipeline state
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            runs: Vec::new(),
            last_checkpoint: None,
        }
    }

    /// Load from JSON file
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }

    /// Save to JSON file
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Start a new run
    pub fn start_run(&mut self, run_id: Option<String>) -> &PipelineRun {
        let id =
            run_id.unwrap_or_else(|| DateTime::now().format("%Y%m%d_%H%M%S").unwrap_or_default());
        let run = PipelineRun {
            id,
            started: DateTime::now(),
            status: RunStatus::Running,
            checkpoints: Vec::new(),
            completed: None,
            ejc_markers: Vec::new(),
        };
        self.runs.push(run);
        // SAFETY: just pushed — index is guaranteed valid
        &self.runs[self.runs.len() - 1]
    }

    /// Add checkpoint to current run
    pub fn checkpoint(&mut self, name: impl Into<String>, data: serde_json::Value) -> Result<()> {
        let cp = Checkpoint {
            name: name.into(),
            timestamp: DateTime::now(),
            data,
        };
        self.last_checkpoint = Some(cp.clone());
        if let Some(run) = self.runs.last_mut() {
            run.checkpoints.push(cp);
            Ok(())
        } else {
            Err(BrainError::ArtifactNotFound("No active run".into()))
        }
    }

    /// Complete current run
    pub fn complete_run(&mut self, status: RunStatus) -> Result<()> {
        if let Some(run) = self.runs.last_mut() {
            run.status = status;
            run.completed = Some(DateTime::now());
            Ok(())
        } else {
            Err(BrainError::ArtifactNotFound("No active run".into()))
        }
    }

    /// Attach EJC markers from spliceosome to current run.
    pub fn attach_ejc_markers(&mut self, markers: Vec<serde_json::Value>) -> Result<()> {
        if let Some(run) = self.runs.last_mut() {
            run.ejc_markers = markers;
            Ok(())
        } else {
            Err(BrainError::ArtifactNotFound("No active run".into()))
        }
    }

    /// Abort current run (NMD degradation pathway).
    pub fn abort_run(&mut self, reason: &str) -> Result<()> {
        self.checkpoint("nmd_abort", serde_json::json!({"reason": reason}))?;
        if let Some(run) = self.runs.last_mut() {
            run.status = RunStatus::Aborted;
            run.completed = Some(DateTime::now());
            Ok(())
        } else {
            Err(BrainError::ArtifactNotFound("No active run".into()))
        }
    }

    /// Get current run
    pub fn current_run(&self) -> Option<&PipelineRun> {
        self.runs.last()
    }

    /// Summary for display
    pub fn summary(&self) -> String {
        let total = self.runs.len();
        let last = self
            .current_run()
            .map(|r| format!("{} ({})", r.id, r.status_str()))
            .unwrap_or_default();
        let cps = self
            .last_checkpoint
            .as_ref()
            .map(|c| c.name.clone())
            .unwrap_or_default();
        format!(
            "Pipeline: {} | Runs: {} | Last: {} | Checkpoint: {}",
            self.name, total, last, cps
        )
    }
}

impl PipelineRun {
    fn status_str(&self) -> &'static str {
        match self.status {
            RunStatus::Running => "running",
            RunStatus::Completed => "completed",
            RunStatus::Failed => "failed",
            RunStatus::Cancelled => "cancelled",
            RunStatus::Aborted => "aborted",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_lifecycle() {
        let mut state = PipelineState::new("test-pipeline");
        state.start_run(None);
        state
            .checkpoint("step1", serde_json::json!({"progress": 50}))
            .unwrap();
        state
            .checkpoint("step2", serde_json::json!({"progress": 100}))
            .unwrap();
        state.complete_run(RunStatus::Completed).unwrap();

        assert_eq!(state.runs.len(), 1);
        assert_eq!(state.runs[0].checkpoints.len(), 2);
        assert_eq!(state.runs[0].status, RunStatus::Completed);
    }

    #[test]
    fn test_summary() {
        let mut state = PipelineState::new("my-pipe");
        state.start_run(Some("run-001".into()));
        let summary = state.summary();
        assert!(summary.contains("my-pipe"));
        assert!(summary.contains("run-001"));
    }

    #[test]
    fn test_checkpoint_without_run() {
        let mut state = PipelineState::new("test");
        let result = state.checkpoint("orphan", serde_json::json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_complete_without_run() {
        let mut state = PipelineState::new("test");
        let result = state.complete_run(RunStatus::Completed);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_runs() {
        let mut state = PipelineState::new("multi");
        state.start_run(Some("run-1".into()));
        state.complete_run(RunStatus::Completed).unwrap();
        state.start_run(Some("run-2".into()));
        state.complete_run(RunStatus::Failed).unwrap();

        assert_eq!(state.runs.len(), 2);
        assert_eq!(state.runs[0].status, RunStatus::Completed);
        assert_eq!(state.runs[1].status, RunStatus::Failed);
    }

    #[test]
    fn test_run_status_serialization() {
        let status = RunStatus::Running;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"running\"");

        let decoded: RunStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, RunStatus::Running);
    }

    #[test]
    fn test_aborted_status() {
        let mut state = PipelineState::new("nmd-test");
        state.start_run(Some("run-abort".into()));
        state.abort_run("phase order violation").unwrap();

        assert_eq!(state.runs[0].status, RunStatus::Aborted);
        assert!(state.runs[0].completed.is_some());
        // Abort records a checkpoint
        assert!(
            state.runs[0]
                .checkpoints
                .iter()
                .any(|c| c.name == "nmd_abort")
        );
    }

    #[test]
    fn test_aborted_serialization() {
        let status = RunStatus::Aborted;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"aborted\"");
    }

    #[test]
    fn test_ejc_markers() {
        let mut state = PipelineState::new("nmd-markers");
        state.start_run(Some("run-ejc".into()));
        let markers = vec![
            serde_json::json!({"phase_id": "investigate", "grounding": 0.5}),
            serde_json::json!({"phase_id": "implement", "grounding": 0.3}),
        ];
        state.attach_ejc_markers(markers).unwrap();

        let run = state.current_run().unwrap();
        assert_eq!(run.ejc_markers.len(), 2);
    }

    #[test]
    fn test_ejc_markers_default_empty() {
        let mut state = PipelineState::new("test");
        state.start_run(None);
        let run = state.current_run().unwrap();
        assert!(run.ejc_markers.is_empty());
    }

    #[test]
    fn test_abort_without_run() {
        let mut state = PipelineState::new("test");
        let result = state.abort_run("no run");
        assert!(result.is_err());
    }

    #[test]
    fn test_attach_markers_without_run() {
        let mut state = PipelineState::new("test");
        let result = state.attach_ejc_markers(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_backward_compat_deserialize() {
        // Existing pipeline JSON without ejc_markers should deserialize fine
        let json = r#"{
            "name": "legacy",
            "runs": [{
                "id": "r1",
                "started": "2026-01-01T00:00:00Z",
                "status": "completed",
                "checkpoints": [],
                "completed": "2026-01-01T00:01:00Z"
            }],
            "last_checkpoint": null
        }"#;
        let state: PipelineState = serde_json::from_str(json).unwrap();
        assert_eq!(state.runs[0].ejc_markers.len(), 0);
    }
}
