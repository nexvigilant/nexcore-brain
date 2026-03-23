//! Artifact types and operations
//!
//! Artifacts are the primary unit of work in the brain system. Each artifact
//! has a current mutable state and can have multiple resolved (immutable) versions.

use nexcore_chrono::DateTime;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Type of artifact being tracked
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactType {
    /// Task definition and progress tracking
    Task,
    /// Implementation plan with steps
    ImplementationPlan,
    /// Progress walkthrough/documentation
    Walkthrough,
    /// Code review notes
    Review,
    /// Research notes
    Research,
    /// Decision log
    Decision,
    /// Standard operating procedure
    Sop,
    /// System specification or component spec
    Specification,
    /// Architecture schematic or wiring diagram
    Schematic,
    /// Audit or inspection report
    Audit,
    /// Reference document, guide, or knowledge pack
    Reference,
    /// Custom artifact type
    Custom,
}

impl fmt::Display for ArtifactType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArtifactType::Task => write!(f, "task"),
            ArtifactType::ImplementationPlan => write!(f, "implementation_plan"),
            ArtifactType::Walkthrough => write!(f, "walkthrough"),
            ArtifactType::Review => write!(f, "review"),
            ArtifactType::Research => write!(f, "research"),
            ArtifactType::Decision => write!(f, "decision"),
            ArtifactType::Sop => write!(f, "sop"),
            ArtifactType::Specification => write!(f, "specification"),
            ArtifactType::Schematic => write!(f, "schematic"),
            ArtifactType::Audit => write!(f, "audit"),
            ArtifactType::Reference => write!(f, "reference"),
            ArtifactType::Custom => write!(f, "custom"),
        }
    }
}

impl FromStr for ArtifactType {
    type Err = crate::error::BrainError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "task" => Ok(ArtifactType::Task),
            "implementation_plan" | "plan" => Ok(ArtifactType::ImplementationPlan),
            "walkthrough" => Ok(ArtifactType::Walkthrough),
            "review" => Ok(ArtifactType::Review),
            "research" => Ok(ArtifactType::Research),
            "decision" => Ok(ArtifactType::Decision),
            "sop" | "procedure" => Ok(ArtifactType::Sop),
            "specification" | "spec" => Ok(ArtifactType::Specification),
            "schematic" | "diagram" | "wiring" => Ok(ArtifactType::Schematic),
            "audit" | "inspection" => Ok(ArtifactType::Audit),
            "reference" | "guide" => Ok(ArtifactType::Reference),
            "custom" => Ok(ArtifactType::Custom),
            _ => Err(crate::error::BrainError::InvalidArtifactType(s.to_string())),
        }
    }
}

impl ArtifactType {
    /// Infer artifact type from filename
    #[must_use]
    pub fn from_filename(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("task") {
            ArtifactType::Task
        } else if lower.contains("plan") || lower.contains("implementation") {
            ArtifactType::ImplementationPlan
        } else if lower.contains("walkthrough") || lower.contains("progress") {
            ArtifactType::Walkthrough
        } else if lower.contains("review") {
            ArtifactType::Review
        } else if lower.contains("research") {
            ArtifactType::Research
        } else if lower.contains("decision") {
            ArtifactType::Decision
        } else if lower.contains("sop") || lower.contains("procedure") {
            ArtifactType::Sop
        } else if lower.contains("audit") || lower.contains("inspection") {
            ArtifactType::Audit
        } else if lower.contains("spec") {
            ArtifactType::Specification
        } else if lower.contains("schematic")
            || lower.contains("wiring")
            || lower.contains("diagram")
        {
            ArtifactType::Schematic
        } else if lower.contains("reference") || lower.contains("guide") {
            ArtifactType::Reference
        } else {
            ArtifactType::Custom
        }
    }
}

/// Metadata for an artifact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    /// Type of artifact
    pub artifact_type: ArtifactType,

    /// Human-readable summary
    pub summary: String,

    /// When the artifact was created
    pub created_at: DateTime,

    /// When the artifact was last updated
    pub updated_at: DateTime,

    /// Current resolved version (0 = never resolved)
    pub current_version: u32,

    /// Optional tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,

    /// Optional custom metadata
    #[serde(default)]
    pub custom: serde_json::Value,
}

impl ArtifactMetadata {
    /// Create new metadata
    #[must_use]
    pub fn new(artifact_type: ArtifactType, summary: impl Into<String>) -> Self {
        let now = DateTime::now();
        Self {
            artifact_type,
            summary: summary.into(),
            created_at: now,
            updated_at: now,
            current_version: 0,
            tags: Vec::new(),
            custom: serde_json::Value::Null,
        }
    }

    /// Update the metadata timestamp
    pub fn touch(&mut self) {
        self.updated_at = DateTime::now();
    }

    /// Increment version and update timestamp
    pub fn increment_version(&mut self) -> u32 {
        self.current_version += 1;
        self.updated_at = DateTime::now();
        self.current_version
    }
}

/// An artifact in the brain system
///
/// Artifacts represent documents that track work progress. They have:
/// - A current mutable state
/// - Zero or more resolved (immutable) versions
/// - Associated metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Artifact name (e.g., "task.md")
    pub name: String,

    /// Type of artifact
    pub artifact_type: ArtifactType,

    /// Current content
    pub content: String,

    /// Current version (0 = unresolved, 1+ = resolved versions exist)
    pub version: u32,

    /// When this artifact was last modified
    pub updated_at: DateTime,
}

impl Artifact {
    /// Create a new artifact
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        artifact_type: ArtifactType,
        content: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            artifact_type,
            content: content.into(),
            version: 0,
            updated_at: DateTime::now(),
        }
    }

    /// Create a new artifact with type inferred from filename
    #[must_use]
    pub fn from_content(name: impl Into<String>, content: impl Into<String>) -> Self {
        let name = name.into();
        let artifact_type = ArtifactType::from_filename(&name);
        Self {
            name,
            artifact_type,
            content: content.into(),
            version: 0,
            updated_at: DateTime::now(),
        }
    }

    /// Update the artifact content
    pub fn update_content(&mut self, content: impl Into<String>) {
        self.content = content.into();
        self.updated_at = DateTime::now();
    }

    /// Generate a summary from content (first non-empty line or first N chars)
    #[must_use]
    pub fn generate_summary(&self) -> String {
        // Find the first non-empty, non-header line
        for line in self.content.lines() {
            let trimmed = line.trim();
            // Skip empty lines and markdown headers
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            // Return first meaningful line, truncated if needed
            let max_len = 100;
            if trimmed.len() > max_len {
                return format!("{}...", &trimmed[..max_len]);
            }
            return trimmed.to_string();
        }
        // Fallback: use artifact name
        format!("{} artifact", self.artifact_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_artifact_type_from_filename() {
        assert_eq!(ArtifactType::from_filename("task.md"), ArtifactType::Task);
        assert_eq!(
            ArtifactType::from_filename("my_task.md"),
            ArtifactType::Task
        );
        assert_eq!(
            ArtifactType::from_filename("implementation_plan.md"),
            ArtifactType::ImplementationPlan
        );
        assert_eq!(
            ArtifactType::from_filename("plan.md"),
            ArtifactType::ImplementationPlan
        );
        assert_eq!(
            ArtifactType::from_filename("walkthrough.md"),
            ArtifactType::Walkthrough
        );
        assert_eq!(
            ArtifactType::from_filename("SOP-DEV-004.md"),
            ArtifactType::Sop
        );
        assert_eq!(
            ArtifactType::from_filename("procedure_manual.md"),
            ArtifactType::Sop
        );
        assert_eq!(
            ArtifactType::from_filename("api_spec.md"),
            ArtifactType::Specification
        );
        assert_eq!(
            ArtifactType::from_filename("wiring_diagram.md"),
            ArtifactType::Schematic
        );
        assert_eq!(
            ArtifactType::from_filename("circuit_schematic.md"),
            ArtifactType::Schematic
        );
        assert_eq!(
            ArtifactType::from_filename("security_audit.md"),
            ArtifactType::Audit
        );
        assert_eq!(
            ArtifactType::from_filename("inspection_report.md"),
            ArtifactType::Audit
        );
        assert_eq!(
            ArtifactType::from_filename("quick_reference.md"),
            ArtifactType::Reference
        );
        assert_eq!(
            ArtifactType::from_filename("user_guide.md"),
            ArtifactType::Reference
        );
        assert_eq!(
            ArtifactType::from_filename("random_file.txt"),
            ArtifactType::Custom
        );
    }

    #[test]
    fn test_artifact_type_from_str() {
        assert_eq!("task".parse::<ArtifactType>().unwrap(), ArtifactType::Task);
        assert_eq!(
            "plan".parse::<ArtifactType>().unwrap(),
            ArtifactType::ImplementationPlan
        );
        assert_eq!(
            "implementation_plan".parse::<ArtifactType>().unwrap(),
            ArtifactType::ImplementationPlan
        );
        assert_eq!("sop".parse::<ArtifactType>().unwrap(), ArtifactType::Sop);
        assert_eq!(
            "procedure".parse::<ArtifactType>().unwrap(),
            ArtifactType::Sop
        );
        assert_eq!(
            "specification".parse::<ArtifactType>().unwrap(),
            ArtifactType::Specification
        );
        assert_eq!(
            "spec".parse::<ArtifactType>().unwrap(),
            ArtifactType::Specification
        );
        assert_eq!(
            "schematic".parse::<ArtifactType>().unwrap(),
            ArtifactType::Schematic
        );
        assert_eq!(
            "diagram".parse::<ArtifactType>().unwrap(),
            ArtifactType::Schematic
        );
        assert_eq!(
            "wiring".parse::<ArtifactType>().unwrap(),
            ArtifactType::Schematic
        );
        assert_eq!(
            "audit".parse::<ArtifactType>().unwrap(),
            ArtifactType::Audit
        );
        assert_eq!(
            "inspection".parse::<ArtifactType>().unwrap(),
            ArtifactType::Audit
        );
        assert_eq!(
            "reference".parse::<ArtifactType>().unwrap(),
            ArtifactType::Reference
        );
        assert_eq!(
            "guide".parse::<ArtifactType>().unwrap(),
            ArtifactType::Reference
        );
    }

    #[test]
    fn test_artifact_new() {
        let artifact = Artifact::new("task.md", ArtifactType::Task, "# My Task\n\nDo something");
        assert_eq!(artifact.name, "task.md");
        assert_eq!(artifact.artifact_type, ArtifactType::Task);
        assert_eq!(artifact.version, 0);
    }

    #[test]
    fn test_artifact_generate_summary() {
        let artifact = Artifact::new(
            "task.md",
            ArtifactType::Task,
            "# My Task\n\nThis is the first meaningful line.",
        );
        assert_eq!(
            artifact.generate_summary(),
            "This is the first meaningful line."
        );
    }

    #[test]
    fn test_metadata_increment_version() {
        let mut metadata = ArtifactMetadata::new(ArtifactType::Task, "Test task");
        assert_eq!(metadata.current_version, 0);

        let v1 = metadata.increment_version();
        assert_eq!(v1, 1);
        assert_eq!(metadata.current_version, 1);

        let v2 = metadata.increment_version();
        assert_eq!(v2, 2);
        assert_eq!(metadata.current_version, 2);
    }

    // ========== CTVP Phase 0: Edge Case Tests ==========

    #[test]
    fn test_artifact_empty_content() {
        let artifact = Artifact::new("task.md", ArtifactType::Task, "");
        assert!(artifact.content.is_empty());

        // Summary should fallback to type name
        let summary = artifact.generate_summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("task"));
    }

    #[test]
    fn test_artifact_unicode_content() {
        let unicode_content = "# 任务 🚀\n\n这是一个测试任务。\n\nПривет мир! 🌍";
        let artifact = Artifact::new("task.md", ArtifactType::Task, unicode_content);

        assert_eq!(artifact.content, unicode_content);
        let summary = artifact.generate_summary();
        assert!(summary.contains("这是一个测试任务"));
    }

    #[test]
    fn test_artifact_very_long_name() {
        // Test with a name longer than typical filesystem limits
        let long_name = format!("{}.md", "a".repeat(300));
        let artifact = Artifact::new(&long_name, ArtifactType::Custom, "content");

        assert_eq!(artifact.name.len(), 303); // 300 'a' + ".md"
        assert_eq!(artifact.artifact_type, ArtifactType::Custom);
    }

    #[test]
    fn test_artifact_special_characters_in_name() {
        let special_names = [
            "task-2024.md",
            "task_v2.1.0.md",
            "my task (draft).md",
            "task#1.md",
        ];

        for name in special_names {
            let artifact = Artifact::from_content(name, "content");
            assert_eq!(artifact.name, name);
        }
    }

    #[test]
    fn test_artifact_type_case_insensitivity() {
        // from_filename should be case-insensitive
        assert_eq!(ArtifactType::from_filename("TASK.MD"), ArtifactType::Task);
        assert_eq!(ArtifactType::from_filename("TaSk.Md"), ArtifactType::Task);
        assert_eq!(
            ArtifactType::from_filename("IMPLEMENTATION_PLAN.MD"),
            ArtifactType::ImplementationPlan
        );
    }

    #[test]
    fn test_artifact_type_from_str_invalid() {
        let result = "invalid_type".parse::<ArtifactType>();
        assert!(result.is_err());

        if let Err(crate::error::BrainError::InvalidArtifactType(s)) = result {
            assert_eq!(s, "invalid_type");
        } else {
            panic!("Expected InvalidArtifactType error");
        }
    }

    #[test]
    fn test_artifact_summary_only_headers() {
        let artifact = Artifact::new(
            "task.md",
            ArtifactType::Task,
            "# Header\n## Subheader\n### Another",
        );
        let summary = artifact.generate_summary();

        // Should fallback when only headers exist
        assert!(summary.contains("task"));
    }

    #[test]
    fn test_artifact_summary_very_long_line() {
        let long_line = "x".repeat(200);
        let content = format!("# Header\n\n{}", long_line);
        let artifact = Artifact::new("task.md", ArtifactType::Task, &content);

        let summary = artifact.generate_summary();
        // Summary should be truncated
        assert!(summary.len() <= 103); // 100 chars + "..."
        assert!(summary.ends_with("..."));
    }

    #[test]
    fn test_metadata_touch_updates_timestamp() {
        let mut metadata = ArtifactMetadata::new(ArtifactType::Task, "Test");
        let original_updated = metadata.updated_at;

        // Small delay to ensure timestamp changes
        std::thread::sleep(std::time::Duration::from_millis(10));
        metadata.touch();

        assert!(metadata.updated_at >= original_updated);
    }

    #[test]
    fn test_artifact_type_display() {
        assert_eq!(format!("{}", ArtifactType::Task), "task");
        assert_eq!(
            format!("{}", ArtifactType::ImplementationPlan),
            "implementation_plan"
        );
        assert_eq!(format!("{}", ArtifactType::Walkthrough), "walkthrough");
        assert_eq!(format!("{}", ArtifactType::Review), "review");
        assert_eq!(format!("{}", ArtifactType::Research), "research");
        assert_eq!(format!("{}", ArtifactType::Decision), "decision");
        assert_eq!(format!("{}", ArtifactType::Sop), "sop");
        assert_eq!(format!("{}", ArtifactType::Specification), "specification");
        assert_eq!(format!("{}", ArtifactType::Schematic), "schematic");
        assert_eq!(format!("{}", ArtifactType::Audit), "audit");
        assert_eq!(format!("{}", ArtifactType::Reference), "reference");
        assert_eq!(format!("{}", ArtifactType::Custom), "custom");
    }

    #[test]
    fn test_metadata_default_values() {
        let metadata = ArtifactMetadata::new(ArtifactType::Custom, "Test");

        assert!(metadata.tags.is_empty());
        assert_eq!(metadata.custom, serde_json::Value::Null);
        assert_eq!(metadata.current_version, 0);
    }

    #[test]
    fn test_artifact_update_content() {
        let mut artifact = Artifact::new("task.md", ArtifactType::Task, "original");
        let original_time = artifact.updated_at;

        std::thread::sleep(std::time::Duration::from_millis(10));
        artifact.update_content("updated");

        assert_eq!(artifact.content, "updated");
        assert!(artifact.updated_at >= original_time);
    }
}
