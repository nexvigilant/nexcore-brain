//! Error types for the brain system
//!
//! Includes timeout and recovery error variants for CTVP Phase 1 safety.

use core::fmt;
use std::time::Duration;

/// Brain system errors
#[derive(Debug)]
pub enum BrainError {
    /// Session not found
    SessionNotFound(String),
    /// Artifact not found
    ArtifactNotFound(String),
    /// Version not found
    VersionNotFound {
        /// Artifact name
        artifact: String,
        /// Requested version
        version: u32,
    },
    /// File not tracked
    FileNotTracked(String),
    /// Invalid artifact type
    InvalidArtifactType(String),
    /// IO error
    Io(std::io::Error),
    /// JSON serialization error
    Json(serde_json::Error),
    /// Serialization error (string message)
    Serialization(String),
    /// UUID parse error
    InvalidUuid(nexcore_id::ParseError),
    /// Path error
    InvalidPath(String),
    /// Already exists
    AlreadyExists(String),
    /// Implicit knowledge error
    ImplicitError(String),
    // ========== CTVP Phase 1: Safety Error Variants ==========
    /// Operation timed out
    Timeout(Duration, String),
    /// Index is corrupted and needs recovery
    IndexCorrupted {
        /// Description of the corruption
        message: String,
        /// Whether automatic recovery is possible
        recoverable: bool,
    },
    /// Brain system is unavailable (graceful degradation)
    BrainUnavailable(String),
    /// Lock acquisition failed
    LockTimeout {
        /// Resource being locked
        resource: String,
        /// How long we waited
        timeout: Duration,
    },
    /// Recovery in progress
    RecoveryInProgress(String),
    /// Partial write detected
    PartialWrite {
        /// Artifact name
        artifact: String,
        /// Details about what's missing
        details: String,
    },
    /// Generic error
    Other(String),
}

impl fmt::Display for BrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SessionNotFound(id) => write!(f, "Session not found: {id}"),
            Self::ArtifactNotFound(name) => write!(f, "Artifact not found: {name}"),
            Self::VersionNotFound { artifact, version } => {
                write!(f, "Version {version} not found for artifact: {artifact}")
            }
            Self::FileNotTracked(path) => write!(f, "File not tracked: {path}"),
            Self::InvalidArtifactType(t) => write!(f, "Invalid artifact type: {t}"),
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Json(e) => write!(f, "JSON error: {e}"),
            Self::Serialization(msg) => write!(f, "Serialization error: {msg}"),
            Self::InvalidUuid(e) => write!(f, "Invalid UUID: {e}"),
            Self::InvalidPath(p) => write!(f, "Invalid path: {p}"),
            Self::AlreadyExists(name) => write!(f, "Already exists: {name}"),
            Self::ImplicitError(msg) => write!(f, "Implicit knowledge error: {msg}"),
            Self::Timeout(d, op) => write!(f, "Operation timed out after {d:?}: {op}"),
            Self::IndexCorrupted {
                message,
                recoverable,
            } => {
                write!(
                    f,
                    "Index corrupted: {message}. Recovery available: {recoverable}"
                )
            }
            Self::BrainUnavailable(reason) => {
                write!(
                    f,
                    "Brain unavailable: {reason}. Operations will continue without persistence."
                )
            }
            Self::LockTimeout { resource, timeout } => {
                write!(f, "Failed to acquire lock on {resource} after {timeout:?}")
            }
            Self::RecoveryInProgress(msg) => write!(f, "Recovery in progress: {msg}"),
            Self::PartialWrite { artifact, details } => {
                write!(f, "Partial write detected for {artifact}: {details}")
            }
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for BrainError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Json(e) => Some(e),
            Self::InvalidUuid(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for BrainError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for BrainError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

impl From<nexcore_id::ParseError> for BrainError {
    fn from(e: nexcore_id::ParseError) -> Self {
        Self::InvalidUuid(e)
    }
}

impl BrainError {
    /// Check if this error is recoverable without user intervention
    #[must_use]
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            BrainError::IndexCorrupted {
                recoverable: true,
                ..
            } | BrainError::PartialWrite { .. }
                | BrainError::BrainUnavailable(_)
        )
    }

    /// Check if this error should trigger graceful degradation
    #[must_use]
    pub fn should_degrade_gracefully(&self) -> bool {
        matches!(
            self,
            BrainError::BrainUnavailable(_)
                | BrainError::Timeout(_, _)
                | BrainError::LockTimeout { .. }
        )
    }

    /// Create a timeout error
    pub fn timeout(duration: Duration, operation: impl Into<String>) -> Self {
        BrainError::Timeout(duration, operation.into())
    }

    /// Create an index corrupted error
    pub fn index_corrupted(message: impl Into<String>, recoverable: bool) -> Self {
        BrainError::IndexCorrupted {
            message: message.into(),
            recoverable,
        }
    }

    /// Create a brain unavailable error (for graceful degradation)
    pub fn unavailable(reason: impl Into<String>) -> Self {
        BrainError::BrainUnavailable(reason.into())
    }

    /// Create a lock timeout error
    pub fn lock_timeout(resource: impl Into<String>, timeout: Duration) -> Self {
        BrainError::LockTimeout {
            resource: resource.into(),
            timeout,
        }
    }

    /// Create a partial write error
    pub fn partial_write(artifact: impl Into<String>, details: impl Into<String>) -> Self {
        BrainError::PartialWrite {
            artifact: artifact.into(),
            details: details.into(),
        }
    }
}

/// Result type for brain operations
pub type Result<T> = std::result::Result<T, BrainError>;

#[cfg(test)]
mod tests {
    use super::*;

    // ========== CTVP Phase 0: Edge Case Tests ==========

    #[test]
    fn test_error_display_session_not_found() {
        let err = BrainError::SessionNotFound("abc-123".into());
        let msg = format!("{}", err);
        assert!(msg.contains("Session not found"));
        assert!(msg.contains("abc-123"));
    }

    #[test]
    fn test_error_display_artifact_not_found() {
        let err = BrainError::ArtifactNotFound("task.md".into());
        let msg = format!("{}", err);
        assert!(msg.contains("Artifact not found"));
        assert!(msg.contains("task.md"));
    }

    #[test]
    fn test_error_display_version_not_found() {
        let err = BrainError::VersionNotFound {
            artifact: "plan.md".into(),
            version: 5,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Version 5"));
        assert!(msg.contains("plan.md"));
    }

    #[test]
    fn test_error_display_file_not_tracked() {
        let err = BrainError::FileNotTracked("/path/to/file.rs".into());
        let msg = format!("{}", err);
        assert!(msg.contains("not tracked"));
        assert!(msg.contains("/path/to/file.rs"));
    }

    #[test]
    fn test_error_display_invalid_artifact_type() {
        let err = BrainError::InvalidArtifactType("unknown_type".into());
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid artifact type"));
        assert!(msg.contains("unknown_type"));
    }

    #[test]
    fn test_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let brain_err: BrainError = io_err.into();

        if let BrainError::Io(inner) = brain_err {
            assert_eq!(inner.kind(), std::io::ErrorKind::NotFound);
        } else {
            panic!("Expected BrainError::Io");
        }
    }

    #[test]
    fn test_error_from_json_error() {
        let json_str = "{ invalid json }";
        let json_result: std::result::Result<serde_json::Value, _> = serde_json::from_str(json_str);
        let json_err = json_result.unwrap_err();

        let brain_err: BrainError = json_err.into();
        if let BrainError::Json(_) = brain_err {
            // Success
        } else {
            panic!("Expected BrainError::Json");
        }
    }

    #[test]
    fn test_error_from_uuid_error() {
        let uuid_result = "not-a-uuid".parse::<nexcore_id::NexId>();
        let uuid_err = uuid_result.unwrap_err();

        let brain_err: BrainError = uuid_err.into();
        if let BrainError::InvalidUuid(_) = brain_err {
            // Success
        } else {
            panic!("Expected BrainError::InvalidUuid");
        }
    }

    #[test]
    fn test_error_display_invalid_path() {
        let err = BrainError::InvalidPath("../relative/path".into());
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid path"));
    }

    #[test]
    fn test_error_display_already_exists() {
        let err = BrainError::AlreadyExists("session-123".into());
        let msg = format!("{}", err);
        assert!(msg.contains("Already exists"));
    }

    #[test]
    fn test_error_display_implicit_error() {
        let err = BrainError::ImplicitError("failed to load preferences".into());
        let msg = format!("{}", err);
        assert!(msg.contains("Implicit knowledge error"));
    }

    #[test]
    fn test_error_display_other() {
        let err = BrainError::Other("Custom error message".into());
        let msg = format!("{}", err);
        assert_eq!(msg, "Custom error message");
    }

    #[test]
    fn test_error_debug_format() {
        let err = BrainError::SessionNotFound("test".into());
        let debug = format!("{:?}", err);
        assert!(debug.contains("SessionNotFound"));
    }

    #[test]
    fn test_result_type_ok() {
        let result: Result<i32> = Ok(42);
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_result_type_err() {
        let result: Result<i32> = Err(BrainError::Other("test".into()));
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unicode_messages() {
        let err = BrainError::SessionNotFound("セッション-日本語-🔥".into());
        let msg = format!("{}", err);
        assert!(msg.contains("セッション"));
        assert!(msg.contains("🔥"));
    }

    // ========== CTVP Phase 1: Safety Error Tests ==========

    #[test]
    fn test_timeout_error() {
        let err = BrainError::timeout(Duration::from_secs(5), "writing artifact");
        let msg = format!("{}", err);
        assert!(msg.contains("timed out"));
        assert!(msg.contains("5s"));
        assert!(msg.contains("writing artifact"));
    }

    #[test]
    fn test_index_corrupted_error() {
        let err = BrainError::index_corrupted("invalid JSON at line 5", true);
        let msg = format!("{}", err);
        assert!(msg.contains("corrupted"));
        assert!(msg.contains("invalid JSON"));
        assert!(msg.contains("true")); // recoverable

        let err_unrecoverable = BrainError::index_corrupted("complete data loss", false);
        let msg = format!("{}", err_unrecoverable);
        assert!(msg.contains("false")); // not recoverable
    }

    #[test]
    fn test_brain_unavailable_error() {
        let err = BrainError::unavailable("disk full");
        let msg = format!("{}", err);
        assert!(msg.contains("unavailable"));
        assert!(msg.contains("disk full"));
        assert!(msg.contains("continue without persistence"));
    }

    #[test]
    fn test_lock_timeout_error() {
        let err = BrainError::lock_timeout("index.json", Duration::from_millis(500));
        let msg = format!("{}", err);
        assert!(msg.contains("lock"));
        assert!(msg.contains("index.json"));
        assert!(msg.contains("500ms"));
    }

    #[test]
    fn test_partial_write_error() {
        let err = BrainError::partial_write("task.md", "metadata missing");
        let msg = format!("{}", err);
        assert!(msg.contains("Partial write"));
        assert!(msg.contains("task.md"));
        assert!(msg.contains("metadata missing"));
    }

    #[test]
    fn test_recovery_in_progress_error() {
        let err = BrainError::RecoveryInProgress("rebuilding index from sessions".into());
        let msg = format!("{}", err);
        assert!(msg.contains("Recovery in progress"));
        assert!(msg.contains("rebuilding"));
    }

    #[test]
    fn test_is_recoverable() {
        // Recoverable errors
        assert!(BrainError::index_corrupted("test", true).is_recoverable());
        assert!(BrainError::partial_write("test", "test").is_recoverable());
        assert!(BrainError::unavailable("test").is_recoverable());

        // Non-recoverable errors
        assert!(!BrainError::index_corrupted("test", false).is_recoverable());
        assert!(!BrainError::SessionNotFound("test".into()).is_recoverable());
        assert!(!BrainError::timeout(Duration::from_secs(1), "test").is_recoverable());
    }

    #[test]
    fn test_should_degrade_gracefully() {
        // Should degrade
        assert!(BrainError::unavailable("test").should_degrade_gracefully());
        assert!(BrainError::timeout(Duration::from_secs(1), "test").should_degrade_gracefully());
        assert!(
            BrainError::lock_timeout("test", Duration::from_secs(1)).should_degrade_gracefully()
        );

        // Should not degrade (need explicit handling)
        assert!(!BrainError::SessionNotFound("test".into()).should_degrade_gracefully());
        assert!(!BrainError::ArtifactNotFound("test".into()).should_degrade_gracefully());
    }

    #[test]
    fn test_error_builder_methods() {
        // Verify all builder methods create correct variants
        let timeout = BrainError::timeout(Duration::from_millis(100), "op");
        assert!(matches!(timeout, BrainError::Timeout(_, _)));

        let corrupted = BrainError::index_corrupted("msg", true);
        assert!(matches!(corrupted, BrainError::IndexCorrupted { .. }));

        let unavailable = BrainError::unavailable("reason");
        assert!(matches!(unavailable, BrainError::BrainUnavailable(_)));

        let lock = BrainError::lock_timeout("res", Duration::from_secs(1));
        assert!(matches!(lock, BrainError::LockTimeout { .. }));

        let partial = BrainError::partial_write("art", "det");
        assert!(matches!(partial, BrainError::PartialWrite { .. }));
    }
}
