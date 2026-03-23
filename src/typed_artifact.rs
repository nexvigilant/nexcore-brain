//! Type-safe artifact persistence wrapper.
//!
//! `TypedArtifact<T>` provides compile-time type safety for brain artifact
//! persistence without requiring the brain crate to import domain types.
//! Domain crates define their own snapshot types and use `TypedArtifact<T>`
//! to persist/restore them through the brain session system.
//!
//! ## T1 Grounding
//!
//! - `TypedArtifact` → π (persistence) + μ (mapping: T ↔ JSON string)
//! - `save` → → (causality: state → artifact)
//! - `load` → ∃ (existence: Option<T>)

use serde::{Serialize, de::DeserializeOwned};
use std::marker::PhantomData;

use crate::artifact::{Artifact, ArtifactType};
use crate::error::{BrainError, Result};
use crate::session::BrainSession;

/// A type-safe wrapper for brain artifact persistence.
///
/// Wraps the untyped `Artifact` API with compile-time type checking.
/// Domain crates construct a `TypedArtifact<MyState>` and call `save`/`load`
/// to persist their state without the brain crate knowing about `MyState`.
///
/// # Usage
///
/// ```rust,ignore
/// use nexcore_brain::TypedArtifact;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Serialize, Deserialize)]
/// struct MyState { count: u32 }
///
/// let artifact = TypedArtifact::<MyState>::new("my-state.json");
/// artifact.save(&session, &MyState { count: 42 })?;
/// let restored: Option<MyState> = artifact.load(&session)?;
/// ```
#[derive(Debug)]
pub struct TypedArtifact<T> {
    name: String,
    _phantom: PhantomData<T>,
}

impl<T> TypedArtifact<T> {
    /// Create a new typed artifact handle with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            _phantom: PhantomData,
        }
    }

    /// Get the artifact name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<T: Serialize> TypedArtifact<T> {
    /// Persist state to the brain session as a JSON artifact.
    ///
    /// Serializes `state` to pretty-printed JSON, wraps it in an
    /// `Artifact::new(name, ArtifactType::Custom, json)`, and saves
    /// via `session.save_artifact()`.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails or the artifact cannot be saved.
    pub fn save(&self, session: &BrainSession, state: &T) -> Result<()> {
        let json = serde_json::to_string_pretty(state)
            .map_err(|e| BrainError::Other(format!("TypedArtifact serialize: {e}")))?;
        let artifact = Artifact::new(&self.name, ArtifactType::Custom, json);
        session.save_artifact(&artifact)
    }
}

impl<T: DeserializeOwned> TypedArtifact<T> {
    /// Load state from the brain session.
    ///
    /// Reads the artifact by name, deserializes the JSON content to `T`.
    /// Returns `Ok(None)` if the artifact does not exist.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails or the session cannot be read.
    pub fn load(&self, session: &BrainSession) -> Result<Option<T>> {
        match session.get_artifact(&self.name, None) {
            Ok(artifact) => {
                let state: T = serde_json::from_str(&artifact.content)
                    .map_err(|e| BrainError::Other(format!("TypedArtifact deserialize: {e}")))?;
                Ok(Some(state))
            }
            Err(BrainError::ArtifactNotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestSnapshot {
        count: u32,
        label: String,
    }

    fn make_test_session(dir: &std::path::Path) -> BrainSession {
        std::fs::create_dir_all(dir).unwrap();
        BrainSession {
            id: nexcore_id::NexId::v4().to_string(),
            created_at: nexcore_chrono::DateTime::now(),
            project: None,
            git_commit: None,
            session_dir: dir.to_path_buf(),
        }
    }

    #[test]
    fn test_typed_artifact_round_trip() {
        let temp = TempDir::new().unwrap();
        let session = make_test_session(&temp.path().join("sess"));

        let artifact = TypedArtifact::<TestSnapshot>::new("test-snapshot.json");

        let original = TestSnapshot {
            count: 42,
            label: "hello".to_string(),
        };

        artifact.save(&session, &original).unwrap();
        let restored = artifact.load(&session).unwrap();

        assert_eq!(restored, Some(original));
    }

    #[test]
    fn test_typed_artifact_load_missing() {
        let temp = TempDir::new().unwrap();
        let session = make_test_session(&temp.path().join("sess"));

        let artifact = TypedArtifact::<TestSnapshot>::new("nonexistent.json");
        let result = artifact.load(&session).unwrap();

        assert_eq!(result, None);
    }

    #[test]
    fn test_typed_artifact_overwrite() {
        let temp = TempDir::new().unwrap();
        let session = make_test_session(&temp.path().join("sess"));

        let artifact = TypedArtifact::<TestSnapshot>::new("state.json");

        let v1 = TestSnapshot {
            count: 1,
            label: "first".to_string(),
        };
        artifact.save(&session, &v1).unwrap();

        let v2 = TestSnapshot {
            count: 2,
            label: "second".to_string(),
        };
        artifact.save(&session, &v2).unwrap();

        let restored = artifact.load(&session).unwrap();
        assert_eq!(restored, Some(v2));
    }

    #[test]
    fn test_typed_artifact_name() {
        let artifact = TypedArtifact::<TestSnapshot>::new("my-artifact.json");
        assert_eq!(artifact.name(), "my-artifact.json");
    }
}
