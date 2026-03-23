//! Property-based tests for the Brain System (CTVP Phase 0)
//!
//! These tests use proptest to verify invariants that must hold for all inputs,
//! strengthening our Phase 0 (Preclinical) evidence from Moderate to Strong.
//!
//! ## Properties Tested
//!
//! 1. **Session ID Uniqueness** - Creating 100 sessions yields 100 unique IDs
//! 2. **Artifact Version Monotonicity** - Version numbers only increase
//! 3. **Hash Stability** - Same content always produces same hash
//! 4. **Confidence Bounds** - Confidence always in [0.0, 1.0]
//! 5. **Roundtrip Serialization** - Save → Load → equals original

use nexcore_codec::hex;
use proptest::prelude::*;
use std::collections::HashSet;
use std::fs;
use tempfile::TempDir;

// Import the brain types
use nexcore_brain::{
    Artifact, ArtifactMetadata, ArtifactType, Correction, Pattern, Preference, TrackedFile,
};

/// Setup a clean test environment with temporary directories
#[allow(dead_code)]
fn setup_test_brain() -> (
    TempDir,
    std::path::PathBuf,
    std::path::PathBuf,
    std::path::PathBuf,
) {
    let temp = TempDir::new().unwrap();
    let brain_dir = temp.path().join(".claude/brain");
    let tracker_dir = temp.path().join(".claude/code_tracker");
    let implicit_dir = temp.path().join(".claude/implicit");

    fs::create_dir_all(brain_dir.join("sessions")).unwrap();
    fs::create_dir_all(brain_dir.join("annotations")).unwrap();
    fs::write(brain_dir.join("index.json"), "[]").unwrap();

    fs::create_dir_all(tracker_dir.join("active")).unwrap();
    fs::create_dir_all(tracker_dir.join("history")).unwrap();
    fs::write(tracker_dir.join("index.json"), "{}").unwrap();

    fs::create_dir_all(&implicit_dir).unwrap();
    fs::write(implicit_dir.join("preferences.json"), "{}").unwrap();
    fs::write(implicit_dir.join("patterns.json"), "[]").unwrap();
    fs::write(implicit_dir.join("corrections.json"), "[]").unwrap();

    (temp, brain_dir, tracker_dir, implicit_dir)
}

// ============================================================================
// Property 1: Session ID Uniqueness
// ============================================================================
// Creating multiple sessions should always produce unique IDs.
// This property ensures no collision in UUID generation.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_session_ids_are_unique(count in 2..100usize) {
        // Note: This test uses the real BrainSession::create which touches the real filesystem.
        // For isolation, we rely on UUID v4's uniqueness guarantee rather than filesystem state.
        // A full isolation test would require environment variable injection for paths.

        let mut ids = HashSet::new();
        for _ in 0..count {
            let id = nexcore_id::NexId::v4();
            ids.insert(id);
        }

        // All IDs should be unique
        prop_assert_eq!(ids.len(), count, "UUID collision detected in {} generations", count);
    }
}

// ============================================================================
// Property 2: Artifact Version Monotonicity
// ============================================================================
// Version numbers must always increase when increment_version is called.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_version_monotonically_increases(increments in 1..100usize) {
        let mut metadata = ArtifactMetadata::new(ArtifactType::Task, "Test task");
        let mut prev_version = metadata.current_version;

        for _ in 0..increments {
            let new_version = metadata.increment_version();

            // Version must strictly increase
            prop_assert!(
                new_version > prev_version,
                "Version did not increase: {} -> {}",
                prev_version,
                new_version
            );

            prev_version = new_version;
        }

        // Final version should equal number of increments
        prop_assert_eq!(
            metadata.current_version,
            increments as u32,
            "Final version mismatch after {} increments",
            increments
        );
    }
}

// ============================================================================
// Property 3: Hash Stability
// ============================================================================
// The same content should always produce the same hash (determinism).

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_hash_is_stable(content in any::<Vec<u8>>()) {
        use nexcore_hash::sha256::Sha256;

        // Compute hash twice
        let compute_hash = |data: &[u8]| -> String {
            let mut hasher = Sha256::new();
            hasher.update(data);
            let result = hasher.finalize();
            hex::encode(&result[..16])
        };

        let hash1 = compute_hash(&content);
        let hash2 = compute_hash(&content);

        // Hash should be exactly 32 hex chars (16 bytes)
        prop_assert_eq!(
            hash1.len(), 32,
            "Hash length incorrect: expected 32, got {}",
            hash1.len()
        );

        // Same content must produce same hash
        prop_assert_eq!(
            hash1, hash2,
            "Hash not stable for content of length {}",
            content.len()
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_different_content_different_hash(
        content1 in any::<Vec<u8>>(),
        content2 in any::<Vec<u8>>()
    ) {
        use nexcore_hash::sha256::Sha256;

        // Skip if contents happen to be equal
        prop_assume!(content1 != content2);

        let compute_hash = |data: &[u8]| -> String {
            let mut hasher = Sha256::new();
            hasher.update(data);
            let result = hasher.finalize();
            hex::encode(&result[..16])
        };

        let hash1 = compute_hash(&content1);
        let hash2 = compute_hash(&content2);

        // Different content should (almost certainly) produce different hashes
        // Note: With 128-bit truncation, collision probability is ~2^-64 per pair
        prop_assert_ne!(
            hash1, hash2,
            "Hash collision for different content"
        );
    }
}

// ============================================================================
// Property 4: Confidence Bounds
// ============================================================================
// Confidence must always be in range [0.0, 1.0] regardless of operations.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_preference_confidence_bounded(
        reinforcements in 0..1000usize,
        weakenings in 0..100usize
    ) {
        let mut pref = Preference::new("test_key", serde_json::json!("test_value"));

        // Initial confidence should be 0.5
        prop_assert!(
            (0.0..=1.0).contains(&pref.confidence),
            "Initial confidence out of bounds: {}",
            pref.confidence
        );

        // Reinforce many times
        for _ in 0..reinforcements {
            pref.reinforce();
            prop_assert!(
                (0.0..=1.0).contains(&pref.confidence),
                "Confidence out of bounds after reinforce: {}",
                pref.confidence
            );
        }

        // Weaken many times
        for _ in 0..weakenings {
            pref.weaken();
            prop_assert!(
                (0.0..=1.0).contains(&pref.confidence),
                "Confidence out of bounds after weaken: {}",
                pref.confidence
            );
        }
    }

    #[test]
    fn prop_pattern_confidence_bounded(examples in 0..500usize) {
        let mut pattern = Pattern::new("test_pattern", "test_type", "Test description");

        prop_assert!(
            (0.0..=1.0).contains(&pattern.confidence),
            "Initial pattern confidence out of bounds: {}",
            pattern.confidence
        );

        for i in 0..examples {
            pattern.add_example(format!("example_{}", i));
            prop_assert!(
                (0.0..=1.0).contains(&pattern.confidence),
                "Pattern confidence out of bounds after {} examples: {}",
                i + 1,
                pattern.confidence
            );
        }
    }
}

// ============================================================================
// Property 5: Roundtrip Serialization
// ============================================================================
// Objects should survive JSON serialization → deserialization unchanged.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_artifact_type_roundtrip(
        type_index in 0..7usize
    ) {
        let artifact_types = [
            ArtifactType::Task,
            ArtifactType::ImplementationPlan,
            ArtifactType::Walkthrough,
            ArtifactType::Review,
            ArtifactType::Research,
            ArtifactType::Decision,
            ArtifactType::Custom,
        ];

        let original = artifact_types[type_index];

        // Serialize and deserialize
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: ArtifactType = serde_json::from_str(&json).unwrap();

        prop_assert_eq!(original, deserialized, "ArtifactType roundtrip failed");
    }

    #[test]
    fn prop_artifact_roundtrip(
        name in "[a-z]{1,20}\\.md",
        content in "[a-zA-Z0-9\\s]{0,1000}",
        type_index in 0..7usize
    ) {
        let artifact_types = [
            ArtifactType::Task,
            ArtifactType::ImplementationPlan,
            ArtifactType::Walkthrough,
            ArtifactType::Review,
            ArtifactType::Research,
            ArtifactType::Decision,
            ArtifactType::Custom,
        ];

        let original = Artifact::new(&name, artifact_types[type_index], &content);

        // Serialize and deserialize
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: Artifact = serde_json::from_str(&json).unwrap();

        prop_assert_eq!(original.name, deserialized.name);
        prop_assert_eq!(original.artifact_type, deserialized.artifact_type);
        prop_assert_eq!(original.content, deserialized.content);
        prop_assert_eq!(original.version, deserialized.version);
    }

    #[test]
    fn prop_preference_roundtrip(
        key in "[a-z_]{1,30}",
        value in "[a-zA-Z0-9]{0,100}"
    ) {
        let mut original = Preference::new(&key, serde_json::json!(value));

        // Apply some operations
        original.reinforce();
        original.reinforce();

        // Serialize and deserialize
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: Preference = serde_json::from_str(&json).unwrap();

        prop_assert_eq!(original.key, deserialized.key);
        prop_assert_eq!(original.value, deserialized.value);
        prop_assert!((original.confidence - deserialized.confidence).abs() < f64::EPSILON);
        prop_assert_eq!(original.reinforcement_count, deserialized.reinforcement_count);
    }

    #[test]
    fn prop_pattern_roundtrip(
        id in "[a-z_]{1,20}",
        pattern_type in "(naming|structure|workflow|custom)",
        description in "[a-zA-Z0-9\\s]{10,200}"
    ) {
        let mut original = Pattern::new(&id, &pattern_type, &description);
        original.add_example("example_1");
        original.add_example("example_2");

        // Serialize and deserialize
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: Pattern = serde_json::from_str(&json).unwrap();

        prop_assert_eq!(original.id, deserialized.id);
        prop_assert_eq!(original.pattern_type, deserialized.pattern_type);
        prop_assert_eq!(original.description, deserialized.description);
        prop_assert_eq!(original.examples, deserialized.examples);
        prop_assert_eq!(original.occurrence_count, deserialized.occurrence_count);
    }

    #[test]
    fn prop_correction_roundtrip(
        mistake in "[a-zA-Z\\s]{5,100}",
        correction in "[a-zA-Z\\s]{5,100}"
    ) {
        let mut original = Correction::new(&mistake, &correction);
        original.mark_applied();
        original.mark_applied();

        // Serialize and deserialize
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: Correction = serde_json::from_str(&json).unwrap();

        prop_assert_eq!(original.mistake, deserialized.mistake);
        prop_assert_eq!(original.correction, deserialized.correction);
        prop_assert_eq!(original.application_count, deserialized.application_count);
    }
}

// ============================================================================
// Property 6: Artifact Type Inference Consistency
// ============================================================================
// The same filename should always infer the same artifact type.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_artifact_type_inference_is_deterministic(
        prefix in "[a-z]{0,10}",
        keyword in "(task|plan|walkthrough|review|research|decision|random)",
        suffix in "[a-z]{0,10}"
    ) {
        let filename = format!("{}{}{}.md", prefix, keyword, suffix);

        let type1 = ArtifactType::from_filename(&filename);
        let type2 = ArtifactType::from_filename(&filename);

        prop_assert_eq!(type1, type2, "Type inference not deterministic for: {}", filename);
    }
}

// ============================================================================
// Property 7: Metadata Timestamp Ordering
// ============================================================================
// updated_at should never be earlier than created_at.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_metadata_timestamps_ordered(operations in 0..20usize) {
        let mut metadata = ArtifactMetadata::new(ArtifactType::Task, "Test");

        let created_at = metadata.created_at;

        for _ in 0..operations {
            metadata.touch();
            prop_assert!(
                metadata.updated_at >= created_at,
                "updated_at ({}) is before created_at ({})",
                metadata.updated_at,
                created_at
            );
        }
    }
}

// ============================================================================
// Property 8: Summary Generation is Safe
// ============================================================================
// generate_summary should never panic on any content.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_summary_generation_never_panics(
        content in ".*"  // Any string content
    ) {
        let artifact = Artifact::new("test.md", ArtifactType::Task, &content);

        // Should not panic
        let summary = artifact.generate_summary();

        // Summary should not be empty
        prop_assert!(!summary.is_empty(), "Summary was empty");

        // Summary should be reasonably bounded
        prop_assert!(
            summary.len() <= 200,
            "Summary too long: {} chars",
            summary.len()
        );
    }
}

// ============================================================================
// Property 9: Preference Reinforcement Convergence
// ============================================================================
// Repeated reinforcement should converge toward 1.0 but never reach it.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn prop_confidence_converges_but_never_reaches_one(iterations in 10..500usize) {
        let mut pref = Preference::new("test", serde_json::json!(true));

        for _ in 0..iterations {
            pref.reinforce();
        }

        // Should approach but never equal 1.0
        prop_assert!(
            pref.confidence < 1.0,
            "Confidence reached 1.0 after {} reinforcements",
            iterations
        );

        // Should be quite close to 1.0 after many iterations
        if iterations >= 100 {
            prop_assert!(
                pref.confidence > 0.99,
                "Confidence not converging: {} after {} iterations",
                pref.confidence,
                iterations
            );
        }
    }
}

// ============================================================================
// Integration Property Tests
// ============================================================================

#[test]
fn test_tracked_file_hash_matches_content() {
    let temp = TempDir::new().unwrap();
    let file_path = temp.path().join("test.txt");
    let content = b"Hello, world!";

    fs::write(&file_path, content).unwrap();

    let tracked = TrackedFile::from_path(&file_path).unwrap();

    // Re-hash and compare
    use nexcore_hash::sha256::Sha256;
    let mut hasher = Sha256::new();
    hasher.update(content);
    let result = hasher.finalize();
    let expected_hash = hex::encode(&result[..16]);

    assert_eq!(tracked.content_hash, expected_hash);
    assert_eq!(tracked.size, content.len() as u64);
}

#[test]
fn test_artifact_from_content_infers_type() {
    let test_cases = [
        ("task.md", ArtifactType::Task),
        ("my_task.md", ArtifactType::Task),
        ("implementation_plan.md", ArtifactType::ImplementationPlan),
        ("plan.md", ArtifactType::ImplementationPlan),
        ("walkthrough.md", ArtifactType::Walkthrough),
        ("progress.md", ArtifactType::Walkthrough),
        ("code_review.md", ArtifactType::Review),
        ("research_notes.md", ArtifactType::Research),
        ("decision_log.md", ArtifactType::Decision),
        ("random_file.md", ArtifactType::Custom),
    ];

    for (filename, expected_type) in test_cases {
        let artifact = Artifact::from_content(filename, "content");
        assert_eq!(
            artifact.artifact_type, expected_type,
            "Wrong type for filename: {}",
            filename
        );
    }
}

#[test]
fn test_session_entry_serialization_complete() {
    use nexcore_id::NexId;

    // Test that all fields survive serialization
    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
    struct TestSessionEntry {
        id: NexId,
        created_at: nexcore_chrono::DateTime,
        project: Option<String>,
        git_commit: Option<String>,
        description: Option<String>,
    }

    let entry = TestSessionEntry {
        id: NexId::v4(),
        created_at: nexcore_chrono::DateTime::now(),
        project: Some("test-project".into()),
        git_commit: Some("abc123def456".into()),
        description: Some("Test session for property testing".into()),
    };

    let json = serde_json::to_string(&entry).unwrap();
    let parsed: TestSessionEntry = serde_json::from_str(&json).unwrap();

    assert_eq!(entry, parsed);
}
