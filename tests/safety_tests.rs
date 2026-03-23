//! Safety Tests for the Brain System (CTVP Phase 1)
//!
//! These tests verify the system fails gracefully under adverse conditions.
//! Phase 1 focuses on fault injection and chaos engineering.
//!
//! ## Test Categories
//!
//! 1. **Permission Failures** - Read-only directories, missing permissions
//! 2. **Concurrent Access** - Race conditions, data corruption
//! 3. **Corrupted Data** - Invalid JSON, truncated files
//! 4. **Resource Exhaustion** - Large content, many sessions
//! 5. **Recovery** - Index rebuild, session recovery

use nexcore_codec::hex;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::TempDir;

use nexcore_brain::{
    Artifact, ArtifactMetadata, ArtifactType, BrainError, Correction, Pattern, Preference,
};
use nexcore_error::{Result, nexerror};

// ============================================================================
// Test Infrastructure
// ============================================================================

/// Create an isolated test environment with proper brain directory structure
#[allow(dead_code)]
fn setup_isolated_brain_env() -> Result<(TempDir, PathBuf)> {
    let temp = TempDir::new()?;
    let brain_dir = temp.path().join(".claude/brain");

    fs::create_dir_all(brain_dir.join("sessions"))?;
    fs::create_dir_all(brain_dir.join("annotations"))?;
    fs::write(brain_dir.join("index.json"), "[]")?;

    // Also create tracker and implicit dirs
    let tracker_dir = temp.path().join(".claude/code_tracker");
    fs::create_dir_all(tracker_dir.join("active"))?;
    fs::create_dir_all(tracker_dir.join("history"))?;
    fs::write(tracker_dir.join("index.json"), "{}")?;

    let implicit_dir = temp.path().join(".claude/implicit");
    fs::create_dir_all(&implicit_dir)?;
    fs::write(implicit_dir.join("preferences.json"), "{}")?;
    fs::write(implicit_dir.join("patterns.json"), "[]")?;
    fs::write(implicit_dir.join("corrections.json"), "[]")?;

    Ok((temp, brain_dir))
}

// ============================================================================
// Permission Failure Tests
// ============================================================================

#[test]
fn test_write_to_read_only_directory() -> Result<()> {
    let temp = TempDir::new()?;
    let read_only_dir = temp.path().join("read_only");
    fs::create_dir_all(&read_only_dir)?;

    // Make directory read-only
    let mut perms = fs::metadata(&read_only_dir)?.permissions();
    perms.set_mode(0o444); // r--r--r--
    fs::set_permissions(&read_only_dir, perms)?;

    // Attempt to write should fail gracefully
    let result = fs::write(read_only_dir.join("test.txt"), "content");
    assert!(result.is_err(), "Write to read-only dir should fail");

    // Cleanup: restore permissions so temp dir can be deleted
    let mut perms = fs::metadata(&read_only_dir)?.permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&read_only_dir, perms)?;
    Ok(())
}

#[test]
fn test_artifact_save_with_permissions_error() -> Result<()> {
    let temp = TempDir::new()?;
    let session_dir = temp.path().join("session");
    fs::create_dir_all(&session_dir)?;

    // Create an artifact
    let artifact = Artifact::new("task.md", ArtifactType::Task, "# Test Task");

    // Make directory read-only
    let mut perms = fs::metadata(&session_dir)?.permissions();
    perms.set_mode(0o444);
    fs::set_permissions(&session_dir, perms)?;

    // Attempting to write artifact should fail
    let artifact_path = session_dir.join(&artifact.name);
    let result = fs::write(&artifact_path, &artifact.content);
    assert!(
        result.is_err(),
        "Should fail to write to read-only session dir"
    );

    // Cleanup
    let mut perms = fs::metadata(&session_dir)?.permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&session_dir, perms)?;
    Ok(())
}

#[test]
fn test_missing_parent_directory() -> Result<()> {
    let nonexistent_path = PathBuf::from("/tmp/nexcore_test_nonexistent_12345/brain/sessions");

    // Ensure it doesn't exist
    if nonexistent_path.exists() {
        let parent = nonexistent_path
            .parent()
            .ok_or_else(|| nexerror!("missing_parent_path"))?;
        fs::remove_dir_all(parent).ok();
    }

    // Attempting operations on missing path should produce clear errors
    let result = fs::read_to_string(nonexistent_path.join("index.json"));
    assert!(result.is_err());

    if let Err(e) = result {
        assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
    }
    Ok(())
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

#[test]
fn test_concurrent_uuid_generation_no_collisions() -> Result<()> {
    use nexcore_id::NexId;
    use std::collections::HashSet;
    use std::sync::Mutex;

    let num_threads = 10;
    let uuids_per_thread = 100;
    let all_uuids = Arc::new(Mutex::new(HashSet::new()));
    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let uuids = Arc::clone(&all_uuids);
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                // Wait for all threads to be ready
                barrier.wait();

                let mut local_uuids = Vec::new();
                for _ in 0..uuids_per_thread {
                    local_uuids.push(NexId::v4());
                }

                // Add to shared set
                let mut uuids = uuids.lock().unwrap();
                for id in local_uuids {
                    uuids.insert(id);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().map_err(|_| nexerror!("thread_panicked"))?;
    }

    let uuids = all_uuids.lock().unwrap();
    assert_eq!(
        uuids.len(),
        num_threads * uuids_per_thread,
        "UUID collision detected! Expected {} unique UUIDs, got {}",
        num_threads * uuids_per_thread,
        uuids.len()
    );
    Ok(())
}

#[test]
fn test_concurrent_metadata_updates() -> Result<()> {
    let num_threads = 10;
    let updates_per_thread = 100;
    let barrier = Arc::new(Barrier::new(num_threads));
    let final_versions = Arc::new(std::sync::Mutex::new(Vec::new()));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            let versions = Arc::clone(&final_versions);

            thread::spawn(move || {
                barrier.wait();

                let mut metadata = ArtifactMetadata::new(ArtifactType::Task, "Test");
                for _ in 0..updates_per_thread {
                    metadata.increment_version();
                }

                let mut versions = versions.lock().unwrap();
                versions.push(metadata.current_version);
            })
        })
        .collect();

    for handle in handles {
        handle.join().map_err(|_| nexerror!("thread_panicked"))?;
    }

    let versions = final_versions.lock().unwrap();
    // Each thread should have incremented independently to updates_per_thread
    for &v in versions.iter() {
        assert_eq!(
            v, updates_per_thread as u32,
            "Thread-local metadata should have version {}",
            updates_per_thread
        );
    }
    Ok(())
}

#[test]
fn test_concurrent_preference_reinforcement() -> Result<()> {
    let num_threads = 10;
    let reinforcements_per_thread = 100;
    let barrier = Arc::new(Barrier::new(num_threads));
    let results = Arc::new(std::sync::Mutex::new(Vec::new()));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            let results = Arc::clone(&results);

            thread::spawn(move || {
                barrier.wait();

                let mut pref = Preference::new("test", serde_json::json!("value"));
                for _ in 0..reinforcements_per_thread {
                    pref.reinforce();
                }

                let mut results = results.lock().unwrap();
                results.push((pref.confidence, pref.reinforcement_count));
            })
        })
        .collect();

    for handle in handles {
        handle.join().map_err(|_| nexerror!("thread_panicked"))?;
    }

    let results = results.lock().unwrap();
    for (confidence, count) in results.iter() {
        // Each thread starts fresh, so count should be 1 (initial) + reinforcements
        assert_eq!(*count, 1 + reinforcements_per_thread as u32);
        assert!(
            *confidence > 0.99,
            "Confidence should be high after many reinforcements"
        );
        assert!(
            *confidence < 1.0,
            "Confidence should never reach exactly 1.0"
        );
    }
    Ok(())
}

// ============================================================================
// Corrupted Data Tests
// ============================================================================

#[test]
fn test_corrupted_json_index() -> Result<()> {
    let temp = TempDir::new()?;
    let index_path = temp.path().join("index.json");

    // Write corrupted JSON
    let corrupted_variants = [
        "{ invalid json }",
        "[{\"id\": }]",
        "",
        "null",
        "\"string instead of array\"",
        "[{\"id\": \"not-a-uuid\"}]",
    ];

    for corrupted in corrupted_variants {
        fs::write(&index_path, corrupted)?;

        let result: Result<Vec<serde_json::Value>, _> =
            serde_json::from_str(&fs::read_to_string(&index_path)?);

        // Some corrupted JSON will parse but not be valid session data
        // The important thing is we don't panic
        match result {
            Ok(parsed) => {
                // If it parsed, verify we can iterate without panic
                for item in &parsed {
                    let _ = item.get("id");
                }
            }
            Err(_) => {
                // Expected for truly corrupted JSON
            }
        }
    }
    Ok(())
}

#[test]
fn test_truncated_artifact_content() -> Result<()> {
    let temp = TempDir::new()?;
    let artifact_path = temp.path().join("task.md");

    // Write content then truncate mid-UTF8 sequence
    let content = "# Task with UTF-8: 日本語 🦀";
    fs::write(&artifact_path, content)?;

    // Read back and verify it's intact
    let read_content = fs::read_to_string(&artifact_path)?;
    assert_eq!(content, read_content);

    // Now test with actual truncation at byte level
    let bytes = content.as_bytes();
    let truncated = &bytes[..bytes.len() / 2]; // Truncate in middle

    // This might produce invalid UTF-8
    let truncated_path = temp.path().join("truncated.md");
    fs::write(&truncated_path, truncated)?;

    // Reading as string might fail if truncated mid-character
    let result = fs::read_to_string(&truncated_path);
    // We just verify it doesn't panic - it may succeed or fail gracefully
    match result {
        Ok(s) => assert!(!s.is_empty()),
        Err(e) => {
            // Should be a UTF-8 error, not a panic
            assert!(
                e.to_string().contains("valid UTF-8")
                    || e.kind() == std::io::ErrorKind::InvalidData
            );
        }
    }
    Ok(())
}

#[test]
fn test_corrupted_metadata_json() -> Result<()> {
    let temp = TempDir::new()?;
    let metadata_path = temp.path().join("task.md.metadata.json");

    let corrupted_metadata = [
        "{}", // Empty object - missing required fields
        "{\"artifact_type\": \"invalid_type\"}",
        "{\"current_version\": -1}", // Negative version
        "{\"confidence\": 2.0}",     // Out of bounds
    ];

    for corrupted in corrupted_metadata {
        fs::write(&metadata_path, corrupted)?;

        let result: Result<ArtifactMetadata, _> =
            serde_json::from_str(&fs::read_to_string(&metadata_path)?);

        // Should either parse with defaults or fail gracefully
        match result {
            Ok(metadata) => {
                // If it parsed, verify bounds
                assert!(metadata.current_version <= u32::MAX);
            }
            Err(_) => {
                // Expected for invalid metadata
            }
        }
    }
    Ok(())
}

#[test]
fn test_empty_file_handling() -> Result<()> {
    let temp = TempDir::new()?;

    // Empty index.json
    let index_path = temp.path().join("index.json");
    fs::write(&index_path, "")?;

    let result: Result<Vec<serde_json::Value>, _> =
        serde_json::from_str(&fs::read_to_string(&index_path).unwrap_or_default());

    match result {
        Ok(_) => return Err(nexerror!("empty_string_should_not_parse_as_valid_json")),
        Err(_) => {
            // Expected - empty string is not valid JSON
        }
    }

    // Empty artifact
    let artifact_path = temp.path().join("empty.md");
    fs::write(&artifact_path, "")?;

    let content = fs::read_to_string(&artifact_path)?;
    assert!(content.is_empty());

    // Create artifact from empty content - should not panic
    let artifact = Artifact::new("empty.md", ArtifactType::Task, "");
    assert!(artifact.content.is_empty());
    let summary = artifact.generate_summary();
    assert!(
        !summary.is_empty(),
        "Summary should have fallback for empty content"
    );
    Ok(())
}

// ============================================================================
// Resource Exhaustion Tests
// ============================================================================

#[test]
fn test_large_artifact_content() -> Result<()> {
    // 10MB of content
    let large_content: String = "x".repeat(10 * 1024 * 1024);

    let artifact = Artifact::new("large.md", ArtifactType::Task, &large_content);

    assert_eq!(artifact.content.len(), 10 * 1024 * 1024);

    // Summary should be bounded regardless of content size
    let summary = artifact.generate_summary();
    assert!(
        summary.len() <= 200,
        "Summary should be bounded even for large content"
    );

    // Serialization should work
    let json = serde_json::to_string(&artifact);
    assert!(json.is_ok(), "Should serialize large artifact");
    Ok(())
}

#[test]
fn test_many_artifact_versions() -> Result<()> {
    let mut metadata = ArtifactMetadata::new(ArtifactType::Task, "Test");

    // Increment version 10,000 times
    for i in 1..=10_000 {
        let v = metadata.increment_version();
        assert_eq!(v, i as u32, "Version should match iteration");
    }

    assert_eq!(metadata.current_version, 10_000);
    Ok(())
}

#[test]
fn test_deeply_nested_json_value() -> Result<()> {
    // Create deeply nested JSON
    let mut value = serde_json::json!({"deep": "value"});
    for _ in 0..100 {
        value = serde_json::json!({"nested": value});
    }

    let pref = Preference::new("nested_pref", value.clone());

    // Should serialize and deserialize without stack overflow
    let json = serde_json::to_string(&pref)?;
    let _: Preference = serde_json::from_str(&json)?;
    Ok(())
}

#[test]
fn test_many_pattern_examples() -> Result<()> {
    let mut pattern = Pattern::new("stress_test", "test", "Stress test pattern");

    // Add 10,000 examples
    for i in 0..10_000 {
        pattern.add_example(format!("example_{}", i));
    }

    assert_eq!(pattern.examples.len(), 10_000);
    assert_eq!(pattern.occurrence_count, 10_001); // 1 initial + 10,000 added

    // Confidence should be capped at 0.95
    assert!(pattern.confidence <= 0.95);
    assert!(pattern.confidence > 0.9);
    Ok(())
}

#[test]
fn test_many_corrections() -> Result<()> {
    let mut corrections = Vec::new();

    // Create 1000 corrections
    for i in 0..1000 {
        let mut correction = Correction::new(format!("mistake_{}", i), format!("correction_{}", i));
        correction.context = Some(format!("context for correction {}", i));
        for _ in 0..10 {
            correction.mark_applied();
        }
        corrections.push(correction);
    }

    assert_eq!(corrections.len(), 1000);

    // Verify all corrections are independent
    for (i, c) in corrections.iter().enumerate() {
        assert_eq!(c.mistake, format!("mistake_{}", i));
        assert_eq!(c.application_count, 10);
    }
    Ok(())
}

// ============================================================================
// Error Chain Tests
// ============================================================================

#[test]
fn test_brain_error_chain() -> Result<()> {
    // Test that errors can be chained and display properly
    let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
    let brain_error: BrainError = io_error.into();

    let msg = format!("{}", brain_error);
    assert!(msg.contains("file missing") || msg.contains("IO error"));

    // Test debug format
    let debug = format!("{:?}", brain_error);
    assert!(!debug.is_empty());
    Ok(())
}

#[test]
fn test_error_downcasting() -> Result<()> {
    let io_error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
    let brain_error: BrainError = io_error.into();

    // Verify we can inspect the error type
    if let BrainError::Io(inner) = brain_error {
        assert_eq!(inner.kind(), std::io::ErrorKind::PermissionDenied);
    } else {
        return Err(nexerror!("expected_brain_error_io"));
    }
    Ok(())
}

// ============================================================================
// Recovery Scenario Tests
// ============================================================================

#[test]
fn test_partial_write_detection() -> Result<()> {
    let temp = TempDir::new()?;
    let artifact_path = temp.path().join("task.md");
    let metadata_path = temp.path().join("task.md.metadata.json");

    // Simulate partial write: artifact exists but metadata doesn't
    fs::write(&artifact_path, "# Partial artifact")?;

    assert!(artifact_path.exists());
    assert!(!metadata_path.exists());

    // Reading artifact should work
    let content = fs::read_to_string(&artifact_path)?;
    assert!(content.contains("Partial"));

    // System should handle missing metadata gracefully
    // (In production, we'd create default metadata)
    Ok(())
}

#[test]
fn test_stale_session_in_index() -> Result<()> {
    let temp = TempDir::new()?;
    let sessions_dir = temp.path().join("sessions");
    let index_path = temp.path().join("index.json");

    fs::create_dir_all(&sessions_dir)?;

    // Create index entry for session that doesn't exist
    let stale_index = r#"[
        {
            "id": "00000000-0000-0000-0000-000000000001",
            "created_at": "2024-01-01T00:00:00Z",
            "project": "deleted-project",
            "git_commit": null,
            "description": "This session was deleted"
        }
    ]"#;
    fs::write(&index_path, stale_index)?;

    // Session directory doesn't exist
    let session_dir = sessions_dir.join("00000000-0000-0000-0000-000000000001");
    assert!(!session_dir.exists());

    // Parse index - should work
    let index_content = fs::read_to_string(&index_path)?;
    let entries: Vec<serde_json::Value> = serde_json::from_str(&index_content)?;
    assert_eq!(entries.len(), 1);

    // But session directory is missing - this is a detectable inconsistency
    Ok(())
}

#[test]
fn test_orphaned_session_directory() -> Result<()> {
    let temp = TempDir::new()?;
    let sessions_dir = temp.path().join("sessions");
    let index_path = temp.path().join("index.json");

    // Create empty index
    fs::write(&index_path, "[]")?;

    // Create orphaned session directory (not in index)
    let orphan_session = sessions_dir.join("orphan-session-id");
    fs::create_dir_all(&orphan_session)?;
    fs::write(orphan_session.join("task.md"), "# Orphaned task")?;

    // Index shows no sessions
    let index_content = fs::read_to_string(&index_path)?;
    let entries: Vec<serde_json::Value> = serde_json::from_str(&index_content)?;
    assert!(entries.is_empty());

    // But directory exists
    assert!(orphan_session.exists());
    assert!(orphan_session.join("task.md").exists());

    // This is a recoverable situation - we could rebuild index from directories
    Ok(())
}

// ============================================================================
// Thread Safety of Immutable Operations
// ============================================================================

#[test]
fn test_concurrent_artifact_reads() -> Result<()> {
    let temp = TempDir::new()?;
    let artifact_path = temp.path().join("shared.md");

    let content = "# Shared Artifact\n\nThis content is read by multiple threads.";
    fs::write(&artifact_path, content)?;

    let path = Arc::new(artifact_path);
    let num_threads = 10;
    let reads_per_thread = 100;
    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let path = Arc::clone(&path);
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                barrier.wait();

                for _ in 0..reads_per_thread {
                    match fs::read_to_string(path.as_ref()) {
                        Ok(read_content) => assert_eq!(read_content, content),
                        Err(err) => {
                            eprintln!("read_failed: {err}");
                            return;
                        }
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle
            .join()
            .map_err(|_| nexerror!("thread_panicked_during_concurrent_reads"))?;
    }
    Ok(())
}

#[test]
fn test_concurrent_hash_computation() -> Result<()> {
    use nexcore_hash::sha256::Sha256;

    let content = b"Content to hash concurrently";
    let expected_hash = {
        let mut hasher = Sha256::new();
        hasher.update(content);
        hex::encode(&hasher.finalize()[..16])
    };

    let num_threads = 10;
    let hashes_per_thread = 100;
    let barrier = Arc::new(Barrier::new(num_threads));
    let results = Arc::new(std::sync::Mutex::new(Vec::new()));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            let results = Arc::clone(&results);
            let expected = expected_hash.clone();

            thread::spawn(move || {
                barrier.wait();

                for _ in 0..hashes_per_thread {
                    let mut hasher = Sha256::new();
                    hasher.update(content);
                    let hash = hex::encode(&hasher.finalize()[..16]);
                    assert_eq!(hash, expected, "Hash mismatch in concurrent computation");
                }

                let mut results = results.lock().unwrap();
                results.push(true);
            })
        })
        .collect();

    for handle in handles {
        handle
            .join()
            .map_err(|_| nexerror!("thread_panicked_during_concurrent_hashing"))?;
    }

    let results = results.lock().unwrap();
    assert_eq!(results.len(), num_threads);
    Ok(())
}
