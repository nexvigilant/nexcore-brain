//! Configuration for the Brain System
//!
//! Provides configurable timeouts and thresholds for CTVP Phase 1 safety.

use std::time::Duration;

/// Brain system configuration
#[derive(Debug, Clone)]
pub struct BrainConfig {
    /// Timeout for file I/O operations
    pub io_timeout: Duration,

    /// Timeout for acquiring locks on shared resources (e.g., index.json)
    pub lock_timeout: Duration,

    /// Maximum number of retry attempts for transient failures
    pub max_retries: u32,

    /// Whether to enable graceful degradation when brain is unavailable
    pub graceful_degradation: bool,

    /// Whether to attempt automatic recovery from corrupted state
    pub auto_recovery: bool,

    /// Maximum artifact size in bytes (10MB default)
    pub max_artifact_size: usize,

    /// Maximum number of sessions to keep in index
    pub max_sessions: usize,

    /// Maximum number of versions to keep per artifact
    pub max_versions_per_artifact: u32,
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            io_timeout: Duration::from_secs(5),
            lock_timeout: Duration::from_secs(2),
            max_retries: 3,
            graceful_degradation: true,
            auto_recovery: true,
            max_artifact_size: 10 * 1024 * 1024, // 10MB
            max_sessions: 1000,
            max_versions_per_artifact: 100,
        }
    }
}

impl BrainConfig {
    /// Create a new configuration with default values
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration optimized for testing (shorter timeouts)
    #[must_use]
    pub fn for_testing() -> Self {
        Self {
            io_timeout: Duration::from_millis(100),
            lock_timeout: Duration::from_millis(50),
            max_retries: 1,
            graceful_degradation: true,
            auto_recovery: true,
            max_artifact_size: 1024 * 1024, // 1MB for tests
            max_sessions: 100,
            max_versions_per_artifact: 10,
        }
    }

    /// Create a configuration optimized for high-reliability (longer timeouts)
    #[must_use]
    pub fn high_reliability() -> Self {
        Self {
            io_timeout: Duration::from_secs(30),
            lock_timeout: Duration::from_secs(10),
            max_retries: 5,
            graceful_degradation: true,
            auto_recovery: true,
            max_artifact_size: 100 * 1024 * 1024, // 100MB
            max_sessions: 10000,
            max_versions_per_artifact: 1000,
        }
    }

    /// Set the I/O timeout
    #[must_use]
    pub fn with_io_timeout(mut self, timeout: Duration) -> Self {
        self.io_timeout = timeout;
        self
    }

    /// Set the lock timeout
    #[must_use]
    pub fn with_lock_timeout(mut self, timeout: Duration) -> Self {
        self.lock_timeout = timeout;
        self
    }

    /// Set the maximum retry count
    #[must_use]
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Enable or disable graceful degradation
    #[must_use]
    pub fn with_graceful_degradation(mut self, enabled: bool) -> Self {
        self.graceful_degradation = enabled;
        self
    }

    /// Enable or disable auto-recovery
    #[must_use]
    pub fn with_auto_recovery(mut self, enabled: bool) -> Self {
        self.auto_recovery = enabled;
        self
    }

    /// Check if an artifact size is within limits
    #[must_use]
    pub fn is_artifact_size_valid(&self, size: usize) -> bool {
        size <= self.max_artifact_size
    }

    /// Check if session count is within limits
    #[must_use]
    pub fn is_session_count_valid(&self, count: usize) -> bool {
        count <= self.max_sessions
    }

    /// Check if version count is within limits
    #[must_use]
    pub fn is_version_count_valid(&self, count: u32) -> bool {
        count <= self.max_versions_per_artifact
    }
}

// Global configuration (thread-local for isolation)
//
// In production, this would typically be set once at startup.
// For testing, each test can set its own configuration.
thread_local! {
    static CONFIG: std::cell::RefCell<BrainConfig> = std::cell::RefCell::new(BrainConfig::default());
}

/// Get a copy of the current configuration
#[must_use]
pub fn get_config() -> BrainConfig {
    CONFIG.with(|c| c.borrow().clone())
}

/// Set the configuration (returns the old configuration)
pub fn set_config(config: BrainConfig) -> BrainConfig {
    CONFIG.with(|c| c.replace(config))
}

/// Reset configuration to defaults
pub fn reset_config() {
    CONFIG.with(|c| *c.borrow_mut() = BrainConfig::default());
}

/// Execute a closure with a temporary configuration
pub fn with_config<F, R>(config: BrainConfig, f: F) -> R
where
    F: FnOnce() -> R,
{
    let old = set_config(config);
    let result = f();
    set_config(old);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BrainConfig::default();

        assert_eq!(config.io_timeout, Duration::from_secs(5));
        assert_eq!(config.lock_timeout, Duration::from_secs(2));
        assert_eq!(config.max_retries, 3);
        assert!(config.graceful_degradation);
        assert!(config.auto_recovery);
        assert_eq!(config.max_artifact_size, 10 * 1024 * 1024);
    }

    #[test]
    fn test_testing_config() {
        let config = BrainConfig::for_testing();

        assert_eq!(config.io_timeout, Duration::from_millis(100));
        assert_eq!(config.lock_timeout, Duration::from_millis(50));
        assert_eq!(config.max_retries, 1);
    }

    #[test]
    fn test_high_reliability_config() {
        let config = BrainConfig::high_reliability();

        assert_eq!(config.io_timeout, Duration::from_secs(30));
        assert_eq!(config.lock_timeout, Duration::from_secs(10));
        assert_eq!(config.max_retries, 5);
    }

    #[test]
    fn test_builder_pattern() {
        let config = BrainConfig::new()
            .with_io_timeout(Duration::from_secs(10))
            .with_lock_timeout(Duration::from_secs(5))
            .with_max_retries(10)
            .with_graceful_degradation(false)
            .with_auto_recovery(false);

        assert_eq!(config.io_timeout, Duration::from_secs(10));
        assert_eq!(config.lock_timeout, Duration::from_secs(5));
        assert_eq!(config.max_retries, 10);
        assert!(!config.graceful_degradation);
        assert!(!config.auto_recovery);
    }

    #[test]
    fn test_artifact_size_validation() {
        let config = BrainConfig::default();

        assert!(config.is_artifact_size_valid(1024)); // 1KB
        assert!(config.is_artifact_size_valid(10 * 1024 * 1024)); // 10MB (exact limit)
        assert!(!config.is_artifact_size_valid(10 * 1024 * 1024 + 1)); // Over limit
    }

    #[test]
    fn test_session_count_validation() {
        let config = BrainConfig::default();

        assert!(config.is_session_count_valid(100));
        assert!(config.is_session_count_valid(1000)); // Exact limit
        assert!(!config.is_session_count_valid(1001)); // Over limit
    }

    #[test]
    fn test_version_count_validation() {
        let config = BrainConfig::default();

        assert!(config.is_version_count_valid(50));
        assert!(config.is_version_count_valid(100)); // Exact limit
        assert!(!config.is_version_count_valid(101)); // Over limit
    }

    #[test]
    fn test_global_config() {
        // Reset to ensure clean state
        reset_config();

        let default = get_config();
        assert_eq!(default.io_timeout, Duration::from_secs(5));

        // Set custom config
        let custom = BrainConfig::for_testing();
        let old = set_config(custom);
        assert_eq!(old.io_timeout, Duration::from_secs(5));

        let current = get_config();
        assert_eq!(current.io_timeout, Duration::from_millis(100));

        // Reset
        reset_config();
        let reset = get_config();
        assert_eq!(reset.io_timeout, Duration::from_secs(5));
    }

    #[test]
    fn test_with_config_scope() {
        reset_config();

        let result = with_config(BrainConfig::for_testing(), || {
            let config = get_config();
            assert_eq!(config.io_timeout, Duration::from_millis(100));
            42
        });

        assert_eq!(result, 42);

        // Config should be restored
        let config = get_config();
        assert_eq!(config.io_timeout, Duration::from_secs(5));
    }
}
