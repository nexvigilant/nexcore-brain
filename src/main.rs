//! Brain CLI - Antigravity-style working memory for Claude Code
//!
//! Usage:
//!   brain session new                    # Create new session
//!   brain session list                   # List all sessions
//!   brain session current                # Get current (latest) session ID
//!   brain session load <id>              # Load session by ID
//!   brain artifact save <file>           # Save artifact to current session
//!   brain artifact resolve <name>        # Create resolved snapshot
//!   brain artifact get <name>            # Get current artifact
//!   brain artifact get <name> --version N  # Get specific version
//!   brain artifact list                  # List artifacts in session
//!   brain artifact diff <name> <v1> <v2> # Diff two versions
//!   brain track <file>                   # Track a file for change detection
//!   brain changed <file>                 # Check if file changed
//!   brain original <file>                # Get original content
//!   brain autopsy run <id>               # Autopsy a single session
//!   brain autopsy backfill               # Retroactive autopsy on all sessions
//!   brain autopsy report                 # Aggregate intelligence summary
//!   brain autopsy directive <id>         # Longitudinal directive report
//!   brain init                           # Initialize brain directories

use clap::{Parser, Subcommand};
use nexcore_brain::{
    Artifact, ArtifactType, BrainSession, CodeTracker, ImplicitKnowledge, attempt_recovery,
    check_brain_availability, check_index_health, detect_partial_writes, initialize_directories,
    rebuild_index_from_sessions, repair_partial_writes,
};
use nexcore_db::pool::DbPool;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "brain")]
#[command(about = "Antigravity-style working memory for Claude Code")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize brain directories
    Init,

    /// Session management
    Session {
        #[command(subcommand)]
        action: SessionAction,
    },

    /// Artifact management
    Artifact {
        #[command(subcommand)]
        action: ArtifactAction,
    },

    /// Track a file for change detection
    Track {
        /// File to track
        file: PathBuf,

        /// Project name (defaults to current directory name)
        #[arg(short, long)]
        project: Option<String>,
    },

    /// Check if a tracked file has changed
    Changed {
        /// File to check
        file: PathBuf,

        /// Project name
        #[arg(short, long)]
        project: Option<String>,
    },

    /// Get original content of a tracked file
    Original {
        /// File to get original content for
        file: PathBuf,

        /// Project name
        #[arg(short, long)]
        project: Option<String>,
    },

    /// Implicit knowledge operations
    Implicit {
        #[command(subcommand)]
        action: ImplicitAction,
    },

    /// Recovery and health check operations
    Recovery {
        #[command(subcommand)]
        action: RecoveryAction,
    },

    /// Directive Autopsy Engine — structured session post-mortem
    Autopsy {
        #[command(subcommand)]
        action: AutopsyAction,
    },
}

#[derive(Subcommand)]
enum SessionAction {
    /// Create a new session
    New {
        /// Project name
        #[arg(short, long)]
        project: Option<String>,

        /// Git commit hash
        #[arg(short, long)]
        commit: Option<String>,

        /// Description
        #[arg(short, long)]
        description: Option<String>,

        /// Inject session ID into context (for hooks)
        #[arg(long)]
        inject_context: bool,
    },

    /// List all sessions
    List,

    /// Get the current (latest) session ID
    Current,

    /// Load a session by ID
    Load {
        /// Session ID
        id: String,
    },

    /// Import sessions from Antigravity brain
    ImportAntigravity,
}

#[derive(Subcommand)]
enum ArtifactAction {
    /// Save an artifact from a file
    Save {
        /// File to save as artifact
        file: PathBuf,

        /// Session ID (defaults to latest)
        #[arg(short, long)]
        session: Option<String>,

        /// Artifact type (task, plan, walkthrough, review, research, decision, custom)
        #[arg(short = 't', long)]
        artifact_type: Option<String>,
    },

    /// Resolve an artifact (create immutable snapshot)
    Resolve {
        /// Artifact name
        name: String,

        /// Session ID (defaults to latest)
        #[arg(short, long)]
        session: Option<String>,
    },

    /// Get an artifact's content
    Get {
        /// Artifact name
        name: String,

        /// Session ID (defaults to latest)
        #[arg(short, long)]
        session: Option<String>,

        /// Specific version number
        #[arg(short, long)]
        version: Option<u32>,
    },

    /// List artifacts in a session
    List {
        /// Session ID (defaults to latest)
        #[arg(short, long)]
        session: Option<String>,
    },

    /// List versions of an artifact
    Versions {
        /// Artifact name
        name: String,

        /// Session ID (defaults to latest)
        #[arg(short, long)]
        session: Option<String>,
    },

    /// Diff two versions of an artifact
    Diff {
        /// Artifact name
        name: String,

        /// First version
        v1: u32,

        /// Second version
        v2: u32,

        /// Session ID (defaults to latest)
        #[arg(short, long)]
        session: Option<String>,
    },
}

#[derive(Subcommand)]
enum ImplicitAction {
    /// Get a preference
    GetPref {
        /// Preference key
        key: String,
    },

    /// Set a preference
    SetPref {
        /// Preference key
        key: String,

        /// Preference value (JSON)
        value: String,
    },

    /// List all preferences
    ListPrefs,

    /// Get statistics
    Stats,
}

#[derive(Subcommand)]
enum RecoveryAction {
    /// Check brain health status
    Check,

    /// Repair partial writes (create missing metadata)
    Repair {
        /// Session ID to repair (defaults to latest)
        #[arg(short, long)]
        session: Option<String>,
    },

    /// Rebuild index from session directories
    RebuildIndex,

    /// Attempt automatic recovery
    Auto,
}

#[derive(Subcommand)]
enum AutopsyAction {
    /// Autopsy a single session
    Run {
        /// Session ID to autopsy
        id: String,
    },

    /// Retroactive autopsy on all un-autopsied sessions
    Backfill,

    /// Aggregate intelligence summary across all autopsied sessions
    Report,

    /// Longitudinal directive report
    Directive {
        /// Directive ID (e.g., "D008")
        id: String,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> nexcore_brain::error::Result<()> {
    match cli.command {
        Commands::Init => {
            initialize_directories()?;
            println!("Brain directories initialized");
            Ok(())
        }
        Commands::Session { action } => handle_session_action(action),
        Commands::Artifact { action } => handle_artifact_action(action),
        Commands::Track { file, project } => handle_track(file, project),
        Commands::Changed { file, project } => handle_changed(file, project),
        Commands::Original { file, project } => handle_original(file, project),
        Commands::Implicit { action } => handle_implicit_action(action),
        Commands::Recovery { action } => handle_recovery_action(action),
        Commands::Autopsy { action } => handle_autopsy_action(action),
    }
}

fn handle_session_action(action: SessionAction) -> nexcore_brain::error::Result<()> {
    match action {
        SessionAction::New {
            project,
            commit,
            description,
            inject_context,
        } => create_new_session(project, commit, description, inject_context),
        SessionAction::List => list_sessions(),
        SessionAction::Current => get_current_session(),
        SessionAction::Load { id } => load_session_by_id(id),
        SessionAction::ImportAntigravity => import_antigravity_sessions(),
    }
}

fn import_antigravity_sessions() -> nexcore_brain::error::Result<()> {
    println!("Scanning Antigravity brain for sessions...");
    let count = BrainSession::import_from_antigravity()?;
    if count > 0 {
        println!("Successfully imported {count} session(s)");
    } else {
        println!("No new sessions found in Antigravity brain");
    }
    Ok(())
}

fn create_new_session(
    project: Option<String>,
    commit: Option<String>,
    description: Option<String>,
    inject_context: bool,
) -> nexcore_brain::error::Result<()> {
    let session = BrainSession::create_with_options(project, commit, description)?;
    let id = session.id;

    if inject_context {
        println!("{{\"brain_session_id\": \"{id}\"}}");
    } else {
        println!("{id}");
    }
    Ok(())
}

fn list_sessions() -> nexcore_brain::error::Result<()> {
    let sessions = BrainSession::list_all()?;
    if sessions.is_empty() {
        println!("No sessions found");
        return Ok(());
    }

    for session in sessions {
        let project = session.project.as_deref().unwrap_or("-");
        let desc = session.description.as_deref().unwrap_or("");
        println!(
            "{}\t{}\t{}\t{}",
            session.id,
            session
                .created_at
                .format("%Y-%m-%d %H:%M")
                .unwrap_or_default(),
            project,
            desc
        );
    }
    Ok(())
}

fn get_current_session() -> nexcore_brain::error::Result<()> {
    let session = BrainSession::load_latest()?;
    println!("{}", session.id);
    Ok(())
}

fn load_session_by_id(id: String) -> nexcore_brain::error::Result<()> {
    let session = BrainSession::load_str(&id)?;
    println!("Loaded session: {}", session.id);
    println!("Created: {}", session.created_at);
    if let Some(p) = &session.project {
        println!("Project: {p}");
    }
    if let Some(c) = &session.git_commit {
        println!("Git commit: {c}");
    }
    Ok(())
}

fn handle_artifact_action(action: ArtifactAction) -> nexcore_brain::error::Result<()> {
    match action {
        ArtifactAction::Save {
            file,
            session,
            artifact_type,
        } => save_artifact_cmd(file, session, artifact_type),
        ArtifactAction::Resolve { name, session } => resolve_artifact_cmd(name, session),
        ArtifactAction::Get {
            name,
            session,
            version,
        } => get_artifact_cmd(name, session, version),
        ArtifactAction::List { session } => list_artifacts_cmd(session),
        ArtifactAction::Versions { name, session } => list_artifact_versions_cmd(name, session),
        ArtifactAction::Diff {
            name,
            v1,
            v2,
            session,
        } => diff_artifact_versions_cmd(name, v1, v2, session),
    }
}

fn save_artifact_cmd(
    file: PathBuf,
    session: Option<String>,
    artifact_type: Option<String>,
) -> nexcore_brain::error::Result<()> {
    let session = get_session(session)?;
    let content = std::fs::read_to_string(&file)?;
    let name = file
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("artifact");

    let art_type = artifact_type
        .map(|t| t.parse::<ArtifactType>())
        .transpose()?
        .unwrap_or_else(|| ArtifactType::from_filename(name));

    let artifact = Artifact::new(name, art_type, content);
    session.save_artifact(&artifact)?;
    println!("Saved artifact: {name}");
    Ok(())
}

fn resolve_artifact_cmd(name: String, session: Option<String>) -> nexcore_brain::error::Result<()> {
    let session = get_session(session)?;
    let version = session.resolve_artifact(&name)?;
    println!("Resolved {name} to version {version}");
    Ok(())
}

fn get_artifact_cmd(
    name: String,
    session: Option<String>,
    version: Option<u32>,
) -> nexcore_brain::error::Result<()> {
    let session = get_session(session)?;
    let artifact = session.get_artifact(&name, version)?;
    print!("{}", artifact.content);
    Ok(())
}

fn list_artifacts_cmd(session: Option<String>) -> nexcore_brain::error::Result<()> {
    let session = get_session(session)?;
    let artifacts = session.list_artifacts()?;
    if artifacts.is_empty() {
        println!("No artifacts in session");
    } else {
        for name in artifacts {
            println!("{name}");
        }
    }
    Ok(())
}

fn list_artifact_versions_cmd(
    name: String,
    session: Option<String>,
) -> nexcore_brain::error::Result<()> {
    let session = get_session(session)?;
    let versions = session.list_versions(&name)?;
    if versions.is_empty() {
        println!("No resolved versions for {name}");
    } else {
        for v in versions {
            println!("{v}");
        }
    }
    Ok(())
}

fn diff_artifact_versions_cmd(
    name: String,
    v1: u32,
    v2: u32,
    session: Option<String>,
) -> nexcore_brain::error::Result<()> {
    let session = get_session(session)?;
    let diff = session.diff_versions(&name, v1, v2)?;
    print!("{diff}");
    Ok(())
}

fn handle_track(file: PathBuf, project: Option<String>) -> nexcore_brain::error::Result<()> {
    let project = project.unwrap_or_else(get_default_project);
    let mut tracker = CodeTracker::new(project, None)?;
    let tracked = tracker.track_file(&file)?;
    println!(
        "Tracked {} (hash: {})",
        file.display(),
        &tracked.content_hash[..16]
    );
    Ok(())
}

fn handle_changed(file: PathBuf, project: Option<String>) -> nexcore_brain::error::Result<()> {
    let project = project.unwrap_or_else(get_default_project);
    let tracker = CodeTracker::load(&project)?;
    let changed = tracker.has_changed(&file)?;
    if changed {
        println!("CHANGED");
        std::process::exit(1);
    } else {
        println!("UNCHANGED");
    }
    Ok(())
}

fn handle_original(file: PathBuf, project: Option<String>) -> nexcore_brain::error::Result<()> {
    let project = project.unwrap_or_else(get_default_project);
    let tracker = CodeTracker::load(&project)?;
    let content = tracker.get_original(&file)?;
    print!("{content}");
    Ok(())
}

fn handle_implicit_action(action: ImplicitAction) -> nexcore_brain::error::Result<()> {
    match action {
        ImplicitAction::GetPref { key } => get_preference_cmd(key),
        ImplicitAction::SetPref { key, value } => set_preference_cmd(key, value),
        ImplicitAction::ListPrefs => list_preferences_cmd(),
        ImplicitAction::Stats => get_implicit_stats_cmd(),
    }
}

fn get_preference_cmd(key: String) -> nexcore_brain::error::Result<()> {
    let knowledge = ImplicitKnowledge::load()?;
    if let Some(pref) = knowledge.get_preference(&key) {
        println!("{}", serde_json::to_string_pretty(&pref.value)?);
    } else {
        eprintln!("Preference not found: {key}");
        std::process::exit(1);
    }
    Ok(())
}

fn set_preference_cmd(key: String, value: String) -> nexcore_brain::error::Result<()> {
    let mut knowledge = ImplicitKnowledge::load()?;
    let value: serde_json::Value = serde_json::from_str(&value)?;
    knowledge.set_preference_value(&key, value);
    knowledge.save()?;
    println!("Set preference: {key}");
    Ok(())
}

fn list_preferences_cmd() -> nexcore_brain::error::Result<()> {
    let knowledge = ImplicitKnowledge::load()?;
    let prefs = knowledge.list_preferences();
    if prefs.is_empty() {
        println!("No preferences set");
        return Ok(());
    }

    for pref in prefs {
        println!(
            "{}\t{:.2}\t{}",
            pref.key,
            pref.confidence,
            serde_json::to_string(&pref.value)?
        );
    }
    Ok(())
}

fn get_implicit_stats_cmd() -> nexcore_brain::error::Result<()> {
    let knowledge = ImplicitKnowledge::load()?;
    let stats = knowledge.stats();
    println!("{}", serde_json::to_string_pretty(&stats)?);
    Ok(())
}

fn handle_recovery_action(action: RecoveryAction) -> nexcore_brain::error::Result<()> {
    match action {
        RecoveryAction::Check => check_recovery_cmd(),
        RecoveryAction::Repair { session } => repair_recovery_cmd(session),
        RecoveryAction::RebuildIndex => rebuild_index_cmd(),
        RecoveryAction::Auto => auto_recovery_cmd(),
    }
}

fn check_recovery_cmd() -> nexcore_brain::error::Result<()> {
    println!("Brain Health Check");
    println!("==================");

    if let Some(reason) = check_brain_availability() {
        println!("Status: DEGRADED");
        println!("Reason: {reason}");
    } else {
        println!("Status: HEALTHY");
    }

    print!("Index: ");
    match check_index_health() {
        Some(reason) => println!("CORRUPTED - {reason}"),
        None => println!("OK"),
    }

    if let Ok(session) = BrainSession::load_latest() {
        print_partial_writes(&session);
    }
    Ok(())
}

fn print_partial_writes(session: &BrainSession) {
    let partials = detect_partial_writes(&session.dir());
    if partials.is_empty() {
        println!("Partial writes: None detected");
    } else {
        println!(
            "Partial writes: {} artifact(s) missing metadata",
            partials.len()
        );
        for name in &partials {
            println!("  - {name}");
        }
    }
}

fn repair_recovery_cmd(session: Option<String>) -> nexcore_brain::error::Result<()> {
    let session = get_session(session)?;
    println!("Repairing partial writes in session: {}", session.id);

    let result = repair_partial_writes(&session.dir())?;
    if result.success {
        println!("Repaired {} artifact(s)", result.recovered_count);
        for detail in &result.details {
            println!("  {detail}");
        }
    } else {
        println!("No repairs needed");
    }

    if !result.warnings.is_empty() {
        println!("Warnings:");
        for warning in &result.warnings {
            println!("  {warning}");
        }
    }
    Ok(())
}

fn rebuild_index_cmd() -> nexcore_brain::error::Result<()> {
    println!("Rebuilding index from session directories...");

    let result = rebuild_index_from_sessions()?;
    if result.success {
        println!("Rebuilt index with {} session(s)", result.recovered_count);
        for detail in &result.details {
            println!("  {detail}");
        }
    } else {
        println!("Index rebuild failed");
    }

    if !result.warnings.is_empty() {
        println!("Warnings:");
        for warning in &result.warnings {
            println!("  {warning}");
        }
    }
    Ok(())
}

fn auto_recovery_cmd() -> nexcore_brain::error::Result<()> {
    println!("Attempting automatic recovery...");

    let result = attempt_recovery()?;
    if result.success {
        if result.recovered_count > 0 {
            println!(
                "Recovery successful: {} item(s) recovered",
                result.recovered_count
            );
        } else {
            println!("No recovery needed - brain is healthy");
        }
        for detail in &result.details {
            println!("  {detail}");
        }
    } else {
        println!("Recovery not performed");
        for warning in &result.warnings {
            println!("  {warning}");
        }
    }
    Ok(())
}

// ── Autopsy ──────────────────────────────────────────────────────────────

fn handle_autopsy_action(action: AutopsyAction) -> nexcore_brain::error::Result<()> {
    match action {
        AutopsyAction::Run { id } => autopsy_run_cmd(id),
        AutopsyAction::Backfill => autopsy_backfill_cmd(),
        AutopsyAction::Report => autopsy_report_cmd(),
        AutopsyAction::Directive { id } => autopsy_directive_cmd(id),
    }
}

/// Helper to open the brain database pool.
fn open_db() -> nexcore_brain::error::Result<DbPool> {
    DbPool::open_default().map_err(|e| {
        nexcore_brain::error::BrainError::Other(
            ["Failed to open brain database: ", &e.to_string()].concat(),
        )
    })
}

fn autopsy_run_cmd(session_id: String) -> nexcore_brain::error::Result<()> {
    let pool = open_db()?;
    pool.with_conn(|conn| {
        let (row, newly_inserted) =
            nexcore_db::autopsy_engine::run_retroactive_single(conn, &session_id)?;

        if newly_inserted {
            println!("Autopsy created for session: {}", row.session_id);
        } else {
            println!("Autopsy already exists for session: {}", row.session_id);
        }

        print_autopsy_summary(&row);
        Ok(())
    })
    .map_err(|e| {
        nexcore_brain::error::BrainError::Other(["Autopsy run failed: ", &e.to_string()].concat())
    })
}

fn autopsy_backfill_cmd() -> nexcore_brain::error::Result<()> {
    let pool = open_db()?;
    pool.with_conn(|conn| {
        let result = nexcore_db::autopsy_engine::run_retroactive(conn)?;

        println!("Autopsy Backfill Complete");
        println!("========================");
        println!("Sessions processed: {}", result.sessions_processed);
        println!("Records inserted:   {}", result.records_inserted);
        println!("Records skipped:    {}", result.records_skipped);

        if !result.errors.is_empty() {
            println!("\nWarnings ({}):", result.errors.len());
            for err in &result.errors {
                println!("  - {err}");
            }
        }

        Ok(())
    })
    .map_err(|e| {
        nexcore_brain::error::BrainError::Other(
            ["Autopsy backfill failed: ", &e.to_string()].concat(),
        )
    })
}

fn autopsy_report_cmd() -> nexcore_brain::error::Result<()> {
    let pool = open_db()?;
    pool.with_conn(|conn| {
        let rows = nexcore_db::autopsy::list_all(conn)?;

        if rows.is_empty() {
            println!("No autopsy records found. Run `brain autopsy backfill` first.");
            return Ok(());
        }

        let total = rows.len();

        // PDP pass rates (exclude not_evaluated)
        let (mut g1_pass, mut g1_total) = (0usize, 0usize);
        let (mut g2_pass, mut g2_total) = (0usize, 0usize);
        let (mut g3_pass, mut g3_total) = (0usize, 0usize);

        // Verdict distribution
        let (mut fully, mut partially, mut not_dem) = (0usize, 0usize, 0usize);

        // Root cause totals
        let (mut rc_prop, mut rc_sowhat, mut rc_why, mut rc_hook) = (0i64, 0i64, 0i64, 0i64);

        // Aggregates
        let mut total_lessons: i64 = 0;
        let mut total_patterns: i64 = 0;
        let mut rho_sum: f64 = 0.0;
        let mut rho_count: usize = 0;

        for row in &rows {
            // PDP gates
            if row.g1_proposition != "not_evaluated" {
                g1_total = g1_total.saturating_add(1);
                if row.g1_proposition == "pass" {
                    g1_pass = g1_pass.saturating_add(1);
                }
            }
            if row.g2_specificity != "not_evaluated" {
                g2_total = g2_total.saturating_add(1);
                if row.g2_specificity == "pass" {
                    g2_pass = g2_pass.saturating_add(1);
                }
            }
            if row.g3_singularity != "not_evaluated" {
                g3_total = g3_total.saturating_add(1);
                if row.g3_singularity == "pass" {
                    g3_pass = g3_pass.saturating_add(1);
                }
            }

            // Verdict
            match row.outcome_verdict.as_deref() {
                Some("fully_demonstrated") => fully = fully.saturating_add(1),
                Some("partially_demonstrated") => partially = partially.saturating_add(1),
                Some("not_demonstrated") => not_dem = not_dem.saturating_add(1),
                _ => {}
            }

            // Root causes
            rc_prop = rc_prop.saturating_add(row.rc_pdp_proposition);
            rc_sowhat = rc_sowhat.saturating_add(row.rc_pdp_so_what);
            rc_why = rc_why.saturating_add(row.rc_pdp_why);
            rc_hook = rc_hook.saturating_add(row.rc_hook_gap);

            // Lessons / patterns
            total_lessons = total_lessons.saturating_add(row.lesson_count);
            total_patterns = total_patterns.saturating_add(row.pattern_count);

            // Rho
            if let Some(rho) = row.rho_session {
                rho_sum += rho;
                rho_count = rho_count.saturating_add(1);
            }
        }

        // Print report
        println!("Autopsy Intelligence Report");
        println!("===========================");
        println!();
        println!("Sessions autopsied: {total}");
        println!();

        // PDP pass rates
        println!("PDP Foundation Gate Pass Rates:");
        print_pass_rate("  G1 Proposition", g1_pass, g1_total);
        print_pass_rate("  G2 Specificity", g2_pass, g2_total);
        print_pass_rate("  G3 Singularity", g3_pass, g3_total);
        println!();

        // Verdict distribution
        println!("Verdict Distribution:");
        println!("  Fully demonstrated:     {fully}");
        println!("  Partially demonstrated: {partially}");
        println!("  Not demonstrated:       {not_dem}");
        println!();

        // Lessons
        let avg_lessons = if total > 0 {
            total_lessons as f64 / total as f64
        } else {
            0.0
        };
        println!("Lessons: {total_lessons} total, {avg_lessons:.1} avg/session");
        println!("Patterns: {total_patterns} total");
        println!();

        // Root causes
        println!("Top Root Causes:");
        let mut causes = [
            ("PDP Proposition", rc_prop),
            ("PDP So What?", rc_sowhat),
            ("PDP Why?", rc_why),
            ("Hook/Tool Gap", rc_hook),
        ];
        causes.sort_by(|a, b| b.1.cmp(&a.1));
        for (name, count) in &causes {
            if *count > 0 {
                println!("  {name}: {count}");
            }
        }
        if causes.iter().all(|(_, c)| *c == 0) {
            println!("  (none recorded)");
        }
        println!();

        // Rho
        if rho_count > 0 {
            let avg_rho = rho_sum / rho_count as f64;
            println!("Compounding: avg rho = {avg_rho:.3} ({rho_count} sessions with data)");
        } else {
            println!("Compounding: no rho data recorded");
        }

        Ok(())
    })
    .map_err(|e| {
        nexcore_brain::error::BrainError::Other(
            ["Autopsy report failed: ", &e.to_string()].concat(),
        )
    })
}

fn autopsy_directive_cmd(directive_id: String) -> nexcore_brain::error::Result<()> {
    let pool = open_db()?;
    pool.with_conn(|conn| {
        let rows = nexcore_db::autopsy::list_by_directive(conn, &directive_id)?;

        if rows.is_empty() {
            println!("No autopsy records found for directive: {directive_id}");
            return Ok(());
        }

        println!("Directive Report: {directive_id}");
        println!("{}", "=".repeat(20 + directive_id.len()));
        println!("Sessions: {}", rows.len());
        println!();

        for (i, row) in rows.iter().enumerate() {
            let phase = row.phase.as_deref().unwrap_or("-");
            let phase_type = row.phase_type.as_deref().unwrap_or("-");
            let verdict = row.outcome_verdict.as_deref().unwrap_or("-");

            let short_id: String = row.session_id.chars().take(12).collect();
            println!(
                "{}. {} | {} ({}) | {}",
                i + 1,
                short_id,
                phase,
                phase_type,
                verdict,
            );

            let rho_str = row
                .rho_session
                .map(|r| {
                    let mut buf = String::with_capacity(8);
                    buf.push_str("rho=");
                    // Manual f64 formatting to avoid format! with variable
                    let truncated = (r * 1000.0) as i64;
                    let whole = truncated / 1000;
                    let frac = (truncated % 1000).unsigned_abs();
                    buf.push_str(&whole.to_string());
                    buf.push('.');
                    if frac < 10 {
                        buf.push_str("00");
                    } else if frac < 100 {
                        buf.push('0');
                    }
                    buf.push_str(&frac.to_string());
                    buf
                })
                .unwrap_or_else(|| "-".to_string());

            println!(
                "   lessons={} patterns={} tools={} mcp={} {}",
                row.lesson_count, row.pattern_count, row.tool_calls_total, row.mcp_calls, rho_str,
            );
        }

        Ok(())
    })
    .map_err(|e| {
        nexcore_brain::error::BrainError::Other(
            ["Directive report failed: ", &e.to_string()].concat(),
        )
    })
}

fn print_autopsy_summary(row: &nexcore_db::autopsy::AutopsyRow) {
    let directive = row.directive_id.as_deref().unwrap_or("-");
    let phase = row.phase.as_deref().unwrap_or("-");
    let verdict = row.outcome_verdict.as_deref().unwrap_or("-");

    println!("  Directive: {directive}");
    println!("  Phase: {phase}");
    println!("  Verdict: {verdict}");
    println!(
        "  Lessons: {} | Patterns: {}",
        row.lesson_count, row.pattern_count
    );
    println!(
        "  Tools: {} total, {} MCP | Hook blocks: {}",
        row.tool_calls_total, row.mcp_calls, row.hook_blocks
    );

    if let Some(rho) = row.rho_session {
        let truncated = (rho * 1000.0) as i64;
        let whole = truncated / 1000;
        let frac = (truncated % 1000).unsigned_abs();
        print!("  Rho: {whole}.");
        if frac < 10 {
            print!("00");
        } else if frac < 100 {
            print!("0");
        }
        println!("{frac}");
    }
}

fn print_pass_rate(label: &str, pass: usize, total: usize) {
    if total == 0 {
        println!("{label}: n/a (no evaluated sessions)");
    } else {
        let pct = (pass as f64 / total as f64) * 100.0;
        let pct_int = pct as u32;
        println!("{label}: {pass}/{total} ({pct_int}%)");
    }
}

fn get_session(id: Option<String>) -> nexcore_brain::error::Result<BrainSession> {
    match id {
        Some(id) => Ok(BrainSession::load_str(&id)?),
        None => Ok(BrainSession::load_latest()?),
    }
}

fn get_default_project() -> String {
    std::env::current_dir()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "default".to_string())
}
