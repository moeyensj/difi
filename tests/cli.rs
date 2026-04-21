//! CLI integration tests.
//!
//! Invokes the compiled `difi` binary via `assert_cmd` against the parquet
//! fixtures under `python/difi/tests/testdata/`. Gated on the `cli` feature
//! because the binary is built behind `required-features = ["cli"]`.

#![cfg(feature = "cli")]

use std::path::{Path, PathBuf};
use std::process::Command;

use assert_cmd::Command as AssertCommand;
use predicates::prelude::*;
use tempfile::TempDir;

fn testdata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("difi")
        .join("tests")
        .join("testdata")
}

fn observations_parquet() -> PathBuf {
    testdata_dir().join("observations.parquet")
}

fn linkage_members_parquet() -> PathBuf {
    testdata_dir().join("linkage_members.parquet")
}

fn cmd() -> AssertCommand {
    AssertCommand::cargo_bin("difi").expect("difi binary not built — needs --features cli")
}

fn expect_file(path: &Path) {
    assert!(path.exists(), "expected output {} to exist", path.display());
    let meta = std::fs::metadata(path).unwrap();
    assert!(
        meta.len() > 0,
        "expected {} to be non-empty",
        path.display()
    );
}

// ---------------------------------------------------------------------------
// analyze-observations (CIFI)
// ---------------------------------------------------------------------------

#[test]
fn analyze_observations_singleton_defaults_on_fixture() {
    let tmp = TempDir::new().unwrap();
    cmd()
        .args(["analyze-observations", "-i"])
        .arg(observations_parquet())
        .arg("-o")
        .arg(tmp.path())
        .assert()
        .success();

    expect_file(&tmp.path().join("all_objects.parquet"));
    expect_file(&tmp.path().join("findable_observations.parquet"));
    expect_file(&tmp.path().join("partition_summaries.parquet"));
    expect_file(&tmp.path().join("run_manifest.json"));
}

#[test]
fn analyze_observations_explicit_singleton_flags() {
    let tmp = TempDir::new().unwrap();
    cmd()
        .args([
            "analyze-observations",
            "--metric",
            "singletons",
            "--min-obs",
            "6",
            "--min-nights",
            "3",
        ])
        .arg("-i")
        .arg(observations_parquet())
        .arg("-o")
        .arg(tmp.path())
        .assert()
        .success();
    expect_file(&tmp.path().join("all_objects.parquet"));
}

#[test]
fn analyze_observations_tracklet_metric() {
    let tmp = TempDir::new().unwrap();
    cmd()
        .args([
            "analyze-observations",
            "--metric",
            "tracklets",
            "--tracklet-min-obs",
            "2",
            "--min-linkage-nights",
            "3",
        ])
        .arg("-i")
        .arg(observations_parquet())
        .arg("-o")
        .arg(tmp.path())
        .assert()
        .success();
    expect_file(&tmp.path().join("partition_summaries.parquet"));
}

#[test]
fn analyze_observations_sliding_partition() {
    let tmp = TempDir::new().unwrap();
    cmd()
        .args([
            "analyze-observations",
            "--partition-mode",
            "sliding",
            "--partition-window",
            "5",
            "--partition-min-nights",
            "3",
        ])
        .arg("-i")
        .arg(observations_parquet())
        .arg("-o")
        .arg(tmp.path())
        .assert()
        .success();
    expect_file(&tmp.path().join("partition_summaries.parquet"));
}

#[test]
fn cifi_alias_resolves_to_analyze_observations() {
    let tmp = TempDir::new().unwrap();
    cmd()
        .arg("cifi")
        .arg("-i")
        .arg(observations_parquet())
        .arg("-o")
        .arg(tmp.path())
        .assert()
        .success();
    expect_file(&tmp.path().join("all_objects.parquet"));
    expect_file(&tmp.path().join("run_manifest.json"));
}

#[test]
fn cifi_help_alias_works() {
    // --help via alias should succeed and mention the command.
    cmd()
        .args(["cifi", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Can I Find It"));
}

// ---------------------------------------------------------------------------
// analyze-linkages (DIFI)
// ---------------------------------------------------------------------------

#[test]
fn analyze_linkages_produces_expected_outputs() {
    let tmp = TempDir::new().unwrap();
    cmd()
        .arg("analyze-linkages")
        .arg("-i")
        .arg(observations_parquet())
        .arg("-l")
        .arg(linkage_members_parquet())
        .arg("-o")
        .arg(tmp.path())
        .assert()
        .success();

    expect_file(&tmp.path().join("all_linkages.parquet"));
    expect_file(&tmp.path().join("all_objects.parquet"));
    expect_file(&tmp.path().join("findable_observations.parquet"));
    expect_file(&tmp.path().join("partition_summaries.parquet"));
    expect_file(&tmp.path().join("run_manifest.json"));
}

#[test]
fn analyze_alias_resolves_to_analyze_linkages() {
    let tmp = TempDir::new().unwrap();
    cmd()
        .arg("analyze")
        .arg("-i")
        .arg(observations_parquet())
        .arg("-l")
        .arg(linkage_members_parquet())
        .arg("-o")
        .arg(tmp.path())
        .assert()
        .success();
    expect_file(&tmp.path().join("all_linkages.parquet"));
}

// ---------------------------------------------------------------------------
// Scenarios TOML (batch mode)
// ---------------------------------------------------------------------------

#[test]
fn scenarios_cli_observations_overrides_toml_defaults() {
    // Precedence contract: `--observations` overrides `[defaults].observations`
    // so you can re-run a TOML against a different input without editing the file.
    // Per-scenario entries still win over the CLI flag.
    let tmp = TempDir::new().unwrap();
    let scenarios_path = tmp.path().join("scenarios.toml");
    let missing = tmp.path().join("missing_defaults.parquet");
    let toml = format!(
        r#"
[defaults]
observations = "{missing}"

[[scenario]]
name = "uses_cli_observations"
metric = "singletons"
"#,
        missing = missing.display(),
    );
    std::fs::write(&scenarios_path, toml).unwrap();

    let out_dir = tmp.path().join("out");
    cmd()
        .arg("cifi")
        .arg("--scenarios")
        .arg(&scenarios_path)
        .arg("-i")
        .arg(observations_parquet())
        .arg("-o")
        .arg(&out_dir)
        .assert()
        .success();

    expect_file(
        &out_dir
            .join("uses_cli_observations")
            .join("all_objects.parquet"),
    );
}

#[test]
fn scenarios_toml_batch_writes_one_subdir_per_scenario() {
    let tmp = TempDir::new().unwrap();
    let scenarios_path = tmp.path().join("scenarios.toml");
    let toml = format!(
        r#"
[defaults]
observations = "{obs}"

[[scenario]]
name = "singleton_defaults"
metric = "singletons"

[[scenario]]
name = "tracklet_defaults"
metric = "tracklets"
"#,
        obs = observations_parquet().display(),
    );
    std::fs::write(&scenarios_path, toml).unwrap();

    let out_dir = tmp.path().join("out");
    cmd()
        .arg("cifi")
        .arg("--scenarios")
        .arg(&scenarios_path)
        .arg("-o")
        .arg(&out_dir)
        .assert()
        .success();

    expect_file(&out_dir.join("run_manifest.json"));
    expect_file(
        &out_dir
            .join("singleton_defaults")
            .join("all_objects.parquet"),
    );
    expect_file(
        &out_dir
            .join("singleton_defaults")
            .join("partition_summaries.parquet"),
    );
    expect_file(
        &out_dir
            .join("tracklet_defaults")
            .join("all_objects.parquet"),
    );
    expect_file(
        &out_dir
            .join("tracklet_defaults")
            .join("partition_summaries.parquet"),
    );
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn missing_observations_exits_with_io_code_2() {
    let tmp = TempDir::new().unwrap();
    let missing = tmp.path().join("does_not_exist.parquet");
    cmd()
        .arg("cifi")
        .arg("-i")
        .arg(&missing)
        .arg("-o")
        .arg(tmp.path().join("out"))
        .assert()
        .code(2)
        .stderr(predicate::str::contains("difi: error"));
}

#[test]
fn sliding_partition_without_window_fails_clearly() {
    let tmp = TempDir::new().unwrap();
    cmd()
        .args(["cifi", "--partition-mode", "sliding"])
        .arg("-i")
        .arg(observations_parquet())
        .arg("-o")
        .arg(tmp.path())
        .assert()
        .failure()
        .stderr(predicate::str::contains("--partition-window"));
}

// ---------------------------------------------------------------------------
// Progress JSON
// ---------------------------------------------------------------------------

#[test]
fn progress_json_emits_valid_ndjson_on_stdout() {
    let tmp = TempDir::new().unwrap();

    // Use std::process::Command to capture stdout separately from stderr.
    let output = Command::new(assert_cmd::cargo::cargo_bin("difi"))
        .args(["--progress-json", "cifi"])
        .arg("-i")
        .arg(observations_parquet())
        .arg("-o")
        .arg(tmp.path())
        .output()
        .expect("spawn difi");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();

    // Every non-empty line must be parseable JSON with an "event" key.
    let mut event_kinds = Vec::new();
    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("invalid JSON line: {line:?} ({e})"));
        let event = v.get("event").and_then(|s| s.as_str()).unwrap_or("");
        assert!(!event.is_empty(), "event missing: {line}");
        event_kinds.push(event.to_string());
    }

    assert!(
        event_kinds.contains(&"start".to_string()),
        "missing start event"
    );
    assert!(
        event_kinds.contains(&"done".to_string()),
        "missing done event"
    );
}

#[test]
fn progress_json_with_failure_emits_error_event_and_stderr_line() {
    let tmp = TempDir::new().unwrap();
    let missing = tmp.path().join("does_not_exist.parquet");

    let output = Command::new(assert_cmd::cargo::cargo_bin("difi"))
        .args(["--progress-json", "cifi"])
        .arg("-i")
        .arg(&missing)
        .arg("-o")
        .arg(tmp.path().join("out"))
        .output()
        .expect("spawn difi");

    assert!(!output.status.success());

    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("difi: error"),
        "stderr should always carry the human error line, got: {stderr}"
    );

    // Last NDJSON line on stdout should be an error event.
    let stdout = String::from_utf8(output.stdout).unwrap();
    let last_json_line = stdout
        .lines()
        .rfind(|l| !l.trim().is_empty())
        .expect("expected at least one JSON line");
    let v: serde_json::Value = serde_json::from_str(last_json_line).unwrap();
    assert_eq!(v.get("event").and_then(|s| s.as_str()), Some("error"));
}

// ---------------------------------------------------------------------------
// Top-level
// ---------------------------------------------------------------------------

#[test]
fn top_level_help_mentions_both_subcommands() {
    cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("analyze-observations"))
        .stdout(predicate::str::contains("analyze-linkages"));
}

#[test]
fn version_flag_prints_package_version() {
    cmd()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains(env!("CARGO_PKG_VERSION")));
}
