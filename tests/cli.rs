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
// Multi-partition analyze-linkages (Phase 3)
// ---------------------------------------------------------------------------

#[test]
fn scenario_manifest_reports_unique_findable_found_and_completeness() {
    // Sliding-window analyze-linkages; assert the manifest carries distinct-
    // object cross-partition counts alongside the per-partition sums, and
    // that the relationships hold:
    //   unique <= sum           (distinct can only be <= total occurrences)
    //   0 <= unique_completeness <= 100
    let tmp = TempDir::new().unwrap();
    cmd()
        .args([
            "analyze-linkages",
            "--partition-mode",
            "sliding",
            "--partition-window",
            "3",
            "--partition-min-nights",
            "2",
        ])
        .arg("-i")
        .arg(observations_parquet())
        .arg("-l")
        .arg(linkage_members_parquet())
        .arg("-o")
        .arg(tmp.path())
        .assert()
        .success();

    let m: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(tmp.path().join("run_manifest.json")).unwrap(),
    )
    .unwrap();
    let s = &m["scenarios"][0];

    let findable = s["findable_count"].as_i64().unwrap();
    let unique_findable = s["unique_findable_count"].as_i64().unwrap();
    assert!(
        unique_findable > 0 && unique_findable <= findable,
        "unique_findable_count ({unique_findable}) must be in (0, findable_count={findable}]"
    );

    let found = s["found_count"].as_i64().unwrap();
    let unique_found = s["unique_found_count"].as_i64().unwrap();
    assert!(
        unique_found <= found,
        "unique_found_count ({unique_found}) must be <= found_count ({found})"
    );
    assert!(
        unique_found <= unique_findable,
        "unique_found_count ({unique_found}) must be <= unique_findable_count ({unique_findable})"
    );

    let uc = s["unique_completeness"].as_f64().unwrap();
    assert!(
        (0.0..=100.0).contains(&uc),
        "unique_completeness ({uc}) should be in [0, 100]"
    );
}

#[test]
fn cifi_only_manifest_has_unique_findable_but_no_unique_found_or_completeness() {
    let tmp = TempDir::new().unwrap();
    cmd()
        .arg("cifi")
        .arg("-i")
        .arg(observations_parquet())
        .arg("-o")
        .arg(tmp.path())
        .assert()
        .success();
    let m: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(tmp.path().join("run_manifest.json")).unwrap(),
    )
    .unwrap();
    let s = &m["scenarios"][0];
    assert!(s["unique_findable_count"].as_i64().unwrap() > 0);
    // CIFI-only skips DIFI, so the unique_found / unique_completeness fields
    // are omitted via `skip_serializing_if = Option::is_none`.
    assert!(s.get("unique_found_count").is_none() || s["unique_found_count"].is_null());
    assert!(s.get("unique_completeness").is_none() || s["unique_completeness"].is_null());
}

#[test]
fn analyze_linkages_sliding_partition_writes_ignored_linkages_and_manifest_warnings() {
    // With a narrow sliding window, many linkages will be wholly outside
    // individual partitions. Those rows should be excluded from
    // all_linkages.parquet and written to ignored_linkages.parquet, with
    // counts recorded in manifest.warnings.
    let tmp = TempDir::new().unwrap();
    cmd()
        .args([
            "analyze-linkages",
            "--partition-mode",
            "sliding",
            "--partition-window",
            "3",
            "--partition-min-nights",
            "2",
        ])
        .arg("-i")
        .arg(observations_parquet())
        .arg("-l")
        .arg(linkage_members_parquet())
        .arg("-o")
        .arg(tmp.path())
        .assert()
        .success()
        .stderr(predicate::str::contains("difi: warning"));

    expect_file(&tmp.path().join("ignored_linkages.parquet"));

    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(tmp.path().join("run_manifest.json")).unwrap(),
    )
    .unwrap();
    let rows = manifest["warnings"]["ignored_linkage_rows"]
        .as_u64()
        .expect("warnings.ignored_linkage_rows should be a number");
    assert!(
        rows > 0,
        "sliding partitions should produce non-zero ignored rows"
    );
}

#[test]
fn analyze_linkages_sliding_partition_produces_multi_partition_output() {
    let tmp = TempDir::new().unwrap();
    cmd()
        .args([
            "analyze-linkages",
            "--partition-mode",
            "sliding",
            "--partition-window",
            "3",
            "--partition-min-nights",
            "2",
        ])
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
    expect_file(&tmp.path().join("partition_summaries.parquet"));

    // Manifest should record the partition mode as sliding.
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(tmp.path().join("run_manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        manifest["scenarios"][0]["partitions"]["mode"]
            .as_str()
            .unwrap(),
        "sliding"
    );
}

// ---------------------------------------------------------------------------
// --cifi-output-dir reuse (Phase 4)
// ---------------------------------------------------------------------------

fn run_cifi_to(tmp: &Path) -> PathBuf {
    let cifi_dir = tmp.join("cifi_out");
    cmd()
        .arg("cifi")
        .arg("-i")
        .arg(observations_parquet())
        .arg("-o")
        .arg(&cifi_dir)
        .assert()
        .success();
    cifi_dir
}

fn run_sliding_cifi_to(tmp: &Path) -> PathBuf {
    let cifi_dir = tmp.join("cifi_out_sliding");
    cmd()
        .args([
            "cifi",
            "--partition-mode",
            "sliding",
            "--partition-window",
            "3",
            "--partition-min-nights",
            "2",
        ])
        .arg("-i")
        .arg(observations_parquet())
        .arg("-o")
        .arg(&cifi_dir)
        .assert()
        .success();
    cifi_dir
}

#[test]
fn analyze_linkages_with_cifi_output_dir_succeeds_and_records_provenance() {
    let tmp = TempDir::new().unwrap();
    let cifi_dir = run_cifi_to(tmp.path());
    let difi_dir = tmp.path().join("difi_out");

    cmd()
        .arg("analyze-linkages")
        .arg("-i")
        .arg(observations_parquet())
        .arg("-l")
        .arg(linkage_members_parquet())
        .arg("--cifi-output-dir")
        .arg(&cifi_dir)
        .arg("-o")
        .arg(&difi_dir)
        .assert()
        .success();

    expect_file(&difi_dir.join("all_linkages.parquet"));
    expect_file(&difi_dir.join("all_objects.parquet"));
    expect_file(&difi_dir.join("partition_summaries.parquet"));
    expect_file(&difi_dir.join("run_manifest.json"));

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(difi_dir.join("run_manifest.json")).unwrap())
            .unwrap();
    assert!(
        manifest["reused_cifi"].is_object(),
        "reused_cifi should be populated"
    );
    assert!(
        manifest["reused_cifi"]["observations_input"]["sha256_prefix"].is_string(),
        "reused_cifi must record the reused observations fingerprint"
    );
    // CIFI phase should be recorded as elapsed=0 (skipped).
    assert_eq!(
        manifest["scenarios"][0]["cifi_elapsed_s"].as_f64().unwrap(),
        0.0
    );
}

#[test]
fn analyze_linkages_reuse_records_snapshot_partition_scheme_not_cli_defaults() {
    // Regression: the reuse path used to record `args.partition.to_manifest()`
    // (always "single" in reuse mode, since partition flags are rejected)
    // instead of the reused CIFI's actual partition scheme. The manifest's
    // scenarios[0].partitions should now reflect the snapshot.
    let tmp = TempDir::new().unwrap();
    let cifi_dir = run_sliding_cifi_to(tmp.path());
    let difi_dir = tmp.path().join("difi_out");

    cmd()
        .arg("analyze-linkages")
        .arg("-i")
        .arg(observations_parquet())
        .arg("-l")
        .arg(linkage_members_parquet())
        .arg("--cifi-output-dir")
        .arg(&cifi_dir)
        .arg("-o")
        .arg(&difi_dir)
        .assert()
        .success();

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(difi_dir.join("run_manifest.json")).unwrap())
            .unwrap();
    let partitions = &manifest["scenarios"][0]["partitions"];
    assert_eq!(
        partitions["mode"].as_str().unwrap(),
        "sliding",
        "reuse manifest should record the snapshot's partition mode (sliding), \
         not the default Single from unused CLI args. got: {partitions:?}"
    );
    assert_eq!(partitions["window"].as_i64().unwrap(), 3);
}

#[test]
fn analyze_linkages_rejects_cifi_dir_plus_partition_flags() {
    let tmp = TempDir::new().unwrap();
    let cifi_dir = run_cifi_to(tmp.path());
    let difi_dir = tmp.path().join("difi_out");

    cmd()
        .arg("analyze-linkages")
        .arg("-i")
        .arg(observations_parquet())
        .arg("-l")
        .arg(linkage_members_parquet())
        .arg("--cifi-output-dir")
        .arg(&cifi_dir)
        .arg("--partition-mode")
        .arg("sliding")
        .arg("--partition-window")
        .arg("5")
        .arg("-o")
        .arg(&difi_dir)
        .assert()
        .failure()
        .stderr(predicate::str::contains("--cifi-output-dir"));
}

#[test]
fn analyze_linkages_rejects_mismatched_observations_fingerprint() {
    let tmp = TempDir::new().unwrap();
    let cifi_dir = run_cifi_to(tmp.path());

    // Copy the real observations and append a byte so the sha256 prefix diverges.
    let fake_obs = tmp.path().join("fake_obs.parquet");
    std::fs::copy(observations_parquet(), &fake_obs).unwrap();
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&fake_obs)
            .unwrap();
        f.write_all(b"tamper").unwrap();
    }

    let difi_dir = tmp.path().join("difi_out");
    cmd()
        .arg("analyze-linkages")
        .arg("-i")
        .arg(&fake_obs)
        .arg("-l")
        .arg(linkage_members_parquet())
        .arg("--cifi-output-dir")
        .arg(&cifi_dir)
        .arg("-o")
        .arg(&difi_dir)
        .assert()
        .failure()
        .stderr(predicate::str::contains("fingerprint mismatch"));
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
