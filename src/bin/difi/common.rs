//! Shared CLI helpers: argument groups, metric/partition builders, manifest,
//! and NDJSON progress event emission.
//!
//! Kept intentionally small — the CLI is glue, not an algorithm surface.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use difi::metrics::FindabilityMetric;
use difi::metrics::singleton::SingletonMetric;
use difi::metrics::tracklet::TrackletMetric;
use difi::partitions::{self, Partition};

// ---------------------------------------------------------------------------
// Global run context
// ---------------------------------------------------------------------------

/// Context threaded through every subcommand: whether to emit NDJSON progress
/// events on stdout, and the timing origin for `elapsed_s`/`total_elapsed_s`.
pub struct RunContext {
    pub progress_json: bool,
    started_at: Instant,
    started_at_unix_s: f64,
    /// The full argv, recorded in the manifest.
    pub command: Vec<String>,
    /// Serializes stdout writes so progress events don't interleave.
    stdout_lock: Mutex<()>,
}

impl RunContext {
    pub fn new(progress_json: bool, command: Vec<String>) -> Self {
        Self {
            progress_json,
            started_at: Instant::now(),
            started_at_unix_s: unix_seconds_now(),
            command,
            stdout_lock: Mutex::new(()),
        }
    }

    pub fn elapsed_s(&self) -> f64 {
        self.started_at.elapsed().as_secs_f64()
    }

    pub fn started_at_unix_s(&self) -> f64 {
        self.started_at_unix_s
    }

    /// Emit a progress event. Under `--progress-json` it goes to stdout as
    /// NDJSON; otherwise it's formatted for humans on stderr.
    pub fn emit(&self, event: ProgressEvent<'_>) {
        if self.progress_json {
            let mut payload = event.to_json();
            payload["ts_unix_s"] = serde_json::json!(unix_seconds_now());
            let line = payload.to_string();
            let _guard = self.stdout_lock.lock().unwrap();
            let mut out = std::io::stdout().lock();
            // Write + newline; ignore EPIPE so `| head` doesn't panic.
            let _ = writeln!(out, "{line}");
            let _ = out.flush();
        } else {
            let _ = writeln!(std::io::stderr(), "{}", event.human());
        }
    }

    /// Emit an error event. Always logs a single-line human message to stderr,
    /// and additionally emits an NDJSON `{"event":"error", ...}` on stdout when
    /// `--progress-json` is set. Piping stdout through `jq` never suppresses
    /// the stderr line.
    pub fn emit_error(&self, err: &anyhow::Error) {
        let _ = writeln!(std::io::stderr(), "difi: error: {err:#}");
        if self.progress_json {
            let payload = serde_json::json!({
                "event": "error",
                "message": format!("{err:#}"),
                "ts_unix_s": unix_seconds_now(),
            });
            let _guard = self.stdout_lock.lock().unwrap();
            let mut out = std::io::stdout().lock();
            let _ = writeln!(out, "{payload}");
            let _ = out.flush();
        }
    }

    /// Emit a structured warning. Always writes a human-readable line to
    /// stderr; under `--progress-json` additionally writes an NDJSON
    /// `{"event":"warning", ...}` line to stdout with the supplied fields.
    pub fn emit_warning(&self, message: &str, fields: serde_json::Value) {
        let _ = writeln!(std::io::stderr(), "difi: warning: {message}");
        if self.progress_json {
            let mut payload = serde_json::json!({
                "event": "warning",
                "message": message,
                "ts_unix_s": unix_seconds_now(),
            });
            if let serde_json::Value::Object(map) = fields
                && let serde_json::Value::Object(ref mut outer) = payload
            {
                for (k, v) in map {
                    outer.insert(k, v);
                }
            }
            let _guard = self.stdout_lock.lock().unwrap();
            let mut out = std::io::stdout().lock();
            let _ = writeln!(out, "{payload}");
            let _ = out.flush();
        }
    }
}

fn unix_seconds_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Progress events
// ---------------------------------------------------------------------------

pub enum ProgressEvent<'a> {
    Start {
        subcommand: &'a str,
        step: &'a str,
        input: &'a str,
    },
    LoadedObservations {
        count: usize,
        elapsed_s: f64,
    },
    ScenarioStart {
        name: &'a str,
        metric: &'a str,
        partitions: usize,
    },
    ScenarioDone {
        name: &'a str,
        findable: i64,
        found: Option<i64>,
        elapsed_s: f64,
    },
    Done {
        total_elapsed_s: f64,
        output_dir: &'a Path,
    },
}

impl ProgressEvent<'_> {
    fn to_json(&self) -> serde_json::Value {
        match self {
            ProgressEvent::Start {
                subcommand,
                step,
                input,
            } => serde_json::json!({
                "event": "start",
                "subcommand": subcommand,
                "step": step,
                "input": input,
            }),
            ProgressEvent::LoadedObservations { count, elapsed_s } => serde_json::json!({
                "event": "loaded_observations",
                "count": count,
                "elapsed_s": elapsed_s,
            }),
            ProgressEvent::ScenarioStart {
                name,
                metric,
                partitions,
            } => serde_json::json!({
                "event": "scenario_start",
                "name": name,
                "metric": metric,
                "partitions": partitions,
            }),
            ProgressEvent::ScenarioDone {
                name,
                findable,
                found,
                elapsed_s,
            } => serde_json::json!({
                "event": "scenario_done",
                "name": name,
                "findable": findable,
                "found": found,
                "elapsed_s": elapsed_s,
            }),
            ProgressEvent::Done {
                total_elapsed_s,
                output_dir,
            } => serde_json::json!({
                "event": "done",
                "total_elapsed_s": total_elapsed_s,
                "output_dir": output_dir.display().to_string(),
            }),
        }
    }

    fn human(&self) -> String {
        match self {
            ProgressEvent::Start {
                subcommand,
                step,
                input,
            } => format!("[{step}] {subcommand}: input={input}"),
            ProgressEvent::LoadedObservations { count, elapsed_s } => {
                format!("loaded {count} observations in {elapsed_s:.2}s")
            }
            ProgressEvent::ScenarioStart {
                name,
                metric,
                partitions,
            } => format!("scenario {name} ({metric}, {partitions} partition(s))"),
            ProgressEvent::ScenarioDone {
                name,
                findable,
                found,
                elapsed_s,
            } => match found {
                Some(f) => format!(
                    "scenario {name} done in {elapsed_s:.2}s: findable={findable} found={f}"
                ),
                None => format!("scenario {name} done in {elapsed_s:.2}s: findable={findable}"),
            },
            ProgressEvent::Done {
                total_elapsed_s,
                output_dir,
            } => format!(
                "done in {:.2}s — wrote to {}",
                total_elapsed_s,
                output_dir.display()
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Metric arguments
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[value(rename_all = "lowercase")]
pub enum MetricKind {
    Singletons,
    Tracklets,
}

impl MetricKind {
    pub fn as_str(self) -> &'static str {
        match self {
            MetricKind::Singletons => "singletons",
            MetricKind::Tracklets => "tracklets",
        }
    }
}

/// Metric configuration for both singleton and tracklet metrics. Unused fields
/// for the selected metric are ignored.
#[derive(Args, Clone, Debug)]
pub struct MetricArgs {
    /// Findability metric.
    #[arg(short = 'm', long = "metric", value_enum, default_value_t = MetricKind::Singletons)]
    pub metric: MetricKind,

    // ---- Singleton-specific ----
    /// Minimum total observations for singleton findability. Also used as the
    /// DIFI linkage threshold in `analyze-linkages`.
    #[arg(long, default_value_t = 6)]
    pub min_obs: usize,

    /// Minimum distinct nights for singleton findability.
    #[arg(long, default_value_t = 3)]
    pub min_nights: usize,

    /// Minimum per-night observations when exactly `min_nights` are present.
    #[arg(long, default_value_t = 1)]
    pub min_nightly_obs_in_min_nights: usize,

    // ---- Tracklet-specific ----
    /// Minimum observations per intra-night tracklet.
    #[arg(long, default_value_t = 2)]
    pub tracklet_min_obs: usize,

    /// Maximum intra-tracklet time separation, in hours.
    #[arg(long, default_value_t = 1.5)]
    pub max_obs_separation_hours: f64,

    /// Minimum distinct nights with valid tracklets.
    #[arg(long, default_value_t = 3)]
    pub min_linkage_nights: usize,

    /// Minimum angular separation between tracklet observations, in arcseconds.
    #[arg(long, default_value_t = 1.0)]
    pub min_obs_angular_separation_arcsec: f64,
}

impl MetricArgs {
    pub fn build(&self) -> Box<dyn FindabilityMetric> {
        match self.metric {
            MetricKind::Singletons => Box::new(SingletonMetric {
                min_obs: self.min_obs,
                min_nights: self.min_nights,
                min_nightly_obs_in_min_nights: self.min_nightly_obs_in_min_nights,
            }),
            MetricKind::Tracklets => Box::new(TrackletMetric {
                tracklet_min_obs: self.tracklet_min_obs,
                max_obs_separation: self.max_obs_separation_hours / 24.0,
                min_linkage_nights: self.min_linkage_nights,
                min_obs_angular_separation: self.min_obs_angular_separation_arcsec,
            }),
        }
    }

    pub fn to_manifest(&self) -> serde_json::Value {
        match self.metric {
            MetricKind::Singletons => serde_json::json!({
                "kind": "singletons",
                "min_obs": self.min_obs,
                "min_nights": self.min_nights,
                "min_nightly_obs_in_min_nights": self.min_nightly_obs_in_min_nights,
            }),
            MetricKind::Tracklets => serde_json::json!({
                "kind": "tracklets",
                "tracklet_min_obs": self.tracklet_min_obs,
                "max_obs_separation_hours": self.max_obs_separation_hours,
                "min_linkage_nights": self.min_linkage_nights,
                "min_obs_angular_separation_arcsec": self.min_obs_angular_separation_arcsec,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Partition arguments
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[value(rename_all = "lowercase")]
pub enum PartitionMode {
    Single,
    Sliding,
    Blocks,
}

#[derive(Args, Clone, Debug)]
pub struct PartitionArgs {
    /// Partition mode.
    #[arg(long = "partition-mode", value_enum, default_value_t = PartitionMode::Single)]
    pub partition_mode: PartitionMode,

    /// Window size in nights. Required for sliding/blocks.
    #[arg(long = "partition-window")]
    pub partition_window: Option<i64>,

    /// Ramp-up cap for sliding windows. Defaults to `--partition-window`.
    #[arg(long = "partition-min-nights")]
    pub partition_min_nights: Option<i64>,
}

impl PartitionArgs {
    /// True when the user left every partition flag at its CLI default.
    /// Used by `analyze-linkages` to reject `--cifi-output-dir` + any
    /// non-default partition flag (the reused CIFI dictates the partitions).
    pub fn is_default(&self) -> bool {
        self.partition_mode == PartitionMode::Single
            && self.partition_window.is_none()
            && self.partition_min_nights.is_none()
    }

    pub fn build(&self, nights: &[i64]) -> Result<Vec<Partition>> {
        match self.partition_mode {
            PartitionMode::Single => Ok(vec![partitions::create_single(nights)?]),
            PartitionMode::Sliding => {
                let window = self.partition_window.ok_or_else(|| {
                    anyhow::anyhow!("--partition-window is required for --partition-mode sliding")
                })?;
                Ok(partitions::create_linking_windows(
                    nights,
                    Some(window),
                    Some(self.partition_min_nights.unwrap_or(window)),
                    true,
                )?)
            }
            PartitionMode::Blocks => {
                let window = self.partition_window.ok_or_else(|| {
                    anyhow::anyhow!("--partition-window is required for --partition-mode blocks")
                })?;
                Ok(partitions::create_linking_windows(
                    nights,
                    Some(window),
                    None,
                    false,
                )?)
            }
        }
    }

    pub fn to_manifest(&self) -> serde_json::Value {
        match self.partition_mode {
            PartitionMode::Single => serde_json::json!({ "mode": "single" }),
            PartitionMode::Sliding => serde_json::json!({
                "mode": "sliding",
                "window": self.partition_window,
                "min_nights": self.partition_min_nights.or(self.partition_window),
            }),
            PartitionMode::Blocks => serde_json::json!({
                "mode": "blocks",
                "window": self.partition_window,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Input fingerprinting
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputFingerprint {
    pub path: String,
    pub size_bytes: u64,
    /// SHA-256 of the first 1 MiB of the file. Cheap integrity check — not a
    /// cryptographic guarantee.
    pub sha256_prefix: String,
}

pub fn fingerprint_input(path: &Path) -> Result<InputFingerprint> {
    let meta = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    let size_bytes = meta.len();
    let sha256_prefix = sha256_first_mib(path)?;
    Ok(InputFingerprint {
        path: path.display().to_string(),
        size_bytes,
        sha256_prefix,
    })
}

fn sha256_first_mib(path: &Path) -> Result<String> {
    const PREFIX_BYTES: u64 = 1 << 20;
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = std::io::BufReader::new(f).take(PREFIX_BYTES);
    let mut buf = Vec::new();
    reader
        .read_to_end(&mut buf)
        .with_context(|| format!("read {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&buf);
    Ok(format!("{:x}", hasher.finalize()))
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct Manifest {
    pub difi_version: String,
    pub command: Vec<String>,
    pub observations_input: InputFingerprint,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub linkages_input: Option<InputFingerprint>,
    /// Provenance of a reused CIFI output, set when the run was invoked with
    /// `--cifi-output-dir`. Omitted otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reused_cifi: Option<ReusedCifiRef>,
    /// Structured warnings surfaced during the run (e.g. ignored linkages).
    /// Omitted when empty.
    #[serde(skip_serializing_if = "WarningsManifest::is_empty")]
    pub warnings: WarningsManifest,
    pub scenarios: Vec<ScenarioManifest>,
    pub host: HostInfo,
    pub started_at_unix_s: f64,
    pub finished_at_unix_s: f64,
}

/// Structured warnings recorded in the manifest.
#[derive(Debug, Clone, Default, Serialize)]
pub struct WarningsManifest {
    /// Total (linkage, partition) rows written to `ignored_linkages.parquet`.
    pub ignored_linkage_rows: usize,
    /// Distinct linkage IDs that appeared in `ignored_linkages.parquet` but
    /// were never classified in any partition — strong signal of a
    /// mismatched `--linkages` file for the partition scheme.
    pub orphan_linkages: usize,
}

impl WarningsManifest {
    pub fn is_empty(&self) -> bool {
        self.ignored_linkage_rows == 0 && self.orphan_linkages == 0
    }
}

/// Lightweight reference to a reused CIFI run's manifest. Captures enough to
/// audit provenance without embedding the full reused manifest.
#[derive(Debug, Clone, Serialize)]
pub struct ReusedCifiRef {
    pub path: String,
    pub manifest_difi_version: String,
    pub manifest_command: Vec<String>,
    pub observations_input: InputFingerprint,
}

/// Read a manifest for fingerprint verification, tolerating extra fields.
/// Carries the reused run's scenario metadata so a reusing run can record the
/// *snapshot's* metric / partition scheme in its own manifest (rather than
/// the current CLI args, which are defaults in reuse mode).
#[derive(Debug, Clone, Deserialize)]
pub struct ManifestForReuse {
    pub difi_version: String,
    pub command: Vec<String>,
    pub observations_input: InputFingerprint,
    #[serde(default)]
    pub scenarios: Vec<ReusedScenario>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReusedScenario {
    #[serde(default)]
    pub metric: serde_json::Value,
    #[serde(default)]
    pub partitions: serde_json::Value,
}

pub fn read_manifest_for_reuse(path: &Path) -> Result<ManifestForReuse> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read manifest {}", path.display()))?;
    let m: ManifestForReuse = serde_json::from_str(&text)
        .with_context(|| format!("parse manifest {}", path.display()))?;
    Ok(m)
}

#[derive(Debug, Clone, Serialize)]
pub struct ScenarioManifest {
    pub name: String,
    pub metric: serde_json::Value,
    pub partitions: serde_json::Value,
    pub cifi_elapsed_s: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub difi_elapsed_s: Option<f64>,
    pub findable_count: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub found_count: Option<i64>,
    pub outputs: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HostInfo {
    pub hostname: Option<String>,
    pub threads: usize,
}

impl HostInfo {
    pub fn capture() -> Self {
        let hostname = std::env::var("HOSTNAME")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| {
                std::fs::read_to_string("/etc/hostname")
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            });
        Self {
            hostname,
            threads: rayon::current_num_threads(),
        }
    }
}

pub fn write_manifest(path: &Path, manifest: &Manifest) -> Result<()> {
    let json = serde_json::to_string_pretty(manifest)?;
    std::fs::write(path, json + "\n")
        .with_context(|| format!("write manifest to {}", path.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Scenarios TOML
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ScenariosFile {
    #[serde(default)]
    pub defaults: ScenarioDefaults,
    #[serde(rename = "scenario", default)]
    pub scenarios: Vec<ScenarioEntry>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ScenarioDefaults {
    pub observations: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScenarioEntry {
    pub name: String,
    pub observations: Option<PathBuf>,

    // Metric (default: singletons)
    #[serde(default = "default_metric_kind")]
    pub metric: MetricKind,

    // Singleton
    #[serde(default = "default_min_obs")]
    pub min_obs: usize,
    #[serde(default = "default_min_nights")]
    pub min_nights: usize,
    #[serde(default = "default_one")]
    pub min_nightly_obs_in_min_nights: usize,

    // Tracklet
    #[serde(default = "default_tracklet_min_obs")]
    pub tracklet_min_obs: usize,
    #[serde(default = "default_max_obs_sep_hours")]
    pub max_obs_separation_hours: f64,
    #[serde(default = "default_min_linkage_nights")]
    pub min_linkage_nights: usize,
    #[serde(default = "default_min_ang_sep")]
    pub min_obs_angular_separation_arcsec: f64,

    // Partition
    #[serde(default = "default_partition_mode")]
    pub partition_mode: PartitionMode,
    pub partition_window: Option<i64>,
    pub partition_min_nights: Option<i64>,
}

fn default_metric_kind() -> MetricKind {
    MetricKind::Singletons
}
fn default_min_obs() -> usize {
    6
}
fn default_min_nights() -> usize {
    3
}
fn default_one() -> usize {
    1
}
fn default_tracklet_min_obs() -> usize {
    2
}
fn default_max_obs_sep_hours() -> f64 {
    1.5
}
fn default_min_linkage_nights() -> usize {
    3
}
fn default_min_ang_sep() -> f64 {
    1.0
}
fn default_partition_mode() -> PartitionMode {
    PartitionMode::Single
}

impl ScenarioEntry {
    pub fn to_metric_args(&self) -> MetricArgs {
        MetricArgs {
            metric: self.metric,
            min_obs: self.min_obs,
            min_nights: self.min_nights,
            min_nightly_obs_in_min_nights: self.min_nightly_obs_in_min_nights,
            tracklet_min_obs: self.tracklet_min_obs,
            max_obs_separation_hours: self.max_obs_separation_hours,
            min_linkage_nights: self.min_linkage_nights,
            min_obs_angular_separation_arcsec: self.min_obs_angular_separation_arcsec,
        }
    }

    pub fn to_partition_args(&self) -> PartitionArgs {
        PartitionArgs {
            partition_mode: self.partition_mode,
            partition_window: self.partition_window,
            partition_min_nights: self.partition_min_nights,
        }
    }
}

pub fn read_scenarios(path: &Path) -> Result<ScenariosFile> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read scenarios file {}", path.display()))?;
    let file: ScenariosFile = toml::from_str(&text)
        .with_context(|| format!("parse scenarios file {}", path.display()))?;
    if file.scenarios.is_empty() {
        bail!(
            "scenarios file {} contains no [[scenario]] entries",
            path.display()
        );
    }
    Ok(file)
}

// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------

pub fn version_string() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

pub fn now_unix_s() -> f64 {
    unix_seconds_now()
}

pub fn argv() -> Vec<String> {
    std::env::args().collect()
}

/// Ensure an output directory exists, creating it if needed.
pub fn ensure_dir(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path)
        .with_context(|| format!("create output directory {}", path.display()))?;
    Ok(())
}
