//! `difi analyze-observations` (alias: `cifi`) — run CIFI on a parquet input.
//!
//! Single-scenario mode: inline flags on the command line.
//! Batch mode: `--scenarios <FILE>.toml`, one named scenario per directory.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Args as ClapArgs;

use difi::cifi::analyze_observations;
use difi::io::{
    columns, read_observations_projected, write_all_objects, write_findable_observations,
    write_partition_summaries,
};
use difi::partitions::PartitionSummary;
use difi::types::{
    AllObjects, FindableObservations, ObservationTable, Observations, StringInterner,
};

use crate::common::{
    self, HostInfo, InputFingerprint, Manifest, MetricArgs, PartitionArgs, ProgressEvent,
    RunContext, ScenarioEntry, ScenarioManifest, ensure_dir, fingerprint_input, now_unix_s,
    read_scenarios, version_string, write_manifest,
};

#[derive(ClapArgs, Debug)]
pub struct Args {
    /// Parquet file with the difi observations schema.
    #[arg(short = 'i', long)]
    pub observations: Option<PathBuf>,

    /// Output directory (created if missing). Single scenarios write files
    /// here directly; batch scenarios write to subdirectories.
    #[arg(short = 'o', long)]
    pub output_dir: PathBuf,

    /// Run a batch of scenarios defined in a TOML file instead of a single
    /// scenario from CLI flags.
    #[arg(long)]
    pub scenarios: Option<PathBuf>,

    #[command(flatten)]
    pub metric: MetricArgs,

    #[command(flatten)]
    pub partition: PartitionArgs,
}

pub fn run(args: Args, ctx: &RunContext) -> Result<()> {
    ensure_dir(&args.output_dir)?;

    if let Some(scenarios_path) = &args.scenarios {
        run_batch(&args, scenarios_path, ctx)
    } else {
        run_single(&args, ctx)
    }
}

// ---------------------------------------------------------------------------
// Single-scenario mode
// ---------------------------------------------------------------------------

fn run_single(args: &Args, ctx: &RunContext) -> Result<()> {
    let obs_path = args.observations.as_ref().ok_or_else(|| {
        anyhow::anyhow!("--observations is required (or pass --scenarios <FILE>)")
    })?;

    ctx.emit(ProgressEvent::Start {
        subcommand: "analyze-observations",
        step: "cifi",
        input: &obs_path.display().to_string(),
    });

    let obs_fp = fingerprint_input(obs_path)?;

    let load_t0 = Instant::now();
    let (obs, id_interner, _obs_code_interner) =
        read_observations_projected(obs_path, Some(columns::CIFI))
            .with_context(|| format!("read observations from {}", obs_path.display()))?;
    ctx.emit(ProgressEvent::LoadedObservations {
        count: obs.len(),
        elapsed_s: load_t0.elapsed().as_secs_f64(),
    });
    if obs.len() == 0 {
        bail!("no observations in {}", obs_path.display());
    }

    let scenario_outputs = run_cifi_scenario(
        "default",
        &obs,
        &id_interner,
        &args.metric,
        &args.partition,
        &args.output_dir,
        ctx,
    )?;

    // Write top-level manifest.
    let manifest = Manifest {
        difi_version: version_string(),
        command: ctx.command.clone(),
        observations_input: obs_fp,
        linkages_input: None,
        scenarios: vec![scenario_outputs],
        host: HostInfo::capture(),
        started_at_unix_s: ctx.started_at_unix_s(),
        finished_at_unix_s: now_unix_s(),
    };
    write_manifest(&args.output_dir.join("run_manifest.json"), &manifest)?;

    ctx.emit(ProgressEvent::Done {
        total_elapsed_s: ctx.elapsed_s(),
        output_dir: &args.output_dir,
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Batch scenarios mode
// ---------------------------------------------------------------------------

fn run_batch(args: &Args, scenarios_path: &Path, ctx: &RunContext) -> Result<()> {
    let scenarios = read_scenarios(scenarios_path)?;

    ctx.emit(ProgressEvent::Start {
        subcommand: "analyze-observations",
        step: "cifi",
        input: &scenarios_path.display().to_string(),
    });

    // Resolve each scenario's observations path (per-scenario override, else
    // defaults, else the CLI --observations flag).
    let mut scenario_manifests = Vec::new();
    let mut fingerprint_cache: BTreeMap<PathBuf, InputFingerprint> = BTreeMap::new();
    let mut obs_cache: BTreeMap<PathBuf, (Observations, StringInterner)> = BTreeMap::new();

    // We track the first observations input for the top-level manifest.
    let mut top_level_obs_fp: Option<InputFingerprint> = None;

    for entry in &scenarios.scenarios {
        let obs_path = resolve_obs_path(entry, &scenarios, args)?;

        // Cache fingerprint per unique input path.
        let fp = if let Some(fp) = fingerprint_cache.get(&obs_path) {
            fp.clone()
        } else {
            let fp = fingerprint_input(&obs_path)?;
            fingerprint_cache.insert(obs_path.clone(), fp.clone());
            fp
        };
        if top_level_obs_fp.is_none() {
            top_level_obs_fp = Some(fp.clone());
        }

        // Load observations (cached per unique path).
        if !obs_cache.contains_key(&obs_path) {
            let load_t0 = Instant::now();
            let (obs, id_interner, _obs_code_interner) =
                read_observations_projected(&obs_path, Some(columns::CIFI))
                    .with_context(|| format!("read observations from {}", obs_path.display()))?;
            ctx.emit(ProgressEvent::LoadedObservations {
                count: obs.len(),
                elapsed_s: load_t0.elapsed().as_secs_f64(),
            });
            if obs.len() == 0 {
                bail!("no observations in {}", obs_path.display());
            }
            obs_cache.insert(obs_path.clone(), (obs, id_interner));
        }
        let (obs, id_interner) = obs_cache.get(&obs_path).unwrap();

        let scen_dir = args.output_dir.join(&entry.name);
        ensure_dir(&scen_dir)?;

        let metric = entry.to_metric_args();
        let partition = entry.to_partition_args();
        let sm = run_cifi_scenario(
            &entry.name,
            obs,
            id_interner,
            &metric,
            &partition,
            &scen_dir,
            ctx,
        )?;
        scenario_manifests.push(sm);
    }

    let obs_fp = top_level_obs_fp.ok_or_else(|| anyhow::anyhow!("no scenarios executed"))?;
    let manifest = Manifest {
        difi_version: version_string(),
        command: ctx.command.clone(),
        observations_input: obs_fp,
        linkages_input: None,
        scenarios: scenario_manifests,
        host: HostInfo::capture(),
        started_at_unix_s: ctx.started_at_unix_s(),
        finished_at_unix_s: now_unix_s(),
    };
    write_manifest(&args.output_dir.join("run_manifest.json"), &manifest)?;

    ctx.emit(ProgressEvent::Done {
        total_elapsed_s: ctx.elapsed_s(),
        output_dir: &args.output_dir,
    });
    Ok(())
}

/// Resolve the observations path for a scenario.
///
/// Precedence (most explicit wins, then command-line overrides file defaults):
///   1. `[[scenario]] observations = "..."` — per-scenario entry
///   2. `--observations` CLI flag — ephemeral override of the file's default
///   3. `[defaults] observations = "..."` — file-level default
fn resolve_obs_path(
    entry: &ScenarioEntry,
    scenarios: &common::ScenariosFile,
    args: &Args,
) -> Result<PathBuf> {
    entry
        .observations
        .clone()
        .or_else(|| args.observations.clone())
        .or_else(|| scenarios.defaults.observations.clone())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "scenario {} has no observations path (set per-scenario, --observations, or [defaults])",
                entry.name
            )
        })
}

// ---------------------------------------------------------------------------
// Core CIFI run (shared by single and batch)
// ---------------------------------------------------------------------------

fn run_cifi_scenario(
    name: &str,
    obs: &Observations,
    id_interner: &StringInterner,
    metric_args: &MetricArgs,
    partition_args: &PartitionArgs,
    out_dir: &Path,
    ctx: &RunContext,
) -> Result<ScenarioManifest> {
    let metric = metric_args.build();
    let partitions = partition_args.build(&obs.night)?;

    ctx.emit(ProgressEvent::ScenarioStart {
        name,
        metric: metric_args.metric.as_str(),
        partitions: partitions.len(),
    });

    let t0 = Instant::now();
    let (all_objects, findable, mut summaries) =
        analyze_observations(obs, Some(&partitions), metric.as_ref())
            .context("analyze_observations failed")?;
    let elapsed = t0.elapsed().as_secs_f64();

    // Sort summaries by start_night for readable output.
    summaries.sort_by_key(|s| s.start_night);

    let findable_count: i64 = summaries.iter().filter_map(|s| s.findable).sum();

    // Write outputs.
    let outputs = write_cifi_outputs(out_dir, &all_objects, &findable, &summaries, id_interner)?;

    ctx.emit(ProgressEvent::ScenarioDone {
        name,
        findable: findable_count,
        found: None,
        elapsed_s: elapsed,
    });

    Ok(ScenarioManifest {
        name: name.to_string(),
        metric: metric_args.to_manifest(),
        partitions: partition_args.to_manifest(),
        cifi_elapsed_s: elapsed,
        difi_elapsed_s: None,
        findable_count,
        found_count: None,
        outputs,
    })
}

pub(crate) fn write_cifi_outputs(
    out_dir: &Path,
    all_objects: &AllObjects,
    findable: &FindableObservations,
    summaries: &[PartitionSummary],
    id_interner: &StringInterner,
) -> Result<BTreeMap<String, String>> {
    let all_objects_path = out_dir.join("all_objects.parquet");
    let findable_path = out_dir.join("findable_observations.parquet");
    let summaries_path = out_dir.join("partition_summaries.parquet");

    write_all_objects(&all_objects_path, all_objects, id_interner)
        .with_context(|| format!("write {}", all_objects_path.display()))?;
    write_findable_observations(&findable_path, findable, id_interner)
        .with_context(|| format!("write {}", findable_path.display()))?;
    write_partition_summaries(&summaries_path, summaries)
        .with_context(|| format!("write {}", summaries_path.display()))?;

    let mut outputs = BTreeMap::new();
    outputs.insert(
        "all_objects".to_string(),
        all_objects_path.display().to_string(),
    );
    outputs.insert(
        "findable_observations".to_string(),
        findable_path.display().to_string(),
    );
    outputs.insert(
        "partition_summaries".to_string(),
        summaries_path.display().to_string(),
    );
    Ok(outputs)
}
