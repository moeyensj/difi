//! `difi analyze-linkages` (alias: `analyze`) — run CIFI then DIFI end-to-end.
//!
//! Single-partition only in v1: partition flags are not exposed. The library's
//! DIFI update path keys AllObjects rows by object_id alone, so multi-partition
//! DIFI would mis-attribute linkage stats across partitions. This matches the
//! Python `analyze_linkages` API.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Args as ClapArgs;

use difi::cifi::analyze_observations as cifi_run;
use difi::difi::analyze_linkages as difi_run;
use difi::io::{columns, read_linkage_members, read_observations_projected, write_all_linkages};
use difi::partitions::{self, PartitionSummary};
use difi::types::{ObservationTable, StringInterner};

use crate::analyze_observations::write_cifi_outputs;
use crate::common::{
    HostInfo, Manifest, MetricArgs, ProgressEvent, RunContext, ScenarioManifest, ensure_dir,
    fingerprint_input, now_unix_s, version_string, write_manifest,
};

#[derive(ClapArgs, Debug)]
pub struct Args {
    /// Parquet file with the difi observations schema.
    #[arg(short = 'i', long)]
    pub observations: PathBuf,

    /// Parquet file with linkage_id + obs_id columns.
    #[arg(short = 'l', long)]
    pub linkages: PathBuf,

    /// Output directory (created if missing).
    #[arg(short = 'o', long)]
    pub output_dir: PathBuf,

    /// Contamination % threshold for the pure/contaminated/mixed split.
    #[arg(long, default_value_t = 20.0)]
    pub contamination_percentage: f64,

    #[command(flatten)]
    pub metric: MetricArgs,
}

pub fn run(args: Args, ctx: &RunContext) -> Result<()> {
    ensure_dir(&args.output_dir)?;

    ctx.emit(ProgressEvent::Start {
        subcommand: "analyze-linkages",
        step: "cifi",
        input: &args.observations.display().to_string(),
    });

    let obs_fp = fingerprint_input(&args.observations)?;
    let linkages_fp = fingerprint_input(&args.linkages)?;

    // --- Load observations ---
    let load_t0 = Instant::now();
    let (obs, mut id_interner, _obs_code_interner) =
        read_observations_projected(&args.observations, Some(columns::CIFI))
            .with_context(|| format!("read observations from {}", args.observations.display()))?;
    ctx.emit(ProgressEvent::LoadedObservations {
        count: obs.len(),
        elapsed_s: load_t0.elapsed().as_secs_f64(),
    });
    if obs.len() == 0 {
        bail!("no observations in {}", args.observations.display());
    }

    // --- Run CIFI (single partition spanning all nights) ---
    let partition = partitions::create_single(&obs.night)?;
    let partitions_slice = vec![partition];

    let metric = args.metric.build();
    let cifi_t0 = Instant::now();
    let (mut all_objects, findable, mut summaries) =
        cifi_run(&obs, Some(&partitions_slice), metric.as_ref())
            .context("analyze_observations (CIFI phase) failed")?;
    let cifi_elapsed = cifi_t0.elapsed().as_secs_f64();

    // We expect exactly one partition summary.
    if summaries.len() != 1 {
        bail!(
            "expected 1 partition summary, got {} — analyze-linkages is single-partition only",
            summaries.len()
        );
    }
    let mut partition_summary: PartitionSummary = summaries.remove(0);

    // --- Load linkage members, sharing the observations' id_interner ---
    let (all_linkages, difi_elapsed) = run_difi_phase(
        &args.linkages,
        &obs,
        &mut id_interner,
        &mut all_objects,
        &mut partition_summary,
        args.metric.min_obs,
        args.contamination_percentage,
        ctx,
    )?;

    // --- Write outputs ---
    let final_summaries = vec![partition_summary.clone()];
    let mut outputs = write_cifi_outputs(
        &args.output_dir,
        &all_objects,
        &findable,
        &final_summaries,
        &id_interner,
    )?;
    let all_linkages_path = args.output_dir.join("all_linkages.parquet");
    write_all_linkages(&all_linkages_path, &all_linkages, &id_interner)
        .with_context(|| format!("write {}", all_linkages_path.display()))?;
    outputs.insert(
        "all_linkages".to_string(),
        all_linkages_path.display().to_string(),
    );

    let findable_count = partition_summary.findable.unwrap_or(0);
    let found_count = partition_summary.found;

    ctx.emit(ProgressEvent::ScenarioDone {
        name: "default",
        findable: findable_count,
        found: found_count,
        elapsed_s: cifi_elapsed + difi_elapsed,
    });

    let manifest = Manifest {
        difi_version: version_string(),
        command: ctx.command.clone(),
        observations_input: obs_fp,
        linkages_input: Some(linkages_fp),
        scenarios: vec![ScenarioManifest {
            name: "default".to_string(),
            metric: args.metric.to_manifest(),
            partitions: serde_json::json!({ "mode": "single" }),
            cifi_elapsed_s: cifi_elapsed,
            difi_elapsed_s: Some(difi_elapsed),
            findable_count,
            found_count,
            outputs,
        }],
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

#[allow(clippy::too_many_arguments)]
fn run_difi_phase(
    linkages_path: &Path,
    obs: &difi::types::Observations,
    id_interner: &mut StringInterner,
    all_objects: &mut difi::types::AllObjects,
    partition_summary: &mut PartitionSummary,
    min_obs: usize,
    contamination_percentage: f64,
    ctx: &RunContext,
) -> Result<(difi::types::AllLinkages, f64)> {
    let t0 = Instant::now();
    let linkage_members = read_linkage_members(linkages_path, id_interner)
        .with_context(|| format!("read linkage members from {}", linkages_path.display()))?;

    ctx.emit(ProgressEvent::ScenarioStart {
        name: "default",
        metric: "difi",
        partitions: 1,
    });

    let all_linkages = difi_run(
        obs,
        &linkage_members,
        all_objects,
        partition_summary,
        min_obs,
        contamination_percentage,
    )
    .context("analyze_linkages (DIFI phase) failed")?;

    Ok((all_linkages, t0.elapsed().as_secs_f64()))
}
