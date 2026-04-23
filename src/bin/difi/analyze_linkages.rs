//! `difi analyze-linkages` (alias: `analyze`) — run CIFI then DIFI.
//!
//! Supports partition flags (sliding/blocks/single) and CIFI-output reuse via
//! `--cifi-output-dir`. The two are mutually exclusive: a reused CIFI carries
//! its own partition structure, so any non-default partition flag paired with
//! `--cifi-output-dir` is a user error.
//!
//! The library's `update_all_objects` now filters `AllObjects` rows by
//! `partition_id`, so looping DIFI over multi-partition `AllObjects` is
//! correctness-safe.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Args as ClapArgs;

use difi::cifi::analyze_observations as cifi_run;
use difi::difi::analyze_linkages as difi_run;
use difi::io::{
    columns, read_all_objects, read_findable_observations, read_linkage_members,
    read_observations_projected, read_partition_summaries, write_all_linkages,
    write_ignored_linkages,
};
use difi::partitions::PartitionSummary;
use difi::types::{
    AllLinkages, AllObjects, FindableObservations, IgnoredLinkages, ObservationTable,
    StringInterner,
};

use crate::analyze_observations::write_cifi_outputs;
use crate::common::{
    HostInfo, InputFingerprint, Manifest, MetricArgs, PartitionArgs, ProgressEvent, ReusedCifiRef,
    RunContext, ScenarioManifest, WarningsManifest, compute_unique_counts, ensure_dir,
    fingerprint_input, now_unix_s, read_manifest_for_reuse, version_string, write_manifest,
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

    /// Reuse a prior CIFI run. When set, skips the internal CIFI pass and
    /// loads `all_objects.parquet`, `partition_summaries.parquet`, and
    /// `findable_observations.parquet` from this directory. Mutually
    /// exclusive with partition flags — the reused CIFI dictates the
    /// partitions.
    #[arg(long)]
    pub cifi_output_dir: Option<PathBuf>,

    /// Contamination % threshold for the pure/contaminated/mixed split.
    #[arg(long, default_value_t = 20.0)]
    pub contamination_percentage: f64,

    #[command(flatten)]
    pub metric: MetricArgs,

    #[command(flatten)]
    pub partition: PartitionArgs,
}

pub fn run(args: Args, ctx: &RunContext) -> Result<()> {
    ensure_dir(&args.output_dir)?;

    // Reuse vs. fresh-CIFI mode is chosen up front so the rest of the flow
    // is linear. Partition flags are rejected in reuse mode: the reused
    // CIFI already encodes its partitioning.
    if let Some(ref cifi_dir) = args.cifi_output_dir {
        if !args.partition.is_default() {
            bail!(
                "--cifi-output-dir provides its own partition structure; \
                remove --partition-mode / --partition-window / --partition-min-nights"
            );
        }
        run_reuse(&args, cifi_dir.clone(), ctx)
    } else {
        run_fresh(&args, ctx)
    }
}

// ---------------------------------------------------------------------------
// Shared: run DIFI across all partitions
// ---------------------------------------------------------------------------

/// Union AllLinkages column-wise from a list of per-partition tables.
fn concat_all_linkages(parts: Vec<AllLinkages>) -> AllLinkages {
    let mut out = AllLinkages::default();
    for p in parts {
        out.linkage_id.extend(p.linkage_id);
        out.partition_id.extend(p.partition_id);
        out.linked_object_id.extend(p.linked_object_id);
        out.num_obs.extend(p.num_obs);
        out.num_obs_outside_partition
            .extend(p.num_obs_outside_partition);
        out.num_members.extend(p.num_members);
        out.pure.extend(p.pure);
        out.pure_complete.extend(p.pure_complete);
        out.contaminated.extend(p.contaminated);
        out.contamination.extend(p.contamination);
        out.mixed.extend(p.mixed);
        out.found_pure.extend(p.found_pure);
        out.found_contaminated.extend(p.found_contaminated);
    }
    out
}

fn concat_ignored(parts: Vec<IgnoredLinkages>) -> IgnoredLinkages {
    let mut out = IgnoredLinkages::default();
    for p in parts {
        out.extend(p);
    }
    out
}

/// Run DIFI once per partition, mutating `all_objects` and each
/// `partition_summary` in place.
fn run_difi_over_partitions(
    obs: &difi::types::Observations,
    linkages_path: &std::path::Path,
    id_interner: &mut StringInterner,
    all_objects: &mut AllObjects,
    summaries: &mut [PartitionSummary],
    min_obs: usize,
    contamination_percentage: f64,
) -> Result<(AllLinkages, IgnoredLinkages, f64)> {
    let t0 = Instant::now();
    let linkage_members = read_linkage_members(linkages_path, id_interner)
        .with_context(|| format!("read linkage members from {}", linkages_path.display()))?;

    let mut per_partition_al: Vec<AllLinkages> = Vec::with_capacity(summaries.len());
    let mut per_partition_ig: Vec<IgnoredLinkages> = Vec::with_capacity(summaries.len());
    for ps in summaries.iter_mut() {
        let (al, ig) = difi_run(
            obs,
            &linkage_members,
            all_objects,
            ps,
            min_obs,
            contamination_percentage,
        )
        .context("analyze_linkages (DIFI phase) failed")?;
        per_partition_al.push(al);
        per_partition_ig.push(ig);
    }

    Ok((
        concat_all_linkages(per_partition_al),
        concat_ignored(per_partition_ig),
        t0.elapsed().as_secs_f64(),
    ))
}

// ---------------------------------------------------------------------------
// Fresh-CIFI mode
// ---------------------------------------------------------------------------

fn run_fresh(args: &Args, ctx: &RunContext) -> Result<()> {
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

    // --- Run CIFI with user-supplied partition config ---
    let partitions = args.partition.build(&obs.night)?;
    ctx.emit(ProgressEvent::ScenarioStart {
        name: "default",
        metric: args.metric.metric.as_str(),
        partitions: partitions.len(),
    });

    let metric = args.metric.build();
    let cifi_t0 = Instant::now();
    let (mut all_objects, findable, mut summaries) =
        cifi_run(&obs, Some(&partitions), metric.as_ref())
            .context("analyze_observations (CIFI phase) failed")?;
    let cifi_elapsed = cifi_t0.elapsed().as_secs_f64();
    summaries.sort_by_key(|s| s.start_night);

    // --- DIFI across all partitions ---
    let (all_linkages, ignored, difi_elapsed) = run_difi_over_partitions(
        &obs,
        &args.linkages,
        &mut id_interner,
        &mut all_objects,
        &mut summaries,
        args.metric.min_obs,
        args.contamination_percentage,
    )?;

    write_outputs_and_manifest(
        args,
        ctx,
        &id_interner,
        &all_objects,
        &findable,
        &summaries,
        &all_linkages,
        &ignored,
        args.metric.to_manifest(),
        args.partition.to_manifest(),
        cifi_elapsed,
        difi_elapsed,
        obs_fp,
        linkages_fp,
        None,
    )?;

    ctx.emit(ProgressEvent::Done {
        total_elapsed_s: ctx.elapsed_s(),
        output_dir: &args.output_dir,
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// CIFI reuse mode
// ---------------------------------------------------------------------------

fn run_reuse(args: &Args, cifi_dir: PathBuf, ctx: &RunContext) -> Result<()> {
    let reused_all_objects_path = cifi_dir.join("all_objects.parquet");
    let reused_summaries_path = cifi_dir.join("partition_summaries.parquet");
    let reused_findable_path = cifi_dir.join("findable_observations.parquet");
    let reused_manifest_path = cifi_dir.join("run_manifest.json");

    for required in [
        &reused_all_objects_path,
        &reused_summaries_path,
        &reused_manifest_path,
    ] {
        if !required.exists() {
            bail!(
                "--cifi-output-dir missing required file: {}",
                required.display()
            );
        }
    }

    ctx.emit(ProgressEvent::Start {
        subcommand: "analyze-linkages",
        step: "difi",
        input: &cifi_dir.display().to_string(),
    });

    let obs_fp = fingerprint_input(&args.observations)?;
    let linkages_fp = fingerprint_input(&args.linkages)?;

    // --- Fingerprint check against reused manifest ---
    let reused_manifest = read_manifest_for_reuse(&reused_manifest_path)?;
    if reused_manifest.observations_input.sha256_prefix != obs_fp.sha256_prefix {
        bail!(
            "observations fingerprint mismatch: current run has sha256_prefix={} \
            but --cifi-output-dir {} was computed against sha256_prefix={}. \
            The reused CIFI is stale for this observations file.",
            obs_fp.sha256_prefix,
            cifi_dir.display(),
            reused_manifest.observations_input.sha256_prefix,
        );
    }

    // --- Load observations (needed for DIFI obs-index lookups) ---
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

    // --- Load reused CIFI artifacts, sharing the observations' interner ---
    let mut all_objects = read_all_objects(&reused_all_objects_path, &mut id_interner)
        .with_context(|| {
            format!(
                "read reused all_objects from {}",
                reused_all_objects_path.display()
            )
        })?;
    let mut summaries = read_partition_summaries(&reused_summaries_path).with_context(|| {
        format!(
            "read reused partition_summaries from {}",
            reused_summaries_path.display()
        )
    })?;
    summaries.sort_by_key(|s| s.start_night);

    let findable: FindableObservations = if reused_findable_path.exists() {
        read_findable_observations(&reused_findable_path, &mut id_interner).with_context(|| {
            format!(
                "read reused findable_observations from {}",
                reused_findable_path.display()
            )
        })?
    } else {
        FindableObservations::default()
    };

    ctx.emit(ProgressEvent::ScenarioStart {
        name: "default",
        metric: "difi",
        partitions: summaries.len(),
    });

    // --- DIFI across all partitions (CIFI phase skipped) ---
    let (all_linkages, ignored, difi_elapsed) = run_difi_over_partitions(
        &obs,
        &args.linkages,
        &mut id_interner,
        &mut all_objects,
        &mut summaries,
        args.metric.min_obs,
        args.contamination_percentage,
    )?;

    // Reuse the snapshot's recorded metric + partition scheme in the written
    // manifest, since the current invocation's --metric / --partition flags
    // are unused (CIFI was skipped).
    let (scenario_metric, scenario_partitions) = match reused_manifest.scenarios.first() {
        Some(s) => (s.metric.clone(), s.partitions.clone()),
        None => (args.metric.to_manifest(), args.partition.to_manifest()),
    };

    write_outputs_and_manifest(
        args,
        ctx,
        &id_interner,
        &all_objects,
        &findable,
        &summaries,
        &all_linkages,
        &ignored,
        scenario_metric,
        scenario_partitions,
        0.0, // cifi was skipped
        difi_elapsed,
        obs_fp,
        linkages_fp,
        Some(ReusedCifiRef {
            path: cifi_dir.display().to_string(),
            manifest_difi_version: reused_manifest.difi_version,
            manifest_command: reused_manifest.command,
            observations_input: reused_manifest.observations_input,
        }),
    )?;

    ctx.emit(ProgressEvent::Done {
        total_elapsed_s: ctx.elapsed_s(),
        output_dir: &args.output_dir,
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared output path
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn write_outputs_and_manifest(
    args: &Args,
    ctx: &RunContext,
    id_interner: &StringInterner,
    all_objects: &AllObjects,
    findable: &FindableObservations,
    summaries: &[PartitionSummary],
    all_linkages: &AllLinkages,
    ignored: &IgnoredLinkages,
    // scenario_metric / scenario_partitions describe the CIFI pass that
    // produced `all_objects` / `summaries`. In fresh mode these come from the
    // current CLI args; in reuse mode the caller reads them off the reused
    // manifest so the written manifest reflects the snapshot's scheme, not
    // the reuse invocation's (defaulted) flags.
    scenario_metric: serde_json::Value,
    scenario_partitions: serde_json::Value,
    cifi_elapsed: f64,
    difi_elapsed: f64,
    obs_fp: InputFingerprint,
    linkages_fp: InputFingerprint,
    reused_cifi: Option<ReusedCifiRef>,
) -> Result<()> {
    let mut outputs = write_cifi_outputs(
        &args.output_dir,
        all_objects,
        findable,
        summaries,
        id_interner,
    )?;
    let all_linkages_path = args.output_dir.join("all_linkages.parquet");
    write_all_linkages(&all_linkages_path, all_linkages, id_interner)
        .with_context(|| format!("write {}", all_linkages_path.display()))?;
    outputs.insert(
        "all_linkages".to_string(),
        all_linkages_path.display().to_string(),
    );

    // Compute warnings + write ignored_linkages.parquet (only when non-empty,
    // so single-partition runs where nothing is ignored stay clean).
    let warnings = compute_warnings(all_linkages, ignored);
    if !ignored.is_empty() {
        let ignored_path = args.output_dir.join("ignored_linkages.parquet");
        write_ignored_linkages(&ignored_path, ignored, id_interner)
            .with_context(|| format!("write {}", ignored_path.display()))?;
        outputs.insert(
            "ignored_linkages".to_string(),
            ignored_path.display().to_string(),
        );
        emit_ignored_warning(ctx, &warnings, &ignored_path);
    }

    let findable_count: i64 = summaries.iter().filter_map(|s| s.findable).sum();
    let found_count: Option<i64> = {
        let total: i64 = summaries.iter().filter_map(|s| s.found).sum();
        if summaries.iter().any(|s| s.found.is_some()) {
            Some(total)
        } else {
            None
        }
    };

    // Cross-partition distinct-object counts and survey-wide completeness.
    // `findable_count` / `found_count` above are SUMS across partitions (an
    // object findable in K partitions contributes K); these `unique_*` fields
    // answer "how many distinct objects did the linker recover?"
    let (unique_findable_count, unique_found_count_raw) = compute_unique_counts(all_objects);
    let unique_found_count = Some(unique_found_count_raw);
    let unique_completeness = if unique_findable_count > 0 {
        Some(unique_found_count_raw as f64 / unique_findable_count as f64 * 100.0)
    } else {
        None
    };

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
        reused_cifi,
        warnings,
        scenarios: vec![ScenarioManifest {
            name: "default".to_string(),
            metric: scenario_metric,
            partitions: scenario_partitions,
            cifi_elapsed_s: cifi_elapsed,
            difi_elapsed_s: Some(difi_elapsed),
            findable_count,
            found_count,
            unique_findable_count,
            unique_found_count,
            unique_completeness,
            outputs,
        }],
        host: HostInfo::capture(),
        started_at_unix_s: ctx.started_at_unix_s(),
        finished_at_unix_s: now_unix_s(),
    };
    write_manifest(&args.output_dir.join("run_manifest.json"), &manifest)?;
    Ok(())
}

/// Count orphan linkages (appear only in IgnoredLinkages, never in AllLinkages)
/// vs. cross-boundary ones (appear in both across different partitions).
fn compute_warnings(all_linkages: &AllLinkages, ignored: &IgnoredLinkages) -> WarningsManifest {
    use std::collections::HashSet;
    let classified: HashSet<u64> = all_linkages.linkage_id.iter().copied().collect();
    let mut ignored_ids: HashSet<u64> = HashSet::new();
    for &lid in &ignored.linkage_id {
        ignored_ids.insert(lid);
    }
    let orphan_linkages = ignored_ids
        .iter()
        .filter(|lid| !classified.contains(lid))
        .count();
    WarningsManifest {
        ignored_linkage_rows: ignored.len(),
        orphan_linkages,
    }
}

fn emit_ignored_warning(ctx: &RunContext, w: &WarningsManifest, path: &std::path::Path) {
    let cross_boundary_rows = w.ignored_linkage_rows.saturating_sub(w.orphan_linkages);
    let message = if w.orphan_linkages > 0 {
        format!(
            "{} linkage×partition row(s) excluded to {} — {} orphan linkage(s) had no \
             observations in ANY partition (likely wrong --linkages file for this partition \
             scheme); {} cross-boundary row(s) were partially outside individual partitions",
            w.ignored_linkage_rows,
            path.display(),
            w.orphan_linkages,
            cross_boundary_rows,
        )
    } else {
        format!(
            "{} cross-boundary linkage×partition row(s) excluded to {} — linkages partially \
             outside individual partitions (usually benign)",
            w.ignored_linkage_rows,
            path.display(),
        )
    };
    ctx.emit_warning(
        &message,
        serde_json::json!({
            "kind": "ignored_linkages",
            "ignored_linkage_rows": w.ignored_linkage_rows,
            "orphan_linkages": w.orphan_linkages,
            "cross_boundary_rows": cross_boundary_rows,
            "path": path.display().to_string(),
        }),
    );
}
