//! CIFI: Can I Find It?
//!
//! Determines which objects are "findable" based on their observation patterns
//! and computes per-object summaries within each partition.

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::error::Result;
use crate::metrics::FindabilityMetric;
use crate::partitions::{self, Partition, PartitionSummary};
use crate::types::{AllObjects, FindableObservations, ObjectSummary, Observations};

/// Analyze observations to determine findability and build object summaries.
///
/// Returns (AllObjects, FindableObservations, Vec<PartitionSummary>).
pub fn analyze_observations(
    observations: &Observations,
    partitions: Option<&[Partition]>,
    metric: &dyn FindabilityMetric,
) -> Result<(AllObjects, FindableObservations, Vec<PartitionSummary>)> {
    if observations.is_empty() {
        return Ok((
            AllObjects::default(),
            FindableObservations::default(),
            Vec::new(),
        ));
    }

    // Default to single partition if none provided
    let owned_partitions;
    let partitions = match partitions {
        Some(p) => p,
        None => {
            owned_partitions = vec![partitions::create_single(&observations.night)?];
            &owned_partitions
        }
    };

    // Create partition summaries
    let mut summaries = partitions::create_summaries(
        &observations.night,
        partitions,
        &observations.night_sorted_indices,
    );

    // Group observation indices by object_id (skip observations without object_id)
    let mut object_indices: HashMap<u64, Vec<usize>> = HashMap::new();
    for i in 0..observations.len() {
        if let Some(obj_id) = observations.object_id[i] {
            object_indices.entry(obj_id).or_default().push(i);
        }
    }

    // Run findability metric per object in parallel
    let findable_results: Vec<_> = object_indices
        .par_iter()
        .flat_map(|(_, indices)| {
            metric.determine_object_findable(indices, observations, partitions)
        })
        .collect();

    // Collect into FindableObservations
    let mut findable = FindableObservations::default();
    for fo in findable_results {
        findable.push(fo);
    }

    // Update partition summaries with findable counts
    for summary in &mut summaries {
        let findable_count = findable
            .partition_id
            .iter()
            .zip(findable.object_id.iter())
            .filter(|&(&pid, _)| pid == summary.id)
            .map(|(_, &oid)| oid)
            .collect::<HashSet<_>>()
            .len();
        summary.findable = Some(findable_count as i64);
    }

    // Build AllObjects: per-object, per-partition summaries
    let findable_set: HashSet<(u64, u64)> = findable
        .partition_id
        .iter()
        .zip(findable.object_id.iter())
        .map(|(&pid, &oid)| (pid, oid))
        .collect();

    let mut all_objects = AllObjects::default();

    for partition in partitions {
        let indices = observations.indices_in_partition(partition.start_night, partition.end_night);

        // Group by object within this partition
        let mut partition_objects: HashMap<u64, Vec<usize>> = HashMap::new();
        for &i in indices {
            if let Some(obj_id) = observations.object_id[i] {
                partition_objects.entry(obj_id).or_default().push(i);
            }
        }

        for (obj_id, obj_indices) in &partition_objects {
            let mut mjd_min = f64::INFINITY;
            let mut mjd_max = f64::NEG_INFINITY;
            let mut observatories = HashSet::new();

            for &i in obj_indices {
                let t = observations.time_mjd[i];
                if t < mjd_min {
                    mjd_min = t;
                }
                if t > mjd_max {
                    mjd_max = t;
                }
                observatories.insert(observations.observatory_code[i]);
            }

            all_objects.push(ObjectSummary {
                object_id: *obj_id,
                partition_id: partition.id,
                mjd_min,
                mjd_max,
                arc_length: mjd_max - mjd_min,
                num_obs: obj_indices.len() as i64,
                num_observatories: observatories.len() as i64,
                findable: Some(findable_set.contains(&(partition.id, *obj_id))),
                found_pure: 0,
                found_contaminated: 0,
                pure: 0,
                pure_complete: 0,
                contaminated: 0,
                contaminant: 0,
                mixed: 0,
                obs_in_pure: 0,
                obs_in_pure_complete: 0,
                obs_in_contaminated: 0,
                obs_as_contaminant: 0,
                obs_in_mixed: 0,
            });
        }
    }

    Ok((all_objects, findable, summaries))
}
