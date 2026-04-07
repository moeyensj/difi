//! DIFI: Did I Find It?
//!
//! Classifies linkages as pure, contaminated, or mixed based on their
//! constituent observations' ground-truth object associations. Updates
//! object summaries with linkage statistics.
//!
//! Key algorithmic improvement over v2: all aggregation is done in a single
//! pass per linkage using HashMap accumulators, replacing the chain of
//! ~10 group_by → aggregate → join operations.

use std::collections::{HashMap, HashSet};

use crate::error::{Error, Result};
use crate::partitions::PartitionSummary;
use crate::types::{
    AllLinkages, AllObjects, LinkageMemberSlices, LinkageMemberTable, LinkageSummary,
    ObservationSlices, ObservationTable, compute_night_sorted_indices, indices_in_partition,
};

/// Classify all linkages within a single partition.
///
/// Accepts any `ObservationTable` + `LinkageMemberTable` implementor.
/// The generic boundary is here; internals use concrete slices.
pub fn classify_linkages(
    observations: &impl ObservationTable,
    linkage_members: &impl LinkageMemberTable,
    partition_summary: &PartitionSummary,
    min_obs: usize,
    contamination_percentage: f64,
) -> Result<AllLinkages> {
    let obs = ObservationSlices::from_table(observations);
    let lm = LinkageMemberSlices::from_table(linkage_members);
    classify_linkages_inner(
        &obs,
        &lm,
        partition_summary,
        min_obs,
        contamination_percentage,
    )
}

/// Internal implementation operating on concrete slices.
fn classify_linkages_inner(
    obs: &ObservationSlices<'_>,
    lm: &LinkageMemberSlices<'_>,
    partition_summary: &PartitionSummary,
    min_obs: usize,
    contamination_percentage: f64,
) -> Result<AllLinkages> {
    // Build obs_id -> index lookup
    let obs_id_to_idx: HashMap<u64, usize> =
        obs.ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // Count observations per object in partition (for pure_complete check)
    let night_sorted = compute_night_sorted_indices(obs.nights);
    let partition_indices = indices_in_partition(
        obs.nights,
        &night_sorted,
        partition_summary.start_night,
        partition_summary.end_night,
    );
    let mut obs_per_object_in_partition: HashMap<u64, usize> = HashMap::new();
    for &i in partition_indices {
        if let Some(obj_id) = obs.object_ids[i] {
            *obs_per_object_in_partition.entry(obj_id).or_default() += 1;
        }
    }

    // Single pass: group linkage members by linkage_id, accumulate per-object counts
    struct LinkageAccum {
        object_counts: HashMap<Option<u64>, usize>,
        total_obs: usize,
        outside_partition: usize,
        num_distinct_objects: usize,
        obs_inside_partition: usize,
    }

    let mut linkage_accums: HashMap<u64, LinkageAccum> = HashMap::new();

    for i in 0..lm.len() {
        let linkage_id = lm.linkage_ids[i];
        let obs_id = lm.obs_ids[i];

        let obs_idx = obs_id_to_idx
            .get(&obs_id)
            .ok_or_else(|| Error::InvalidInput(format!("Observation ID {} not found", obs_id)))?;

        let object_id = obs.object_ids[*obs_idx];
        let night = obs.nights[*obs_idx];
        let in_partition =
            night >= partition_summary.start_night && night <= partition_summary.end_night;

        let accum = linkage_accums
            .entry(linkage_id)
            .or_insert_with(|| LinkageAccum {
                object_counts: HashMap::new(),
                total_obs: 0,
                outside_partition: 0,
                num_distinct_objects: 0,
                obs_inside_partition: 0,
            });

        let entry = accum.object_counts.entry(object_id).or_insert(0);
        if *entry == 0 {
            accum.num_distinct_objects += 1;
        }
        *entry += 1;
        accum.total_obs += 1;

        if !in_partition {
            accum.outside_partition += 1;
        } else {
            accum.obs_inside_partition += 1;
        }
    }

    // Classify each linkage
    let mut all_linkages = AllLinkages::default();

    for (linkage_id, accum) in &linkage_accums {
        // Find dominant object (highest count)
        let (dominant_object, dominant_count) = accum
            .object_counts
            .iter()
            .max_by_key(|&(_, &count)| count)
            .map(|(obj, &count)| (*obj, count))
            .unwrap();

        let contamination = if accum.total_obs > 0 {
            (1.0 - dominant_count as f64 / accum.total_obs as f64) * 100.0
        } else {
            0.0
        };

        let pure = contamination == 0.0;
        let contaminated = !pure && contamination <= contamination_percentage;
        let mixed = !pure && !contaminated;

        let linked_object_id = if pure || contaminated {
            dominant_object
        } else {
            None
        };

        // Check pure_complete: does this linkage contain all partition observations
        // of the linked object?
        let pure_complete = if pure {
            if let Some(obj_id) = linked_object_id {
                let expected = obs_per_object_in_partition
                    .get(&obj_id)
                    .copied()
                    .unwrap_or(0);
                expected > 0 && accum.obs_inside_partition == expected
            } else {
                false
            }
        } else {
            false
        };

        let found_pure = pure && accum.total_obs >= min_obs;
        let found_contaminated = contaminated && dominant_count >= min_obs;

        all_linkages.push(LinkageSummary {
            linkage_id: *linkage_id,
            partition_id: partition_summary.id,
            linked_object_id,
            num_obs: accum.total_obs as i64,
            num_obs_outside_partition: accum.outside_partition as i64,
            num_members: accum.num_distinct_objects as i64,
            pure,
            pure_complete,
            contaminated,
            contamination,
            mixed,
            found_pure,
            found_contaminated,
        });
    }

    Ok(all_linkages)
}

/// Update AllObjects with linkage classification statistics.
///
/// Single-pass algorithm: for each linkage member, look up which linkage it
/// belongs to, determine the linkage classification, and accumulate counters
/// on the appropriate object.
pub fn update_all_objects(
    all_objects: &mut AllObjects,
    observations: &impl ObservationTable,
    linkage_members: &impl LinkageMemberTable,
    all_linkages: &AllLinkages,
    min_obs: usize,
) {
    let obs = ObservationSlices::from_table(observations);
    let lm = LinkageMemberSlices::from_table(linkage_members);
    update_all_objects_inner(all_objects, &obs, &lm, all_linkages, min_obs);
}

/// Internal implementation operating on concrete slices.
fn update_all_objects_inner(
    all_objects: &mut AllObjects,
    obs: &ObservationSlices<'_>,
    lm: &LinkageMemberSlices<'_>,
    all_linkages: &AllLinkages,
    min_obs: usize,
) {
    // Build object_id -> AllObjects index lookup
    let mut obj_to_idx: HashMap<u64, usize> = HashMap::new();
    for (i, &obj_id) in all_objects.object_id.iter().enumerate() {
        obj_to_idx.insert(obj_id, i);
    }

    // Build linkage_id -> AllLinkages index lookup
    let mut linkage_to_idx: HashMap<u64, usize> = HashMap::new();
    for (i, &lid) in all_linkages.linkage_id.iter().enumerate() {
        linkage_to_idx.insert(lid, i);
    }

    // Build obs_id -> object_id lookup
    let obs_to_obj: HashMap<u64, Option<u64>> = obs
        .ids
        .iter()
        .zip(obs.object_ids.iter())
        .map(|(&id, &obj)| (id, obj))
        .collect();

    // Group linkage members by (linkage_id, object_id) and count
    struct MembershipInfo {
        linkage_idx: usize,
        count: usize,
    }

    let mut membership: HashMap<(u64, u64), MembershipInfo> = HashMap::new();

    for i in 0..lm.len() {
        let linkage_id = lm.linkage_ids[i];
        let obs_id = lm.obs_ids[i];

        let object_id = match obs_to_obj.get(&obs_id).and_then(|&o| o) {
            Some(id) => id,
            None => continue,
        };

        let linkage_idx = match linkage_to_idx.get(&linkage_id) {
            Some(&idx) => idx,
            None => continue,
        };

        let entry = membership
            .entry((linkage_id, object_id))
            .or_insert_with(|| MembershipInfo {
                linkage_idx,
                count: 0,
            });
        entry.count += 1;
    }

    // Accumulate per-object statistics
    let mut found_pure_linkages: HashMap<u64, HashSet<u64>> = HashMap::new();
    let mut found_contaminated_linkages: HashMap<u64, HashSet<u64>> = HashMap::new();

    for ((linkage_id, object_id), info) in &membership {
        let obj_idx = match obj_to_idx.get(object_id) {
            Some(&idx) => idx,
            None => continue,
        };

        let li = info.linkage_idx;
        let is_linked_object = all_linkages.linked_object_id[li] == Some(*object_id);

        if all_linkages.pure[li] && is_linked_object {
            all_objects.pure[obj_idx] += 1;
            all_objects.obs_in_pure[obj_idx] += info.count as i64;

            if all_linkages.pure_complete[li] {
                all_objects.pure_complete[obj_idx] += 1;
                all_objects.obs_in_pure_complete[obj_idx] += info.count as i64;
            }

            if info.count >= min_obs {
                found_pure_linkages
                    .entry(*object_id)
                    .or_default()
                    .insert(*linkage_id);
            }
        }

        if all_linkages.contaminated[li] {
            if is_linked_object {
                all_objects.contaminated[obj_idx] += 1;
                all_objects.obs_in_contaminated[obj_idx] += info.count as i64;

                if info.count >= min_obs {
                    found_contaminated_linkages
                        .entry(*object_id)
                        .or_default()
                        .insert(*linkage_id);
                }
            } else {
                all_objects.contaminant[obj_idx] += 1;
                all_objects.obs_as_contaminant[obj_idx] += info.count as i64;
            }
        }

        if all_linkages.mixed[li] {
            all_objects.mixed[obj_idx] += 1;
            all_objects.obs_in_mixed[obj_idx] += info.count as i64;
        }
    }

    // Set found counts (number of distinct linkages)
    for (&obj_id, linkages) in &found_pure_linkages {
        if let Some(&idx) = obj_to_idx.get(&obj_id) {
            all_objects.found_pure[idx] = linkages.len() as i64;
        }
    }
    for (&obj_id, linkages) in &found_contaminated_linkages {
        if let Some(&idx) = obj_to_idx.get(&obj_id) {
            all_objects.found_contaminated[idx] = linkages.len() as i64;
        }
    }
}

/// Full DIFI analysis: classify linkages and update object summaries.
///
/// Accepts any `ObservationTable` + `LinkageMemberTable` implementor.
///
/// Returns the classified `AllLinkages`. Mutates `all_objects` and
/// `partition_summary` in place.
pub fn analyze_linkages(
    observations: &impl ObservationTable,
    linkage_members: &impl LinkageMemberTable,
    all_objects: &mut AllObjects,
    partition_summary: &mut PartitionSummary,
    min_obs: usize,
    contamination_percentage: f64,
) -> Result<AllLinkages> {
    let all_linkages = classify_linkages(
        observations,
        linkage_members,
        partition_summary,
        min_obs,
        contamination_percentage,
    )?;

    update_all_objects(
        all_objects,
        observations,
        linkage_members,
        &all_linkages,
        min_obs,
    );

    // Update partition summary
    let mut found_objects: HashSet<u64> = HashSet::new();
    let mut pure_known = 0i64;
    let mut pure_unknown = 0i64;
    let mut contaminated_count = 0i64;
    let mut mixed_count = 0i64;

    for i in 0..all_linkages.len() {
        if all_linkages.pure[i] {
            if all_linkages.linked_object_id[i].is_some() {
                pure_known += 1;
                found_objects.insert(all_linkages.linked_object_id[i].unwrap());
            } else {
                pure_unknown += 1;
            }
        }
        if all_linkages.contaminated[i] {
            contaminated_count += 1;
        }
        if all_linkages.mixed[i] {
            mixed_count += 1;
        }
    }

    let found = found_objects.len() as i64;
    let findable = partition_summary.findable.unwrap_or(0);
    let completeness = if findable > 0 {
        (found as f64 / findable as f64) * 100.0
    } else {
        found as f64 * 100.0
    };

    partition_summary.found = Some(found);
    partition_summary.completeness = Some(completeness);
    partition_summary.pure_known = Some(pure_known);
    partition_summary.pure_unknown = Some(pure_unknown);
    partition_summary.contaminated = Some(contaminated_count);
    partition_summary.mixed = Some(mixed_count);

    Ok(all_linkages)
}
