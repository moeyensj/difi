//! DIFI: Did I Find It?
//!
//! Classifies linkages as pure, contaminated, or mixed based on their
//! constituent observations' ground-truth object associations. Updates
//! object summaries with linkage statistics.
//!
//! Key algorithmic improvements over v2:
//! - Single-pass aggregation replacing ~10 group_by → aggregate → join ops
//! - Sorted ID index + binary search instead of HashMap for obs lookups
//!   (eliminates ~50 bytes/entry HashMap at survey scale)

use std::collections::{HashMap, HashSet};

use crate::error::{Error, Result};
use crate::partitions::PartitionSummary;
use crate::types::{
    AllLinkages, AllObjects, IgnoredLinkage, IgnoredLinkageReason, IgnoredLinkages,
    LinkageMemberSlices, LinkageMemberTable, LinkageSummary, NO_OBJECT, ObservationSlices,
    ObservationTable, compute_id_sorted_indices, compute_night_sorted_indices,
    indices_in_partition, lookup_by_id,
};

/// Classify all linkages within a single partition.
///
/// Returns `(AllLinkages, IgnoredLinkages)`. A linkage whose observations all
/// fall outside the partition's night range is excluded from `AllLinkages` and
/// reported in `IgnoredLinkages` with reason `NoObservationsInPartition` —
/// including such "phantom" classifications in the output would produce
/// misleading rows where `num_obs_outside_partition == num_obs` and double-
/// count under cross-partition aggregation.
///
/// Accepts any `ObservationTable` + `LinkageMemberTable` implementor.
/// The generic boundary is here; internals use concrete slices.
pub fn classify_linkages(
    observations: &impl ObservationTable,
    linkage_members: &impl LinkageMemberTable,
    partition_summary: &PartitionSummary,
    min_obs: usize,
    contamination_percentage: f64,
) -> Result<(AllLinkages, IgnoredLinkages)> {
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
) -> Result<(AllLinkages, IgnoredLinkages)> {
    // Build sorted ID index for O(log n) lookups (replaces HashMap)
    let id_sorted = compute_id_sorted_indices(obs.ids);

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
        let obj_id = obs.object_ids[i];
        if obj_id != NO_OBJECT {
            *obs_per_object_in_partition.entry(obj_id).or_default() += 1;
        }
    }

    // Single pass: group linkage members by linkage_id, accumulate per-object counts.
    // `object_counts_in_partition` mirrors `object_counts` but scoped to obs
    // whose night falls inside [start_night, end_night], so per-partition
    // found_* determinations use in-partition counts rather than whole-linkage
    // totals (which could let a linkage with few in-partition obs inflate
    // completeness above 100%).
    struct LinkageAccum {
        object_counts: HashMap<u64, usize>,
        object_counts_in_partition: HashMap<u64, usize>,
        total_obs: usize,
        outside_partition: usize,
        num_distinct_objects: usize,
        obs_inside_partition: usize,
    }

    let mut linkage_accums: HashMap<u64, LinkageAccum> = HashMap::new();

    for i in 0..lm.len() {
        let linkage_id = lm.linkage_ids[i];
        let obs_id = lm.obs_ids[i];

        // Binary search instead of HashMap lookup
        let obs_idx = lookup_by_id(obs.ids, &id_sorted, obs_id)
            .ok_or_else(|| Error::InvalidInput(format!("Observation ID {obs_id} not found")))?;

        let object_id = obs.object_ids[obs_idx];
        let night = obs.nights[obs_idx];
        let in_partition =
            night >= partition_summary.start_night && night <= partition_summary.end_night;

        let accum = linkage_accums
            .entry(linkage_id)
            .or_insert_with(|| LinkageAccum {
                object_counts: HashMap::new(),
                object_counts_in_partition: HashMap::new(),
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
            *accum
                .object_counts_in_partition
                .entry(object_id)
                .or_insert(0) += 1;
        }
    }

    // Classify each linkage
    let mut all_linkages = AllLinkages::default();
    let mut ignored = IgnoredLinkages::default();

    for (linkage_id, accum) in &linkage_accums {
        // Linkages with zero observations inside the partition are "phantom"
        // classifications — they'd claim pure/contaminated/mixed status for a
        // linkage that doesn't actually touch this partition. Emit an
        // IgnoredLinkage record instead so callers can surface user errors
        // (wrong linkage file) without polluting AllLinkages.
        if accum.obs_inside_partition == 0 {
            ignored.push(IgnoredLinkage {
                linkage_id: *linkage_id,
                partition_id: partition_summary.id,
                reason: IgnoredLinkageReason::NoObservationsInPartition,
                num_obs: accum.total_obs as i64,
                num_members: accum.num_distinct_objects as i64,
            });
            continue;
        }

        // Find dominant object (highest count)
        let (dominant_object, dominant_count) = accum
            .object_counts
            .iter()
            .max_by_key(|&(_, &count)| count)
            .map(|(&obj, &count)| (obj, count))
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
            NO_OBJECT
        };

        // Check pure_complete: does this linkage contain all partition observations
        // of the linked object?
        let pure_complete = if pure && linked_object_id != NO_OBJECT {
            let expected = obs_per_object_in_partition
                .get(&linked_object_id)
                .copied()
                .unwrap_or(0);
            expected > 0 && accum.obs_inside_partition == expected
        } else {
            false
        };

        // "Found" semantics are partition-scoped: a linkage only counts as
        // finding an object in partition P if it has >= min_obs observations
        // of that object *inside* P. Using whole-linkage counts (total_obs /
        // dominant_count) lets cross-boundary linkages inflate
        // partition_summary.completeness above 100%. For single-partition
        // runs these are identical (no obs outside the partition).
        let dominant_count_in_partition = accum
            .object_counts_in_partition
            .get(&dominant_object)
            .copied()
            .unwrap_or(0);
        let found_pure = pure && accum.obs_inside_partition >= min_obs;
        let found_contaminated = contaminated && dominant_count_in_partition >= min_obs;

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

    Ok((all_linkages, ignored))
}

/// Update AllObjects with linkage classification statistics for a single partition.
///
/// `partition_summary.id` scopes the update: only `AllObjects` rows whose
/// `partition_id` matches are touched. This lets callers loop over partitions
/// safely when `AllObjects` contains rows from a multi-partition CIFI run.
///
/// Uses sorted index + binary search for obs lookups instead of HashMap.
pub fn update_all_objects(
    all_objects: &mut AllObjects,
    observations: &impl ObservationTable,
    linkage_members: &impl LinkageMemberTable,
    all_linkages: &AllLinkages,
    partition_summary: &PartitionSummary,
    min_obs: usize,
) {
    let obs = ObservationSlices::from_table(observations);
    let lm = LinkageMemberSlices::from_table(linkage_members);
    update_all_objects_inner(
        all_objects,
        &obs,
        &lm,
        all_linkages,
        partition_summary.id,
        min_obs,
    );
}

/// Internal implementation operating on concrete slices.
fn update_all_objects_inner(
    all_objects: &mut AllObjects,
    obs: &ObservationSlices<'_>,
    lm: &LinkageMemberSlices<'_>,
    all_linkages: &AllLinkages,
    partition_id: u64,
    min_obs: usize,
) {
    // Build sorted ID index for obs lookups
    let id_sorted = compute_id_sorted_indices(obs.ids);

    // Build object_id -> AllObjects index lookup, restricted to the target
    // partition. Without this filter, a multi-partition AllObjects would
    // collapse rows (same object_id across partitions would overwrite) and
    // a later partition's linkage stats would be written to an unrelated row.
    let mut obj_to_idx: HashMap<u64, usize> = HashMap::new();
    for (i, (&obj_id, &pid)) in all_objects
        .object_id
        .iter()
        .zip(all_objects.partition_id.iter())
        .enumerate()
    {
        if pid == partition_id {
            obj_to_idx.insert(obj_id, i);
        }
    }

    // Build linkage_id -> AllLinkages index lookup (small — proportional to linkages)
    let mut linkage_to_idx: HashMap<u64, usize> = HashMap::new();
    for (i, &lid) in all_linkages.linkage_id.iter().enumerate() {
        linkage_to_idx.insert(lid, i);
    }

    // Group linkage members by (linkage_id, object_id) and count
    struct MembershipInfo {
        linkage_idx: usize,
        count: usize,
    }

    let mut membership: HashMap<(u64, u64), MembershipInfo> = HashMap::new();

    for i in 0..lm.len() {
        let linkage_id = lm.linkage_ids[i];
        let obs_id = lm.obs_ids[i];

        // Binary search for object_id
        let object_id = match lookup_by_id(obs.ids, &id_sorted, obs_id) {
            Some(idx) => {
                let oid = obs.object_ids[idx];
                if oid == NO_OBJECT {
                    continue;
                }
                oid
            }
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
        let is_linked_object = all_linkages.linked_object_id[li] == *object_id;

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
/// Returns `(AllLinkages, IgnoredLinkages)`. Mutates `all_objects` and
/// `partition_summary` in place.
pub fn analyze_linkages(
    observations: &impl ObservationTable,
    linkage_members: &impl LinkageMemberTable,
    all_objects: &mut AllObjects,
    partition_summary: &mut PartitionSummary,
    min_obs: usize,
    contamination_percentage: f64,
) -> Result<(AllLinkages, IgnoredLinkages)> {
    let (all_linkages, ignored) = classify_linkages(
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
        partition_summary,
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
            // pure_known / pure_unknown describe the linkage population and
            // stay on raw pure-ness.
            let lid = all_linkages.linked_object_id[i];
            if lid != NO_OBJECT {
                pure_known += 1;
            } else {
                pure_unknown += 1;
            }
        }
        // `found` / `completeness` use the stricter found_pure: an object is
        // "found" in this partition only via a pure linkage with >= min_obs
        // observations inside the partition. Keeps completeness bounded by
        // the linker's in-partition evidence rather than whole-linkage totals.
        if all_linkages.found_pure[i] {
            let lid = all_linkages.linked_object_id[i];
            if lid != NO_OBJECT {
                found_objects.insert(lid);
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

    Ok((all_linkages, ignored))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{LinkageMembers, LinkageSummary, ObjectSummary, Observations};

    fn ps(id: u64, start: i64, end: i64) -> PartitionSummary {
        PartitionSummary {
            id,
            start_night: start,
            end_night: end,
            observations: 0,
            findable: Some(1),
            found: None,
            completeness: None,
            pure_known: None,
            pure_unknown: None,
            contaminated: None,
            mixed: None,
        }
    }

    fn os(object_id: u64, partition_id: u64) -> ObjectSummary {
        ObjectSummary {
            object_id,
            partition_id,
            mjd_min: 0.0,
            mjd_max: 0.0,
            arc_length: 0.0,
            num_obs: 0,
            num_observatories: 0,
            findable: Some(true),
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
        }
    }

    #[test]
    fn update_all_objects_only_touches_target_partition_rows() {
        // Multi-partition AllObjects: object 42 has a row in partitions 0 and 1.
        // Calling update_all_objects with partition_summary.id = 0 must not
        // touch row 1. This is the exact regression the Phase 1 fix prevents.
        let mut all_objects = AllObjects::default();
        all_objects.push(os(42, 0));
        all_objects.push(os(42, 1));

        // One pure linkage classifying object 42 within partition 0.
        let mut all_linkages = AllLinkages::default();
        all_linkages.push(LinkageSummary {
            linkage_id: 100,
            partition_id: 0,
            linked_object_id: 42,
            num_obs: 6,
            num_obs_outside_partition: 0,
            num_members: 1,
            pure: true,
            pure_complete: false,
            contaminated: false,
            contamination: 0.0,
            mixed: false,
            found_pure: true,
            found_contaminated: false,
        });

        let obs = Observations::new(
            (0u64..6).collect(),
            vec![60000.0; 6],
            vec![0.0; 6],
            vec![0.0; 6],
            vec![0u32; 6],
            vec![42u64; 6],
            vec![1, 1, 2, 2, 3, 3],
        );
        let lm = LinkageMembers {
            linkage_id: vec![100u64; 6],
            obs_id: (0u64..6).collect(),
        };

        let partition_0 = ps(0, 1, 5);
        update_all_objects(&mut all_objects, &obs, &lm, &all_linkages, &partition_0, 6);

        // Partition 0 row should be updated
        assert_eq!(all_objects.partition_id[0], 0);
        assert_eq!(all_objects.pure[0], 1, "partition 0 row should be updated");
        assert_eq!(all_objects.found_pure[0], 1);
        assert_eq!(all_objects.obs_in_pure[0], 6);

        // Partition 1 row must be byte-for-byte unchanged (all zeros as initialized)
        assert_eq!(all_objects.partition_id[1], 1);
        assert_eq!(
            all_objects.pure[1], 0,
            "partition 1 row must not be touched"
        );
        assert_eq!(all_objects.found_pure[1], 0);
        assert_eq!(all_objects.obs_in_pure[1], 0);
        assert_eq!(all_objects.contaminated[1], 0);
        assert_eq!(all_objects.mixed[1], 0);
    }

    #[test]
    fn classify_linkages_excludes_wholly_external_and_reports_as_ignored() {
        // Linkage L1 has all 6 obs in partition 0 (nights 1-5).
        // Linkage L2 has all 6 obs in partition 1 (nights 6-10).
        // Running classify_linkages targeting partition 0 should put L1 in
        // AllLinkages and L2 in IgnoredLinkages — NOT both in AllLinkages with
        // phantom num_obs_outside_partition == num_obs rows.
        let obs = Observations::new(
            (0u64..12).collect(),
            vec![60000.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0u32; 12],
            vec![42u64; 12],
            vec![1, 1, 2, 2, 3, 3, 6, 6, 7, 7, 8, 8],
        );
        let lm = LinkageMembers {
            linkage_id: vec![
                100, 100, 100, 100, 100, 100, // L1 → obs 0..5 (partition 0)
                200, 200, 200, 200, 200, 200, // L2 → obs 6..11 (partition 1)
            ],
            obs_id: (0u64..12).collect(),
        };

        let (all, ignored) = classify_linkages(&obs, &lm, &ps(0, 1, 5), 6, 20.0).unwrap();

        assert_eq!(all.len(), 1, "only L1 should be classified for partition 0");
        assert_eq!(all.linkage_id[0], 100);
        assert_eq!(
            all.num_obs_outside_partition[0], 0,
            "L1's obs are all inside partition 0"
        );

        assert_eq!(ignored.len(), 1, "L2 should be reported as ignored");
        assert_eq!(ignored.linkage_id[0], 200);
        assert_eq!(ignored.partition_id[0], 0);
        assert_eq!(ignored.num_obs[0], 6);
        assert_eq!(
            ignored.reason[0],
            crate::types::IgnoredLinkageReason::NoObservationsInPartition
        );
    }

    #[test]
    fn update_all_objects_accumulates_across_sequential_partition_calls() {
        // Two partitions, two disjoint linkages. Call update_all_objects once
        // per partition; each row should end up with only its partition's
        // contribution.
        let mut all_objects = AllObjects::default();
        all_objects.push(os(42, 0));
        all_objects.push(os(42, 1));

        // Linkage L100 (pure, partition 0) covers obs 0..5; L200 covers 6..11.
        let mut all_linkages = AllLinkages::default();
        for (lid, pid) in [(100u64, 0u64), (200, 1)] {
            all_linkages.push(LinkageSummary {
                linkage_id: lid,
                partition_id: pid,
                linked_object_id: 42,
                num_obs: 6,
                num_obs_outside_partition: 0,
                num_members: 1,
                pure: true,
                pure_complete: false,
                contaminated: false,
                contamination: 0.0,
                mixed: false,
                found_pure: true,
                found_contaminated: false,
            });
        }

        let obs = Observations::new(
            (0u64..12).collect(),
            vec![60000.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0u32; 12],
            vec![42u64; 12],
            vec![1, 1, 2, 2, 3, 3, 6, 6, 7, 7, 8, 8],
        );
        let lm_p0 = LinkageMembers {
            linkage_id: vec![100u64; 6],
            obs_id: (0u64..6).collect(),
        };
        let lm_p1 = LinkageMembers {
            linkage_id: vec![200u64; 6],
            obs_id: (6u64..12).collect(),
        };

        update_all_objects(
            &mut all_objects,
            &obs,
            &lm_p0,
            &all_linkages,
            &ps(0, 1, 5),
            6,
        );
        update_all_objects(
            &mut all_objects,
            &obs,
            &lm_p1,
            &all_linkages,
            &ps(1, 6, 10),
            6,
        );

        assert_eq!(all_objects.pure[0], 1);
        assert_eq!(all_objects.obs_in_pure[0], 6);
        assert_eq!(all_objects.pure[1], 1);
        assert_eq!(all_objects.obs_in_pure[1], 6);
    }
}
