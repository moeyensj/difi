//! Singleton findability metric.
//!
//! An object is findable if it has at least `min_obs` observations
//! spanning at least `min_nights` distinct nights.

use std::collections::HashMap;

use crate::partitions::Partition;
use crate::types::{FindableObservation, ObservationSlices};

use super::FindabilityMetric;

/// Singleton findability metric.
///
/// An object is findable if it has at least `min_obs` observations
/// across at least `min_nights` distinct nights within a partition.
///
/// When exactly `min_nights` nights are present, each night must have
/// at least `min_nightly_obs_in_min_nights` observations.
pub struct SingletonMetric {
    pub min_obs: usize,
    pub min_nights: usize,
    pub min_nightly_obs_in_min_nights: usize,
}

impl Default for SingletonMetric {
    fn default() -> Self {
        Self {
            min_obs: 6,
            min_nights: 3,
            min_nightly_obs_in_min_nights: 1,
        }
    }
}

impl FindabilityMetric for SingletonMetric {
    fn determine_object_findable(
        &self,
        obs_indices: &[usize],
        observations: &ObservationSlices<'_>,
        partitions: &[Partition],
    ) -> Vec<FindableObservation> {
        let mut results = Vec::new();

        for partition in partitions {
            // Filter to observations within this partition
            let partition_indices: Vec<usize> = obs_indices
                .iter()
                .copied()
                .filter(|&i| {
                    observations.nights[i] >= partition.start_night
                        && observations.nights[i] <= partition.end_night
                })
                .collect();

            if partition_indices.len() < self.min_obs {
                continue;
            }

            // Count observations per night
            let mut night_counts: HashMap<i64, usize> = HashMap::new();
            for &i in &partition_indices {
                *night_counts.entry(observations.nights[i]).or_default() += 1;
            }

            let num_nights = night_counts.len();
            if num_nights < self.min_nights {
                continue;
            }

            // Sort nights chronologically
            let mut nights_sorted: Vec<i64> = night_counts.keys().copied().collect();
            nights_sorted.sort_unstable();

            // When exactly min_nights nights, check per-night observation counts
            if num_nights == self.min_nights {
                let all_meet_min = night_counts
                    .values()
                    .all(|&c| c >= self.min_nightly_obs_in_min_nights);
                if !all_meet_min {
                    continue;
                }
            }

            // Discovery night: the chronologically earliest night at which
            // we have accumulated min_obs observations across min_nights nights.
            // Walk nights in chronological order, accumulating observations.
            let mut cumulative_obs = 0usize;
            let mut cumulative_nights = 0usize;
            let mut discovery_night = None;
            let mut discovery_obs = Vec::new();

            for &night in &nights_sorted {
                let count = night_counts[&night];
                cumulative_obs += count;
                cumulative_nights += 1;

                // Collect observation IDs from this night
                for &i in &partition_indices {
                    if observations.nights[i] == night && discovery_obs.len() < self.min_obs {
                        discovery_obs.push(observations.ids[i]);
                    }
                }

                if cumulative_nights >= self.min_nights && cumulative_obs >= self.min_obs {
                    discovery_night = Some(night);
                    break;
                }
            }

            if let Some(dn) = discovery_night {
                discovery_obs.truncate(self.min_obs);
                results.push(FindableObservation {
                    partition_id: partition.id,
                    object_id: observations.object_ids[partition_indices[0]].unwrap(),
                    discovery_night: Some(dn),
                    obs_ids: Some(discovery_obs),
                });
            }
        }

        results
    }
}
