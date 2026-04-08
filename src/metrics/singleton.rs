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
                    object_id: observations.object_ids[partition_indices[0]],
                    discovery_night: Some(dn),
                    obs_ids: Some(discovery_obs),
                });
            }
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ObservationSlices;

    fn make_obs_slices<'a>(
        ids: &'a [u64],
        times: &'a [f64],
        ra: &'a [f64],
        dec: &'a [f64],
        nights: &'a [i64],
        object_ids: &'a [u64],
        obs_codes: &'a [u32],
    ) -> ObservationSlices<'a> {
        ObservationSlices {
            ids,
            times_mjd: times,
            ra,
            dec,
            nights,
            object_ids,
            observatory_codes: obs_codes,
        }
    }

    #[test]
    fn test_singleton_not_enough_obs() {
        let ids: Vec<u64> = (0..3).collect();
        let times = vec![60000.0, 60001.0, 60002.0];
        let ra = vec![0.0; 3];
        let dec = vec![0.0; 3];
        let nights = vec![1, 2, 3];
        let object_ids = vec![0u64; 3];
        let obs_codes = vec![0u32; 3];
        let obs = make_obs_slices(&ids, &times, &ra, &dec, &nights, &object_ids, &obs_codes);

        let metric = SingletonMetric {
            min_obs: 6,
            min_nights: 3,
            min_nightly_obs_in_min_nights: 1,
        };
        let partition = Partition {
            id: 0,
            start_night: 1,
            end_night: 3,
        };

        let result = metric.determine_object_findable(&[0, 1, 2], &obs, &[partition]);
        assert!(result.is_empty(), "3 obs < min_obs=6, not findable");
    }

    #[test]
    fn test_singleton_not_enough_nights() {
        let ids: Vec<u64> = (0..6).collect();
        let times = vec![60000.0; 6];
        let ra = vec![0.0; 6];
        let dec = vec![0.0; 6];
        let nights = vec![1, 1, 1, 1, 1, 1]; // all same night
        let object_ids = vec![0u64; 6];
        let obs_codes = vec![0u32; 6];
        let obs = make_obs_slices(&ids, &times, &ra, &dec, &nights, &object_ids, &obs_codes);

        let metric = SingletonMetric {
            min_obs: 6,
            min_nights: 3,
            min_nightly_obs_in_min_nights: 1,
        };
        let partition = Partition {
            id: 0,
            start_night: 1,
            end_night: 1,
        };

        let result = metric.determine_object_findable(&[0, 1, 2, 3, 4, 5], &obs, &[partition]);
        assert!(result.is_empty(), "1 night < min_nights=3, not findable");
    }

    #[test]
    fn test_singleton_findable() {
        let ids: Vec<u64> = (0..6).collect();
        let times = vec![60000.0, 60000.1, 60001.0, 60001.1, 60002.0, 60002.1];
        let ra = vec![0.0; 6];
        let dec = vec![0.0; 6];
        let nights = vec![1, 1, 2, 2, 3, 3];
        let object_ids = vec![0u64; 6];
        let obs_codes = vec![0u32; 6];
        let obs = make_obs_slices(&ids, &times, &ra, &dec, &nights, &object_ids, &obs_codes);

        let metric = SingletonMetric::default(); // min_obs=6, min_nights=3
        let partition = Partition {
            id: 0,
            start_night: 1,
            end_night: 3,
        };

        let result = metric.determine_object_findable(&[0, 1, 2, 3, 4, 5], &obs, &[partition]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].discovery_night, Some(3));
    }

    #[test]
    fn test_singleton_min_nightly_obs_enforced() {
        // Exactly min_nights=3 nights, but one night has 0 obs below threshold
        let ids: Vec<u64> = (0..7).collect();
        let times = vec![0.0; 7];
        let ra = vec![0.0; 7];
        let dec = vec![0.0; 7];
        let nights = vec![1, 1, 1, 2, 2, 2, 3]; // night 3 has only 1 obs
        let object_ids = vec![0u64; 7];
        let obs_codes = vec![0u32; 7];
        let obs = make_obs_slices(&ids, &times, &ra, &dec, &nights, &object_ids, &obs_codes);

        let metric = SingletonMetric {
            min_obs: 6,
            min_nights: 3,
            min_nightly_obs_in_min_nights: 2, // need 2/night when exactly 3 nights
        };
        let partition = Partition {
            id: 0,
            start_night: 1,
            end_night: 3,
        };

        let result = metric.determine_object_findable(&[0, 1, 2, 3, 4, 5, 6], &obs, &[partition]);
        assert!(
            result.is_empty(),
            "Night 3 has 1 obs < min_nightly=2, not findable"
        );
    }
}
