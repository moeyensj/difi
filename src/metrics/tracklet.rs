//! Tracklet findability metric.
//!
//! An object is findable if it has tracklets (intra-night observation groups)
//! on at least `min_linkage_nights` distinct nights, where each tracklet has
//! at least `tracklet_min_obs` observations within `max_obs_separation` time
//! and with at least `min_obs_angular_separation` angular separation.

use std::collections::HashMap;

use crate::metrics::haversine_distance;
use crate::partitions::Partition;
use crate::types::{FindableObservation, ObservationSlices};

use super::FindabilityMetric;

/// Tracklet-based findability metric.
pub struct TrackletMetric {
    /// Minimum observations per tracklet (intra-night group).
    pub tracklet_min_obs: usize,
    /// Maximum time separation between consecutive observations in a
    /// tracklet, in fractional days.
    pub max_obs_separation: f64,
    /// Minimum number of distinct nights with valid tracklets.
    pub min_linkage_nights: usize,
    /// Minimum angular separation (arcseconds) between any two observations
    /// in a tracklet to confirm real motion.
    pub min_obs_angular_separation: f64,
}

impl Default for TrackletMetric {
    fn default() -> Self {
        Self {
            tracklet_min_obs: 2,
            max_obs_separation: 1.5 / 24.0,
            min_linkage_nights: 3,
            min_obs_angular_separation: 1.0,
        }
    }
}

/// Find indices of observations that are within `max_separation_days` of
/// at least one other observation (assumes `times` is sorted ascending).
fn find_temporally_close_indices(times: &[f64], max_separation_days: f64) -> Vec<usize> {
    if times.len() < 2 {
        return Vec::new();
    }
    let max_sep_minutes = max_separation_days * 24.0 * 60.0;
    let mut valid = vec![false; times.len()];

    for i in 0..times.len() - 1 {
        let dt_minutes = (times[i + 1] - times[i]) * 24.0 * 60.0;
        if dt_minutes <= max_sep_minutes {
            valid[i] = true;
            valid[i + 1] = true;
        }
    }

    valid
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v { Some(i) } else { None })
        .collect()
}

/// Find indices of observations that have at least `min_separation_arcsec`
/// angular separation from another observation on the same night.
///
/// Improvement over the Python version: checks all pairs within a night,
/// not just consecutive observations.
fn find_angularly_separated_indices(
    nights: &[i64],
    ra: &[f64],
    dec: &[f64],
    min_separation_arcsec: f64,
) -> Vec<bool> {
    let min_sep_deg = min_separation_arcsec / 3600.0;
    let mut valid = vec![false; nights.len()];

    // Group indices by night
    let mut night_groups: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &night) in nights.iter().enumerate() {
        night_groups.entry(night).or_default().push(i);
    }

    for indices in night_groups.values() {
        // Check all pairs within this night
        for a in 0..indices.len() {
            for b in (a + 1)..indices.len() {
                let ia = indices[a];
                let ib = indices[b];
                let dist = haversine_distance(ra[ia], dec[ia], ra[ib], dec[ib]);
                if dist >= min_sep_deg {
                    valid[ia] = true;
                    valid[ib] = true;
                }
            }
        }
    }

    valid
}

impl FindabilityMetric for TrackletMetric {
    fn determine_object_findable(
        &self,
        obs_indices: &[usize],
        observations: &ObservationSlices<'_>,
        partitions: &[Partition],
    ) -> Vec<FindableObservation> {
        let mut results = Vec::new();
        let total_required = self.tracklet_min_obs * self.min_linkage_nights;

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

            if partition_indices.len() < total_required {
                continue;
            }

            // Extract local arrays for this partition's observations
            let times: Vec<f64> = partition_indices
                .iter()
                .map(|&i| observations.times_mjd[i])
                .collect();
            let nights: Vec<i64> = partition_indices
                .iter()
                .map(|&i| observations.nights[i])
                .collect();
            let ra: Vec<f64> = partition_indices
                .iter()
                .map(|&i| observations.ra[i])
                .collect();
            let dec: Vec<f64> = partition_indices
                .iter()
                .map(|&i| observations.dec[i])
                .collect();
            let ids: Vec<u64> = partition_indices
                .iter()
                .map(|&i| observations.ids[i])
                .collect();

            let mut valid_mask = vec![true; partition_indices.len()];

            if self.tracklet_min_obs > 1 {
                // Filter by temporal proximity
                let temporal_valid = find_temporally_close_indices(&times, self.max_obs_separation);
                valid_mask = vec![false; partition_indices.len()];
                for &i in &temporal_valid {
                    valid_mask[i] = true;
                }

                // Check per-night observation counts
                let mut night_counts: HashMap<i64, usize> = HashMap::new();
                for (i, &night) in nights.iter().enumerate() {
                    if valid_mask[i] {
                        *night_counts.entry(night).or_default() += 1;
                    }
                }

                // Remove observations on nights without enough tracklet members
                for (i, &night) in nights.iter().enumerate() {
                    if valid_mask[i] {
                        let count = night_counts.get(&night).copied().unwrap_or(0);
                        if count < self.tracklet_min_obs {
                            valid_mask[i] = false;
                        }
                    }
                }

                // Filter by angular separation
                if self.min_obs_angular_separation > 0.0 {
                    let valid_nights: Vec<i64> = nights
                        .iter()
                        .zip(valid_mask.iter())
                        .filter_map(|(&n, &v)| if v { Some(n) } else { None })
                        .collect();
                    let valid_ra: Vec<f64> = ra
                        .iter()
                        .zip(valid_mask.iter())
                        .filter_map(|(&r, &v)| if v { Some(r) } else { None })
                        .collect();
                    let valid_dec: Vec<f64> = dec
                        .iter()
                        .zip(valid_mask.iter())
                        .filter_map(|(&d, &v)| if v { Some(d) } else { None })
                        .collect();

                    let angular_valid = find_angularly_separated_indices(
                        &valid_nights,
                        &valid_ra,
                        &valid_dec,
                        self.min_obs_angular_separation,
                    );

                    // Map angular validity back to original mask
                    let mut angular_iter = angular_valid.iter();
                    for v in valid_mask.iter_mut() {
                        if *v {
                            *v = *angular_iter.next().unwrap();
                        }
                    }
                }
            }

            // Count valid observations and unique nights
            let valid_count: usize = valid_mask.iter().filter(|&&v| v).count();
            let mut valid_unique_nights: Vec<i64> = nights
                .iter()
                .zip(valid_mask.iter())
                .filter_map(|(&n, &v)| if v { Some(n) } else { None })
                .collect();
            valid_unique_nights.sort_unstable();
            valid_unique_nights.dedup();

            if valid_count < total_required || valid_unique_nights.len() < self.min_linkage_nights {
                continue;
            }

            // Discovery night: the min_linkage_nights-th unique night
            let discovery_night = valid_unique_nights[self.min_linkage_nights - 1];

            // Collect observation IDs up to and including discovery night
            let discovery_obs: Vec<u64> = ids
                .iter()
                .zip(valid_mask.iter())
                .zip(nights.iter())
                .filter_map(|((&id, &v), &n)| {
                    if v && n <= discovery_night {
                        Some(id)
                    } else {
                        None
                    }
                })
                .collect();

            results.push(FindableObservation {
                partition_id: partition.id,
                object_id: observations.object_ids[partition_indices[0]],
                discovery_night: Some(discovery_night),
                obs_ids: Some(discovery_obs),
            });
        }

        results
    }
}
