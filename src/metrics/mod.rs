//! Findability metrics.
//!
//! A findability metric determines whether an object is "findable" — i.e.,
//! whether it has enough observations in the right pattern to be discoverable
//! by a linking algorithm.

pub mod singleton;
pub mod tracklet;

use crate::partitions::Partition;
use crate::types::{FindableObservation, ObservationSlices};

/// Trait for findability metrics.
///
/// Implementations determine whether a single object (represented by its
/// observation indices) is findable within each partition.
///
/// Takes `ObservationSlices` (concrete borrowed slices) so generics
/// don't propagate into metric implementations.
pub trait FindabilityMetric: Send + Sync {
    /// Determine whether the given observations (belonging to a single object)
    /// satisfy the findability criteria within each partition.
    ///
    /// `obs_indices` are indices into the `observations` slices for this object.
    ///
    /// Returns one `FindableObservation` per partition where the object is
    /// findable.
    fn determine_object_findable(
        &self,
        obs_indices: &[usize],
        observations: &ObservationSlices<'_>,
        partitions: &[Partition],
    ) -> Vec<FindableObservation>;
}

/// Haversine great-circle distance between two points on a sphere.
///
/// All inputs and output in degrees.
pub fn haversine_distance(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let ra1 = ra1.to_radians();
    let dec1 = dec1.to_radians();
    let ra2 = ra2.to_radians();
    let dec2 = dec2.to_radians();

    let dlon = ra2 - ra1;
    let dlat = dec2 - dec1;

    let a = (dlat / 2.0).sin().powi(2) + dec1.cos() * dec2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    c.to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine_same_point() {
        let d = haversine_distance(45.0, 30.0, 45.0, 30.0);
        assert!(d.abs() < 1e-12);
    }

    #[test]
    fn test_haversine_poles() {
        // North pole to south pole = 180 degrees
        let d = haversine_distance(0.0, 90.0, 0.0, -90.0);
        assert!((d - 180.0).abs() < 1e-10);
    }

    #[test]
    fn test_haversine_quarter_circle() {
        // Along equator, 90 degrees apart
        let d = haversine_distance(0.0, 0.0, 90.0, 0.0);
        assert!((d - 90.0).abs() < 1e-10);
    }
}
