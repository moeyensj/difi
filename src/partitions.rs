//! Partition creation and summary.
//!
//! Partitions divide observations into time windows for analysis.
//! Each partition is defined by a start and end night (inclusive).

use crate::error::{Error, Result};

/// A partition defining a time window of observations.
#[derive(Debug, Clone)]
pub struct Partition {
    pub id: u64,
    pub start_night: i64,
    pub end_night: i64,
}

/// Summary statistics for a partition.
#[derive(Debug, Clone)]
pub struct PartitionSummary {
    pub id: u64,
    pub start_night: i64,
    pub end_night: i64,
    pub observations: i64,
    pub findable: Option<i64>,
    pub found: Option<i64>,
    pub completeness: Option<f64>,
    pub pure_known: Option<i64>,
    pub pure_unknown: Option<i64>,
    pub contaminated: Option<i64>,
    pub mixed: Option<i64>,
}

/// Create a single partition spanning all given nights.
pub fn create_single(nights: &[i64]) -> Result<Partition> {
    if nights.is_empty() {
        return Err(Error::InvalidInput(
            "Cannot create partition from empty nights".to_string(),
        ));
    }
    let min_night = *nights.iter().min().unwrap();
    let max_night = *nights.iter().max().unwrap();
    Ok(Partition {
        id: 0,
        start_night: min_night,
        end_night: max_night,
    })
}

/// Create non-overlapping or sliding linking windows.
///
/// If `detection_window` is None, returns a single partition spanning all nights.
/// If `sliding` is true, windows slide by one night with a ramp-up from `min_nights`.
/// If `sliding` is false, windows are non-overlapping blocks of `detection_window` nights.
pub fn create_linking_windows(
    nights: &[i64],
    detection_window: Option<i64>,
    min_nights: Option<i64>,
    sliding: bool,
) -> Result<Vec<Partition>> {
    if nights.is_empty() {
        return Err(Error::InvalidInput(
            "Cannot create partitions from empty nights".to_string(),
        ));
    }
    let min_night = *nights.iter().min().unwrap();
    let max_night = *nights.iter().max().unwrap();

    let detection_window = match detection_window {
        None => return Ok(vec![create_single(nights)?]),
        Some(dw) => {
            if dw >= (max_night - min_night + 1) {
                max_night - min_night + 1
            } else {
                dw
            }
        }
    };

    let min_nights = min_nights.unwrap_or(detection_window);
    if detection_window < min_nights {
        return Err(Error::InvalidInput(
            "Detection window must be >= min_nights".to_string(),
        ));
    }

    let mut partitions = Vec::new();

    if sliding {
        let mut i: u64 = 0;
        let mut start_night = min_night;
        let mut end_night = start_night + min_nights - 1;
        loop {
            if end_night > max_night {
                break;
            }
            partitions.push(Partition {
                id: i,
                start_night,
                end_night,
            });
            i += 1;
            end_night += 1;
            if end_night - detection_window == start_night {
                start_night += 1;
            }
        }
    } else {
        let mut i: u64 = 0;
        let mut start = min_night;
        while start <= max_night {
            let end = (start + detection_window - 1).min(max_night);
            partitions.push(Partition {
                id: i,
                start_night: start,
                end_night: end,
            });
            i += 1;
            start += detection_window;
        }
    }

    Ok(partitions)
}

/// Create partition summaries by counting observations per partition.
///
/// Uses a pre-sorted night index for O(log n) lookups per partition.
pub fn create_summaries(
    obs_nights: &[i64],
    partitions: &[Partition],
    night_sorted_indices: &[usize],
) -> Vec<PartitionSummary> {
    partitions
        .iter()
        .map(|p| {
            let lo = night_sorted_indices.partition_point(|&i| obs_nights[i] < p.start_night);
            let hi = night_sorted_indices.partition_point(|&i| obs_nights[i] <= p.end_night);
            let num_obs = (hi - lo) as i64;

            PartitionSummary {
                id: p.id,
                start_night: p.start_night,
                end_night: p.end_night,
                observations: num_obs,
                findable: None,
                found: None,
                completeness: None,
                pure_known: None,
                pure_unknown: None,
                contaminated: None,
                mixed: None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_single() {
        let nights = vec![5, 3, 7, 1, 9];
        let p = create_single(&nights).unwrap();
        assert_eq!(p.start_night, 1);
        assert_eq!(p.end_night, 9);
    }

    #[test]
    fn test_create_single_empty() {
        assert!(create_single(&[]).is_err());
    }

    #[test]
    fn test_create_linking_windows_non_overlapping() {
        let nights: Vec<i64> = (0..10).collect();
        let partitions = create_linking_windows(&nights, Some(3), None, false).unwrap();
        assert_eq!(partitions.len(), 4); // [0,2], [3,5], [6,8], [9,9]
        assert_eq!(partitions[0].start_night, 0);
        assert_eq!(partitions[0].end_night, 2);
        assert_eq!(partitions[3].start_night, 9);
        assert_eq!(partitions[3].end_night, 9);
    }

    #[test]
    fn test_create_linking_windows_none() {
        let nights = vec![1, 5, 10];
        let partitions = create_linking_windows(&nights, None, None, false).unwrap();
        assert_eq!(partitions.len(), 1);
        assert_eq!(partitions[0].start_night, 1);
        assert_eq!(partitions[0].end_night, 10);
    }
}
