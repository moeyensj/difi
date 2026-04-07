//! Integration test for TrackletMetric findability.

use std::path::PathBuf;

use difi::cifi::analyze_observations;
use difi::io::read_observations;
use difi::metrics::tracklet::TrackletMetric;

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("difi")
        .join("tests")
        .join("testdata")
}

#[test]
fn test_tracklet_metric_on_test_data() {
    let obs_path = test_data_dir().join("observations.parquet");
    let (obs, _, _) = read_observations(&obs_path).unwrap();

    let metric = TrackletMetric {
        tracklet_min_obs: 2,
        max_obs_separation: 1.5 / 24.0, // 1.5 hours
        min_linkage_nights: 3,
        min_obs_angular_separation: 1.0, // 1 arcsecond
    };

    let (all_objects, findable, summaries) =
        analyze_observations(&obs, None, &metric).unwrap();

    // Test data has 5 objects with 3 observations per night across 10 nights.
    // With tracklet_min_obs=2, max_obs_separation=1.5h, and 3 obs/night,
    // objects should be findable if they have enough temporal proximity
    // and angular separation between observations.

    println!("TrackletMetric results:");
    println!("  Objects: {}", all_objects.len());
    println!("  Findable: {}", findable.len());
    if !summaries.is_empty() {
        println!(
            "  Partition findable: {}",
            summaries[0].findable.unwrap_or(0)
        );
    }

    // We should get some results (objects exist with observations)
    assert!(!all_objects.is_empty(), "Should have objects");

    // All objects should have 30 observations each
    for i in 0..all_objects.len() {
        assert_eq!(all_objects.num_obs[i], 30);
    }
}

#[test]
fn test_tracklet_metric_strict_params() {
    let obs_path = test_data_dir().join("observations.parquet");
    let (obs, _, _) = read_observations(&obs_path).unwrap();

    // Very strict params — require 5 obs per tracklet, should find fewer objects
    let strict_metric = TrackletMetric {
        tracklet_min_obs: 5,
        max_obs_separation: 0.5 / 24.0, // 30 minutes
        min_linkage_nights: 3,
        min_obs_angular_separation: 1.0,
    };

    let (_, findable_strict, _) =
        analyze_observations(&obs, None, &strict_metric).unwrap();

    // Default params for comparison
    let default_metric = TrackletMetric::default();
    let (_, findable_default, _) =
        analyze_observations(&obs, None, &default_metric).unwrap();

    // Stricter params should find <= objects than default
    assert!(
        findable_strict.len() <= findable_default.len(),
        "Stricter params should not find more objects: strict={} default={}",
        findable_strict.len(),
        findable_default.len()
    );
}
