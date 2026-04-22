//! Integration test: read test fixtures, run CIFI + DIFI, verify results.

use std::path::PathBuf;

use difi::cifi::analyze_observations;
use difi::difi::analyze_linkages;
use difi::io::{read_linkage_members, read_observations};
use difi::metrics::singleton::SingletonMetric;

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("difi")
        .join("tests")
        .join("testdata")
}

#[test]
fn test_read_observations() {
    let obs_path = test_data_dir().join("observations.parquet");
    let (obs, id_interner, obs_code_interner) = read_observations(&obs_path).unwrap();

    assert_eq!(obs.id.len(), 150, "Expected 150 observations");
    assert!(!id_interner.is_empty());
    assert!(!obs_code_interner.is_empty());

    // All observations should have nights
    assert!(obs.night.iter().all(|&n| n > 0));
    // RA should be in [0, 360]
    assert!(obs.ra.iter().all(|&r| (0.0..=360.0).contains(&r)));
    // Dec should be in [-90, 90]
    assert!(obs.dec.iter().all(|&d| (-90.0..=90.0).contains(&d)));
}

#[test]
fn test_read_linkage_members() {
    let obs_path = test_data_dir().join("observations.parquet");
    let lm_path = test_data_dir().join("linkage_members.parquet");

    let (_, mut id_interner, _) = read_observations(&obs_path).unwrap();
    let lm = read_linkage_members(&lm_path, &mut id_interner).unwrap();

    assert_eq!(lm.linkage_id.len(), 293, "Expected 293 linkage members");
}

#[test]
fn test_cifi_singleton() {
    let obs_path = test_data_dir().join("observations.parquet");
    let (obs, _id_interner, _) = read_observations(&obs_path).unwrap();

    let metric = SingletonMetric {
        min_obs: 6,
        min_nights: 3,
        min_nightly_obs_in_min_nights: 1,
    };

    let (all_objects, findable, summaries) = analyze_observations(&obs, None, &metric).unwrap();

    // We should have objects and findable observations
    assert!(!all_objects.is_empty(), "Expected some objects");
    assert!(!findable.is_empty(), "Expected some findable observations");
    assert_eq!(summaries.len(), 1, "Single partition expected");

    // All findable objects should have a discovery night
    assert!(findable.discovery_night.iter().all(|dn| dn.is_some()));

    println!("Objects: {}", all_objects.len());
    println!("Findable: {}", findable.len());
    println!(
        "Partition: nights {}-{}, {} obs, {} findable",
        summaries[0].start_night,
        summaries[0].end_night,
        summaries[0].observations,
        summaries[0].findable.unwrap_or(0)
    );
}

#[test]
fn test_full_pipeline() {
    let obs_path = test_data_dir().join("observations.parquet");
    let lm_path = test_data_dir().join("linkage_members.parquet");

    let (obs, _id_interner, _) = read_observations(&obs_path).unwrap();
    let (_, mut id_interner2, _) = read_observations(&obs_path).unwrap();
    let lm = read_linkage_members(&lm_path, &mut id_interner2).unwrap();

    // CIFI
    let metric = SingletonMetric::default();
    let (mut all_objects, _findable, mut summaries) =
        analyze_observations(&obs, None, &metric).unwrap();

    assert!(!all_objects.is_empty());
    assert_eq!(summaries.len(), 1);

    // DIFI
    let (all_linkages, _ignored) =
        analyze_linkages(&obs, &lm, &mut all_objects, &mut summaries[0], 6, 20.0).unwrap();

    // Should have classified some linkages
    assert!(!all_linkages.is_empty(), "Expected some linkages");

    // Count classifications
    let num_pure: usize = all_linkages.pure.iter().filter(|&&p| p).count();
    let num_contaminated: usize = all_linkages.contaminated.iter().filter(|&&c| c).count();
    let num_mixed: usize = all_linkages.mixed.iter().filter(|&&m| m).count();

    println!("Linkages: {} total", all_linkages.len());
    println!("  Pure: {num_pure}");
    println!("  Contaminated: {num_contaminated}");
    println!("  Mixed: {num_mixed}");
    println!(
        "Completeness: {:.1}%",
        summaries[0].completeness.unwrap_or(0.0)
    );

    // Every linkage should be exactly one of pure/contaminated/mixed
    for i in 0..all_linkages.len() {
        let classifications = [
            all_linkages.pure[i],
            all_linkages.contaminated[i],
            all_linkages.mixed[i],
        ];
        let count = classifications.iter().filter(|&&c| c).count();
        assert_eq!(
            count, 1,
            "Linkage {i} has {count} classifications (expected exactly 1)"
        );
    }

    // Pure and contaminated linkages should have a linked_object_id
    for i in 0..all_linkages.len() {
        if all_linkages.pure[i] || all_linkages.contaminated[i] {
            assert!(
                all_linkages.linked_object_id[i] != difi::types::NO_OBJECT,
                "Pure/contaminated linkage {i} should have linked_object_id"
            );
        }
    }

    // Mixed linkages should NOT have a linked_object_id
    for i in 0..all_linkages.len() {
        if all_linkages.mixed[i] {
            assert!(
                all_linkages.linked_object_id[i] == difi::types::NO_OBJECT,
                "Mixed linkage {i} should not have linked_object_id"
            );
        }
    }
}
