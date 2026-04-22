//! Parity test: verify exact agreement with Python v2 difi output.
//!
//! Expected values come from running Python v2 difi on the same test fixtures
//! with SingletonMetric(min_obs=6, min_nights=3, min_nightly_obs_in_min_nights=1)
//! and analyze_linkages(min_obs=6, contamination_percentage=20.0).

use std::collections::HashMap;
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

fn run_pipeline() -> (
    difi::types::AllObjects,
    difi::types::AllLinkages,
    Vec<difi::partitions::PartitionSummary>,
    difi::types::StringInterner,
) {
    let obs_path = test_data_dir().join("observations.parquet");
    let lm_path = test_data_dir().join("linkage_members.parquet");

    let (obs, id_interner, _) = read_observations(&obs_path).unwrap();
    let mut id_interner2 = id_interner.clone();
    let lm = read_linkage_members(&lm_path, &mut id_interner2).unwrap();

    let metric = SingletonMetric {
        min_obs: 6,
        min_nights: 3,
        min_nightly_obs_in_min_nights: 1,
    };

    let (mut all_objects, _findable, mut summaries) =
        analyze_observations(&obs, None, &metric).unwrap();

    let (all_linkages, _ignored) =
        analyze_linkages(&obs, &lm, &mut all_objects, &mut summaries[0], 6, 20.0).unwrap();

    (all_objects, all_linkages, summaries, id_interner2)
}

// ── CIFI parity ──────────────────────────────────────────────────────

#[test]
fn test_cifi_counts() {
    let (all_objects, _, summaries, _) = run_pipeline();

    assert_eq!(all_objects.len(), 5, "5 objects");
    assert_eq!(summaries[0].findable, Some(5), "5 findable");

    // All objects should have 30 observations
    for i in 0..all_objects.len() {
        assert_eq!(all_objects.num_obs[i], 30);
        assert_eq!(all_objects.findable[i], Some(true));
    }
}

// ── DIFI parity ──────────────────────────────────────────────────────

#[test]
fn test_difi_summary_counts() {
    let (_, all_linkages, summaries, _) = run_pipeline();

    assert_eq!(all_linkages.len(), 20);

    let n_pure: usize = all_linkages.pure.iter().filter(|&&p| p).count();
    let n_pure_complete: usize = all_linkages.pure_complete.iter().filter(|&&p| p).count();
    let n_contaminated: usize = all_linkages.contaminated.iter().filter(|&&c| c).count();
    let n_mixed: usize = all_linkages.mixed.iter().filter(|&&m| m).count();
    let n_found_pure: usize = all_linkages.found_pure.iter().filter(|&&f| f).count();
    let n_found_contaminated: usize = all_linkages
        .found_contaminated
        .iter()
        .filter(|&&f| f)
        .count();

    assert_eq!(n_pure, 10, "10 pure linkages");
    assert_eq!(n_pure_complete, 5, "5 pure complete linkages");
    assert_eq!(n_contaminated, 2, "2 contaminated linkages");
    assert_eq!(n_mixed, 8, "8 mixed linkages");
    assert_eq!(n_found_pure, 10, "10 found pure");
    assert_eq!(n_found_contaminated, 2, "2 found contaminated");

    // Partition summary
    assert_eq!(summaries[0].found, Some(5));
    assert!((summaries[0].completeness.unwrap() - 100.0).abs() < 0.01);
    assert_eq!(summaries[0].pure_known, Some(10));
    assert_eq!(summaries[0].pure_unknown, Some(0));
    assert_eq!(summaries[0].contaminated, Some(2));
    assert_eq!(summaries[0].mixed, Some(8));
}

#[test]
fn test_difi_per_linkage() {
    let (_, all_linkages, _, interner) = run_pipeline();

    // Expected per-linkage results from Python v2
    // (linkage_name, linked_object, num_obs, contamination%, type, pure_complete)
    #[allow(clippy::type_complexity)]
    let expected: Vec<(&str, Option<&str>, i64, f64, &str, bool)> = vec![
        ("linkage_pure_00000", Some("00000"), 30, 0.0, "pure", true),
        ("linkage_pure_00001", Some("00001"), 30, 0.0, "pure", true),
        ("linkage_pure_00002", Some("00002"), 30, 0.0, "pure", true),
        ("linkage_pure_00003", Some("00003"), 30, 0.0, "pure", true),
        ("linkage_pure_00004", Some("00004"), 30, 0.0, "pure", true),
        (
            "linkage_pure_incomplete_00000",
            Some("00000"),
            6,
            0.0,
            "pure",
            false,
        ),
        (
            "linkage_pure_incomplete_00001",
            Some("00001"),
            10,
            0.0,
            "pure",
            false,
        ),
        (
            "linkage_pure_incomplete_00002",
            Some("00002"),
            7,
            0.0,
            "pure",
            false,
        ),
        (
            "linkage_pure_incomplete_00003",
            Some("00003"),
            8,
            0.0,
            "pure",
            false,
        ),
        (
            "linkage_pure_incomplete_00004",
            Some("00004"),
            10,
            0.0,
            "pure",
            false,
        ),
        (
            "linkage_partial_00000",
            Some("00000"),
            12,
            8.3,
            "contaminated",
            false,
        ),
        (
            "linkage_partial_00001",
            Some("00001"),
            12,
            16.7,
            "contaminated",
            false,
        ),
        ("linkage_partial_00002", None, 12, 25.0, "mixed", false),
        ("linkage_partial_00003", None, 12, 41.7, "mixed", false),
        ("linkage_partial_00004", None, 12, 50.0, "mixed", false),
        ("linkage_mixed_00000", None, 9, 66.7, "mixed", false),
        ("linkage_mixed_00001", None, 7, 57.1, "mixed", false),
        ("linkage_mixed_00002", None, 9, 66.7, "mixed", false),
        ("linkage_mixed_00003", None, 9, 66.7, "mixed", false),
        ("linkage_mixed_00004", None, 8, 62.5, "mixed", false),
    ];

    // Build lookup by linkage name
    let mut rust_linkages: HashMap<String, usize> = HashMap::new();
    for i in 0..all_linkages.len() {
        let name = interner
            .resolve(all_linkages.linkage_id[i])
            .unwrap()
            .to_string();
        rust_linkages.insert(name, i);
    }

    for (name, exp_linked, exp_nobs, exp_contam, exp_type, exp_pure_complete) in &expected {
        let i = *rust_linkages
            .get(*name)
            .unwrap_or_else(|| panic!("Missing linkage: {name}"));

        // Check num_obs
        assert_eq!(
            all_linkages.num_obs[i], *exp_nobs,
            "{name}: num_obs mismatch"
        );

        // Check type classification
        let actual_type = if all_linkages.pure[i] {
            "pure"
        } else if all_linkages.contaminated[i] {
            "contaminated"
        } else {
            "mixed"
        };
        assert_eq!(actual_type, *exp_type, "{name}: type mismatch");

        // Check pure_complete
        assert_eq!(
            all_linkages.pure_complete[i], *exp_pure_complete,
            "{name}: pure_complete mismatch"
        );

        // Check contamination percentage (within rounding tolerance)
        assert!(
            (all_linkages.contamination[i] - exp_contam).abs() < 0.15,
            "{name}: contamination {:.1} != {:.1}",
            all_linkages.contamination[i],
            exp_contam
        );

        // Check linked_object_id
        let actual_linked = interner
            .resolve(all_linkages.linked_object_id[i])
            .map(|s| s.to_string());
        let exp_linked_str = exp_linked.map(|s| s.to_string());
        assert_eq!(
            actual_linked, exp_linked_str,
            "{name}: linked_object_id mismatch"
        );
    }
}

#[test]
fn test_difi_per_object() {
    let (all_objects, _, _, interner) = run_pipeline();

    // Expected per-object results from Python v2
    // (object_id, found_pure, found_contam, pure, pure_complete, contaminated, contaminant, mixed)
    #[allow(clippy::type_complexity)]
    let expected: Vec<(&str, i64, i64, i64, i64, i64, i64, i64)> = vec![
        ("00000", 2, 1, 2, 1, 1, 0, 6),
        ("00001", 2, 1, 2, 1, 1, 0, 5),
        ("00002", 2, 0, 2, 1, 0, 0, 4),
        ("00003", 2, 0, 2, 1, 0, 2, 7),
        ("00004", 2, 0, 2, 1, 0, 1, 3),
    ];

    // Build lookup by object name
    let mut rust_objects: HashMap<String, usize> = HashMap::new();
    for i in 0..all_objects.len() {
        let name = interner
            .resolve(all_objects.object_id[i])
            .unwrap()
            .to_string();
        rust_objects.insert(name, i);
    }

    for (oid, fp, fc, p, pc, c, ct, m) in &expected {
        let i = *rust_objects
            .get(*oid)
            .unwrap_or_else(|| panic!("Missing object: {oid}"));

        assert_eq!(all_objects.found_pure[i], *fp, "{oid}: found_pure");
        assert_eq!(
            all_objects.found_contaminated[i], *fc,
            "{oid}: found_contaminated"
        );
        assert_eq!(all_objects.pure[i], *p, "{oid}: pure");
        assert_eq!(all_objects.pure_complete[i], *pc, "{oid}: pure_complete");
        assert_eq!(all_objects.contaminated[i], *c, "{oid}: contaminated");
        assert_eq!(all_objects.contaminant[i], *ct, "{oid}: contaminant");
        assert_eq!(all_objects.mixed[i], *m, "{oid}: mixed");
    }
}
