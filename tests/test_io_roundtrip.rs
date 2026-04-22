//! Test Parquet write → read round-trip for output types.

use std::path::PathBuf;

use difi::cifi::analyze_observations;
use difi::difi::analyze_linkages;
use difi::io::{
    read_all_objects, read_findable_observations, read_linkage_members, read_observations,
    read_partition_summaries, write_all_linkages, write_all_objects, write_findable_observations,
    write_partition_summaries,
};
use difi::metrics::singleton::SingletonMetric;

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("difi")
        .join("tests")
        .join("testdata")
}

#[test]
fn test_write_all_objects() {
    let obs_path = test_data_dir().join("observations.parquet");
    let lm_path = test_data_dir().join("linkage_members.parquet");

    let (obs, id_interner, _) = read_observations(&obs_path).unwrap();
    let mut id_interner2 = id_interner.clone();
    let lm = read_linkage_members(&lm_path, &mut id_interner2).unwrap();

    let metric = SingletonMetric::default();
    let (mut all_objects, findable, mut summaries) =
        analyze_observations(&obs, None, &metric).unwrap();

    let (all_linkages, _ignored) =
        analyze_linkages(&obs, &lm, &mut all_objects, &mut summaries[0], 6, 20.0).unwrap();

    let dir = tempfile::tempdir().unwrap();

    // Write all output types
    write_all_objects(
        &dir.path().join("all_objects.parquet"),
        &all_objects,
        &id_interner2,
    )
    .unwrap();
    write_all_linkages(
        &dir.path().join("all_linkages.parquet"),
        &all_linkages,
        &id_interner2,
    )
    .unwrap();
    write_partition_summaries(&dir.path().join("partition_summaries.parquet"), &summaries).unwrap();
    write_findable_observations(
        &dir.path().join("findable.parquet"),
        &findable,
        &id_interner2,
    )
    .unwrap();

    // Verify files exist and are non-empty
    for name in &[
        "all_objects.parquet",
        "all_linkages.parquet",
        "partition_summaries.parquet",
        "findable.parquet",
    ] {
        let path = dir.path().join(name);
        assert!(path.exists(), "{name} should exist");
        assert!(
            std::fs::metadata(&path).unwrap().len() > 0,
            "{name} should be non-empty"
        );
    }

    // Read back and verify row counts via parquet reader
    let verify_rows = |path: &std::path::Path, expected: usize, label: &str| {
        let file = std::fs::File::open(path).unwrap();
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let total: usize = reader.map(|b| b.unwrap().num_rows()).sum();
        assert_eq!(total, expected, "{label} row count");
    };

    verify_rows(
        &dir.path().join("all_objects.parquet"),
        all_objects.len(),
        "all_objects",
    );
    verify_rows(
        &dir.path().join("all_linkages.parquet"),
        all_linkages.len(),
        "all_linkages",
    );
    verify_rows(
        &dir.path().join("partition_summaries.parquet"),
        summaries.len(),
        "partition_summaries",
    );
    verify_rows(
        &dir.path().join("findable.parquet"),
        findable.len(),
        "findable",
    );
}

// ---------------------------------------------------------------------------
// Phase 2 readers: write → read round-trip verifying value equality
// ---------------------------------------------------------------------------

#[test]
fn test_read_all_objects_roundtrip() {
    let obs_path = test_data_dir().join("observations.parquet");
    let (obs, id_interner, _) = read_observations(&obs_path).unwrap();

    let metric = SingletonMetric::default();
    let (written, _findable, _summaries) = analyze_observations(&obs, None, &metric).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("all_objects.parquet");
    write_all_objects(&path, &written, &id_interner).unwrap();

    // Reader must re-intern object_ids into a shared interner to line up
    // u64 IDs with the observations session.
    let (_obs2, mut id_interner2, _) = read_observations(&obs_path).unwrap();
    let read = read_all_objects(&path, &mut id_interner2).unwrap();

    assert_eq!(read.len(), written.len());

    // Resolve interned IDs back to strings on both sides and compare.
    let written_objs: Vec<&str> = written
        .object_id
        .iter()
        .map(|&id| id_interner.resolve(id).unwrap_or(""))
        .collect();
    let read_objs: Vec<&str> = read
        .object_id
        .iter()
        .map(|&id| id_interner2.resolve(id).unwrap_or(""))
        .collect();
    assert_eq!(written_objs, read_objs, "object_id strings must match");

    assert_eq!(read.partition_id, written.partition_id);
    assert_eq!(read.mjd_min, written.mjd_min);
    assert_eq!(read.mjd_max, written.mjd_max);
    assert_eq!(read.num_obs, written.num_obs);
    assert_eq!(read.num_observatories, written.num_observatories);
    assert_eq!(read.findable, written.findable);
    // DIFI-side fields should be zero since no analyze_linkages was called.
    assert!(read.pure.iter().all(|&x| x == 0));
    assert!(read.found_pure.iter().all(|&x| x == 0));
}

#[test]
fn test_read_partition_summaries_roundtrip() {
    let obs_path = test_data_dir().join("observations.parquet");
    let lm_path = test_data_dir().join("linkage_members.parquet");

    let (obs, id_interner, _) = read_observations(&obs_path).unwrap();
    let mut id_interner2 = id_interner.clone();
    let lm = read_linkage_members(&lm_path, &mut id_interner2).unwrap();

    let metric = SingletonMetric::default();
    let (mut all_objects, _findable, mut summaries) =
        analyze_observations(&obs, None, &metric).unwrap();
    let _ = analyze_linkages(&obs, &lm, &mut all_objects, &mut summaries[0], 6, 20.0).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("partition_summaries.parquet");
    write_partition_summaries(&path, &summaries).unwrap();

    let read = read_partition_summaries(&path).unwrap();
    assert_eq!(read.len(), summaries.len());
    for (r, w) in read.iter().zip(summaries.iter()) {
        assert_eq!(r.id, w.id);
        assert_eq!(r.start_night, w.start_night);
        assert_eq!(r.end_night, w.end_night);
        assert_eq!(r.observations, w.observations);
        assert_eq!(r.findable, w.findable);
        assert_eq!(r.found, w.found);
        assert_eq!(r.completeness, w.completeness);
        assert_eq!(r.pure_known, w.pure_known);
        assert_eq!(r.pure_unknown, w.pure_unknown);
        assert_eq!(r.contaminated, w.contaminated);
        assert_eq!(r.mixed, w.mixed);
    }
}

#[test]
fn test_read_findable_observations_roundtrip() {
    let obs_path = test_data_dir().join("observations.parquet");
    let (obs, id_interner, _) = read_observations(&obs_path).unwrap();

    let metric = SingletonMetric::default();
    let (_all_objects, findable, _summaries) = analyze_observations(&obs, None, &metric).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("findable_observations.parquet");
    write_findable_observations(&path, &findable, &id_interner).unwrap();

    let (_obs2, mut id_interner2, _) = read_observations(&obs_path).unwrap();
    let read = read_findable_observations(&path, &mut id_interner2).unwrap();

    assert_eq!(read.len(), findable.len());
    assert_eq!(read.partition_id, findable.partition_id);
    assert_eq!(read.discovery_night, findable.discovery_night);
    // Writer drops obs_ids; reader fills with None. Document via test.
    assert!(read.obs_ids.iter().all(|v| v.is_none()));

    // Verify interner-resolved strings match on both sides.
    let w_strs: Vec<&str> = findable
        .object_id
        .iter()
        .map(|&id| id_interner.resolve(id).unwrap_or(""))
        .collect();
    let r_strs: Vec<&str> = read
        .object_id
        .iter()
        .map(|&id| id_interner2.resolve(id).unwrap_or(""))
        .collect();
    assert_eq!(w_strs, r_strs);
}
