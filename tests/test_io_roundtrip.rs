//! Test Parquet write → read round-trip for output types.

use std::path::PathBuf;

use difi::cifi::analyze_observations;
use difi::difi::analyze_linkages;
use difi::io::{
    read_linkage_members, read_observations, write_all_linkages, write_all_objects,
    write_findable_observations, write_partition_summaries,
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

    let all_linkages =
        analyze_linkages(&obs, &lm, &mut all_objects, &mut summaries[0], 6, 20.0).unwrap();

    let dir = tempfile::tempdir().unwrap();

    // Write all output types
    write_all_objects(&dir.path().join("all_objects.parquet"), &all_objects, &id_interner2).unwrap();
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
