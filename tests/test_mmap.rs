//! Test memory-mapped observation access.

use std::path::PathBuf;

use difi::cifi::analyze_observations;
use difi::io::read_observations;
use difi::metrics::singleton::SingletonMetric;
use difi::mmap::{MmapObservations, load_observations_cached, write_cache};
use difi::types::ObservationTable;

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("difi")
        .join("tests")
        .join("testdata")
}

#[test]
fn test_mmap_roundtrip() {
    let obs_path = test_data_dir().join("observations.parquet");
    let (obs, interner, _) = read_observations(&obs_path).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let cache_dir = dir.path().join("test_cache");

    // Write cache
    write_cache(&cache_dir, &obs, &interner).unwrap();

    // Load from cache
    let (mmap_obs, mmap_interner) = MmapObservations::from_cache(&cache_dir).unwrap();

    // Verify lengths match
    assert_eq!(mmap_obs.len(), obs.len());
    assert_eq!(mmap_interner.len(), interner.len());

    // Verify data matches
    assert_eq!(mmap_obs.ids(), obs.ids());
    assert_eq!(mmap_obs.nights(), obs.nights());
    assert_eq!(mmap_obs.object_ids(), obs.object_ids());
    assert_eq!(mmap_obs.ra(), obs.ra());
    assert_eq!(mmap_obs.dec(), obs.dec());
    assert_eq!(mmap_obs.times_mjd(), obs.times_mjd());
    assert_eq!(mmap_obs.observatory_codes(), obs.observatory_codes());
}

#[test]
fn test_mmap_cifi_parity() {
    // Run CIFI on both in-memory and mmap'd observations, verify same results
    let obs_path = test_data_dir().join("observations.parquet");
    let (obs, interner, _) = read_observations(&obs_path).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let cache_dir = dir.path().join("test_cache");
    write_cache(&cache_dir, &obs, &interner).unwrap();
    let (mmap_obs, _) = MmapObservations::from_cache(&cache_dir).unwrap();

    let metric = SingletonMetric::default();

    let (objects_mem, findable_mem, summaries_mem) =
        analyze_observations(&obs, None, &metric).unwrap();
    let (objects_mmap, findable_mmap, summaries_mmap) =
        analyze_observations(&mmap_obs, None, &metric).unwrap();

    assert_eq!(objects_mem.len(), objects_mmap.len());
    assert_eq!(findable_mem.len(), findable_mmap.len());
    assert_eq!(summaries_mem.len(), summaries_mmap.len());
    assert_eq!(summaries_mem[0].findable, summaries_mmap[0].findable);
}

#[test]
fn test_load_cached_creates_and_reuses() {
    let obs_path = test_data_dir().join("observations.parquet");

    let dir = tempfile::tempdir().unwrap();
    let parquet_copy = dir.path().join("observations.parquet");
    std::fs::copy(&obs_path, &parquet_copy).unwrap();

    // First call: creates cache
    let (obs1, interner1) = load_observations_cached(&parquet_copy).unwrap();
    assert_eq!(obs1.len(), 150);

    // Cache directory should exist
    let cache_dir = dir.path().join("observations.parquet.difi_cache");
    assert!(cache_dir.exists());

    // Second call: reuses cache (should be fast, same results)
    let (obs2, interner2) = load_observations_cached(&parquet_copy).unwrap();
    assert_eq!(obs2.len(), obs1.len());
    assert_eq!(interner2.len(), interner1.len());
    assert_eq!(obs2.ids(), obs1.ids());
}
