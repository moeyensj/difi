use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::PathBuf;

use difi::cifi::analyze_observations;
use difi::difi::analyze_linkages;
use difi::io::{read_linkage_members, read_observations};
use difi::metrics::singleton::SingletonMetric;
use difi::metrics::tracklet::TrackletMetric;
use difi::types::{LinkageMembers, Observations, StringInterner};

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("difi")
        .join("tests")
        .join("testdata")
}

fn load_test_data() -> (Observations, LinkageMembers, StringInterner) {
    let obs_path = test_data_dir().join("observations.parquet");
    let lm_path = test_data_dir().join("linkage_members.parquet");

    let (obs, id_interner, _) = read_observations(&obs_path).unwrap();
    let mut id_interner2 = id_interner.clone();
    let lm = read_linkage_members(&lm_path, &mut id_interner2).unwrap();

    (obs, lm, id_interner2)
}

/// Generate synthetic observations: `n_objects` objects with `obs_per_night`
/// observations per night across `n_nights` nights.
fn generate_synthetic_data(
    n_objects: usize,
    n_nights: usize,
    obs_per_night: usize,
) -> Observations {
    let total = n_objects * n_nights * obs_per_night;
    let mut id = Vec::with_capacity(total);
    let mut time_mjd = Vec::with_capacity(total);
    let mut ra = Vec::with_capacity(total);
    let mut dec = Vec::with_capacity(total);
    let mut observatory_code = Vec::with_capacity(total);
    let mut object_id = Vec::with_capacity(total);
    let mut night = Vec::with_capacity(total);

    let mut obs_counter = 0u64;
    for obj in 0..n_objects {
        let base_ra = (obj as f64 * 10.0) % 360.0;
        let base_dec = ((obj as f64 * 7.0) % 180.0) - 90.0;

        for n in 0..n_nights {
            for o in 0..obs_per_night {
                id.push(obs_counter);
                obs_counter += 1;
                let mjd = 60000.0 + n as f64 + o as f64 / (obs_per_night as f64 * 24.0);
                time_mjd.push(mjd);
                // Small motion per night
                ra.push(base_ra + n as f64 * 0.01 + o as f64 * 0.001);
                dec.push(base_dec + n as f64 * 0.005);
                observatory_code.push(0);
                object_id.push(obj as u64);
                night.push(60000 + n as i64);
            }
        }
    }

    Observations::new(id, time_mjd, ra, dec, observatory_code, object_id, night)
}

fn bench_cifi_singleton(c: &mut Criterion) {
    let (obs, _, _) = load_test_data();
    let metric = SingletonMetric::default();

    c.bench_function("cifi_singleton_5obj_150obs", |b| {
        b.iter(|| {
            black_box(analyze_observations(&obs, None, &metric).unwrap());
        })
    });
}

fn bench_cifi_tracklet(c: &mut Criterion) {
    let (obs, _, _) = load_test_data();
    let metric = TrackletMetric::default();

    c.bench_function("cifi_tracklet_5obj_150obs", |b| {
        b.iter(|| {
            black_box(analyze_observations(&obs, None, &metric).unwrap());
        })
    });
}

fn bench_full_pipeline(c: &mut Criterion) {
    let (obs, lm, _) = load_test_data();

    c.bench_function("full_pipeline_5obj_20linkages", |b| {
        b.iter(|| {
            let metric = SingletonMetric::default();
            let (mut all_objects, _, mut summaries) =
                analyze_observations(&obs, None, &metric).unwrap();
            let all_linkages =
                analyze_linkages(&obs, &lm, &mut all_objects, &mut summaries[0], 6, 20.0).unwrap();
            black_box(all_linkages);
        })
    });
}

fn bench_cifi_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("cifi_singleton_scaling");
    let metric = SingletonMetric::default();

    for n_objects in [10, 100, 1000] {
        let obs = generate_synthetic_data(n_objects, 10, 3);
        group.bench_with_input(BenchmarkId::new("objects", n_objects), &obs, |b, obs| {
            b.iter(|| {
                black_box(analyze_observations(obs, None, &metric).unwrap());
            })
        });
    }

    group.finish();
}

fn bench_io_read(c: &mut Criterion) {
    let obs_path = test_data_dir().join("observations.parquet");

    c.bench_function("io_read_observations_150", |b| {
        b.iter(|| {
            black_box(read_observations(&obs_path).unwrap());
        })
    });
}

criterion_group!(
    benches,
    bench_cifi_singleton,
    bench_cifi_tracklet,
    bench_full_pipeline,
    bench_cifi_scaling,
    bench_io_read,
);
criterion_main!(benches);
