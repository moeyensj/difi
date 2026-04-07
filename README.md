# difi

Did I Find It?

[![Rust CI](https://github.com/moeyensj/difi/actions/workflows/rust.yml/badge.svg)](https://github.com/moeyensj/difi/actions/workflows/rust.yml)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/152989392.svg)](https://zenodo.org/badge/latestdoi/152989392)
[![Built with Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-D97757?logo=anthropic&logoColor=white&style=flat-square)](https://claude.ai)

<a href="https://b612foundation.org/asteroid-institute/"><img src="https://img.shields.io/badge/Asteroid%20Institute-b612foundation.org-1a1a2e?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48bGluZSB4MT0iMiIgeTE9IjEyIiB4Mj0iMjIiIHkyPSIxMiIvPjxwYXRoIGQ9Ik0xMiAyYTE1LjMgMTUuMyAwIDAgMSA0IDEwIDE1LjMgMTUuMyAwIDAgMS00IDEwIDE1LjMgMTUuMyAwIDAgMS00LTEwIDE1LjMgMTUuMyAwIDAgMSA0LTEweiIvPjwvc3ZnPg==&logoColor=white&style=flat-square" alt="Asteroid Institute"></a>
<a href="https://dirac.astro.washington.edu/"><img src="https://img.shields.io/badge/DIRAC%20Institute-dirac.astro.washington.edu-1a1a2e?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48bGluZSB4MT0iMiIgeTE9IjEyIiB4Mj0iMjIiIHkyPSIxMiIvPjxwYXRoIGQ9Ik0xMiAyYTE1LjMgMTUuMyAwIDAgMSA0IDEwIDE1LjMgMTUuMyAwIDAgMS00IDEwIDE1LjMgMTUuMyAwIDAgMS00LTEwIDE1LjMgMTUuMyAwIDAgMSA0LTEweiIvPjwvc3ZnPg==&logoColor=white&style=flat-square" alt="DIRAC Institute"></a>

---

difi evaluates the completeness and purity of astronomical object linkage results
from software such as [THOR](https://github.com/moeyensj/thor),
[HelioLinC](https://github.com/lsst-dm/heliolinc2), or
[MOPS](https://github.com/lsst/mops_daymops). Given observations with known
object associations and a set of predicted linkages, difi determines which
objects were successfully discovered and how clean each linkage is.

This is difi v3 — a ground-up rewrite in Rust with Python bindings. It uses
[Rayon](https://github.com/rayon-rs/rayon) for parallelism and
[Apache Arrow](https://arrow.apache.org/) for zero-copy data interchange.

## Installation

### Python (recommended)

Requires Python 3.10+ and a Rust toolchain (1.85+).

```bash
pip install maturin
git clone https://github.com/moeyensj/difi.git && cd difi
maturin develop --release
```

Verify:

```python
>>> import difi
>>> difi.__version__
'3.0.0.dev0'
```

### Rust only

```bash
cargo build --release
```

## Input format

difi requires two input tables:

### Observations

Each row is a single detection with an optional ground-truth object label.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | string | no | Unique observation identifier |
| `time` | struct{days: i64, nanos: i64} | no | Epoch as MJD days + nanoseconds |
| `ra` | f64 | no | Right ascension (degrees, 0-360) |
| `dec` | f64 | no | Declination (degrees, -90 to 90) |
| `ra_sigma` | f64 | yes | RA uncertainty (degrees) |
| `dec_sigma` | f64 | yes | Dec uncertainty (degrees) |
| `observatory_code` | string | no | Observatory/telescope identifier |
| `object_id` | string | yes | Ground-truth object label (null if unknown) |
| `night` | i64 | no | Local observing night identifier |

### Linkage members

Each row maps a predicted linkage to one of its constituent observations.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `linkage_id` | string | no | Linkage identifier |
| `obs_id` | string | no | Observation identifier (foreign key to observations.id) |

Both tables are read from Parquet files or passed as quivr/PyArrow objects.

## Usage

### Python

Each function accepts either a file path (str/Path) or a quivr Table object.

```python
from difi import analyze_observations, analyze_linkages

# Step 1: CIFI — determine which objects are findable
# Pass file paths...
cifi_result = analyze_observations("observations.parquet")

# ...or quivr Tables
from difi import Observations
observations = Observations.from_parquet("observations.parquet")
cifi_result = analyze_observations(observations)

# Step 2: DIFI — classify linkages and compute completeness
difi_result = analyze_linkages(
    "observations.parquet",
    "linkage_members.parquet",
    min_obs=6,
    contamination_percentage=20.0,
)
print(f"Completeness: {difi_result['completeness']:.1f}%")
print(f"Pure: {difi_result['num_pure']}, Mixed: {difi_result['num_mixed']}")
```

#### Findability metrics

The `metric` parameter controls what observation pattern makes an object "findable":

```python
from difi import analyze_observations, SingletonMetric, TrackletMetric

# Singleton metric (default): object needs >= min_obs detections
# across >= min_nights distinct nights
result = analyze_observations("observations.parquet")  # uses SingletonMetric()

# Configure the metric
metric = SingletonMetric(min_obs=10, min_nights=5, min_nightly_obs_in_min_nights=2)
result = analyze_observations("observations.parquet", metric=metric)

# Tracklet metric: intra-night tracklets with temporal proximity
# and angular motion on >= min_linkage_nights nights
metric = TrackletMetric(
    tracklet_min_obs=2,
    max_obs_separation=1.5 / 24,       # 1.5 hours in days
    min_linkage_nights=3,
    min_obs_angular_separation=1.0,     # arcseconds
)
result = analyze_observations("observations.parquet", metric=metric)

# String shorthands use default parameters
result = analyze_observations("observations.parquet", metric="tracklets")
```

In Rust, metrics are structs with the same parameters:

```rust
use difi::metrics::singleton::SingletonMetric;
use difi::metrics::tracklet::TrackletMetric;

let singleton = SingletonMetric {
    min_obs: 6,
    min_nights: 3,
    min_nightly_obs_in_min_nights: 2,
};

let tracklet = TrackletMetric {
    tracklet_min_obs: 2,
    max_obs_separation: 1.5 / 24.0,  // days
    min_linkage_nights: 3,
    min_obs_angular_separation: 1.0,  // arcseconds
};
```

### Rust

```rust
use difi::cifi::analyze_observations;
use difi::difi::analyze_linkages;
use difi::io::{read_observations, read_linkage_members};
use difi::metrics::singleton::SingletonMetric;

// Load from Parquet
let (obs, mut interner, _) = read_observations(Path::new("observations.parquet"))?;
let lm = read_linkage_members(Path::new("linkage_members.parquet"), &mut interner)?;

// Step 1: CIFI — determine findability
let metric = SingletonMetric::default();
let (mut all_objects, findable, mut summaries) =
    analyze_observations(&obs, None, &metric)?;

// Step 2: DIFI — classify linkages
let all_linkages = analyze_linkages(
    &obs, &lm, &mut all_objects, &mut summaries[0], 6, 20.0,
)?;
```

### Cross-crate usage (e.g. from THOR)

difi defines `ObservationTable` and `LinkageMemberTable` traits. Implement
them for your own types to call difi directly without data conversion:

```rust
impl difi::types::ObservationTable for MyObservations {
    fn len(&self) -> usize { self.ids.len() }
    fn ids(&self) -> &[u64] { &self.ids }
    fn nights(&self) -> &[i64] { &self.nights }
    fn object_ids(&self) -> &[u64] { &self.object_ids }
    // ...
}

let (objects, findable, summaries) =
    difi::cifi::analyze_observations(&my_obs, None, &metric)?;
```

## Pipeline

difi operates in two phases:

1. **CIFI (Can I Find It?)** — Determines which objects are "findable" based on
   observation patterns. Supports two metrics:
   - **SingletonMetric**: >= `min_obs` observations across >= `min_nights` nights
   - **TrackletMetric**: intra-night tracklets with temporal and angular separation constraints

2. **DIFI (Did I Find It?)** — Classifies each linkage as exactly one of:
   - **Pure** — all observations belong to one object
   - **Pure complete** — pure + contains all partition observations of that object
   - **Contaminated** — mostly one object, contamination <= threshold
   - **Mixed** — too contaminated to attribute to a single object

   **Completeness** = (found objects / findable objects) x 100%

## Partitions

Partitions divide observations into time windows for analysis. By default,
difi creates a single partition spanning all nights. For multi-window analysis
(e.g. sliding detection windows), create partitions explicitly:

```python
from difi import Partitions

# Single partition spanning all nights (default behavior)
partitions = Partitions.create_single(observations.night)

# Non-overlapping windows of 15 nights
partitions = Partitions.create_linking_windows(
    observations.night,
    detection_window=15,
)

# Sliding windows: 15-night window slides by 1 night, with a ramp-up
# starting from min_nights=3
partitions = Partitions.create_linking_windows(
    observations.night,
    detection_window=15,
    min_nights=3,
    sliding=True,
)
```

In Rust:

```rust
use difi::partitions::{create_single, create_linking_windows};

// Single partition
let partition = create_single(&nights)?;

// Non-overlapping 15-night windows
let partitions = create_linking_windows(&nights, Some(15), None, false)?;

// Sliding windows with ramp-up from 3 nights
let partitions = create_linking_windows(&nights, Some(15), Some(3), true)?;
```

## Performance

Benchmarked on the neomod_quads survey dataset (166M observations, 15,935 objects):

| Scale | Python v2 | Rust v3 | Speedup |
|-------|-----------|---------|---------|
| 55M obs (30 nights) | 23.0s | 0.42s | 55x |
| 111M obs (60 nights) | 67.3s | 0.85s | 80x |
| 166M obs (90 nights) | 132.9s | 1.24s | 107x |

Memory at 100M observations: ~3.2 GB (DIFI), ~5.6 GB (CIFI with tracklets).

## Development

```bash
# Build
cargo build

# Test (18 tests including exact Python v2 parity checks)
cargo test

# Lint
cargo clippy --all-targets --all-features

# Benchmarks
cargo bench

# Build Python package
maturin develop --release
```

## Acknowledgments

This work was supported by the [Asteroid Institute](https://b612foundation.org/asteroid-institute/)
(a program of the B612 Foundation) and the [DIRAC Institute](https://dirac.astro.washington.edu/)
at the University of Washington.

## License

BSD 3-Clause. See [LICENSE.md](LICENSE.md) for details.
