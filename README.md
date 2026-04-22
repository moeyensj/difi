# difi

Did I Find It?

[![Rust CI](https://github.com/moeyensj/difi/actions/workflows/rust.yml/badge.svg)](https://github.com/moeyensj/difi/actions/workflows/rust.yml)
[![Coverage Status](https://coveralls.io/repos/github/moeyensj/difi/badge.svg?branch=main)](https://coveralls.io/github/moeyensj/difi?branch=main)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/152989392.svg)](https://zenodo.org/badge/latestdoi/152989392)
[![Built with Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-D97757?logo=anthropic&logoColor=white&style=flat-square)](https://claude.ai)  
<a href="https://b612.ai/"><img src="https://img.shields.io/badge/Asteroid%20Institute-b612foundation.org-1a1a2e?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48bGluZSB4MT0iMiIgeTE9IjEyIiB4Mj0iMjIiIHkyPSIxMiIvPjxwYXRoIGQ9Ik0xMiAyYTE1LjMgMTUuMyAwIDAgMSA0IDEwIDE1LjMgMTUuMyAwIDAgMS00IDEwIDE1LjMgMTUuMyAwIDAgMS00LTEwIDE1LjMgMTUuMyAwIDAgMSA0LTEweiIvPjwvc3ZnPg==&logoColor=white&style=flat-square" alt="Asteroid Institute"></a>
<a href="https://dirac.astro.washington.edu/"><img src="https://img.shields.io/badge/DIRAC%20Institute-dirac.astro.washington.edu-1a1a2e?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48bGluZSB4MT0iMiIgeTE9IjEyIiB4Mj0iMjIiIHkyPSIxMiIvPjxwYXRoIGQ9Ik0xMiAyYTE1LjMgMTUuMyAwIDAgMSA0IDEwIDE1LjMgMTUuMyAwIDAgMS00IDEwIDE1LjMgMTUuMyAwIDAgMS00LTEwIDE1LjMgMTUuMyAwIDAgMSA0LTEweiIvPjwvc3ZnPg==&logoColor=white&style=flat-square" alt="DIRAC Institute"></a>

---

difi evaluates the completeness and purity of astronomical object linkage results
from software such as [THOR](https://github.com/moeyensj/thor),
[HelioLinC](https://github.com/lsst-dm/heliolinc2), or
[MOPS](https://github.com/lsst/mops_daymops). Given observations with known
object associations and a set of predicted linkages, difi determines which
objects were successfully discovered and how clean each linkage is.

This is difi v2 — a ground-up rewrite in Rust with Python bindings. It uses
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
>>> difi.__version__  # matches the published rc
```

### Rust only

```bash
cargo build --release
```

### CLI

The `difi` binary ships with the crate behind a `cli` Cargo feature (so
library consumers don't pull clap/toml/anyhow transitively):

```bash
cargo install difi-rs --features cli
# or from a checkout
cargo build --release --features cli   # binary: ./target/release/difi
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
# num_ignored_linkages counts linkages excluded because they had no
# observations inside the analysis partition (non-zero signals a mismatch
# between your --linkages file and the observation set).
print(f"Ignored: {difi_result['num_ignored_linkages']}")
```

#### Findability metrics

The `metric` parameter controls what observation pattern makes an object "findable":

```python
from difi import analyze_observations

# Singleton metric (default): object needs >= min_obs detections
# across >= min_nights distinct nights
result = analyze_observations(
    "observations.parquet",
    metric="singletons",
    min_obs=6,
    min_nights=3,
)

# Tracklet metric: object needs intra-night tracklets (multiple
# detections within max_obs_separation hours showing angular motion)
# on >= min_nights distinct nights
result = analyze_observations(
    "observations.parquet",
    metric="tracklets",
    min_nights=3,
)
```

In Rust, metrics are structs with full configuration:

```rust
use difi::metrics::singleton::SingletonMetric;
use difi::metrics::tracklet::TrackletMetric;

// Singleton: 6 obs across 3 nights, at least 2 obs/night when exactly 3 nights
let singleton = SingletonMetric {
    min_obs: 6,
    min_nights: 3,
    min_nightly_obs_in_min_nights: 2,
};

// Tracklet: 2+ obs per tracklet within 1.5 hours, 1" angular separation,
// tracklets on 3+ nights
let tracklet = TrackletMetric {
    tracklet_min_obs: 2,
    max_obs_separation: 1.5 / 24.0,  // days
    min_linkage_nights: 3,
    min_obs_angular_separation: 1.0,  // arcseconds
};
```

### CLI

A thin wrapper over the library, for shell pipelines and reproducible runs.
Subcommand names mirror the Python verbs; short aliases keep shell use ergonomic.

```bash
# CIFI: findability from observations  (alias: `difi cifi`)
difi analyze-observations \
    -i observations.parquet \
    -o out/ \
    --metric singletons --min-obs 6 --min-nights 3

# CIFI + DIFI: classify linkages end-to-end  (alias: `difi analyze`)
difi analyze-linkages \
    -i observations.parquet \
    -l linkage_members.parquet \
    -o out/ \
    --contamination-percentage 20
```

Outputs in `<output-dir>/`:

| File | Written by | Contents |
|---|---|---|
| `all_objects.parquet` | both | One row per (object, partition) — CIFI findability flag plus DIFI linkage stats merged in |
| `findable_observations.parquet` | both | One row per findable (object, partition) with discovery night |
| `partition_summaries.parquet` | both | One row per partition with observation / findable / found / completeness counts |
| `all_linkages.parquet` | `analyze-linkages` | One row per classified (linkage, partition) with pure/contaminated/mixed flags |
| `ignored_linkages.parquet` | `analyze-linkages` (only when non-empty) | Linkages excluded from classification, with reason + partition |
| `run_manifest.json` | both | argv, input SHA-256 prefixes, host, per-scenario timings, `warnings` counts, optional `reused_cifi` provenance |

#### Partitioned CIFI

```bash
# Sliding 30-night windows
difi cifi -i observations.parquet -o out/ \
    --partition-mode sliding --partition-window 30

# Tracklets with non-overlapping 15-night blocks
difi cifi -i observations.parquet -o out/ \
    --metric tracklets --min-linkage-nights 3 \
    --partition-mode blocks --partition-window 15
```

Partition flags also apply to `analyze-linkages` — the CLI loops DIFI over
each partition's summary and writes a combined `all_linkages.parquet` keyed by
`partition_id`. Linkages whose observations fall entirely outside a given
partition are excluded from `all_linkages.parquet` and reported in a separate
`ignored_linkages.parquet`; the manifest's `warnings` section surfaces counts
so a run with an unexpectedly high `orphan_linkages` value (linkages that
never intersect any partition) flags a likely mismatched `--linkages` file.

#### Reusing a CIFI snapshot

CIFI is the expensive phase on survey-scale inputs. Run it once, reuse across
multiple linkage sets:

```bash
# Produce a reusable CIFI snapshot
difi cifi -i observations.parquet -o cifi_snapshot/

# Classify two independent linkage sets against the same CIFI work
difi analyze-linkages -i observations.parquet -l thor_linkages.parquet \
    --cifi-output-dir cifi_snapshot/ -o difi_thor/
difi analyze-linkages -i observations.parquet -l precovery_linkages.parquet \
    --cifi-output-dir cifi_snapshot/ -o difi_precovery/
```

`--cifi-output-dir` is mutually exclusive with partition flags (the snapshot
encodes its own partitions). A SHA-256 prefix of the observations file is
stored in each manifest; mismatches between the snapshot and the current
observations fail fast with a clear error.

#### Batch scenarios

Declare scenarios in TOML for findability sweeps (LSST baselines, etc.):

```toml
# lsst_findability.toml
[defaults]
observations = "/path/to/observations.parquet"

[[scenario]]
name = "singleton_6obs_3nights"
metric = "singletons"
min_obs = 6
min_nights = 3

[[scenario]]
name = "tracklet_3pairs_15nights"
metric = "tracklets"
min_linkage_nights = 3
partition_mode = "sliding"
partition_window = 15
```

```bash
difi cifi --scenarios lsst_findability.toml -o results/
# results/<scenario>/all_objects.parquet, partition_summaries.parquet, ...
# results/run_manifest.json summarizes every scenario
```

Per-scenario `observations = "..."` overrides `[defaults]`.

#### Machine-readable progress

`--progress-json` emits one NDJSON event per line on stdout; human text still
goes to stderr.

```bash
difi --progress-json cifi -i observations.parquet -o out/ \
    | jq -c 'select(.event == "scenario_done")'
```

Errors always produce a human line on stderr; under `--progress-json` an
`{"event":"error", ...}` line is additionally written to stdout so machine
consumers see them too.

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

// Step 2: DIFI — classify linkages. Returns (AllLinkages, IgnoredLinkages).
// Linkages whose observations all fall outside summaries[0]'s night range
// are redirected to `ignored` with reason NoObservationsInPartition, instead
// of producing phantom pure/contaminated/mixed rows.
let (all_linkages, ignored) = analyze_linkages(
    &obs, &lm, &mut all_objects, &mut summaries[0], 6, 20.0,
)?;
```

For multi-partition DIFI, loop over `summaries` and concatenate each partition's
`AllLinkages` / `IgnoredLinkages`. The `update_all_objects` call inside
`analyze_linkages` scopes its writes to the current partition's rows in
`AllObjects`, so multi-partition loops are safe.

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

   **Completeness** = (found objects / findable objects) × 100%, where
   *found* counts objects for which at least one pure linkage contains
   ≥ `min_obs` observations **inside the partition**. Single-partition runs see
   no difference from the whole-linkage interpretation; multi-partition runs
   stay bounded by in-partition evidence instead of inflating when a
   cross-boundary linkage has enough total obs but few inside any one window.

## Performance

Benchmarked on the neomod_quads survey dataset (166M observations, 15,935 objects):

| Scale | Python (v2rc3) | Rust (v2rc4) | Speedup |
|-------|-----------|---------|---------|
| 55M obs (30 nights) | 23.0s | 0.42s | 55x |
| 111M obs (60 nights) | 67.3s | 0.85s | 80x |
| 166M obs (90 nights) | 132.9s | 1.24s | 107x |

Memory at 100M observations: ~3.2 GB (DIFI), ~5.6 GB (CIFI with tracklets).

## Development

```bash
# Build the library
cargo build

# Build the library + CLI binary
cargo build --features cli

# Verify lib-only build pulls no CLI deps
cargo build --no-default-features

# Full test suite (library + CLI integration tests)
cargo test --features cli

# Lint
cargo clippy --all-targets --features cli -- -D warnings

# Verify Cargo.toml and pyproject.toml versions agree (CI runs this on every
# PR; the publish workflow also checks tag vs both files before any upload)
./scripts/check_versions.sh

# Benchmarks
cargo bench

# Build Python package
maturin develop --release
```

## Acknowledgments

This work was supported by the [Asteroid Institute](https://asteroidinstitute.org/)
(a program of the [B612 Foundation](https://b612foundation.org/)) and the
[DIRAC Institute](https://dirac.astro.washington.edu/) at the University of Washington.

## License

BSD 3-Clause. See [LICENSE.md](LICENSE.md) for details.
