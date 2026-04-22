window.BENCHMARK_DATA = {
  "lastUpdate": 1776886494493,
  "repoUrl": "https://github.com/moeyensj/difi",
  "entries": {
    "difi Benchmarks": [
      {
        "commit": {
          "author": {
            "email": "moeyensj@gmail.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "committer": {
            "email": "moeyensj@gmail.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "distinct": true,
          "id": "04ddaa197e195caccc025397585d928bedd794b9",
          "message": "Rename crate to difi-rs for crates.io and fix macos runner\n\nPackage name difi-rs (lib name stays difi for use difi::... imports).\nReplace deprecated macos-13 with macos-latest in publish workflow.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-08T10:06:04-07:00",
          "tree_id": "1ca0b75df997ee525b0d0852bb35221be68aefd5",
          "url": "https://github.com/moeyensj/difi/commit/04ddaa197e195caccc025397585d928bedd794b9"
        },
        "date": 1775668498619,
        "tool": "cargo",
        "benches": [
          {
            "name": "cifi_singleton_5obj_150obs",
            "value": 67183,
            "range": "± 1796",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_tracklet_5obj_150obs",
            "value": 79544,
            "range": "± 2863",
            "unit": "ns/iter"
          },
          {
            "name": "full_pipeline_5obj_20linkages",
            "value": 187033,
            "range": "± 2980",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/10",
            "value": 107890,
            "range": "± 8544",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/100",
            "value": 499850,
            "range": "± 6971",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/1000",
            "value": 3474521,
            "range": "± 64218",
            "unit": "ns/iter"
          },
          {
            "name": "io_read_observations_150",
            "value": 294389,
            "range": "± 1908",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "moeyensj@gmail.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "committer": {
            "email": "moeyensj@gmail.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "distinct": true,
          "id": "1567cb43564a3cd7f5410fac71ad1f5f8d55bbe3",
          "message": "Rename crate to difi-rs, publish on tag push, fix macos runner\n\nPackage name difi-rs for crates.io (lib name stays difi for imports).\nTrigger publish workflow on v* tag push instead of GitHub release.\nReplace deprecated macos-13 with macos-latest for wheel builds.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-08T10:17:40-07:00",
          "tree_id": "1ca0b75df997ee525b0d0852bb35221be68aefd5",
          "url": "https://github.com/moeyensj/difi/commit/1567cb43564a3cd7f5410fac71ad1f5f8d55bbe3"
        },
        "date": 1775668809997,
        "tool": "cargo",
        "benches": [
          {
            "name": "cifi_singleton_5obj_150obs",
            "value": 80240,
            "range": "± 2315",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_tracklet_5obj_150obs",
            "value": 95181,
            "range": "± 2395",
            "unit": "ns/iter"
          },
          {
            "name": "full_pipeline_5obj_20linkages",
            "value": 161144,
            "range": "± 5220",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/10",
            "value": 97435,
            "range": "± 4297",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/100",
            "value": 489701,
            "range": "± 7156",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/1000",
            "value": 3522007,
            "range": "± 30200",
            "unit": "ns/iter"
          },
          {
            "name": "io_read_observations_150",
            "value": 207396,
            "range": "± 5250",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "moeyensj@gmail.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "committer": {
            "email": "moeyensj@gmail.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "distinct": true,
          "id": "e4a05b4964b67f2328b2add7b1eeac999b8e189e",
          "message": "Update B612 Foundation links to b612.ai\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-08T10:26:01-07:00",
          "tree_id": "d032a28041ce3555d77250c671d1102d75d6b61a",
          "url": "https://github.com/moeyensj/difi/commit/e4a05b4964b67f2328b2add7b1eeac999b8e189e"
        },
        "date": 1775669293379,
        "tool": "cargo",
        "benches": [
          {
            "name": "cifi_singleton_5obj_150obs",
            "value": 64684,
            "range": "± 11210",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_tracklet_5obj_150obs",
            "value": 79991,
            "range": "± 3475",
            "unit": "ns/iter"
          },
          {
            "name": "full_pipeline_5obj_20linkages",
            "value": 190818,
            "range": "± 2930",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/10",
            "value": 111008,
            "range": "± 6532",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/100",
            "value": 488219,
            "range": "± 18373",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/1000",
            "value": 3383073,
            "range": "± 22055",
            "unit": "ns/iter"
          },
          {
            "name": "io_read_observations_150",
            "value": 271320,
            "range": "± 2110",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "moeyensj@gmail.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "committer": {
            "email": "moeyensj@gmail.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "distinct": true,
          "id": "03c95381f731f84e3c4f6d159a14838b9983899b",
          "message": "Rename crate to difi-rs, bump to v2.0.0-rc5, fix publish workflow\n\nPackage name difi-rs for crates.io (lib name stays difi for imports).\nTrigger publish on v* tag push. Replace deprecated macos-13 runner.\nUpdate B612 Foundation links to b612.ai.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-08T11:28:40-07:00",
          "tree_id": "3aab0634d2155dd3e8e112a74f3dc0f7f2cb9923",
          "url": "https://github.com/moeyensj/difi/commit/03c95381f731f84e3c4f6d159a14838b9983899b"
        },
        "date": 1775673061450,
        "tool": "cargo",
        "benches": [
          {
            "name": "cifi_singleton_5obj_150obs",
            "value": 64564,
            "range": "± 4947",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_tracklet_5obj_150obs",
            "value": 80067,
            "range": "± 7932",
            "unit": "ns/iter"
          },
          {
            "name": "full_pipeline_5obj_20linkages",
            "value": 191896,
            "range": "± 1552",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/10",
            "value": 112329,
            "range": "± 10104",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/100",
            "value": 495948,
            "range": "± 6827",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/1000",
            "value": 3408825,
            "range": "± 31632",
            "unit": "ns/iter"
          },
          {
            "name": "io_read_observations_150",
            "value": 271542,
            "range": "± 2025",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "moeyensj@gmail.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "committer": {
            "email": "moeyensj@gmail.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "distinct": true,
          "id": "f87511030ed886a030aa473af1117ebdbaf9c063",
          "message": "Rename crate to difi-rs, bump to v2.0.0-rc5, fix publish workflow\n\nPackage name difi-rs for crates.io (lib name stays difi for imports).\nTrigger publish on v* tag push. Replace deprecated macos-13 runner.\nUpdate B612 Foundation links to b612.ai.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-08T11:38:16-07:00",
          "tree_id": "e86d7bf4b97f0491bc7660bc71496fc6c250d14b",
          "url": "https://github.com/moeyensj/difi/commit/f87511030ed886a030aa473af1117ebdbaf9c063"
        },
        "date": 1775673632097,
        "tool": "cargo",
        "benches": [
          {
            "name": "cifi_singleton_5obj_150obs",
            "value": 64648,
            "range": "± 7161",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_tracklet_5obj_150obs",
            "value": 80392,
            "range": "± 5534",
            "unit": "ns/iter"
          },
          {
            "name": "full_pipeline_5obj_20linkages",
            "value": 190606,
            "range": "± 3367",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/10",
            "value": 111033,
            "range": "± 7866",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/100",
            "value": 490237,
            "range": "± 8819",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/1000",
            "value": 3399966,
            "range": "± 33744",
            "unit": "ns/iter"
          },
          {
            "name": "io_read_observations_150",
            "value": 269836,
            "range": "± 1380",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "moeyensj@gmail.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "committer": {
            "email": "moeyensj@users.noreply.github.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "distinct": true,
          "id": "b4ebf50da117a070aed85bade1cf7c5498637ab5",
          "message": "Add difi CLI binary\n\nNew `difi` binary gated behind the `cli` Cargo feature so lib-only\nconsumers (thor-rust, precovery) don't compile clap/toml/anyhow/sha2.\n\nSubcommands:\n- `difi analyze-observations` (alias `cifi`) — runs CIFI, writes\n  all_objects / findable_observations / partition_summaries parquet\n  plus a run_manifest.json with version, argv, input sha256-prefix,\n  host info, and per-scenario timings.\n- `difi analyze-linkages` (alias `analyze`) — runs CIFI then DIFI in\n  one shot, adds all_linkages.parquet.\n\nAlso:\n- `--scenarios <FILE>.toml` on analyze-observations for batch CIFI\n  runs with per-scenario observations/metric/partition overrides.\n- `--progress-json` emits NDJSON progress events on stdout; human\n  messages always go to stderr regardless. A panic hook additionally\n  emits `{\"event\":\"error\"}` on stdout before the default hook runs,\n  so the NDJSON contract survives library panics.\n- Scenario observations-path precedence: per-scenario entry >\n  --observations CLI flag > [defaults] (most explicit wins; CLI\n  overrides file defaults for ephemeral re-runs).\n- Exit codes per proposal: 1 (config/args), 2 (I/O), 3 (data); chain\n  inspection prefers the library's classified error over a raw\n  io::Error, since parquet/arrow errors can carry an inner io::Error.\n- README updated with a CLI section (install, subcommand examples,\n  partitions, scenarios TOML, --progress-json) and feature-gated dev\n  commands.\n\nTests (`tests/cli.rs`, 16 cases via assert_cmd + predicates): subcommand\noutputs, cifi / analyze aliases, sliding partitions, scenarios TOML,\nCLI-observations overrides TOML defaults, missing-input returns exit\ncode 2, missing-window error path, NDJSON validity, error events\ndual-emitted to stdout+stderr, --help / --version.\n\nLibrary (`src/lib.rs` and below) untouched — the CLI is pure glue.\n\nVerified:\n- `cargo build --no-default-features`   # no CLI deps leak into lib builds\n- `cargo build --features cli`\n- `cargo fmt -- --check`\n- `cargo clippy --all-targets --all-features`\n- `cargo test --features cli`           # 30 passed (16 new + 14 existing)\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-22T11:22:41-07:00",
          "tree_id": "5a2b280c2f2e06669c4a10d1755715ba667bf56a",
          "url": "https://github.com/moeyensj/difi/commit/b4ebf50da117a070aed85bade1cf7c5498637ab5"
        },
        "date": 1776882478683,
        "tool": "cargo",
        "benches": [
          {
            "name": "cifi_singleton_5obj_150obs",
            "value": 68472,
            "range": "± 1337",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_tracklet_5obj_150obs",
            "value": 80791,
            "range": "± 2281",
            "unit": "ns/iter"
          },
          {
            "name": "full_pipeline_5obj_20linkages",
            "value": 146282,
            "range": "± 18634",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/10",
            "value": 109701,
            "range": "± 4304",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/100",
            "value": 481763,
            "range": "± 19962",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/1000",
            "value": 3802278,
            "range": "± 282991",
            "unit": "ns/iter"
          },
          {
            "name": "io_read_observations_150",
            "value": 295833,
            "range": "± 1766",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "moeyensj@gmail.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "committer": {
            "email": "moeyensj@users.noreply.github.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "distinct": true,
          "id": "d680bfac1894fc550ed64967094ffc0235f280e2",
          "message": "Add version-consistency check to guard Cargo/pyproject/tag drift\n\nToday's rc6 publish partially failed: crates.io published difi-rs\nv2.0.0-rc6 fine, but PyPI 400-rejected the upload because pyproject.toml\nwas still pinned at \"2.0.0rc5\" (from the prior release). maturin built\nwheels named difi-2.0.0rc5-*.whl against an rc6 Cargo.toml, PyPI said\n\"File already exists\", and the Publish workflow failed. The same class of\ndrift also explains why v2.0.0rc4 never got a tag (publish failed mid-way\nand was abandoned).\n\nThis adds a shared check and wires it into both workflows so the drift\ncannot reach a release again.\n\n- `scripts/check_versions.sh` — single source of truth. Compares\n  Cargo.toml (SemVer, e.g. `2.0.0-rc6`) against pyproject.toml (PEP 440,\n  e.g. `2.0.0rc6`) on the PEP 440 normal form. Optionally takes a tag\n  argument and verifies it matches both. Emits `::error::` annotations\n  for GitHub Actions and a plain ✓ line for local use.\n\n- `rust.yml` — runs the check on every PR inside the existing\n  build-lint job, immediately after checkout. Drift is caught at PR time,\n  not release time.\n\n- `publish.yml` — new `preflight` job runs the check against the pushed\n  tag before any artifact-producing job starts. publish-crate,\n  build-wheels, and build-sdist all `needs: preflight`, so\n  crates.io/PyPI cannot receive a build from an inconsistent tree.\n\nAlso bumps `pyproject.toml` from rc5 → rc6 so the check passes on this\nPR and pyproject matches the rc6 state of Cargo.toml on main. This does\nnot re-publish rc6 (rc6 is live on crates.io and the PyPI slot is blocked\nby rc5); the next release tag — rc7, when partitions lands — will be the\nfirst to exercise the preflight end-to-end.\n\nVerified locally:\n- `./scripts/check_versions.sh` → pass\n- `./scripts/check_versions.sh v2.0.0rc6` → pass\n- `./scripts/check_versions.sh v2.0.0rc7` → fail (exit 1, two error lines)\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-22T12:32:38-07:00",
          "tree_id": "56ea5c2d76c8c8f5918da1c52313011c3e7d494d",
          "url": "https://github.com/moeyensj/difi/commit/d680bfac1894fc550ed64967094ffc0235f280e2"
        },
        "date": 1776886493616,
        "tool": "cargo",
        "benches": [
          {
            "name": "cifi_singleton_5obj_150obs",
            "value": 65178,
            "range": "± 3390",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_tracklet_5obj_150obs",
            "value": 81651,
            "range": "± 7922",
            "unit": "ns/iter"
          },
          {
            "name": "full_pipeline_5obj_20linkages",
            "value": 142336,
            "range": "± 1181",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/10",
            "value": 108730,
            "range": "± 2847",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/100",
            "value": 462927,
            "range": "± 5260",
            "unit": "ns/iter"
          },
          {
            "name": "cifi_singleton_scaling/objects/1000",
            "value": 3438611,
            "range": "± 29838",
            "unit": "ns/iter"
          },
          {
            "name": "io_read_observations_150",
            "value": 276356,
            "range": "± 23892",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}