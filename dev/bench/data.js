window.BENCHMARK_DATA = {
  "lastUpdate": 1775669294126,
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
      }
    ]
  }
}