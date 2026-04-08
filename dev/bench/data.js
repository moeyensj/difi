window.BENCHMARK_DATA = {
  "lastUpdate": 1775668499570,
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
      }
    ]
  }
}