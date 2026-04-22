//! Python bindings for difi via PyO3.
//!
//! Exposes CIFI and DIFI analysis as Python functions.
//! Data interchange uses Arrow RecordBatches (zero-copy via FFI).
//!
//! Gated behind the `python` feature flag.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::cifi;
use crate::difi as difi_mod;
use crate::io;
use crate::metrics::FindabilityMetric;
use crate::metrics::singleton::SingletonMetric;
use crate::metrics::tracklet::TrackletMetric;

/// The difi Python module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_observations, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_linkages, m)?)?;
    Ok(())
}

/// Return the difi version string.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Parse a JSON metric config string into a FindabilityMetric.
///
/// Expected format:
///   {"type": "singletons", "min_obs": 6, "min_nights": 3, "min_nightly_obs_in_min_nights": 1}
///   {"type": "tracklets", "tracklet_min_obs": 2, "max_obs_separation": 0.0625, ...}
fn parse_metric(metric_json: &str) -> PyResult<Box<dyn FindabilityMetric>> {
    let config: serde_json::Value = serde_json::from_str(metric_json)
        .map_err(|e| PyRuntimeError::new_err(format!("Invalid metric JSON: {e}")))?;

    let metric_type = config
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| PyRuntimeError::new_err("Metric JSON must have a 'type' field"))?;

    match metric_type {
        "singletons" => {
            let min_obs = config.get("min_obs").and_then(|v| v.as_u64()).unwrap_or(6) as usize;
            let min_nights = config
                .get("min_nights")
                .and_then(|v| v.as_u64())
                .unwrap_or(3) as usize;
            let min_nightly = config
                .get("min_nightly_obs_in_min_nights")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize;
            Ok(Box::new(SingletonMetric {
                min_obs,
                min_nights,
                min_nightly_obs_in_min_nights: min_nightly,
            }))
        }
        "tracklets" => {
            let tracklet_min_obs = config
                .get("tracklet_min_obs")
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as usize;
            let max_obs_separation = config
                .get("max_obs_separation")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.5 / 24.0);
            let min_linkage_nights = config
                .get("min_linkage_nights")
                .and_then(|v| v.as_u64())
                .unwrap_or(3) as usize;
            let min_angular = config
                .get("min_obs_angular_separation")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
            Ok(Box::new(TrackletMetric {
                tracklet_min_obs,
                max_obs_separation,
                min_linkage_nights,
                min_obs_angular_separation: min_angular,
            }))
        }
        _ => Err(PyRuntimeError::new_err(format!(
            "Unknown metric type: {metric_type}"
        ))),
    }
}

/// Can I Find It? — Determine findability of objects in observations.
///
/// Arguments:
///   observations_path: Path to observations Parquet file.
///   metric_json: JSON string with metric type and parameters.
///
/// Returns: dict with object/findable counts and per-object details.
#[pyfunction]
#[pyo3(signature = (observations_path, metric_json))]
fn analyze_observations(
    py: Python<'_>,
    observations_path: &str,
    metric_json: &str,
) -> PyResult<Py<PyDict>> {
    let obs_path = std::path::Path::new(observations_path);

    let (obs, id_interner, _) = io::read_observations(obs_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read observations: {e}")))?;

    let metric_impl = parse_metric(metric_json)?;

    let (all_objects, findable, summaries) =
        cifi::analyze_observations(&obs, None, metric_impl.as_ref())
            .map_err(|e| PyRuntimeError::new_err(format!("CIFI failed: {e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("num_objects", all_objects.len())?;
    dict.set_item("num_findable", findable.len())?;
    dict.set_item("num_partitions", summaries.len())?;

    if !summaries.is_empty() {
        dict.set_item("findable", summaries[0].findable)?;
    }

    // Include per-object details
    let objects_list: Vec<_> = (0..all_objects.len())
        .map(|i| {
            let d = PyDict::new(py);
            let _ = d.set_item(
                "object_id",
                id_interner.resolve(all_objects.object_id[i]).unwrap_or(""),
            );
            let _ = d.set_item("num_obs", all_objects.num_obs[i]);
            let _ = d.set_item("findable", all_objects.findable[i]);
            let _ = d.set_item("arc_length", all_objects.arc_length[i]);
            d
        })
        .collect();
    dict.set_item("objects", objects_list)?;

    Ok(dict.unbind())
}

/// Did I Find It? — Classify linkages and compute completeness.
///
/// Arguments:
///   observations_path: Path to observations Parquet file.
///   linkage_members_path: Path to linkage members Parquet file.
///   metric_json: JSON string with metric type and parameters.
///   min_obs: Minimum observations for "found" (default: 6).
///   contamination_percentage: Max contamination % (default: 20.0).
///
/// Returns: dict with classification counts and completeness.
#[pyfunction]
#[pyo3(signature = (observations_path, linkage_members_path, metric_json, min_obs=6, contamination_percentage=20.0))]
fn analyze_linkages(
    py: Python<'_>,
    observations_path: &str,
    linkage_members_path: &str,
    metric_json: &str,
    min_obs: usize,
    contamination_percentage: f64,
) -> PyResult<Py<PyDict>> {
    let obs_path = std::path::Path::new(observations_path);
    let lm_path = std::path::Path::new(linkage_members_path);

    let (obs, id_interner, _) = io::read_observations(obs_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read observations: {e}")))?;
    let mut id_interner2 = id_interner.clone();
    let lm = io::read_linkage_members(lm_path, &mut id_interner2)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read linkage members: {e}")))?;

    // CIFI
    let metric_impl = parse_metric(metric_json)?;

    let (mut all_objects, _findable, mut summaries) =
        cifi::analyze_observations(&obs, None, metric_impl.as_ref())
            .map_err(|e| PyRuntimeError::new_err(format!("CIFI failed: {e}")))?;

    if summaries.is_empty() {
        return Err(PyRuntimeError::new_err("No partitions created"));
    }

    // DIFI
    let (all_linkages, ignored_linkages) = difi_mod::analyze_linkages(
        &obs,
        &lm,
        &mut all_objects,
        &mut summaries[0],
        min_obs,
        contamination_percentage,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("DIFI failed: {e}")))?;

    let n_pure: usize = all_linkages.pure.iter().filter(|&&p| p).count();
    let n_contaminated: usize = all_linkages.contaminated.iter().filter(|&&c| c).count();
    let n_mixed: usize = all_linkages.mixed.iter().filter(|&&m| m).count();

    let dict = PyDict::new(py);
    dict.set_item("num_linkages", all_linkages.len())?;
    dict.set_item("num_pure", n_pure)?;
    dict.set_item("num_contaminated", n_contaminated)?;
    dict.set_item("num_mixed", n_mixed)?;
    dict.set_item("num_ignored_linkages", ignored_linkages.len())?;
    dict.set_item("completeness", summaries[0].completeness)?;
    dict.set_item("found", summaries[0].found)?;
    dict.set_item("findable", summaries[0].findable)?;

    // Per-linkage details
    let linkages_list: Vec<_> = (0..all_linkages.len())
        .map(|i| {
            let d = PyDict::new(py);
            let _ = d.set_item(
                "linkage_id",
                id_interner2
                    .resolve(all_linkages.linkage_id[i])
                    .unwrap_or(""),
            );
            let _ = d.set_item(
                "linked_object_id",
                id_interner2
                    .resolve(all_linkages.linked_object_id[i])
                    .unwrap_or(""),
            );
            let _ = d.set_item("num_obs", all_linkages.num_obs[i]);
            let _ = d.set_item("contamination", all_linkages.contamination[i]);
            let _ = d.set_item("pure", all_linkages.pure[i]);
            let _ = d.set_item("contaminated", all_linkages.contaminated[i]);
            let _ = d.set_item("mixed", all_linkages.mixed[i]);
            let _ = d.set_item("pure_complete", all_linkages.pure_complete[i]);
            d
        })
        .collect();
    dict.set_item("linkages", linkages_list)?;

    Ok(dict.unbind())
}
