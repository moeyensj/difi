//! Python bindings for difi via PyO3.
//!
//! Exposes CIFI and DIFI analysis as Python functions.
//! Data interchange uses Arrow RecordBatches (zero-copy via FFI).
//!
//! Gated behind the `python` feature flag.

use pyo3::prelude::*;

/// The difi Python module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

/// Return the difi version string.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
