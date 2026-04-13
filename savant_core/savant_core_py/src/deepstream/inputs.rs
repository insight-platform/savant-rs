//! Python bindings for DeepStream input adapters.
//!
//! `FlexibleDecoder` is the Rust-side replacement for the removed
//! `MultiStreamDecoder`. PyO3 bindings will be added once the Rust API is
//! stable.

use pyo3::prelude::*;

/// Register input adapter types on ``savant_rs.deepstream``.
pub fn register_inputs_classes(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
