pub mod capi;
/// The draw specification used to draw objects on the frame when they are visualized.
pub mod draw_spec;
pub mod logging;
pub mod match_query;
pub mod pipeline;
/// # Basic objects
///
pub mod primitives;
pub mod telemetry;
pub mod test;
/// # Utility functions
///
pub mod utils;
pub mod zmq;

use pyo3::prelude::*;

/// Returns the version of the package set in Cargo.toml
///
/// Returns
/// -------
/// str
///   The version of the package.
///
#[pyfunction]
pub fn version() -> String {
    savant_core::version()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {}
}
