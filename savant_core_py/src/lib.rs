pub mod capi;
/// The draw specification used to draw objects on the frame when they are visualized.
pub mod draw_spec;
pub mod logging;
pub mod match_query;
pub mod pipeline;
/// # Basic objects
///
pub mod primitives;
pub mod test;
/// # Utility functions
///
pub mod utils;
pub mod zmq;


use pyo3::prelude::*;





/// Initializes Jaeger tracer.
///
/// Params
/// ------
/// service_name: str
///   The name of the service used by logger.
/// endpoint: str
///   The endpoint of the Jaeger collector.
///
#[pyfunction]
pub fn init_jaeger_tracer(service_name: &str, endpoint: &str) {
    savant_core::telemetry::init_jaeger_tracer(service_name, endpoint);
}

/// Initializes Noop tracer.
///
/// This is useful when the telemetry is not required.
///
#[pyfunction]
pub fn init_noop_tracer() {
    savant_core::telemetry::init_noop_tracer();
}

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
