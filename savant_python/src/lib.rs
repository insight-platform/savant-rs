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

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pymodule;

use crate::logging::{set_log_level, LogLevel};
use crate::match_query::video_object_query;

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
fn init_jaeger_tracer(service_name: &str, endpoint: &str) {
    savant_core::telemetry::init_jaeger_tracer(service_name, endpoint);
}

/// Returns the version of the package set in Cargo.toml
///
#[pyfunction]
pub fn version() -> String {
    savant_core::version()
}

/// Returns version in CRC32 format
///
#[pyfunction]
pub fn version_crc32() -> u32 {
    savant_core::version_crc32()
}

#[pymodule]
fn savant_rs(py: Python, m: &PyModule) -> PyResult<()> {
    let log_env_var_name = "RUST_LOG";
    let log_env_var_level = "trace";
    if std::env::var(log_env_var_name).is_err() {
        std::env::set_var(log_env_var_name, log_env_var_level);
    }
    pretty_env_logger::try_init()
        .map_err(|_| PyRuntimeError::new_err("Failed to initialize logger"))?;
    set_log_level(LogLevel::Error);

    m.add_function(wrap_pyfunction!(init_jaeger_tracer, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(version_crc32, m)?)?;

    m.add_wrapped(wrap_pymodule!(primitives::primitives))?;
    m.add_wrapped(wrap_pymodule!(pipeline::pipeline))?;
    m.add_wrapped(wrap_pymodule!(primitives::geometry))?;
    m.add_wrapped(wrap_pymodule!(primitives::draw_spec))?;
    m.add_wrapped(wrap_pymodule!(utils::utils))?;
    m.add_wrapped(wrap_pymodule!(utils::symbol_mapper_module))?;
    m.add_wrapped(wrap_pymodule!(utils::udf_api_module))?;
    m.add_wrapped(wrap_pymodule!(utils::serialization_module))?;
    m.add_wrapped(wrap_pymodule!(video_object_query))?;
    m.add_wrapped(wrap_pymodule!(logging::logging))?;

    let sys = PyModule::import(py, "sys")?;
    let sys_modules: &PyDict = sys.getattr("modules")?.downcast()?;

    sys_modules.set_item("savant_rs.primitives", m.getattr("primitives")?)?;
    sys_modules.set_item("savant_rs.pipeline", m.getattr("pipeline")?)?;

    sys_modules.set_item("savant_rs.primitives.geometry", m.getattr("geometry")?)?;
    sys_modules.set_item("savant_rs.draw_spec", m.getattr("draw_spec")?)?;
    sys_modules.set_item("savant_rs.utils", m.getattr("utils")?)?;
    sys_modules.set_item("savant_rs.logging", m.getattr("logging")?)?;

    sys_modules.set_item(
        "savant_rs.utils.symbol_mapper",
        m.getattr("symbol_mapper_module")?,
    )?;

    sys_modules.set_item("savant_rs.utils.udf_api", m.getattr("udf_api_module")?)?;

    sys_modules.set_item(
        "savant_rs.utils.serialization",
        m.getattr("serialization_module")?,
    )?;

    sys_modules.set_item(
        "savant_rs.video_object_query",
        m.getattr("video_object_query")?,
    )?;

    Ok(())
}
