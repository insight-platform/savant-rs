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

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pymodule;

use crate::logging::{set_log_level, LogLevel};

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

/// Initializes Noop tracer.
///
/// This is useful when the telemetry is not required.
///
#[pyfunction]
fn init_noop_tracer() {
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

#[pymodule]
fn savant_rs(py: Python, m: &PyModule) -> PyResult<()> {
    let log_env_var_name = "LOGLEVEL";
    let log_env_var_level = "trace";
    if std::env::var(log_env_var_name).is_err() {
        std::env::set_var(log_env_var_name, log_env_var_level);
    }
    pretty_env_logger::try_init_custom_env(log_env_var_name)
        .map_err(|_| PyRuntimeError::new_err("Failed to initialize logger"))?;
    set_log_level(LogLevel::Error);

    m.add_function(wrap_pyfunction!(init_jaeger_tracer, m)?)?; // PYI
    m.add_function(wrap_pyfunction!(init_noop_tracer, m)?)?; // PYI
    m.add_function(wrap_pyfunction!(version, m)?)?; // PYI

    m.add_wrapped(wrap_pymodule!(primitives::primitives))?;
    m.add_wrapped(wrap_pymodule!(pipeline::pipeline))?;
    m.add_wrapped(wrap_pymodule!(primitives::geometry))?;
    m.add_wrapped(wrap_pymodule!(draw_spec::draw_spec))?; // PYI
    m.add_wrapped(wrap_pymodule!(utils::utils))?; // PYI
    m.add_wrapped(wrap_pymodule!(utils::symbol_mapper_module))?;
    m.add_wrapped(wrap_pymodule!(utils::udf_api_module))?;
    m.add_wrapped(wrap_pymodule!(utils::serialization_module))?;
    m.add_wrapped(wrap_pymodule!(match_query::match_query))?;
    m.add_wrapped(wrap_pymodule!(logging::logging))?; // PYI
    m.add_wrapped(wrap_pymodule!(zmq::zmq))?; // PYI

    let sys = PyModule::import(py, "sys")?;
    let sys_modules: &PyDict = sys.getattr("modules")?.downcast()?;

    sys_modules.set_item("savant_rs.primitives", m.getattr("primitives")?)?;
    sys_modules.set_item("savant_rs.pipeline", m.getattr("pipeline")?)?;
    sys_modules.set_item("savant_rs.pipeline2", m.getattr("pipeline")?)?;

    sys_modules.set_item("savant_rs.primitives.geometry", m.getattr("geometry")?)?;
    sys_modules.set_item("savant_rs.draw_spec", m.getattr("draw_spec")?)?;
    sys_modules.set_item("savant_rs.utils", m.getattr("utils")?)?;
    sys_modules.set_item("savant_rs.logging", m.getattr("logging")?)?;
    sys_modules.set_item("savant_rs.zmq", m.getattr("zmq")?)?;

    sys_modules.set_item(
        "savant_rs.utils.symbol_mapper",
        m.getattr("symbol_mapper_module")?,
    )?;

    sys_modules.set_item("savant_rs.utils.udf_api", m.getattr("udf_api_module")?)?;

    sys_modules.set_item(
        "savant_rs.utils.serialization",
        m.getattr("serialization_module")?,
    )?;

    sys_modules.set_item("savant_rs.match_query", m.getattr("match_query")?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {}
}
