/// # C API for Savant Rust Library
///
pub mod capi;
pub mod cplugin;
/// # Basic objects
///
pub mod primitives;
pub mod test;
/// # Utility functions
///
pub mod utils;

use lazy_static::lazy_static;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pymodule;

use primitives::message::video::query::py::video_object_query;

lazy_static! {
    static ref VERSION_CRC32: u32 = crc32fast::hash(env!("CARGO_PKG_VERSION").as_bytes());
}

/// Returns the version of the package set in Cargo.toml
///
#[pyfunction]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

/// Returns version in CRC32 format
///
#[pyfunction]
pub fn version_crc32() -> u32 {
    *VERSION_CRC32
}

pub fn version_to_bytes_le() -> [u8; 4] {
    VERSION_CRC32.to_le_bytes()
}

pub fn bytes_le_to_version(bytes: [u8; 4]) -> u32 {
    u32::from_le_bytes(bytes)
}

#[pymodule]
fn savant_rs(py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(version_crc32, m)?)?;

    m.add_wrapped(wrap_pymodule!(primitives::primitives))?;
    m.add_wrapped(wrap_pymodule!(primitives::geometry))?;
    m.add_wrapped(wrap_pymodule!(primitives::draw_spec))?;
    m.add_wrapped(wrap_pymodule!(utils::utils))?;
    m.add_wrapped(wrap_pymodule!(utils::symbol_mapper_module))?;
    m.add_wrapped(wrap_pymodule!(utils::udf_api_module))?;
    m.add_wrapped(wrap_pymodule!(utils::numpy_module))?;
    m.add_wrapped(wrap_pymodule!(utils::serialization_module))?;
    m.add_wrapped(wrap_pymodule!(video_object_query))?;

    let sys = PyModule::import(py, "sys")?;
    let sys_modules: &PyDict = sys.getattr("modules")?.downcast()?;

    sys_modules.set_item("savant_rs.primitives", m.getattr("primitives")?)?;
    sys_modules.set_item("savant_rs.primitives.geometry", m.getattr("geometry")?)?;
    sys_modules.set_item("savant_rs.draw_spec", m.getattr("draw_spec")?)?;
    sys_modules.set_item("savant_rs.utils", m.getattr("utils")?)?;

    sys_modules.set_item(
        "savant_rs.utils.symbol_mapper",
        m.getattr("symbol_mapper_module")?,
    )?;

    sys_modules.set_item("savant_rs.utils.udf_api", m.getattr("udf_api_module")?)?;
    sys_modules.set_item("savant_rs.utils.numpy", m.getattr("numpy_module")?)?;

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
