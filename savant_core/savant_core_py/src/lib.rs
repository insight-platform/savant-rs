pub mod atomic_counter;
pub mod capi;
#[cfg(feature = "deepstream")]
pub mod deepstream;
/// The draw specification used to draw objects on the frame when they are visualized.
pub mod draw_spec;
pub mod gst;
#[cfg(feature = "gst")]
pub mod gstreamer;
pub mod logging;
pub mod match_query;
pub mod metrics;
#[cfg(feature = "deepstream")]
pub mod nvinfer;
#[cfg(feature = "deepstream")]
pub mod nvtracker;
#[cfg(feature = "deepstream")]
pub mod picasso;
pub mod pipeline;
/// # Basic objects
///
pub mod primitives;
pub mod telemetry;
pub mod test;
/// # Utility functions
///
pub mod utils;
pub mod webserver;
pub mod zmq;

use pyo3::prelude::*;

use hashbrown::HashMap;
use lazy_static::lazy_static;
use parking_lot::RwLock;
use pyo3::exceptions::PyValueError;

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

/// Returns ``True`` when the library was compiled in release mode,
/// ``False`` when compiled in debug mode.
///
/// Returns
/// -------
/// bool
///   ``True`` for a release build, ``False`` for a debug build.
///
#[pyfunction]
pub fn is_release_build() -> bool {
    !cfg!(debug_assertions)
}

lazy_static! {
    pub static ref REGISTERED_HANDLERS: RwLock<HashMap<String, Py<PyAny>>> =
        RwLock::new(HashMap::new());
}

#[pyfunction]
pub fn register_handler(name: &str, handler: Bound<'_, PyAny>) -> PyResult<()> {
    let mut handlers = REGISTERED_HANDLERS.write();
    let unbound = handler.unbind();
    handlers.insert(name.to_string(), unbound);
    Ok(())
}

#[pyfunction]
pub fn unregister_handler(name: &str) -> PyResult<()> {
    let mut handlers = REGISTERED_HANDLERS.write();
    let res = handlers.remove(name);
    if res.is_none() {
        return Err(PyValueError::new_err(format!(
            "Handler with name {name} not found"
        )));
    }
    Ok(())
}
