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
#[cfg(feature = "gst")]
pub mod retina_rtsp;
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

// NOTE: `Py<PyAny>` values live in process-global state outside of any particular `Python<'py>`
// token. PyO3 makes `Py<T>` `Send`; `Drop` may acquire the GIL to decrement the CPython refcount.
//
// SAFETY / threading contract:
// - `register_handler` and `unregister_handler` are `#[pyfunction]`s: they run with the GIL held,
//   so refcount updates and drops of values removed from the map happen on the GIL-held path.
// - Rust code may `read()` the map and clone `Py<PyAny>` for use while holding the GIL (typical
//   pattern: acquire GIL, read handler, call into Python). It must **not** `remove` / `clear` /
//   drop map-owned `Py<PyAny>` on a thread that does not hold the GIL while other code might hold
//   the same `RwLock` and wait on the GIL — that ordering can deadlock.
// - Prefer [`clear_all_handlers`] from Python (e.g. `atexit`) or with the GIL held in Rust tests.
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
    let removed = handlers.remove(name);
    if removed.is_none() {
        return Err(PyValueError::new_err(format!(
            "Handler with name {name} not found"
        )));
    }
    // `removed` (`Option` inner `Py<PyAny>`) is dropped here with the GIL still held — correct
    // for CPython refcount decrements.
    Ok(())
}

/// Clears every registered handler entry, dropping all stored `Py<PyAny>` values.
///
/// Call with the **GIL held** when possible (this function is safe to invoke without it — PyO3 will
/// attach — but dropping many handlers off-GIL adds latency and the same lock-ordering caveats as
/// in the `REGISTERED_HANDLERS` documentation apply).
pub fn clear_all_handlers() {
    log::info!("Clearing all registered Python handlers");
    let mut handlers = REGISTERED_HANDLERS.write();
    handlers.clear();
    log::info!("All registered Python handlers cleared");
}

#[pyfunction(name = "clear_all_handlers")]
pub fn clear_all_handlers_py() -> PyResult<()> {
    clear_all_handlers();
    Ok(())
}
