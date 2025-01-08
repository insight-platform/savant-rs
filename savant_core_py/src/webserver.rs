use pyo3::exceptions::PySystemError;
use pyo3::prelude::*;

/// Starts embedded webserver providing status, shutdown and metrics features.
///
/// Parameters
/// ----------
/// port : int
///
#[pyfunction]
pub fn init_webserver(port: u16) -> PyResult<()> {
    savant_core::webserver::init_webserver(port)
        .map_err(|e| PySystemError::new_err(e.to_string()))?;
    Ok(())
}

/// Stops the embedded webserver.
///
#[pyfunction]
pub fn stop_webserver() -> PyResult<()> {
    savant_core::webserver::stop_webserver();
    Ok(())
}

/// Sets the token to be used to shut down the webserver.
///
/// Parameters
/// ----------
/// token : str
///
#[pyfunction]
pub fn set_shutdown_token(token: String) {
    savant_core::webserver::set_shutdown_token(token);
}

/// Returns the status of the webserver.
///
/// Returns
/// -------
/// bool
///   True if the webserver installed shutdown status, False otherwise.
///
#[pyfunction]
pub fn is_shutdown_set() -> bool {
    savant_core::webserver::is_shutdown_set()
}
