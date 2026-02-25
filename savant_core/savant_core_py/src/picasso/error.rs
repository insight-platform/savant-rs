use picasso::prelude::PicassoError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;

/// Convert a [`PicassoError`] into a Python exception.
pub fn to_py_err(e: PicassoError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}
