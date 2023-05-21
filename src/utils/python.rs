use pyo3::marker::Ungil;
use pyo3::prelude::*;
#[inline(always)]
pub fn no_gil<T, F>(f: F) -> T
where
    F: Ungil + FnOnce() -> T,
    T: Ungil,
{
    Python::with_gil(|py| py.allow_threads(f))
}
