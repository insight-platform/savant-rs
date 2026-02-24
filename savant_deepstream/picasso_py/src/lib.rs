use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_native")]
fn picasso_rs(_: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
