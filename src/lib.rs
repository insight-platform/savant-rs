pub mod primitives;

use pyo3::prelude::*;

#[pymodule]
fn savant_primitives(_py: Python, m: &PyModule) -> PyResult<()> {
    let _ = env_logger::try_init();
    m.add_class::<primitives::Point>()?;
    m.add_class::<primitives::PolygonalArea>()?;
    Ok(())
}
