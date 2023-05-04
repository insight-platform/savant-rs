pub mod primitives;

use primitives::point::Point;
use primitives::polygonal_area::PolygonalArea;
use pyo3::prelude::*;

#[pymodule]
fn savant_primitives(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    // let _ = env_logger::try_init();
    m.add_class::<Point>()?;
    m.add_class::<PolygonalArea>()?;
    Ok(())
}
