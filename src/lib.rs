pub mod primitives;
pub mod tests_py_access;

use crate::tests_py_access::{
    CopyWrapper, Internal, InternalNoClone, ProxyWrapper, TakeWrapper, Wrapper,
};
use primitives::{Intersection, IntersectionKind, Point, PolygonalArea, Segment};
use pyo3::prelude::*;

#[pymodule]
fn savant_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    // let _ = env_logger::try_init();
    m.add_class::<Point>()?;
    m.add_class::<Segment>()?;
    m.add_class::<IntersectionKind>()?;
    m.add_class::<Intersection>()?;
    m.add_class::<PolygonalArea>()?;
    m.add_class::<Internal>()?;
    m.add_class::<InternalNoClone>()?;
    m.add_class::<Wrapper>()?;
    m.add_class::<CopyWrapper>()?;
    m.add_class::<TakeWrapper>()?;
    m.add_class::<ProxyWrapper>()?;
    Ok(())
}
