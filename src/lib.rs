pub mod primitives;
pub mod tests_pyo3_access;

use crate::tests_pyo3_access::{
    CopyWrapper, Internal, InternalMtx, InternalNoClone, ProxyWrapper, TakeWrapper, Wrapper,
};

use primitives::*;
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
    m.add_class::<BBox>()?;
    m.add_class::<Attribute>()?;
    m.add_class::<Value>()?;
    m.add_class::<Object>()?;
    m.add_class::<ProxyObject>()?;
    m.add_class::<ParentObject>()?;
    m.add_class::<VideoFrame>()?;
    m.add_class::<EndOfStream>()?;

    m.add_class::<Internal>()?;
    m.add_class::<InternalNoClone>()?;
    m.add_class::<InternalMtx>()?;
    m.add_class::<Wrapper>()?;
    m.add_class::<CopyWrapper>()?;
    m.add_class::<TakeWrapper>()?;
    m.add_class::<ProxyWrapper>()?;
    Ok(())
}
