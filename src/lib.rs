pub mod primitives;
pub mod test;
pub mod tests_pyo3_access;
pub mod utils;

use crate::tests_pyo3_access::{
    CopyWrapper, Internal, InternalMtx, InternalNoClone, ProxyWrapper, TakeWrapper, Wrapper,
};

use primitives::*;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pymodule;

#[pymodule]
fn savant_rs(py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<Point>()?;
    m.add_class::<Segment>()?;
    m.add_class::<IntersectionKind>()?;
    m.add_class::<Intersection>()?;
    m.add_class::<PolygonalArea>()?;
    m.add_class::<BBox>()?;
    m.add_class::<Attribute>()?;
    m.add_class::<Value>()?;
    m.add_class::<Object>()?;
    m.add_class::<ParentObject>()?;
    m.add_class::<ProxyVideoFrame>()?;
    m.add_class::<EndOfStream>()?;
    m.add_class::<Message>()?;
    m.add_class::<PyVideoFrameContent>()?;

    m.add_class::<Internal>()?;
    m.add_class::<InternalNoClone>()?;
    m.add_class::<InternalMtx>()?;
    m.add_class::<Wrapper>()?;
    m.add_class::<CopyWrapper>()?;
    m.add_class::<TakeWrapper>()?;
    m.add_class::<ProxyWrapper>()?;

    m.add_wrapped(wrap_pymodule!(utils::utils))?;
    let sys = PyModule::import(py, "sys")?;
    let sys_modules: &PyDict = sys.getattr("modules")?.downcast()?;
    sys_modules.set_item("savant_rs.utils", m.getattr("utils")?)?;

    Ok(())
}
