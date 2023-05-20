pub mod primitives;
pub mod test;
pub mod tests_pyo3_access;
pub mod utils;

use crate::tests_pyo3_access::{
    CopyWrapper, Internal, InternalMtx, InternalNoClone, ProxyWrapper, TakeWrapper, Wrapper,
};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pymodule;

use primitives::message::video::object::query::py::video_object_query;

#[pymodule]
fn savant_rs(py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    m.add_class::<Internal>()?;
    m.add_class::<InternalNoClone>()?;
    m.add_class::<InternalMtx>()?;
    m.add_class::<Wrapper>()?;
    m.add_class::<CopyWrapper>()?;
    m.add_class::<TakeWrapper>()?;
    m.add_class::<ProxyWrapper>()?;

    m.add_wrapped(wrap_pymodule!(primitives::primitives))?;
    m.add_wrapped(wrap_pymodule!(utils::utils))?;
    m.add_wrapped(wrap_pymodule!(video_object_query))?;

    let sys = PyModule::import(py, "sys")?;
    let sys_modules: &PyDict = sys.getattr("modules")?.downcast()?;

    sys_modules.set_item("savant_rs.primitives", m.getattr("primitives")?)?;
    sys_modules.set_item("savant_rs.utils", m.getattr("utils")?)?;
    sys_modules.set_item(
        "savant_rs.video_object_query",
        m.getattr("video_object_query")?,
    )?;

    Ok(())
}
