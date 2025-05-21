use pyo3::{exceptions::PyRuntimeError, prelude::*};
use savant_core::primitives::rust;

use crate::err_to_pyo3;

use super::VideoObject;

#[pyclass]
#[derive(Debug, Clone)]
pub struct VideoObjectTree(pub(crate) rust::VideoObjectTree);

#[pymethods]
impl VideoObjectTree {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn walk_objects(&self, callable: &Bound<'_, PyAny>) -> PyResult<()> {
        let callable = |object: &rust::VideoObject,
                        parent: Option<&rust::VideoObject>,
                        result: Option<&Py<PyAny>>|
         -> anyhow::Result<Py<PyAny>> {
            let current_object = VideoObject(object.clone());
            let parent_object = parent.map(|p| VideoObject(p.clone()));
            let result = callable.call1((current_object, parent_object, result))?;
            let result = result.unbind();
            Ok(result)
        };
        err_to_pyo3!(self.0.walk_objects(callable), PyRuntimeError)
    }
}
